import glob
import io
import logging
import os

import numpy as np
import png  # From the 'pypng' library
import tensorflow as tf
import tensorflow_text as text
import torch
from PIL import Image

# --- Suppress TF warnings ---
tf.get_logger().setLevel(logging.ERROR)


# --- 1. Your Custom Helper Function ---
def png_to_tfexample(image_array: np.ndarray) -> tf.train.Example:
    """Creates a tf.train.Example from a NumPy array."""
    # Convert the image to float32 and shift the minimum value to zero
    image = image_array.astype(np.float32)
    image -= image.min()

    if image_array.dtype == np.uint8:
        # For uint8 images, no rescaling is needed
        pixel_array = image.astype(np.uint8)
        bitdepth = 8
    else:
        # For other data types, scale image to use the full 16-bit range
        max_val = image.max()
        if max_val > 0:
            image *= 65535 / max_val  # Scale to 16-bit range
        pixel_array = image.astype(np.uint16)
        bitdepth = 16

    # Ensure the array is 2-D (grayscale image)
    if pixel_array.ndim != 2:
        raise ValueError(f"Array must be 2-D. Actual dimensions: {pixel_array.ndim}")

    # Encode the array as a PNG image
    output = io.BytesIO()
    png.Writer(
        width=pixel_array.shape[1], height=pixel_array.shape[0], greyscale=True, bitdepth=bitdepth
    ).write(output, pixel_array.tolist())
    png_bytes = output.getvalue()

    # Create a tf.train.Example and assign the features
    example = tf.train.Example()
    features = example.features.feature
    features["image/encoded"].bytes_list.value.append(png_bytes)
    features["image/format"].bytes_list.value.append(b"png")

    return example

# huggingface-cli download google/cxr-foundation --local-dir cxr-foundation

# --- 2. Setup and Load TensorFlow Models ---
LOCAL_DIR = "./cxr-foundation/"  # Ensure this points to where you downloaded the models

print("Loading TensorFlow models...")
elixrc_model = tf.saved_model.load(os.path.join(LOCAL_DIR, "elixr-c-v2-pooled"))
elixrc_infer = elixrc_model.signatures["serving_default"]

qformer_model = tf.saved_model.load(os.path.join(LOCAL_DIR, "pax-elixr-b-text"))
qformer_infer = qformer_model.signatures["serving_default"]


# --- 3. Setup Input and Output ---
image_folder = "./chest_xrays"  # Change this to your directory containing the X-rays
image_paths = glob.glob(os.path.join(image_folder, "*.png"))

extracted_features_dict = {}

print(f"Found {len(image_paths)} images. Starting extraction...")


# --- 4. Extraction Loop ---
for path in image_paths:
    filename = os.path.basename(path)

    # Load and prep image (convert to Grayscale just in case)
    img = Image.open(path).convert("L")
    img_array = np.array(img)

    # Use your custom function to serialize
    serialized_img_tf_example = png_to_tfexample(img_array).SerializeToString()

    # ELIXR-C Pass (Image to ELIXR-C embeddings)
    elixrc_output = elixrc_infer(input_example=tf.constant([serialized_img_tf_example]))
    elixrc_embedding = elixrc_output["feature_maps_0"]

    # ELIXR-B Pass (Q-Former)
    qformer_input = {
        "image_feature": elixrc_embedding,
        "ids": tf.constant(np.zeros((1, 1, 128), dtype=np.int32)),
        "paddings": tf.constant(np.zeros((1, 1, 128), dtype=np.float32)),
    }
    qformer_output = qformer_infer(**qformer_input)

    # Extract tensor and convert TF -> NumPy -> PyTorch
    embedding_np = qformer_output["all_contrastive_img_emb"].numpy()
    embedding_pt = torch.from_numpy(embedding_np)

    # Store in dictionary
    extracted_features_dict[filename] = embedding_pt
    print(f"Processed: {filename}")


# --- 5. Save for PyTorch ---
output_file = "elixr_b_features.pt"
torch.save(extracted_features_dict, output_file)
print(f"Successfully saved all features to '{output_file}'!")
