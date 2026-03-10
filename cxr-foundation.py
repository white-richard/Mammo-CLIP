from __future__ import annotations

"""
CXR Foundation embedding extraction script.
Uses Google's ELIXR-C + ELIXR-B (QFormer) models from HuggingFace to produce
image embeddings from a chest X-ray PNG.

Requirements (install separately, not in pyproject.toml):
    pip install tensorflow tensorflow-io huggingface_hub matplotlib
"""

import argparse
import logging
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import get_token, login, snapshot_download
from PIL import Image

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def ensure_hf_login() -> None:
    """Prompt for HuggingFace login if no token is cached."""
    if get_token() is None:
        print("No HuggingFace token found. Please log in.")
        login()


# ---------------------------------------------------------------------------
# TF / model helpers  (imported lazily so the script fails fast with a clear
# message if tensorflow is not installed)
# ---------------------------------------------------------------------------


def _import_tf():
    try:
        import tensorflow as tf

        tf.get_logger().setLevel(logging.ERROR)

        # Allow TF to allocate GPU memory incrementally instead of
        # grabbing all available VRAM up front (which causes OOM).
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        return tf
    except ImportError:
        raise ImportError(
            "TensorFlow is required but not installed.\nInstall it with:  pip install tensorflow tensorflow-text"
        )


def _import_tf_text():
    try:
        import tensorflow_text  # noqa: F401 — registers SentencepieceOp and other custom ops
    except ImportError as e:
        if "tensorflow_text" in str(e):
            raise ImportError(
                "tensorflow-text is required but not installed.\n"
                "Install it with:  pip install tensorflow-text\n"
                "It must match your TensorFlow version exactly."
            ) from e
        raise


def png_to_tfexample(image_array: np.ndarray):
    """Convert a grayscale numpy array to a tf.train.Example proto."""
    tf = _import_tf()

    image_uint8 = image_array.astype(np.uint8)
    encoded = tf.image.encode_png(image_uint8[..., np.newaxis]).numpy()

    feature = {
        "image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded])),
        "image/format": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"png"])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def download_models(local_dir: str) -> None:
    """Download ELIXR-C and ELIXR-B model files from HuggingFace Hub."""
    print(f"Downloading models to '{local_dir}' …")
    snapshot_download(
        repo_id="google/cxr-foundation",
        local_dir=local_dir,
        allow_patterns=["elixr-c-v2-pooled/*", "pax-elixr-b-text/*"],
    )
    print("Download complete.")


def load_image(image_path: str) -> np.ndarray:
    """Open a PNG/JPEG and return a grayscale numpy array."""
    img = Image.open(image_path).convert("L")  # grayscale
    return np.array(img)


def run_elixrc(image_array: np.ndarray, model_dir: str):
    """Run ELIXR-C to produce interim image embeddings."""
    _import_tf_text()
    tf = _import_tf()

    serialized = png_to_tfexample(image_array).SerializeToString()

    model = tf.saved_model.load(os.path.join(model_dir, "elixr-c-v2-pooled"))
    infer = model.signatures["serving_default"]

    output = infer(input_example=tf.constant([serialized]))
    embedding = output["feature_maps_0"].numpy()
    print("ELIXR-C interim embedding shape:", embedding.shape)
    return embedding


def run_elixrb(elixrc_embedding: np.ndarray, model_dir: str):
    """Run ELIXR-B QFormer to produce final contrastive image embeddings."""
    tf = _import_tf()

    model = tf.saved_model.load(os.path.join(model_dir, "pax-elixr-b-text"))
    infer = model.signatures["serving_default"]

    qformer_input = {
        "image_feature": elixrc_embedding.tolist(),
        "ids": np.zeros((1, 1, 128), dtype=np.int32).tolist(),
        "paddings": np.zeros((1, 1, 128), dtype=np.float32).tolist(),
    }

    output = infer(**qformer_input)
    embeddings = output["all_contrastive_img_emb"]
    print("ELIXR-B embedding shape:", embeddings.shape)
    return embeddings


def visualize_embedding(embeddings, save_path=None):
    """Plot the ELIXR-B embedding matrix."""
    plt.figure(figsize=(12, 4))
    plt.imshow(embeddings[0], cmap="gray", aspect="auto")
    plt.colorbar()
    plt.title("Visualization of ELIXR-B embedding output")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Embedding plot saved to '{save_path}'.")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CXR Foundation embeddings from a chest X-ray image."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="Chest_Xray_PA_3-8-2010.png",
        help="Path to the input chest X-ray image (default: Chest_Xray_PA_3-8-2010.png)",
    )
    parser.add_argument(
        "--model-dir",
        default="./hf",
        help="Directory to download / load model files from (default: ./hf)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip model download (assumes models are already present in --model-dir)",
    )
    parser.add_argument(
        "--save-plot",
        metavar="PATH",
        default="tmp/embedding_plot.png",
        help="Save the embedding visualisation to this file instead of displaying it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    
    pathlib.Path(args.save_plot).parent.mkdir(parents=True, exist_ok=True)
    ensure_hf_login()

    if not args.skip_download:
        download_models(args.model_dir)

    print(f"Loading image: {args.image}")
    image_array = load_image(args.image)
    print(f"Image shape (grayscale): {image_array.shape}")

    elixrc_embedding = run_elixrc(image_array, args.model_dir)
    elixrb_embeddings = run_elixrb(elixrc_embedding, args.model_dir)

    visualize_embedding(elixrb_embeddings, save_path=args.save_plot)


if __name__ == "__main__":
    main()
