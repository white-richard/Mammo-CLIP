"""
Extract vision features from MammoCLIP or CXR-Foundation and save them for
downstream use.

Features are saved as a .pt file containing:
    {
        "features":   Float tensor of shape (N, D),
        "img_paths":  list[str]    of length N,
        "metadata":   pd.DataFrame of shape (N, C)  -- all CSV columns,
        "feature_dim": int,
        "dataset":    str,
        "arch":       str,
        "label_col":  str          -- name of the label column in metadata,
        "split":      str,
    }

Labels can be recovered via: metadata[label_col].values

Usage — MammoCLIP (default)
-----
python ./src/codebase/extract_features.py \
  --model mammoclip \
  --data-dir "$HOME/.code/datasets/vindr-mammo" \
  --img-dir "images_png" \
  --csv-file "vindr_detection_v1_folds.csv" \
  --clip_chk_pt_path "$HOME/.code/model_weights/mammoclip/mammoclip-b5-model-best-epoch-7.tar" \
  --dataset "ViNDr" \
  --split "all" \
  --arch "upmc_breast_clip_det_b5_period_n_lp" \
  --output-file "features/vindr_abnormal_features.pt" \
  --batch-size 16 \
  --num-workers 4

Usage — CXR-Foundation
-----
python ./src/codebase/extract_features.py \
  --model cxr-foundation \
  --data-dir "$HOME/.code/datasets/vindr-mammo" \
  --img-dir "images_png" \
  --csv-file "vindr_detection_v1_folds.csv" \
  --dataset "ViNDr" \
  --split "all" \
  --output-file "features/cxr_elixr_features.pt" \
  --batch-size 16 \
  --num-workers 4

Notes:
- The CXR-Foundation backend uses TensorFlow saved_model artifacts which are loaded
  at runtime. The wrapper converts images to TF Examples, runs the TF models,
  converts outputs to NumPy and then to PyTorch tensors so the rest of the
  extraction pipeline (dataloader, saving .pt payload) remains unchanged.

"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow_text as text
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

text = None
# torch.multiprocessing.set_sharing_strategy('file_system')

sys.path.insert(0, str(Path(__file__).parent))

from Classifiers.models.breast_clip_classifier import BreastClipClassifier
from utils import seed_all


class MammoFeatureDataset(Dataset):
    """
    Generic mammography dataset for feature extraction (MammoCLIP models).

    Supports VinDr and RSNA conventions out of the box:
        VinDr: <data_dir>/<img_dir>/<patient_id>/<image_id>          (no extension added)
        RSNA:  <data_dir>/<img_dir>/<patient_id>/<image_id>.png

    For other datasets pass ``custom_path_fn`` (not available via CLI; use
    this module as a library instead).
    """

    ARCH_USE_PIL = {
        "upmc_breast_clip_det_b5_period_n_ft",
        "upmc_vindr_breast_clip_det_b5_period_n_ft",
        "upmc_breast_clip_det_b5_period_n_lp",
        "upmc_vindr_breast_clip_det_b5_period_n_lp",
        "upmc_breast_clip_det_b2_period_n_ft",
        "upmc_vindr_breast_clip_det_b2_period_n_ft",
        "upmc_breast_clip_det_b2_period_n_lp",
        "upmc_vindr_breast_clip_det_b2_period_n_lp",
    }

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        img_dir: str,
        dataset: str,
        arch: str,
        mean: float = 0.3089279,
        std: float = 0.25053555408335154,
    ):
        self.df = df.reset_index(drop=True)
        self.dir_path = data_dir / img_dir
        self.dataset = dataset.lower()
        self.arch = arch.lower()
        self.mean = mean
        self.std = std
        self.use_pil = self.arch in {a.lower() for a in self.ARCH_USE_PIL}

    def __len__(self):
        return len(self.df)

    def _build_path(self, row) -> Path:
        patient_id = str(row["patient_id"])
        image_id = str(row["image_id"])

        if self.dataset == "rsna":
            # RSNA images are stored as .png but the csv omits the extension
            if not image_id.endswith(".png"):
                image_id = image_id + ".png"
            return self.dir_path / patient_id / image_id

        elif self.dataset == "vindr":
            # VinDr images already include .png in the csv
            if not image_id.endswith(".png"):
                image_id = image_id + ".png"
            return self.dir_path / patient_id / image_id

        else:
            # Generic fallback: try as-is, then with .png appended
            p = self.dir_path / patient_id / image_id
            if p.exists():
                return p
            return Path(str(p) + ".png")

    def _load_and_preprocess(self, img_path: Path) -> torch.Tensor:
        # if self.use_pil:
        #     img = Image.open(img_path).convert("RGB")
        #     img = np.array(img, dtype=np.float32)
        if self.use_pil:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                img = np.array(img_rgb, dtype=np.float32)
        else:
            import cv2

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32)

        # Normalise to [0, 1] then standardise
        img -= img.min()
        if img.max() > 0:
            img /= img.max()
        img = (img - self.mean) / self.std
        tensor = torch.tensor(img, dtype=torch.float32)

        if self.use_pil:
            # RGB → (3, H, W) already float
            tensor = tensor.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        else:
            tensor = tensor.unsqueeze(0)  # (1, H, W)

        return tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._build_path(row)

        try:
            img = self._load_and_preprocess(img_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load image at {img_path}: {e}") from e

        # Pass all CSV columns through as a plain dict of Python scalars/strings
        row_meta = {col: row[col] for col in self.df.columns}

        return {
            "image": img,
            "img_path": str(img_path),
            "meta": row_meta,
        }


def collate_fn(batch):
    # Collate metadata: list of per-column lists
    meta_keys = batch[0]["meta"].keys()
    meta = {k: [b["meta"][k] for b in batch] for k in meta_keys}

    return {
        "image": torch.stack([b["image"] for b in batch]),
        "img_path": [b["img_path"] for b in batch],
        "meta": meta,
    }


def load_dataframe(data_dir: Path, csv_file: str, dataset: str, split: str) -> pd.DataFrame:
    """
    Load and (optionally) filter the CSV for a particular split.

    split can be:
        "all"       – return every row
        "train"     – training split only
        "test"      – test / validation split only
    """
    df = pd.read_csv(data_dir / csv_file).fillna(0)
    print(f"Full dataframe shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    if split == "all":
        return df

    dataset_lower = dataset.lower()

    if dataset_lower == "vindr":
        split_col = "split"
        split_map = {"train": "training", "test": "test"}
        wanted = split_map.get(split, split)
        if split_col in df.columns:
            df = df[df[split_col] == wanted].reset_index(drop=True)
        else:
            raise ValueError(f"Expected column '{split_col}' not found in VinDr CSV.")
            print(f"[warning] no '{split_col}' column found; returning all rows.")

    elif dataset_lower == "rsna":
        # RSNA uses numeric fold column; fold 0 is validation, 1/2 are training
        fold_col = "fold"
        if fold_col in df.columns:
            if split == "train":
                df = df[df[fold_col].isin([1, 2])].reset_index(drop=True)
            elif split == "test":
                df = df[df[fold_col] == 0].reset_index(drop=True)
        else:
            raise ValueError(f"Expected column '{fold_col}' not found in RSNA CSV.")
            print(f"[warning] no '{fold_col}' column found; returning all rows.")

    else:
        raise NotImplementedError(
            f"Dataset '{dataset}' not recognised for automatic split filtering. Implement this case"
        )
        # Generic: look for a common split column
        for col in ("split", "fold", "subset"):
            if col in df.columns:
                print(f"[info] filtering on column '{col}' == '{split}'")
                df = df[df[col] == split].reset_index(drop=True)
                break
        else:
            print("[warning] no split column found; returning all rows.")

    print(f"Filtered dataframe shape ({split}): {df.shape}")
    return df


def build_dataloader(df: pd.DataFrame, args) -> DataLoader:
    dataset = MammoFeatureDataset(
        df=df,
        data_dir=Path(args.data_dir),
        img_dir=args.img_dir,
        dataset=args.dataset,
        arch=args.arch,
        mean=args.mean,
        std=args.std,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _png_to_tfexample(image_array: np.ndarray):
    """
    Helper: create a tf.train.Example from a NumPy grayscale image.
    Implemented as a local helper to avoid importing TF globally at module import time.
    """
    import io as _io

    try:
        import png as _png
    except Exception as e:
        raise ImportError(
            "The 'pypng' package is required for CXR-Foundation preprocessing."
        ) from e

    import numpy as _np
    import tensorflow as _tf

    image = image_array.astype(_np.float32)
    image -= image.min()

    if image_array.dtype == _np.uint8:
        pixel_array = image.astype(_np.uint8)
        bitdepth = 8
    else:
        max_val = image.max()
        if max_val > 0:
            image *= 65535 / max_val
        pixel_array = image.astype(_np.uint16)
        bitdepth = 16

    if pixel_array.ndim != 2:
        raise ValueError(f"Array must be 2-D. Actual dimensions: {pixel_array.ndim}")

    output = _io.BytesIO()
    _png.Writer(
        width=pixel_array.shape[1], height=pixel_array.shape[0], greyscale=True, bitdepth=bitdepth
    ).write(output, pixel_array.tolist())
    png_bytes = output.getvalue()

    example = _tf.train.Example()
    features = example.features.feature
    features["image/encoded"].bytes_list.value.append(png_bytes)
    features["image/format"].bytes_list.value.append(b"png")
    return example


class CXRFoundationWrapper:
    """
    Minimal wrapper around the CXR-Foundation TensorFlow saved_models.

    Responsibilities:
    - Load TF saved_model artifacts from a local directory.
    - Provide a method `encode_from_paths(paths)` that accepts a list of image
      file paths, runs the TF models, and returns a torch.Tensor of shape (B, D).
    """

    def __init__(self, local_dir: str):
        # Local imports so the module can be used even if TF isn't installed
        import os as _os

        try:
            import tensorflow as _tf
        except Exception as e:
            raise ImportError("TensorFlow is required for the 'cxr-foundation' backend.") from e
        import numpy as _np

        self._tf = _tf
        self._np = _np
        self._local_dir = local_dir

        # Load saved models (paths expected within local_dir)
        elixr_path = _os.path.join(local_dir, "elixr-c-v2-pooled")
        qformer_path = _os.path.join(local_dir, "pax-elixr-b-text")
        print(f"Loading CXR-Foundation ELIXR model from: {elixr_path}")
        self._elixrc_model = _tf.saved_model.load(elixr_path)
        self._elixrc_infer = self._elixrc_model.signatures["serving_default"]
        print(f"Loading CXR-Foundation Q-Former model from: {qformer_path}")
        self._qformer_model = _tf.saved_model.load(qformer_path)
        self._qformer_infer = self._qformer_model.signatures["serving_default"]

    def encode_from_paths(
        self, paths: list[str], device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """
        Given a list of image file paths, run the TF models in a batch
        and return a torch Tensor of shape (B, D).
        """
        from PIL import Image as _Image
        # Assuming torch is imported at the top of your file

        tf = self._tf
        np = self._np

        if not paths:
            return torch.empty((0, 0), dtype=torch.float32)

        batch_size = len(paths)
        serialized_examples = []

        # 1. Prepare the batch of serialized TF Examples
        for p in paths:
            with _Image.open(p) as img:
                img_gray = img.convert("L")
                arr = np.array(img_gray)
            # Assuming _png_to_tfexample is defined elsewhere in your file
            ex = _png_to_tfexample(arr)
            serialized_examples.append(ex.SerializeToString())

        # Create a single TF tensor containing the whole batch of strings
        # Shape will be (B,)
        serialized_tf = tf.constant(serialized_examples)

        # 2. Batched ELIXR-C inference
        elixr_output = self._elixrc_infer(input_example=serialized_tf)
        elixr_embedding = elixr_output["feature_maps_0"]

        # 3. Batched Q-Former inference
        # Adjust dummy inputs to match the actual batch size instead of hardcoding '1'
        ids = tf.constant(np.zeros((batch_size, 1, 128), dtype=np.int32))
        paddings = tf.constant(np.zeros((batch_size, 1, 128), dtype=np.float32))

        qformer_output = self._qformer_infer(
            image_feature=elixr_embedding, ids=ids, paddings=paddings
        )

        # 4. Process the output batch
        emb_np = qformer_output["all_contrastive_img_emb"].numpy()
        emb_np = np.asarray(emb_np, dtype=np.float32)

        # Handle possible shapes like (B, 1, D) or (B, D)
        if emb_np.ndim == 3:
            emb_np = emb_np.reshape(batch_size, -1)

        # Convert the whole batch to a PyTorch tensor at once
        emb_pt = torch.from_numpy(emb_np)

        return emb_pt


def load_model(clip_chk_pt_path: str, arch: str, device: torch.device) -> BreastClipClassifier:
    """
    Load the MammoCLIP image encoder via BreastClipClassifier.
    The linear classifier head is present but unused during extraction.
    """
    print(f"Loading checkpoint: {clip_chk_pt_path}")
    ckpt = torch.load(clip_chk_pt_path, map_location="cpu")

    # BreastClipClassifier expects an args-like object for arch / freeze logic
    class _Args:
        pass

    _args = _Args()
    _args.arch = arch

    model = BreastClipClassifier(_args, ckpt=ckpt, n_class=1)
    model = model.to(device)
    model.eval()
    print(f"Model loaded. Image encoder type: {model.get_image_encoder_type()}")
    print(f"Feature dimension: {model.image_encoder.out_dim}")
    return model


# ---------------------------------------------------------------------------
# Extraction loops
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_features(
    model: BreastClipClassifier, loader: DataLoader, device: torch.device, debug_mode: bool = False
):
    all_features = []
    all_paths = []
    all_meta: dict[str, list] = {}

    for i, batch in enumerate(tqdm(loader, desc="Extracting features", unit="batch")):
        paths = batch["img_path"]
        meta = batch["meta"]

        # If the model provides a TF-backed encoding entrypoint, use it.
        # CXRFoundation wrapper implements `encode_from_paths`.
        if hasattr(model, "encode_from_paths"):
            # encode_from_paths returns a torch.Tensor on CPU
            features = model.encode_from_paths(paths, device=device)
        else:
            images = batch["image"].to(device)
            # Handle swin encoder: needs (B, H, W, C)
            if model.get_image_encoder_type().lower() == "swin":
                images = images.squeeze(1).permute(0, 3, 1, 2)
            features = model.encode_image(images)  # (B, D)
            # Move to CPU for consistent saving
            features = features.cpu()

        # Ensure features are on CPU for concatenation / saving
        if features.device != torch.device("cpu"):
            features = features.cpu()

        all_features.append(features)
        all_paths.extend(paths)

        # Accumulate metadata columns
        for k, v in meta.items():
            all_meta.setdefault(k, []).extend(v)

        if debug_mode and i >= 2:
            print("[debug_mode] Stopping early after 3 batches.")
            break

    metadata_df = pd.DataFrame(all_meta)
    return torch.cat(all_features, dim=0), all_paths, metadata_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def config():
    parser = argparse.ArgumentParser(
        description="Extract vision features (MammoCLIP or CXR Foundation) and save them to disk."
    )
    parser.add_argument(
        "--model",
        default="mammoclip",
        choices=["mammoclip", "cxr-foundation"],
        help="Which model to use for feature extraction (default: mammoclip)",
    )
    parser.add_argument("--data-dir", required=True, type=str, help="Root directory of the dataset")
    parser.add_argument(
        "--img-dir",
        default="images_png",
        type=str,
        help="Sub-directory containing images (relative to --data-dir)",
    )
    parser.add_argument(
        "--csv-file",
        required=True,
        type=str,
        help="CSV file with patient_id, image_id, labels, split columns",
    )
    parser.add_argument(
        "--clip_chk_pt_path",
        default=None,
        type=str,
        help="Path to MammoCLIP checkpoint (.tar) — required when --model mammoclip",
    )
    parser.add_argument(
        "--cxr_dir",
        default="./cxr-foundation",
        type=str,
        help="Local directory containing CXR-Foundation saved_model artifacts (required when --model cxr-foundation)",
    )

    parser.add_argument(
        "--dataset", default="ViNDr", type=str, help="Dataset name: ViNDr | RSNA | <custom>"
    )
    parser.add_argument(
        "--split",
        default="all",
        type=str,
        choices=["all", "train", "test"],
        help="Which split to extract features for",
    )
    parser.add_argument(
        "--label", default="abnormal", type=str, help="Column name to use as the label"
    )
    parser.add_argument(
        "--arch",
        default="upmc_breast_clip_det_b5_period_n_lp",
        type=str,
        help="Architecture name (used to determine image loading strategy)",
    )
    parser.add_argument(
        "--output-file", default="features.pt", type=str, help="Output .pt file path"
    )
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument(
        "--mean", default=0.3089279, type=float, help="Per-image normalisation mean"
    )
    parser.add_argument(
        "--std", default=0.25053555408335154, type=float, help="Per-image normalisation std"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Stop after 3 batches — useful for quick end-to-end testing",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Compute device",
    )
    return parser.parse_args()


def main():
    args = config()
    seed_all(args.seed)

    # ---- Load & filter dataframe ----------------------------------------
    df = load_dataframe(
        data_dir=Path(args.data_dir),
        csv_file=args.csv_file,
        dataset=args.dataset,
        split=args.split,
    )

    # ---- Build dataloader -----------------------------------------------
    loader = build_dataloader(df, args)
    print(f"Number of samples : {len(loader.dataset)}")
    print(f"Number of batches : {len(loader)}")

    # ---- Load model & extract --------------------------------------------
    if args.model == "mammoclip":
        if not args.clip_chk_pt_path:
            raise ValueError("--clip_chk_pt_path is required when --model mammoclip")
        device = torch.device(
            args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
        )
        print(f"Using device: {device}")
        model = load_model(args.clip_chk_pt_path, args.arch, device)
        features, img_paths, metadata_df = extract_features(
            model, loader, device, debug_mode=args.debug_mode
        )
        arch_tag = args.arch

    elif args.model == "cxr-foundation":
        # CXR-Foundation is TF-based. We use a wrapper that executes TF saved_models,
        # converts outputs to NumPy and then to PyTorch tensors so the downstream code
        # (saving to .pt) remains identical.
        cxr_dir = args.cxr_dir
        print(f"Using CXR-Foundation saved_models at: {cxr_dir}")
        # For TF-backed model we run TF; Torch device here is unused since TF runs independently.
        device = torch.device("cpu")
        model = CXRFoundationWrapper(cxr_dir)
        features, img_paths, metadata_df = extract_features(
            model, loader, device, debug_mode=args.debug_mode
        )
        arch_tag = "cxr-foundation"

    print("\nExtraction complete.")
    print(f"  features shape  : {features.shape}")
    print(f"  metadata columns: {list(metadata_df.columns)}")
    if args.label in metadata_df.columns:
        label_vals = metadata_df[args.label]
        print(f"  label col       : {args.label!r}")
        print(f"  unique labels   : {sorted(label_vals.unique().tolist())}")
    else:
        print(f"  [warning] label column {args.label!r} not found in metadata")

    # ---- Save -------------------------------------------------------------
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "features": features,  # (N, D)  float32
        "img_paths": img_paths,  # list[str]
        "metadata": metadata_df,  # pd.DataFrame, all CSV columns, length N
        # Labels: metadata[label_col] — avoids baking in a single column early
        "feature_dim": features.shape[1],
        "dataset": args.dataset,
        "arch": arch_tag,
        "split": args.split,
        "label_col": args.label,
    }
    torch.save(payload, output_path)
    print(f"\nSaved features to: {output_path}")
    print(f"  feature_dim : {features.shape[1]}")
    print(f"  N samples   : {features.shape[0]}")


if __name__ == "__main__":
    main()
