"""
Extract vision features from MammoCLIP or CXR Foundation and save them for
downstream use.

Features are saved as a .pt file containing:
    {
        "features":   Float tensor of shape (N, D),
        "labels":     Long tensor  of shape (N,),
        "img_paths":  list[str]    of length N,
        "metadata":   pd.DataFrame of shape (N, C)  -- all CSV columns,
        "feature_dim": int,
        "dataset":    str,
        "arch":       str,
    }

Usage — MammoCLIP (default)
-----
python ./src/codebase/extract_features.py \
  --data-dir "$HOME/.code/datasets/vindr-mammo" \
  --img-dir "images_png" \
  --csv-file "vindr_detection_v1_folds_abnormal.csv" \
  --clip_chk_pt_path "$HOME/.code/model_weights/mammoclip/mammoclip-b5-model-best-epoch-7.tar" \
  --dataset "ViNDr" \
  --split "all" \
  --label "abnormal" \
  --arch "upmc_breast_clip_det_b5_period_n_lp" \
  --output-file "features/vindr_abnormal_features.pt" \
  --batch-size 16 \
  --num-workers 4

Usage — CXR Foundation (Google ELIXR-C + ELIXR-B)
-----
python ./src/codebase/extract_features.py \
  --model cxr-foundation \
  --data-dir "$HOME/.code/datasets/vindr-mammo" \
  --img-dir "images_png" \
  --csv-file "vindr_detection_v1_folds_abnormal.csv" \
  --cxr-foundation-dir "$HOME/.code/model_weights/cxr-foundation/hf" \
  --dataset "ViNDr" \
  --split "all" \
  --label "abnormal" \
  --output-file "features/vindr_abnormal_cxr_foundation_features.pt" \
  --num-workers 4

Requirements for CXR Foundation (install separately):
    pip install tensorflow tensorflow-text huggingface_hub
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from Classifiers.models.breast_clip_classifier import BreastClipClassifier
from utils import seed_all

# ---------------------------------------------------------------------------
# CXR Foundation helpers (TensorFlow-based; imported lazily)
# ---------------------------------------------------------------------------


def _import_tf():
    try:
        import tensorflow as tf

        tf.get_logger().setLevel(logging.ERROR)

        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        return tf
    except ImportError:
        raise ImportError(
            "TensorFlow is required for CXR Foundation but not installed.\n"
            "Install it with:  pip install tensorflow tensorflow-text"
        )


def _import_tf_text():
    try:
        import tensorflow_text  # noqa: F401 — registers custom ops
    except ImportError as e:
        if "tensorflow_text" in str(e):
            raise ImportError(
                "tensorflow-text is required but not installed.\n"
                "Install it with:  pip install tensorflow-text\n"
                "It must match your TensorFlow version exactly."
            ) from e
        raise


def _png_to_tfexample(image_array: np.ndarray):
    """Convert a grayscale uint8 numpy array to a serialised tf.train.Example."""
    tf = _import_tf()
    image_uint8 = image_array.astype(np.uint8)
    encoded = tf.image.encode_png(image_uint8[..., np.newaxis]).numpy()
    feature = {
        "image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded])),
        "image/format": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"png"])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _run_elixrc(image_array: np.ndarray, model_dir: str) -> np.ndarray:
    """Run ELIXR-C to produce interim image embeddings."""
    _import_tf_text()
    tf = _import_tf()
    serialized = _png_to_tfexample(image_array).SerializeToString()
    model = tf.saved_model.load(os.path.join(model_dir, "elixr-c-v2-pooled"))
    infer = model.signatures["serving_default"]
    output = infer(input_example=tf.constant([serialized]))
    return output["feature_maps_0"].numpy()


def _run_elixrb(elixrc_embedding: np.ndarray, model_dir: str) -> np.ndarray:
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
    # shape: (1, num_query_tokens, embed_dim) — flatten to (1, D) by mean-pooling
    embeddings = output["all_contrastive_img_emb"].numpy()
    return embeddings


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
        label: str,
        mean: float = 0.3089279,
        std: float = 0.25053555408335154,
    ):
        self.df = df.reset_index(drop=True)
        self.dir_path = data_dir / img_dir
        self.dataset = dataset.lower()
        self.arch = arch.lower()
        self.label = label
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
        if self.use_pil:
            img = Image.open(img_path).convert("RGB")
            img = np.array(img, dtype=np.float32)
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

        label_val = row.get(self.label, -1)
        label = torch.tensor(int(label_val), dtype=torch.long)

        # Pass all CSV columns through as a plain dict of Python scalars/strings
        row_meta = {col: row[col] for col in self.df.columns}

        return {
            "image": img,
            "label": label,
            "img_path": str(img_path),
            "meta": row_meta,
        }


def collate_fn(batch):
    # Collate metadata: list of per-column lists
    meta_keys = batch[0]["meta"].keys()
    meta = {k: [b["meta"][k] for b in batch] for k in meta_keys}

    return {
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
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


class CxrFoundationDataset(Dataset):
    """
    Mammography dataset for CXR Foundation feature extraction.

    Images are loaded as grayscale uint8 numpy arrays and kept on the CPU;
    the TF SavedModel handles all preprocessing internally.  The path-building
    logic mirrors ``MammoFeatureDataset``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        img_dir: str,
        dataset: str,
        label: str,
    ):
        self.df = df.reset_index(drop=True)
        self.dir_path = data_dir / img_dir
        self.dataset = dataset.lower()
        self.label = label

    def __len__(self):
        return len(self.df)

    def _build_path(self, row) -> Path:
        patient_id = str(row["patient_id"])
        image_id = str(row["image_id"])
        if self.dataset in ("vindr", "rsna"):
            if not image_id.endswith(".png"):
                image_id = image_id + ".png"
            return self.dir_path / patient_id / image_id
        p = self.dir_path / patient_id / image_id
        if p.exists():
            return p
        return Path(str(p) + ".png")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._build_path(row)

        try:
            img_array = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
        except Exception as e:
            raise RuntimeError(f"Failed to load image at {img_path}: {e}") from e

        label_val = row.get(self.label, -1)
        label = torch.tensor(int(label_val), dtype=torch.long)
        row_meta = {col: row[col] for col in self.df.columns}

        return {
            "image_array": img_array,  # numpy uint8, kept as-is for TF model
            "label": label,
            "img_path": str(img_path),
            "meta": row_meta,
        }


def collate_fn_cxr(batch):
    """Collate for CXR Foundation: images stay as a list of numpy arrays."""
    meta_keys = batch[0]["meta"].keys()
    meta = {k: [b["meta"][k] for b in batch] for k in meta_keys}
    return {
        "image_arrays": [b["image_array"] for b in batch],
        "label": torch.stack([b["label"] for b in batch]),
        "img_path": [b["img_path"] for b in batch],
        "meta": meta,
    }


def build_dataloader(df: pd.DataFrame, args) -> DataLoader:
    if args.model == "cxr-foundation":
        dataset = CxrFoundationDataset(
            df=df,
            data_dir=Path(args.data_dir),
            img_dir=args.img_dir,
            dataset=args.dataset,
            label=args.label,
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,  # numpy arrays, not tensors
            drop_last=False,
            collate_fn=collate_fn_cxr,
        )

    dataset = MammoFeatureDataset(
        df=df,
        data_dir=Path(args.data_dir),
        img_dir=args.img_dir,
        dataset=args.dataset,
        arch=args.arch,
        label=args.label,
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


def load_cxr_foundation_models(model_dir: str):
    """
    Load both ELIXR-C and ELIXR-B TF SavedModels and return a tuple of their
    ``serving_default`` inference callables.

    Returns
    -------
    (elixrc_infer, elixrb_infer, tf_module)
        tf_module is returned so the caller can construct TF tensors.
    """
    _import_tf_text()
    tf = _import_tf()

    print(f"Loading CXR Foundation models from: {model_dir}")
    elixrc_model = tf.saved_model.load(os.path.join(model_dir, "elixr-c-v2-pooled"))
    elixrb_model = tf.saved_model.load(os.path.join(model_dir, "pax-elixr-b-text"))
    elixrc_infer = elixrc_model.signatures["serving_default"]
    elixrb_infer = elixrb_model.signatures["serving_default"]
    print("CXR Foundation models loaded.")
    return elixrc_infer, elixrb_infer, tf


# ---------------------------------------------------------------------------
# Extraction loops
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_features(model: BreastClipClassifier, loader: DataLoader, device: torch.device):
    all_features = []
    all_labels = []
    all_paths = []
    all_meta: dict[str, list] = {}

    for batch in tqdm(loader, desc="Extracting features", unit="batch"):
        images = batch["image"].to(device)
        labels = batch["label"]
        paths = batch["img_path"]
        meta = batch["meta"]

        # Handle swin encoder: needs (B, H, W, C)
        if model.get_image_encoder_type().lower() == "swin":
            images = images.squeeze(1).permute(0, 3, 1, 2)

        features = model.encode_image(images)  # (B, D)
        features = features.cpu()

        all_features.append(features)
        all_labels.append(labels)
        all_paths.extend(paths)

        # Accumulate metadata columns
        for k, v in meta.items():
            all_meta.setdefault(k, []).extend(v)

    metadata_df = pd.DataFrame(all_meta)
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0), all_paths, metadata_df


def extract_features_cxr_foundation(elixrc_infer, elixrb_infer, tf, loader: DataLoader):
    """
    Extract features using the CXR Foundation (ELIXR-C → ELIXR-B) pipeline.

    ELIXR-C is called once per batch with all serialised examples stacked
    together.  ELIXR-B still runs per-image because its QFormer input shape
    is fixed to batch-size 1 in the SavedModel signature.  The ELIXR-B output
    has shape ``(1, num_query_tokens, embed_dim)``; we mean-pool over the
    token dimension to get a single vector per image.
    """
    all_features = []
    all_labels = []
    all_paths = []
    all_meta: dict[str, list] = {}

    for batch in tqdm(loader, desc="Extracting CXR Foundation features", unit="batch"):
        image_arrays = batch["image_arrays"]  # list of numpy uint8 arrays
        labels = batch["label"]
        paths = batch["img_path"]
        meta = batch["meta"]

        # --- ELIXR-C + ELIXR-B: one call per image ---
        batch_features = []
        for img_array in image_arrays:
            serialized = _png_to_tfexample(img_array).SerializeToString()
            elixrc_out = elixrc_infer(input_example=tf.constant([serialized]))
            elixrc_emb = elixrc_out["feature_maps_0"]  # (1, H', W', C') — keep as tf.Tensor
            elixrb_out = elixrb_infer(
                image_feature=elixrc_emb,
                ids=tf.zeros((1, 1, 128), dtype=tf.int32),
                paddings=tf.zeros((1, 1, 128), dtype=tf.float32),
            )
            # (1, num_query_tokens, embed_dim) → mean-pool → (embed_dim,)
            emb = elixrb_out["all_contrastive_img_emb"].numpy()  # (1, T, D)
            emb = emb.mean(axis=1).squeeze(0)  # (D,)
            batch_features.append(emb)

        features = torch.tensor(np.stack(batch_features), dtype=torch.float32)  # (B, D)

        all_features.append(features)
        all_labels.append(labels)
        all_paths.extend(paths)

        for k, v in meta.items():
            all_meta.setdefault(k, []).extend(v)

    metadata_df = pd.DataFrame(all_meta)
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0), all_paths, metadata_df


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
        "--cxr-foundation-dir",
        default="./hf",
        type=str,
        help=(
            "Directory containing downloaded CXR Foundation SavedModels "
            "(elixr-c-v2-pooled/ and pax-elixr-b-text/ subdirs). "
            "Only used when --model cxr-foundation."
        ),
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
        "--device",
        default="cuda",
        type=str,
        help="Compute device for MammoCLIP: cuda | cpu (CXR Foundation runs on TF/GPU automatically)",
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
    if args.model == "cxr-foundation":
        elixrc_infer, elixrb_infer, tf = load_cxr_foundation_models(args.cxr_foundation_dir)
        features, labels, img_paths, metadata_df = extract_features_cxr_foundation(
            elixrc_infer, elixrb_infer, tf, loader
        )
        arch_tag = "cxr-foundation"
    else:
        if not args.clip_chk_pt_path:
            raise ValueError("--clip_chk_pt_path is required when --model mammoclip")
        device = torch.device(
            args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
        )
        print(f"Using device: {device}")
        model = load_model(args.clip_chk_pt_path, args.arch, device)
        features, labels, img_paths, metadata_df = extract_features(model, loader, device)
        arch_tag = args.arch

    print("\nExtraction complete.")
    print(f"  features shape  : {features.shape}")
    print(f"  labels shape    : {labels.shape}")
    print(f"  unique labels   : {labels.unique().tolist()}")
    print(f"  metadata columns: {list(metadata_df.columns)}")

    # ---- Save -------------------------------------------------------------
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "features": features,  # (N, D)  float32
        "labels": labels,  # (N,)    int64
        "img_paths": img_paths,  # list[str]
        "metadata": metadata_df,  # pd.DataFrame, all CSV columns, length N
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
