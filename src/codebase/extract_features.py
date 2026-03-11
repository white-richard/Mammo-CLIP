"""
Extract vision features from MammoCLIP and save them for
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

"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
        images = batch["image"].to(device)
        paths = batch["img_path"]
        meta = batch["meta"]

        # Handle swin encoder: needs (B, H, W, C)
        if model.get_image_encoder_type().lower() == "swin":
            images = images.squeeze(1).permute(0, 3, 1, 2)

        features = model.encode_image(images)  # (B, D)
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
        choices=["mammoclip"],
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
