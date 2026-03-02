"""
Example: loading pre-extracted MammoCLIP features into a PyTorch DataLoader.

The .pt file produced by extract_features.py has the following structure:

    {
        "features":    Float tensor  (N, D)  – one vector per image,
        "labels":      Long tensor   (N,)    – integer class labels,
        "img_paths":   list[str]     length N,
        "feature_dim": int,
        "dataset":     str,
        "arch":        str,
        "split":       str,
        "label_col":   str,
    }

Usage (standalone)
------------------
python ./src/codebase/load_features.py \
    --features-file features/vindr_abnormal_features.pt \
    --batch-size 64

Usage (as a library)
--------------------
from load_features import MammoFeaturesDataset, build_features_dataloader

train_loader = build_features_dataloader(
    features_file="features/vindr_abnormal_train_features.pt",
    batch_size=64,
    shuffle=True,
    num_workers=4,
)

for batch in train_loader:
    features = batch["features"]   # (B, D) float32
    labels   = batch["labels"]     # (B,)   int64
    paths    = batch["img_paths"]  # list[str] length B
    ...
"""

import argparse
from pathlib import Path
from typing import Callable, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class MammoFeaturesDataset(Dataset):
    """
    PyTorch Dataset that wraps the .pt feature file saved by extract_features.py.

    Parameters
    ----------
    features_file:
        Path to the .pt file produced by extract_features.py.
    transform:
        Optional callable applied to each feature vector (e.g. L2-norm,
        PCA projection, etc.).  Receives and must return a 1-D float Tensor.
    target_transform:
        Optional callable applied to each scalar label Tensor.
    """

    def __init__(
        self,
        features_file: str | Path,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
        target_transform: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        features_file = Path(features_file)
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")

        payload = torch.load(features_file, map_location="cpu")
        self._validate_payload(payload)

        self.features: Tensor = payload["features"].float()  # (N, D)
        self.labels: Tensor = payload["labels"].long()  # (N,)
        self.img_paths: list[str] = payload["img_paths"]

        # Informational metadata (not used during training, but handy)
        self.feature_dim: int = payload.get("feature_dim", self.features.shape[1])
        self.dataset_name: str = payload.get("dataset", "unknown")
        self.arch: str = payload.get("arch", "unknown")
        self.split: str = payload.get("split", "unknown")
        self.label_col: str = payload.get("label_col", "unknown")

        self.transform = transform
        self.target_transform = target_transform

        assert len(self.features) == len(self.labels) == len(self.img_paths), (
            "Mismatch between features, labels, and img_paths lengths."
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _validate_payload(payload: dict) -> None:
        required = ("features", "labels", "img_paths")
        missing = [k for k in required if k not in payload]
        if missing:
            raise KeyError(
                f"The .pt file is missing required keys: {missing}. "
                "Was it created by extract_features.py?"
            )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        feature = self.features[idx]  # (D,)
        label = self.labels[idx]  # scalar

        if self.transform is not None:
            feature = self.transform(feature)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return {
            "features": feature,
            "labels": label,
            "img_paths": self.img_paths[idx],
        }

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n={len(self)}, "
            f"feature_dim={self.feature_dim}, "
            f"dataset={self.dataset_name!r}, "
            f"split={self.split!r}, "
            f"label_col={self.label_col!r}"
            f")"
        )


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def collate_features(batch: list[dict]) -> dict:
    """Default collate function – stacks tensors and gathers paths into a list."""
    return {
        "features": torch.stack([b["features"] for b in batch]),  # (B, D)
        "labels": torch.stack([b["labels"] for b in batch]),  # (B,)
        "img_paths": [b["img_paths"] for b in batch],  # list[str]
    }


def build_features_dataloader(
    features_file: str | Path,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 4,
    transform: Optional[Callable[[Tensor], Tensor]] = None,
    target_transform: Optional[Callable[[Tensor], Tensor]] = None,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Build a DataLoader directly from a .pt features file.

    Parameters
    ----------
    features_file:
        Path to the .pt file produced by extract_features.py.
    batch_size:
        Number of samples per batch.
    shuffle:
        Whether to shuffle the data each epoch. Set True for training.
    num_workers:
        Number of DataLoader worker processes.
    transform:
        Optional transform applied to each feature vector.
    target_transform:
        Optional transform applied to each label.
    pin_memory:
        Pin memory for faster GPU transfer.
    drop_last:
        Drop the last incomplete batch.

    Returns
    -------
    DataLoader whose batches are dicts with keys
        "features"  – Tensor (B, D)
        "labels"    – Tensor (B,)
        "img_paths" – list[str]
    """
    dataset = MammoFeaturesDataset(
        features_file=features_file,
        transform=transform,
        target_transform=target_transform,
    )
    print(dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_features,
    )


def _demo(features_file: str | Path, batch_size: int, num_workers: int) -> None:
    loader = build_features_dataloader(
        features_file=features_file,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    dataset: MammoFeaturesDataset = loader.dataset  # type: ignore[assignment]
    n_pos = int((dataset.labels == 1).sum().item())
    n_neg = int((dataset.labels == 0).sum().item())

    print(f"\n{'=' * 55}")
    print(f"  Dataset  : {dataset.dataset_name}  ({dataset.split} split)")
    print(f"  Label    : {dataset.label_col}")
    print(f"  Arch     : {dataset.arch}")
    print(f"  Samples  : {len(dataset)}  (pos={n_pos}, neg={n_neg})")
    print(f"  Feat dim : {dataset.feature_dim}")
    print(f"  Batches  : {len(loader)}")
    print(f"{'=' * 55}\n")

    print("Iterating over the first 3 batches …")
    for i, batch in enumerate(loader):
        if i >= 3:
            break
        feats = batch["features"]
        labels = batch["labels"]
        paths = batch["img_paths"]
        print(
            f"  batch {i}: features {tuple(feats.shape)}  "
            f"labels {tuple(labels.shape)}  "
            f"first path: {Path(paths[0]).name}"
        )

    print("\nDone.")


def _config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and inspect a .pt feature file produced by extract_features.py."
    )
    parser.add_argument(
        "--features-file", required=True, type=str, help="Path to the .pt features file"
    )
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = _config()
    _demo(
        features_file=args.features_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
