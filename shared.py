"""Shared utilities for Bluetooth AoA indoor localization project.

Provides common data loading, field parsing, preprocessing, and dataset
utilities used across all experiment scripts.
"""
import inspect
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Device configuration — automatically uses CUDA when available
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "Dataset"

# ---------------------------------------------------------------------------
# Per-scene helpers
# ---------------------------------------------------------------------------
SCENE_IDS: List[str] = ["s13", "s20", "s27", "s34"]


def get_scene_paths(scene_id: str, dataset_dir: Path = None) -> Tuple[Path, Path]:
    """Return (train_path, test_path) for a given scene."""
    d = dataset_dir or DATASET_DIR
    train_path = d / "train" / f"train_data-{scene_id}-40-60-seq1.pt"
    test_path = d / "test" / f"test_data-{scene_id}-40-60-seq1.pt"
    return train_path, test_path


def discover_scenes(dataset_dir: Path = None) -> List[str]:
    """Discover scene IDs from training data directory."""
    d = dataset_dir or DATASET_DIR
    train_dir = d / "train"
    scenes = []
    for f in sorted(train_dir.glob("train_data-*.pt")):
        parts = f.stem.split("-")
        scenes.append(parts[1])
    return scenes


def load_all_scenes(scenes: List[str] = None,
                    dataset_dir: Path = None,
                    allow_unsafe: bool = True
                    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Load all per-scene datasets and return dicts of {scene_id: tensor}."""
    scenes = scenes or SCENE_IDS
    train_data: Dict[str, torch.Tensor] = {}
    test_data: Dict[str, torch.Tensor] = {}
    for sid in scenes:
        tp, ep = get_scene_paths(sid, dataset_dir)
        train_data[sid] = safe_load(tp, allow_unsafe=allow_unsafe)
        test_data[sid] = safe_load(ep, allow_unsafe=allow_unsafe)
    return train_data, test_data


def combine_scene_tensors(scene_data: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate per-scene tensors into a single tensor."""
    return torch.cat(list(scene_data.values()), dim=0)

# ---------------------------------------------------------------------------
# Feature layout: 986-dimensional input vector
# ---------------------------------------------------------------------------
FIELD_SLICES: Dict[str, Tuple[int, int]] = {
    "g1_pos": (0, 3),
    "g2_pos": (3, 6),
    "g3_pos": (6, 9),
    "timestamp": (9, 10),
    "area": (10, 11),
    "gt_pos": (11, 14),
    "g1_spec": (14, 14 + 324),
    "g2_spec": (14 + 324, 14 + 648),
    "g3_spec": (14 + 648, 14 + 972),
}


# ---------------------------------------------------------------------------
# Safe tensor loading
# ---------------------------------------------------------------------------
def _supports_weights_only() -> bool:
    return "weights_only" in inspect.signature(torch.load).parameters


def safe_load(path, allow_unsafe: bool = False) -> torch.Tensor:
    """Load a .pt file with weights_only when supported."""
    try:
        if _supports_weights_only():
            return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        pass
    if not allow_unsafe:
        raise RuntimeError(
            "This PyTorch version does not support weights_only=True; "
            "rerun with --allow-unsafe if you trust the file."
        )
    warnings.warn("Loading without weights_only=True; only do this for trusted files.")
    return torch.load(str(path), map_location="cpu")


def split_fields(tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Split [N, 1, 986] tensor into a dict of named field tensors."""
    if tensor.dim() != 3 or tensor.size(1) != 1:
        raise ValueError("Expected tensor shape [N,1,F].")
    flat = tensor[:, 0, :]
    return {name: flat[:, sl[0]: sl[1]] for name, sl in FIELD_SLICES.items()}


# ---------------------------------------------------------------------------
# Statistics (computed from training set only)
# ---------------------------------------------------------------------------
def compute_stats(train_fields: Dict[str, torch.Tensor]) -> dict:
    """Compute normalisation statistics from training data.

    Only the timestamp requires normalisation because its value range
    (0–4100) is orders of magnitude larger than other features.
    Spatial spectra are already in [0, 1] from the spectrum construction
    process, and gateway positions are small values near zero.
    """
    stats: dict = {}
    ts = train_fields["timestamp"]
    stats["ts_mean"] = ts.mean(dim=0)
    stats["ts_std"] = ts.std(dim=0).clamp(min=1e-6)
    stats["num_samples"] = train_fields["g1_pos"].size(0)
    return stats


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(fields: Dict[str, torch.Tensor], stats: dict):
    """Build model inputs **without** spectrum normalisation.

    Rationale
    ---------
    * Spatial spectra are already bounded in [0, 1] by the spatial-spectrum
      construction procedure (beamforming power normalisation), so additional
      Z-score normalisation is unnecessary and could distort the spectral
      shape.
    * Gateway positions are small values (≈ [-1, 1]) and are used as-is.
    * Only the timestamp is Z-score normalised because its raw range
      (0–4100) would otherwise dominate gradient updates.

    Returns
    -------
    flat_feats : (N, 982)  — [g_pos(9), ts_norm(1), spec1(324), spec2(324), spec3(324)]
    spec_seq   : (N, 324, 3) — three gateway spectra stacked
    meta       : (N, 10)     — [g_pos(9), ts_norm(1)]
    targets    : (N, 2)      — ground-truth (x, y)
    """
    g_pos = torch.cat([fields["g1_pos"], fields["g2_pos"], fields["g3_pos"]], dim=1)
    ts = (fields["timestamp"] - stats["ts_mean"]) / stats["ts_std"]
    specs = [fields["g1_spec"], fields["g2_spec"], fields["g3_spec"]]

    spec_seq = torch.stack(specs, dim=-1)  # (N, 324, 3)
    flat_feats = torch.cat([g_pos, ts] + specs, dim=1)
    meta = torch.cat([g_pos, ts], dim=1)
    targets = fields["gt_pos"][:, :2]
    return flat_feats, spec_seq, meta, targets


# ---------------------------------------------------------------------------
# Dataset & DataLoader helpers
# ---------------------------------------------------------------------------
class PositionDataset(Dataset):
    """Wraps the four preprocessed tensors into a map-style dataset."""

    def __init__(self, flat_feats, spec_seq, meta, targets):
        self.flat_feats = flat_feats
        self.spec_seq = spec_seq
        self.meta = meta
        self.targets = targets

    def __len__(self):
        return self.targets.size(0)

    def __getitem__(self, idx):
        return self.flat_feats[idx], self.spec_seq[idx], self.meta[idx], self.targets[idx]


def make_loaders(dataset, val_ratio: float = 0.2, batch_size: int = 64, seed: int = 42):
    """Split *dataset* into train / val loaders with a fixed random seed."""
    N = len(dataset)
    val_size = int(val_ratio * N)
    idx = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def save_stats(stats: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in stats.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def save_json(path: str, content: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)
