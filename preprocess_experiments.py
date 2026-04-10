"""Preprocessing strategy comparison experiment.

Compares three normalisation strategies to justify the project's decision
to skip spatial-spectrum normalisation:

  1. fully_raw    — no normalisation at all
  2. ts_only      — only timestamp normalised  (adopted approach)
  3. block_zscore — timestamp + spectra normalised (legacy approach)

Uses combined cross-scene training data. A lightweight MLP is trained for
5 epochs under each strategy (×3 seeds) to measure validation and test MAE.
Results stored as JSON for visualization.py to consume.
"""
import argparse
import json
import math
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from shared import (
    DEVICE, FIELD_SLICES, split_fields, SCENE_IDS,
    get_scene_paths, PROJECT_ROOT,
    load_all_scenes, combine_scene_tensors,
)

OUT_ROOT = PROJECT_ROOT / "FigData" / "PreprocessExperiments"
SEEDS = [42, 43, 44]


# ------------------------------------------------------------------
# Statistics (training-set only)
# ------------------------------------------------------------------
def compute_stats(train_fields: Dict[str, torch.Tensor]) -> dict:
    stats: dict = {}
    ts = train_fields["timestamp"]
    stats["ts_mean"] = ts.mean(dim=0)
    stats["ts_std"] = ts.std(dim=0).clamp(min=1e-6)
    for i, key in enumerate(["g1_spec", "g2_spec", "g3_spec"], start=1):
        s = train_fields[key]
        stats[f"spec{i}_mean"] = s.mean(dim=0)
        stats[f"spec{i}_std"] = s.std(dim=0).clamp(min=1e-6)
    return stats


# ------------------------------------------------------------------
# Feature builders
# ------------------------------------------------------------------
def build_features(fields, stats, strategy: str) -> torch.Tensor:
    g_pos = torch.cat([fields["g1_pos"], fields["g2_pos"], fields["g3_pos"]], dim=1)
    ts = fields["timestamp"]
    specs = [fields["g1_spec"], fields["g2_spec"], fields["g3_spec"]]

    if strategy == "fully_raw":
        return torch.cat([g_pos, ts] + specs, dim=1)

    if strategy == "ts_only":
        ts = (ts - stats["ts_mean"]) / stats["ts_std"]
        return torch.cat([g_pos, ts] + specs, dim=1)

    if strategy == "block_zscore":
        ts = (ts - stats["ts_mean"]) / stats["ts_std"]
        norm_specs = []
        for i, s in enumerate(specs):
            norm_specs.append((s - stats[f"spec{i+1}_mean"]) / stats[f"spec{i+1}_std"])
        return torch.cat([g_pos, ts] + norm_specs, dim=1)

    raise ValueError(f"Unknown strategy: {strategy}")


# ------------------------------------------------------------------
# Lightweight dataset / model / training
# ------------------------------------------------------------------
class PairDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class SimpleRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


def _eval(model, loader, device):
    loss_fn = nn.MSELoss()
    model.eval()
    mse_sum = mae_sum = n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            mse_sum += loss_fn(pred, y).item() * x.size(0)
            mae_sum += (pred - y).abs().mean().item() * x.size(0)
            n += x.size(0)
    return {"mse": mse_sum / n, "mae": mae_sum / n}


def train_one(features, targets, test_features, test_targets, epochs=5, lr=1e-3, batch_size=64, seed=42):
    N = features.size(0)
    val_size = int(0.2 * N)
    idx = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    train_ds = PairDataset(features[idx[val_size:]], targets[idx[val_size:]])
    val_ds = PairDataset(features[idx[:val_size]], targets[idx[:val_size]])
    test_ds = PairDataset(test_features, test_targets)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size)
    test_ld = DataLoader(test_ds, batch_size=batch_size)

    device = DEVICE
    model = SimpleRegressor(features.size(1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_state, best_val = None, math.inf

    for _ in range(epochs):
        model.train()
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        val_m = _eval(model, val_ld, device)
        if val_m["mse"] < best_val:
            best_val = val_m["mse"]
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    val_f = _eval(model, val_ld, device)
    test_f = _eval(model, test_ld, device)
    return {"val_mse": val_f["mse"], "val_mae": val_f["mae"], "test_mse": test_f["mse"], "test_mae": test_f["mae"]}


# ------------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------------
def run_experiments(train_fields, test_fields, out_dir, label=""):
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = train_fields["gt_pos"][:, :2]
    test_targets = test_fields["gt_pos"][:, :2]
    stats = compute_stats(train_fields)

    strategies = ["fully_raw", "ts_only", "block_zscore"]
    results = {}

    for name in strategies:
        feats = build_features(train_fields, stats, name)
        test_feats = build_features(test_fields, stats, name)
        per_seed = []
        for seed in SEEDS:
            m = train_one(feats, targets, test_feats, test_targets, epochs=5, seed=seed)
            per_seed.append({"seed": seed, **m})

        vm = torch.tensor([s["val_mae"] for s in per_seed])
        tm = torch.tensor([s["test_mae"] for s in per_seed])
        results[name] = {
            "per_seed": per_seed,
            "val_mae_mean": float(vm.mean()),
            "val_mae_std": float(vm.std(unbiased=False)),
            "test_mae_mean": float(tm.mean()),
            "test_mae_std": float(tm.std(unbiased=False)),
        }
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}{name}: val_mae={results[name]['val_mae_mean']:.4f}±{results[name]['val_mae_std']:.4f}, "
              f"test_mae={results[name]['test_mae_mean']:.4f}±{results[name]['test_mae_std']:.4f}")

    with open(out_dir / "preprocess_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Results saved to", out_dir)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="Scene IDs to include (default: all)")
    ap.add_argument("--output-dir", default=str(OUT_ROOT))
    ap.add_argument("--allow-unsafe", action="store_true")
    args = ap.parse_args()

    scenes = args.scenes or SCENE_IDS
    out_dir = Path(args.output_dir)

    # Load & combine all scene data
    train_tensors, test_tensors = load_all_scenes(scenes, allow_unsafe=True)
    combined_train = combine_scene_tensors(train_tensors)
    combined_test = combine_scene_tensors(test_tensors)
    train_fields = split_fields(combined_train)
    test_fields = split_fields(combined_test)

    print(f"\nCombined training: {combined_train.size(0)} samples")
    print(f"Combined test: {combined_test.size(0)} samples")

    results = run_experiments(train_fields, test_fields, out_dir, label="combined")

    # Also save summary for visualization.py
    summary = {s: {"val_mae": results[s]["val_mae_mean"], "test_mae": results[s]["test_mae_mean"]}
               for s in results}
    with open(out_dir / "all_scenes_preprocess.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
