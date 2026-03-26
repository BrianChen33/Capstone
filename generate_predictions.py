#!/usr/bin/env python3
"""Generate prediction .npy files from the trained model for downstream analysis."""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from shared import (
    DEVICE, safe_load, split_fields, compute_stats,
    preprocess, PositionDataset,
)
from models import build_model


def main():
    device = DEVICE
    print(f"Using device: {device}")

    # Load & preprocess
    train_tensor = safe_load("Dataset/train_combined.pt", allow_unsafe=True)
    test_tensor = safe_load("Dataset/test_combined.pt", allow_unsafe=True)
    train_fields = split_fields(train_tensor)
    test_fields = split_fields(test_tensor)

    stats = compute_stats(train_fields)
    flat_test, seq_test, meta_test, y_test = preprocess(test_fields, stats)

    # Load trained model
    with open("artifacts/training_report.json") as f:
        report = json.load(f)
    model_kind = report["model"]

    model = build_model(model_kind, flat_dim=flat_test.size(1), meta_dim=meta_test.size(1)).to(device)
    model.load_state_dict(torch.load("artifacts/best_model.pt", map_location=device, weights_only=True))
    model.eval()

    # Predict
    loader = DataLoader(
        PositionDataset(flat_test, seq_test, meta_test, y_test),
        batch_size=256, shuffle=False,
    )
    all_preds, all_targets = [], []
    with torch.no_grad():
        for flat_feats, spec_seq, meta, y in loader:
            pred = model(flat_feats.to(device), spec_seq.to(device), meta.to(device))
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    timestamps = test_fields["timestamp"].numpy().flatten()

    # Pad to 3D (x, y, z=0) for visualize_enhanced.py compatibility
    preds3d = np.column_stack([preds, np.zeros(len(preds))])
    targets3d = np.column_stack([targets, np.zeros(len(targets))])

    out_dir = "artifacts"
    np.save(os.path.join(out_dir, "test_predictions.npy"), preds3d)
    np.save(os.path.join(out_dir, "test_targets.npy"), targets3d)
    np.save(os.path.join(out_dir, "test_timestamps.npy"), timestamps)

    # Metrics
    errors = np.sqrt(np.sum((preds - targets) ** 2, axis=1))
    x_err = np.abs(preds[:, 0] - targets[:, 0])
    y_err = np.abs(preds[:, 1] - targets[:, 1])

    print(f"Samples: {len(errors)}")
    print(f"MAE (Euclidean): {errors.mean():.4f} m")
    print(f"RMSE: {np.sqrt(np.mean(errors ** 2)):.4f} m")
    print(f"Median Error: {np.median(errors):.4f} m")
    print(f"90th Percentile: {np.percentile(errors, 90):.4f} m")
    print(f"95th Percentile: {np.percentile(errors, 95):.4f} m")
    print(f"Within 0.3m: {100 * np.mean(errors < 0.3):.1f}%")
    print(f"Within 0.5m: {100 * np.mean(errors < 0.5):.1f}%")
    print(f"Within 1.0m: {100 * np.mean(errors < 1.0):.1f}%")
    print(f"Per-axis MAE: X={x_err.mean():.4f}m, Y={y_err.mean():.4f}m")

    # Bucketing
    buckets = [(0, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, float("inf"))]
    labels = ["0-0.3m", "0.3-0.5m", "0.5-1.0m", "1.0-2.0m", ">2.0m"]
    print("\nError Distribution:")
    for (lo, hi), lab in zip(buckets, labels):
        mask = (errors >= lo) & (errors < hi)
        cnt = mask.sum()
        pct = 100 * cnt / len(errors)
        avg = errors[mask].mean() if cnt > 0 else 0
        print(f"  {lab}: count={cnt}, pct={pct:.1f}%, avg={avg:.4f}m")

    metrics = {
        "num_samples": int(len(errors)),
        "mae_euclidean": float(errors.mean()),
        "mse": float(np.mean(errors ** 2)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "median_error": float(np.median(errors)),
        "std_error": float(np.std(errors)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
        "within_0_3m": float(np.mean(errors < 0.3)),
        "within_0_5m": float(np.mean(errors < 0.5)),
        "within_1_0m": float(np.mean(errors < 1.0)),
        "x_mae": float(x_err.mean()),
        "y_mae": float(y_err.mean()),
    }
    with open(os.path.join(out_dir, "detailed_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to {out_dir}/detailed_metrics.json")


if __name__ == "__main__":
    main()
