#!/usr/bin/env python3
"""Generate prediction .npy files from the unified cross-scene model."""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from shared import (
    DEVICE, safe_load, split_fields, compute_stats,
    preprocess, PositionDataset, SCENE_IDS, get_scene_paths,
    load_all_scenes, combine_scene_tensors,
)
from models import build_model


def main():
    device = DEVICE
    print(f"Using device: {device}")
    out_dir = "artifacts"

    # ── Load all scene data ──────────────────────────────────────────
    train_tensors, test_tensors = load_all_scenes()
    combined_train = combine_scene_tensors(train_tensors)
    train_fields = split_fields(combined_train)
    stats = compute_stats(train_fields)

    # ── Load unified model ───────────────────────────────────────────
    report_path = os.path.join(out_dir, "training_report.json")
    with open(report_path) as f:
        report = json.load(f)
    model_kind = report["model"]

    # Need a sample to figure out dimensions
    sample_fields = split_fields(list(test_tensors.values())[0])
    flat_s, _, meta_s, _ = preprocess(sample_fields, stats)

    model = build_model(model_kind, flat_dim=flat_s.size(1), meta_dim=meta_s.size(1)).to(device)
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt"),
                                     map_location=device, weights_only=True))
    model.eval()

    # ── Per-scene predictions ────────────────────────────────────────
    all_metrics = {}
    for scene_id in SCENE_IDS:
        test_fields = split_fields(test_tensors[scene_id])
        flat_test, seq_test, meta_test, y_test = preprocess(test_fields, stats)
        timestamps = test_fields["timestamp"].numpy().flatten()

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

        # Pad to 3D (x, y, z=0)
        preds3d = np.column_stack([preds, np.zeros(len(preds))])
        targets3d = np.column_stack([targets, np.zeros(len(targets))])

        scene_dir = os.path.join(out_dir, scene_id)
        os.makedirs(scene_dir, exist_ok=True)
        np.save(os.path.join(scene_dir, "test_predictions.npy"), preds3d)
        np.save(os.path.join(scene_dir, "test_targets.npy"), targets3d)
        np.save(os.path.join(scene_dir, "test_timestamps.npy"), timestamps)

        # Metrics
        errors = np.sqrt(np.sum((preds - targets) ** 2, axis=1))
        x_err = np.abs(preds[:, 0] - targets[:, 0])
        y_err = np.abs(preds[:, 1] - targets[:, 1])

        print(f"\n[{scene_id}] Samples: {len(errors)}")
        print(f"  MAE (Euclidean): {errors.mean():.4f} m")
        print(f"  RMSE: {np.sqrt(np.mean(errors ** 2)):.4f} m")
        print(f"  Median Error: {np.median(errors):.4f} m")
        print(f"  90th Percentile: {np.percentile(errors, 90):.4f} m")
        print(f"  95th Percentile: {np.percentile(errors, 95):.4f} m")
        print(f"  Within 0.3m: {100 * np.mean(errors < 0.3):.1f}%")
        print(f"  Within 0.5m: {100 * np.mean(errors < 0.5):.1f}%")
        print(f"  Within 1.0m: {100 * np.mean(errors < 1.0):.1f}%")
        print(f"  Per-axis MAE: X={x_err.mean():.4f}m, Y={y_err.mean():.4f}m")

        metrics = {
            "scene": scene_id,
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
        with open(os.path.join(scene_dir, "detailed_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        all_metrics[scene_id] = metrics

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PREDICTION SUMMARY (Unified Model)")
    print(f"{'='*60}")
    print(f"{'Scene':<10} {'MAE':>8} {'RMSE':>8} {'Median':>8} {'<0.5m':>8}")
    for sid, m in all_metrics.items():
        print(f"{sid:<10} {m['mae_euclidean']:>8.4f} {m['rmse']:>8.4f} "
              f"{m['median_error']:>8.4f} {m['within_0_5m']*100:>7.1f}%")

    with open(os.path.join(out_dir, "all_scenes_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved to {out_dir}/all_scenes_metrics.json")


if __name__ == "__main__":
    main()
