#!/usr/bin/env python3
"""Ablation study — evaluates feature-group importance by zeroing out components.

Uses MLP for speed; the goal is *relative* importance, not absolute accuracy.
Reuses the shared preprocessing and training utilities.
"""
import json
import os

import torch
from torch.utils.data import DataLoader

from shared import (
    DEVICE, safe_load, split_fields, compute_stats,
    preprocess, PositionDataset, make_loaders,
)
from models import build_model
from train import train_model, evaluate

EPOCHS = 6
BATCH = 64
LR = 1e-3
SEED = 42


def run_one(kind, flat_tr, seq_tr, meta_tr, y_tr,
            flat_te, seq_te, meta_te, y_te, label=""):
    device = DEVICE
    ds = PositionDataset(flat_tr, seq_tr, meta_tr, y_tr)
    loaders = make_loaders(ds, val_ratio=0.2, batch_size=BATCH, seed=SEED)
    model = build_model(kind, flat_dim=flat_tr.size(1), meta_dim=meta_tr.size(1)).to(device)
    info = train_model(model, loaders, device, EPOCHS, LR)
    test_ds = PositionDataset(flat_te, seq_te, meta_te, y_te)
    test_loader = DataLoader(test_ds, batch_size=BATCH)
    test_m = evaluate(model, test_loader, device)
    print(f"  [{label}] val_mse={info['best_val_mse']:.4f}  "
          f"test_mse={test_m['mse']:.4f}  test_mae={test_m['mae']:.4f}")
    return {"val_mse": info["best_val_mse"], "test_mse": test_m["mse"], "test_mae": test_m["mae"]}


def zero_columns(t, cols):
    t = t.clone()
    for c in cols:
        if c < t.size(1):
            t[:, c] = 0.0
    return t


def main():
    torch.manual_seed(SEED)
    device = DEVICE
    print(f"Using device: {device}")

    train_tensor = safe_load("Dataset/train_combined.pt", allow_unsafe=True)
    test_tensor = safe_load("Dataset/test_combined.pt", allow_unsafe=True)
    train_fields = split_fields(train_tensor)
    test_fields = split_fields(test_tensor)
    stats = compute_stats(train_fields)

    flat_tr, seq_tr, meta_tr, y_tr = preprocess(train_fields, stats)
    flat_te, seq_te, meta_te, y_te = preprocess(test_fields, stats)

    results = {}
    kind = "mlp"

    # Baseline
    print("=== Baseline (full) ===")
    results["baseline"] = run_one(kind, flat_tr, seq_tr, meta_tr, y_tr,
                                  flat_te, seq_te, meta_te, y_te, "baseline")

    # No timestamp — index 9 in flat and meta
    print("=== Ablation: no timestamp ===")
    results["no_timestamp"] = run_one(
        kind,
        zero_columns(flat_tr, [9]), seq_tr, zero_columns(meta_tr, [9]), y_tr,
        zero_columns(flat_te, [9]), seq_te, zero_columns(meta_te, [9]), y_te,
        "no_timestamp",
    )

    # No gateway positions — indices 0-8 in flat and meta
    print("=== Ablation: no gateway positions ===")
    gp_cols = list(range(9))
    results["no_gateway_pos"] = run_one(
        kind,
        zero_columns(flat_tr, gp_cols), seq_tr, zero_columns(meta_tr, gp_cols), y_tr,
        zero_columns(flat_te, gp_cols), seq_te, zero_columns(meta_te, gp_cols), y_te,
        "no_gateway_pos",
    )

    # No spectrum G1 — flat indices 10..333, seq channel 0
    print("=== Ablation: no spectrum G1 ===")
    s1 = list(range(10, 10 + 324))
    seq_tr_a = seq_tr.clone(); seq_tr_a[:, :, 0] = 0.0
    seq_te_a = seq_te.clone(); seq_te_a[:, :, 0] = 0.0
    results["no_spectrum_g1"] = run_one(
        kind, zero_columns(flat_tr, s1), seq_tr_a, meta_tr, y_tr,
        zero_columns(flat_te, s1), seq_te_a, meta_te, y_te, "no_spectrum_g1",
    )

    # No spectrum G2
    print("=== Ablation: no spectrum G2 ===")
    s2 = list(range(10 + 324, 10 + 648))
    seq_tr_a = seq_tr.clone(); seq_tr_a[:, :, 1] = 0.0
    seq_te_a = seq_te.clone(); seq_te_a[:, :, 1] = 0.0
    results["no_spectrum_g2"] = run_one(
        kind, zero_columns(flat_tr, s2), seq_tr_a, meta_tr, y_tr,
        zero_columns(flat_te, s2), seq_te_a, meta_te, y_te, "no_spectrum_g2",
    )

    # No spectrum G3
    print("=== Ablation: no spectrum G3 ===")
    s3 = list(range(10 + 648, 10 + 972))
    seq_tr_a = seq_tr.clone(); seq_tr_a[:, :, 2] = 0.0
    seq_te_a = seq_te.clone(); seq_te_a[:, :, 2] = 0.0
    results["no_spectrum_g3"] = run_one(
        kind, zero_columns(flat_tr, s3), seq_tr_a, meta_tr, y_tr,
        zero_columns(flat_te, s3), seq_te_a, meta_te, y_te, "no_spectrum_g3",
    )

    # No spectrum (all)
    print("=== Ablation: no spectrum (all) ===")
    all_spec = list(range(10, 10 + 972))
    results["no_spectrum_all"] = run_one(
        kind,
        zero_columns(flat_tr, all_spec), torch.zeros_like(seq_tr), meta_tr, y_tr,
        zero_columns(flat_te, all_spec), torch.zeros_like(seq_te), meta_te, y_te,
        "no_spectrum_all",
    )

    # Summary
    print("\n=== ABLATION SUMMARY ===")
    print(f"{'Config':<20} {'Val MSE':>10} {'Test MSE':>10} {'Test MAE':>10}")
    for name, m in results.items():
        print(f"{name:<20} {m['val_mse']:>10.4f} {m['test_mse']:>10.4f} {m['test_mae']:>10.4f}")

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to artifacts/ablation_results.json")


if __name__ == "__main__":
    main()
