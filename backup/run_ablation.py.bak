#!/usr/bin/env python3
"""Quick ablation study reusing train.py pipeline. Outputs results to stdout and JSON."""
import json, os, warnings
import torch
from torch import nn
from torch.utils.data import DataLoader
from train import (
    FIELD_SLICES, safe_load, split_fields, compute_stats,
    preprocess_block_zscore, PositionDataset, make_loaders,
    build_model, train_model, evaluate,
)

EPOCHS = 6
BATCH = 64
LR = 1e-3
SEED = 42


def run_one(kind, flat_train, seq_train, meta_train, y_train,
            flat_test, seq_test, meta_test, y_test, label=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = PositionDataset(flat_train, seq_train, meta_train, y_train)
    loaders = make_loaders(ds, 0.2, BATCH)
    model = build_model(kind, flat_dim=flat_train.size(1), meta_dim=meta_train.size(1)).to(device)
    info = train_model(model, loaders, device, EPOCHS, LR)
    test_ds = PositionDataset(flat_test, seq_test, meta_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH)
    test_metrics = evaluate(model, test_loader, device)
    print(f"  [{label}] val_mse={info['best_val_mse']:.4f} test_mse={test_metrics['mse']:.4f} test_mae={test_metrics['mae']:.4f}")
    return {"val_mse": info["best_val_mse"], "test_mse": test_metrics["mse"], "test_mae": test_metrics["mae"]}


def zero_columns(t, cols):
    """Zero out specific column indices in a tensor."""
    t = t.clone()
    for c in cols:
        if c < t.size(1):
            t[:, c] = 0.0
    return t


def main():
    torch.manual_seed(SEED)
    train_tensor = safe_load("Dataset/train_combined.pt", allow_unsafe=True)
    test_tensor = safe_load("Dataset/test_combined.pt", allow_unsafe=True)
    train_fields = split_fields(train_tensor)
    test_fields = split_fields(test_tensor)
    stats = compute_stats(train_fields)

    flat_tr, seq_tr, meta_tr, y_tr = preprocess_block_zscore(train_fields, stats)
    flat_te, seq_te, meta_te, y_te = preprocess_block_zscore(test_fields, stats)

    results = {}
    kind = "mlp"  # MLP for fast ablation; test relative importance not abs performance

    # Baseline (full features)
    print("=== Baseline (full) ===")
    results["baseline"] = run_one(kind, flat_tr, seq_tr, meta_tr, y_tr,
                                   flat_te, seq_te, meta_te, y_te, "baseline")

    # Ablation 1: Remove timestamp (zero out timestamp in meta & flat)
    print("=== Ablation: no timestamp ===")
    # timestamp is at index 9 in flat_feats; meta = [g_pos(9), ts(1)]
    flat_tr_a = zero_columns(flat_tr, [9])
    flat_te_a = zero_columns(flat_te, [9])
    meta_tr_a = zero_columns(meta_tr, [9])  # ts is the last col of meta (index 9)
    meta_te_a = zero_columns(meta_te, [9])
    results["no_timestamp"] = run_one(kind, flat_tr_a, seq_tr, meta_tr_a, y_tr,
                                       flat_te_a, seq_te, meta_te_a, y_te, "no_timestamp")

    # Ablation 2: Remove gateway positions (zero out first 9 of flat & all of meta's gpos)
    print("=== Ablation: no gateway positions ===")
    flat_tr_a = zero_columns(flat_tr, list(range(9)))
    flat_te_a = zero_columns(flat_te, list(range(9)))
    meta_tr_a = zero_columns(meta_tr, list(range(9)))
    meta_te_a = zero_columns(meta_te, list(range(9)))
    results["no_gateway_pos"] = run_one(kind, flat_tr_a, seq_tr, meta_tr_a, y_tr,
                                         flat_te_a, seq_te, meta_te_a, y_te, "no_gateway_pos")

    # Ablation 3: Remove spectrum gateway 1 (zero out spec1 in seq & flat)
    print("=== Ablation: no spectrum G1 ===")
    # In flat: g_pos(9) + ts(1) + spec1(324) + spec2(324) + spec3(324)
    spec1_flat = list(range(10, 10+324))
    flat_tr_a = zero_columns(flat_tr, spec1_flat)
    flat_te_a = zero_columns(flat_te, spec1_flat)
    seq_tr_a = seq_tr.clone(); seq_tr_a[:, :, 0] = 0.0
    seq_te_a = seq_te.clone(); seq_te_a[:, :, 0] = 0.0
    results["no_spectrum_g1"] = run_one(kind, flat_tr_a, seq_tr_a, meta_tr, y_tr,
                                         flat_te_a, seq_te_a, meta_te, y_te, "no_spectrum_g1")

    # Ablation 4: Remove spectrum gateway 2
    print("=== Ablation: no spectrum G2 ===")
    spec2_flat = list(range(10+324, 10+648))
    flat_tr_a = zero_columns(flat_tr, spec2_flat)
    flat_te_a = zero_columns(flat_te, spec2_flat)
    seq_tr_a = seq_tr.clone(); seq_tr_a[:, :, 1] = 0.0
    seq_te_a = seq_te.clone(); seq_te_a[:, :, 1] = 0.0
    results["no_spectrum_g2"] = run_one(kind, flat_tr_a, seq_tr_a, meta_tr, y_tr,
                                         flat_te_a, seq_te_a, meta_te, y_te, "no_spectrum_g2")

    # Ablation 5: Remove spectrum gateway 3
    print("=== Ablation: no spectrum G3 ===")
    spec3_flat = list(range(10+648, 10+972))
    flat_tr_a = zero_columns(flat_tr, spec3_flat)
    flat_te_a = zero_columns(flat_te, spec3_flat)
    seq_tr_a = seq_tr.clone(); seq_tr_a[:, :, 2] = 0.0
    seq_te_a = seq_te.clone(); seq_te_a[:, :, 2] = 0.0
    results["no_spectrum_g3"] = run_one(kind, flat_tr_a, seq_tr_a, meta_tr, y_tr,
                                         flat_te_a, seq_te_a, meta_te, y_te, "no_spectrum_g3")

    # Ablation 6: Remove ALL spectra (meta-only)
    print("=== Ablation: no spectrum (all) ===")
    all_spec = list(range(10, 10+972))
    flat_tr_a = zero_columns(flat_tr, all_spec)
    flat_te_a = zero_columns(flat_te, all_spec)
    seq_tr_a = torch.zeros_like(seq_tr)
    seq_te_a = torch.zeros_like(seq_te)
    results["no_spectrum_all"] = run_one(kind, flat_tr_a, seq_tr_a, meta_tr, y_tr,
                                          flat_te_a, seq_te_a, meta_te, y_te, "no_spectrum_all")

    # Summary
    print("\n=== ABLATION SUMMARY ===")
    print(f"{'Config':<20} {'Val MSE':>10} {'Test MSE':>10} {'Test MAE':>10}")
    for name, m in results.items():
        print(f"{name:<20} {m['val_mse']:>10.4f} {m['test_mse']:>10.4f} {m['test_mae']:>10.4f}")

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to artifacts/ablation_results.json")


if __name__ == "__main__":
    main()
