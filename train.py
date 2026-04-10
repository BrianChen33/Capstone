"""Training script for Bluetooth AoA indoor localization.

Loads all per-scene datasets, combines training data for cross-scene
generalization, trains one unified model, and evaluates per-scene.
"""
import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from shared import (
    DEVICE, safe_load, split_fields, compute_stats,
    preprocess, PositionDataset, make_loaders, save_stats, save_json,
    SCENE_IDS, get_scene_paths, load_all_scenes, combine_scene_tensors,
)
from models import build_model


def train_model(model, loaders, device, epochs, lr, weight_decay=0):
    """Train with early-stopping on validation MSE and restore best weights."""
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader, val_loader = loaders
    best_state = None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for flat_feats, spec_seq, meta, y in train_loader:
            flat_feats, spec_seq, meta, y = (
                flat_feats.to(device), spec_seq.to(device),
                meta.to(device), y.to(device),
            )
            opt.zero_grad()
            pred = model(flat_feats, spec_seq, meta)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        model.eval()
        mse_sum = mae_sum = n = 0
        with torch.no_grad():
            for flat_feats, spec_seq, meta, y in val_loader:
                flat_feats, spec_seq, meta, y = (
                    flat_feats.to(device), spec_seq.to(device),
                    meta.to(device), y.to(device),
                )
                pred = model(flat_feats, spec_seq, meta)
                mse_sum += loss_fn(pred, y).item() * y.size(0)
                mae_sum += (pred - y).abs().mean().item() * y.size(0)
                n += y.size(0)
        val_mse = mse_sum / n
        val_mae = mae_sum / n
        print(
            f"Epoch {epoch}/{epochs}  "
            f"train_mse={train_loss / len(train_loader):.4f}  "
            f"val_mse={val_mse:.4f}  val_mae={val_mae:.4f}"
        )
        if val_mse < best_val:
            best_val = val_mse
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_mse": best_val}


def evaluate(model, loader, device):
    """Compute MSE and MAE on a DataLoader."""
    loss_fn = nn.MSELoss()
    model.eval()
    mse_sum = mae_sum = n = 0
    with torch.no_grad():
        for flat_feats, spec_seq, meta, y in loader:
            flat_feats, spec_seq, meta, y = (
                flat_feats.to(device), spec_seq.to(device),
                meta.to(device), y.to(device),
            )
            pred = model(flat_feats, spec_seq, meta)
            mse_sum += loss_fn(pred, y).item() * y.size(0)
            mae_sum += (pred - y).abs().mean().item() * y.size(0)
            n += y.size(0)
    return {"mse": mse_sum / n, "mae": mae_sum / n}


def main():
    parser = argparse.ArgumentParser(description="Train positioning model.")
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="Scene IDs to include (default: all)")
    parser.add_argument("--model", default="transformer", choices=["mlp", "cnn", "transformer"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--allow-unsafe-load", action="store_true")
    args = parser.parse_args()

    device = DEVICE
    print(f"Using device: {device}")

    scenes = args.scenes or SCENE_IDS

    # ── Load all per-scene datasets ──────────────────────────────────
    train_tensors, test_tensors = load_all_scenes(
        scenes, allow_unsafe=args.allow_unsafe_load or True,
    )

    # ── Combine training data across scenes ──────────────────────────
    combined_train = combine_scene_tensors(train_tensors)
    train_fields = split_fields(combined_train)
    stats = compute_stats(train_fields)

    flat_train, seq_train, meta_train, y_train = preprocess(train_fields, stats)

    print(f"\nCombined training set: {combined_train.size(0)} samples from {len(scenes)} scenes")

    # ── Train single unified model ───────────────────────────────────
    dataset = PositionDataset(flat_train, seq_train, meta_train, y_train)
    loaders = make_loaders(dataset, val_ratio=args.val_ratio, batch_size=args.batch_size)
    model = build_model(args.model, flat_dim=flat_train.size(1), meta_dim=meta_train.size(1)).to(device)
    train_info = train_model(model, loaders, device=device, epochs=args.epochs, lr=args.lr)

    # ── Save unified model & stats ───────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
    save_stats(stats, os.path.join(args.output_dir, "dataset_stats.json"))

    # ── Evaluate per-scene on test data ──────────────────────────────
    all_results = {}
    for scene_id in scenes:
        test_fields = split_fields(test_tensors[scene_id])
        flat_test, seq_test, meta_test, y_test = preprocess(test_fields, stats)
        test_loader = DataLoader(
            PositionDataset(flat_test, seq_test, meta_test, y_test),
            batch_size=args.batch_size,
        )
        test_metrics = evaluate(model, test_loader, device)
        print(f"[{scene_id}] Test MSE={test_metrics['mse']:.4f}  Test MAE={test_metrics['mae']:.4f}")
        all_results[scene_id] = test_metrics

    # ── Save training report ─────────────────────────────────────────
    save_json(
        os.path.join(args.output_dir, "training_report.json"),
        {
            "model": args.model,
            "scenes": scenes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "val_ratio": args.val_ratio,
            "train_samples": combined_train.size(0),
            "best_val_mse": train_info["best_val_mse"],
            "per_scene_test": all_results,
        },
    )

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY (Unified Cross-Scene Model)")
    print(f"{'='*60}")
    print(f"{'Scene':<10} {'Test MSE':>10} {'Test MAE':>10}")
    for sid, m in all_results.items():
        print(f"{sid:<10} {m['mse']:>10.4f} {m['mae']:>10.4f}")
    save_json(os.path.join(args.output_dir, "all_scenes_summary.json"), all_results)


if __name__ == "__main__":
    main()
