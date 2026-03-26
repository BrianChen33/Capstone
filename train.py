"""Training script for Bluetooth AoA indoor localization.

Loads the combined dataset, preprocesses features (no spectrum normalisation),
trains the selected model architecture, and saves the best checkpoint along
with training statistics.
"""
import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from shared import (
    DEVICE, safe_load, split_fields, compute_stats,
    preprocess, PositionDataset, make_loaders, save_stats, save_json,
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
    parser.add_argument("--train-path", default="Dataset/train_combined.pt")
    parser.add_argument("--test-path", default="Dataset/test_combined.pt")
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

    # Load and preprocess
    train_tensor = safe_load(args.train_path, allow_unsafe=args.allow_unsafe_load)
    test_tensor = safe_load(args.test_path, allow_unsafe=args.allow_unsafe_load)
    train_fields = split_fields(train_tensor)
    test_fields = split_fields(test_tensor)

    stats = compute_stats(train_fields)
    flat_train, seq_train, meta_train, y_train = preprocess(train_fields, stats)
    flat_test, seq_test, meta_test, y_test = preprocess(test_fields, stats)

    # Train
    dataset = PositionDataset(flat_train, seq_train, meta_train, y_train)
    loaders = make_loaders(dataset, val_ratio=args.val_ratio, batch_size=args.batch_size)
    model = build_model(args.model, flat_dim=flat_train.size(1), meta_dim=meta_train.size(1)).to(device)
    train_model(model, loaders, device=device, epochs=args.epochs, lr=args.lr)

    # Evaluate on test set
    test_loader = DataLoader(
        PositionDataset(flat_test, seq_test, meta_test, y_test),
        batch_size=args.batch_size,
    )
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test MSE={test_metrics['mse']:.4f}  Test MAE={test_metrics['mae']:.4f}")

    # Save artefacts
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
    save_stats(stats, os.path.join(args.output_dir, "dataset_stats.json"))
    save_json(
        os.path.join(args.output_dir, "training_report.json"),
        {
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "val_ratio": args.val_ratio,
            "test_metrics": test_metrics,
        },
    )


if __name__ == "__main__":
    main()
