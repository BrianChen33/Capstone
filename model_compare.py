"""Model comparison: MLP vs CNN vs Transformer under fixed compute budget.

Uses the shared preprocessing pipeline (no spectrum normalisation) and
unified model definitions.  Each architecture is trained for exactly
FIXED_EPOCHS epochs with FIXED_LR to ensure a fair comparison.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from shared import (
    DEVICE, safe_load, split_fields, compute_stats,
    preprocess, PositionDataset, make_loaders,
)
from models import build_model
from train import train_model, evaluate

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_PT_PATH = PROJECT_ROOT / "Dataset" / "train_combined.pt"
OUT_DIR = PROJECT_ROOT / "FigData" / "ModelCompare" / "combined"

FIXED_LR = 1e-3
FIXED_EPOCHS = 6
FIXED_SEED = 42


def run_compare(train_path: Path, allow_unsafe: bool, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(exist_ok=True)

    device = DEVICE
    print(f"Using device: {device}")

    # Load & preprocess (training data only — no test set needed for comparison)
    train_tensor = safe_load(train_path, allow_unsafe=allow_unsafe)
    train_fields = split_fields(train_tensor)
    stats = compute_stats(train_fields)
    flat_train, spec_train, meta_train, y_train = preprocess(train_fields, stats)

    dataset = PositionDataset(flat_train, spec_train, meta_train, y_train)
    torch.manual_seed(FIXED_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FIXED_SEED)
    loaders = make_loaders(dataset, val_ratio=0.2, batch_size=32, seed=FIXED_SEED)

    meta_dim = meta_train.size(1)
    flat_dim = flat_train.size(1)

    model_names = ["mlp", "cnn", "transformer"]
    results = {}

    for name in model_names:
        print(f"\nTraining {name} (lr={FIXED_LR:g}, epochs={FIXED_EPOCHS})")
        torch.manual_seed(FIXED_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FIXED_SEED)

        model = build_model(name, flat_dim=flat_dim, meta_dim=meta_dim).to(device)
        train_model(model, loaders, device, FIXED_EPOCHS, FIXED_LR, weight_decay=1e-4)

        # Evaluate best-checkpoint model on validation set
        _, val_loader = loaders
        val_metrics = evaluate(model, val_loader, device)
        results[name] = {
            "lr": float(FIXED_LR),
            "epochs": int(FIXED_EPOCHS),
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
        }
        print(f"  {name}: val_mse={val_metrics['mse']:.4f}  val_mae={val_metrics['mae']:.4f}")

    # Bar chart
    labels = list(results.keys())
    val_mae_vals = [results[k]["val_mae"] for k in labels]
    colors = ["#4a90e2", "#50e3c2", "#f5a623"]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, val_mae_vals, color=colors)
    plt.ylabel("Validation MAE (m)")
    plt.title("Model Comparison (fixed budget, 6 epochs)")
    plt.tight_layout()
    plt.savefig(fig_dir / "model_val_mae.png", dpi=150)
    plt.close()

    with open(out_dir / "model_compare_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to", out_dir)
    return results


def main():
    ap = argparse.ArgumentParser(description="Compare model architectures for positioning.")
    ap.add_argument("--train-path", default=str(TRAIN_PT_PATH))
    ap.add_argument("--output-dir", default=str(OUT_DIR))
    ap.add_argument("--allow-unsafe", action="store_true")
    args = ap.parse_args()
    run_compare(Path(args.train_path), args.allow_unsafe, Path(args.output_dir))


if __name__ == "__main__":
    main()
