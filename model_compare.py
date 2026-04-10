"""Model comparison: MLP vs CNN vs Transformer under fixed compute budget.

Uses combined cross-scene training data and the shared preprocessing
pipeline. Each architecture is trained for FIXED_EPOCHS epochs with
FIXED_LR to ensure a fair comparison. Results stored as JSON for
visualization.py to consume.
"""
import argparse
import json
from pathlib import Path

import torch

from shared import (
    DEVICE, split_fields, compute_stats,
    preprocess, PositionDataset, make_loaders,
    SCENE_IDS, get_scene_paths, PROJECT_ROOT,
    load_all_scenes, combine_scene_tensors,
)
from models import build_model
from train import train_model, evaluate

OUT_ROOT = PROJECT_ROOT / "FigData" / "ModelCompare"

FIXED_LR = 1e-3
FIXED_EPOCHS = 6
FIXED_SEED = 42


def main():
    ap = argparse.ArgumentParser(description="Compare model architectures for positioning.")
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="Scene IDs to include (default: all)")
    ap.add_argument("--output-dir", default=str(OUT_ROOT))
    ap.add_argument("--allow-unsafe", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    print(f"Using device: {device}")

    scenes = args.scenes or SCENE_IDS

    # ── Load & combine training data ─────────────────────────────────
    train_tensors, test_tensors = load_all_scenes(scenes, allow_unsafe=True)
    combined_train = combine_scene_tensors(train_tensors)
    train_fields = split_fields(combined_train)
    stats = compute_stats(train_fields)
    flat_train, spec_train, meta_train, y_train = preprocess(train_fields, stats)

    dataset = PositionDataset(flat_train, spec_train, meta_train, y_train)
    torch.manual_seed(FIXED_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FIXED_SEED)
    loaders = make_loaders(dataset, val_ratio=0.2, batch_size=32, seed=FIXED_SEED)

    meta_dim = meta_train.size(1)
    flat_dim = flat_train.size(1)

    print(f"\nCombined training set: {combined_train.size(0)} samples")

    model_names = ["mlp", "cnn", "transformer"]
    results = {}

    for name in model_names:
        print(f"\nTraining {name} (lr={FIXED_LR:g}, epochs={FIXED_EPOCHS})")
        torch.manual_seed(FIXED_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FIXED_SEED)

        model = build_model(name, flat_dim=flat_dim, meta_dim=meta_dim).to(device)
        train_model(model, loaders, device, FIXED_EPOCHS, FIXED_LR, weight_decay=1e-4)

        _, val_loader = loaders
        val_metrics = evaluate(model, val_loader, device)

        # Also evaluate per-scene on test data
        per_scene = {}
        for sid in scenes:
            tf = split_fields(test_tensors[sid])
            ft, st, mt, yt = preprocess(tf, stats)
            from torch.utils.data import DataLoader
            tl = DataLoader(PositionDataset(ft, st, mt, yt), batch_size=64)
            tm = evaluate(model, tl, device)
            per_scene[sid] = {"test_mse": tm["mse"], "test_mae": tm["mae"]}

        results[name] = {
            "lr": float(FIXED_LR),
            "epochs": int(FIXED_EPOCHS),
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "per_scene": per_scene,
        }
        print(f"  {name}: val_mse={val_metrics['mse']:.4f}  val_mae={val_metrics['mae']:.4f}")

    with open(out_dir / "model_compare_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to", out_dir)


if __name__ == "__main__":
    main()
