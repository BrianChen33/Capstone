"""Utilities to (1) check whether g2/g3 fields are effectively identical, and (2) export .pt tensors to CSV.

Usage examples (PowerShell):
    # Check g2/g3 similarity on both train and test
    python verify_and_export.py --train-path train_data-s02-80-20-seq1.pt --test-path test_data-s02-80-20-seq1.pt

    # Also check spectral fields
    python verify_and_export.py --train-path train_data-s02-80-20-seq1.pt --check-spec

    # Export a tensor to CSV
    python verify_and_export.py --pt-path train_data-s02-80-20-seq1.pt --csv-out artifacts/train.csv
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, List

import torch

from explore_dataset import FIELD_SLICES, safe_load, split_fields


def _diff_stats(a: torch.Tensor, b: torch.Tensor, tol: float) -> Dict[str, float]:
    # 计算逐样本的最大绝对差，判断是否“近似相等”
    diff = (a - b).abs()
    flat = diff.reshape(len(diff), -1)
    per_row_max = flat.max(dim=1).values
    return {
        "max_abs_diff": float(flat.max()),
        "mean_abs_diff": float(flat.mean()),
        "rows_all_close": int((per_row_max <= tol).sum()),
        "rows_total": int(len(flat)),
        "rows_all_close_ratio": float(((per_row_max <= tol).float().mean())),
    }


def _analyze_one(label: str, path: Path, allow_unsafe: bool, tol: float, check_spec: bool) -> Dict[str, Dict[str, float]]:
    # 加载单个 .pt，比较 g2/g3 的位置与（可选）频谱
    tensor = safe_load(path, allow_unsafe=allow_unsafe)
    fields = split_fields(tensor)
    result = {"pos": _diff_stats(fields["g2_pos"], fields["g3_pos"], tol)}
    if check_spec:
        result["spec"] = _diff_stats(fields["g2_spec"], fields["g3_spec"], tol)
    print(f"[{label}] g2_vs_g3 (pos): {result['pos']}")
    if check_spec:
        print(f"[{label}] g2_vs_g3 (spec): {result['spec']}")
    return result


def _build_columns() -> List[str]:
    # 构造 CSV 列名，便于后续写出
    cols: List[str] = []
    cols += [f"g1_pos_{axis}" for axis in ["x", "y", "z"]]
    cols += [f"g2_pos_{axis}" for axis in ["x", "y", "z"]]
    cols += [f"g3_pos_{axis}" for axis in ["x", "y", "z"]]
    cols += ["timestamp"]
    cols += ["area"]
    cols += [f"gt_pos_{axis}" for axis in ["x", "y", "z"]]
    cols += [f"g1_spec_{i}" for i in range(324)]
    cols += [f"g2_spec_{i}" for i in range(324)]
    cols += [f"g3_spec_{i}" for i in range(324)]
    return cols


def _export_csv(path: Path, csv_out: Path, allow_unsafe: bool) -> None:
    # 将 [N,1,986] 展平成 CSV，包含列名
    tensor = safe_load(path, allow_unsafe=allow_unsafe)
    if tensor.dim() != 3 or tensor.size(1) != 1:
        raise ValueError(f"Expected tensor shape [N,1,F], got {tuple(tensor.shape)}")
    flat = tensor[:, 0, :].cpu().numpy()
    cols = _build_columns()
    if flat.shape[1] != len(cols):
        raise ValueError(f"Feature dimension mismatch: data has {flat.shape[1]} cols, expected {len(cols)}")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(flat.tolist())
    print(f"Saved CSV to {csv_out} with shape {flat.shape}")


def main() -> None:
    # CLI：可选择对比 g2/g3，或将任意 .pt 导出为 CSV
    p = argparse.ArgumentParser(description="Check g2/g3 equality and export .pt tensors to CSV.")
    p.add_argument("--train-path", type=Path, help="Path to train tensor for g2/g3 check")
    p.add_argument("--test-path", type=Path, help="Path to test tensor for g2/g3 check")
    p.add_argument("--pt-path", type=Path, help="If set, export this tensor to CSV")
    p.add_argument("--csv-out", type=Path, help="CSV output path when --pt-path is provided")
    p.add_argument("--allow-unsafe", action="store_true", help="Allow unsafe torch.load if weights_only unsupported")
    p.add_argument("--tol", type=float, default=1e-8, help="Tolerance for considering rows identical (default 1e-8)")
    p.add_argument("--check-spec", action="store_true", help="Also compare g2_spec vs g3_spec")
    args = p.parse_args()

    if not args.train_path and not args.pt_path:
        p.error("Provide --train-path for g2/g3 check and/or --pt-path for CSV export")

    if args.train_path:
        _analyze_one("train", args.train_path, args.allow_unsafe, args.tol, args.check_spec)
    if args.test_path:
        _analyze_one("test", args.test_path, args.allow_unsafe, args.tol, args.check_spec)

    if args.pt_path:
        if not args.csv_out:
            p.error("--csv-out is required when --pt-path is set")
        _export_csv(args.pt_path, args.csv_out, args.allow_unsafe)


if __name__ == "__main__":
    main()
