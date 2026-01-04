"""Dataset exploration utilities for Bluetooth positioning tensors.
Generates descriptive statistics and diagnostic visualizations.
"""
import argparse
import inspect
import json
import warnings
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Layout: g1_pos(3) + g2_pos(3) + g3_pos(3) + timestamp(1) + area(1) + gt_pos(3)
#       + g1_spec(324) + g2_spec(324) + g3_spec(324) = 986
FIELD_SLICES: Dict[str, Tuple[int, int]] = {
    "g1_pos": (0, 3),
    "g2_pos": (3, 6),
    "g3_pos": (6, 9),
    "timestamp": (9, 10),
    "area": (10, 11),
    "gt_pos": (11, 14),
    "g1_spec": (14, 14 + 324),
    "g2_spec": (14 + 324, 14 + 648),
    "g3_spec": (14 + 648, 14 + 972),
}


def _supports_weights_only() -> bool:
    # 检查 torch.load 是否支持 weights_only
    return "weights_only" in inspect.signature(torch.load).parameters


def safe_load(path: Path, allow_unsafe: bool) -> torch.Tensor:
    try:
        if _supports_weights_only():
            return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        pass
    if not allow_unsafe:
        raise RuntimeError(
            "Current PyTorch does not support weights_only=True; rerun with --allow-unsafe if you trust the file."
        )
    warnings.warn("Loading without weights_only=True; only do this for trusted files.")
    return torch.load(path, map_location="cpu")


def split_fields(tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    # 拆 [N,1,986] → 按字段切片
    if tensor.dim() != 3 or tensor.size(1) != 1:
        raise ValueError("Expected tensor shape [N,1,F].")
    flat = tensor[:, 0, :]
    return {name: flat[:, sl[0] : sl[1]] for name, sl in FIELD_SLICES.items()}


def summarize_numeric(data: torch.Tensor) -> Dict[str, float]:
    # 生成基础统计（均值/方差/分位数），供 summary.json 使用
    np_data = data.cpu().numpy().reshape(len(data), -1)
    return {
        "shape": list(data.shape),
        "mean": float(np_data.mean()),
        "std": float(np_data.std()),
        "min": float(np_data.min()),
        "max": float(np_data.max()),
        "p25": float(np.quantile(np_data, 0.25)),
        "p50": float(np.quantile(np_data, 0.50)),
        "p75": float(np.quantile(np_data, 0.75)),
    }


def ensure_outdir(path: Path) -> Path:
    # 创建目录后返回原 Path
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_ground_truth_xy(gt_pos: torch.Tensor, timestamp: torch.Tensor, out_path: Path) -> None:
    # 绘制 GT 轨迹，按时间上色
    gt_np = gt_pos.cpu().numpy()
    ts_np = timestamp.cpu().numpy().flatten()
    plt.figure(figsize=(8, 7))
    sc = plt.scatter(gt_np[:, 0], gt_np[:, 1], c=ts_np, cmap="viridis", s=6, alpha=0.7)
    plt.colorbar(sc, label="timestamp")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Ground-truth trajectory colored by timestamp")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_timestamp_hist(timestamp: torch.Tensor, out_path: Path) -> None:
    # 时间戳分布直方图
    ts_np = timestamp.cpu().numpy().flatten()
    plt.figure(figsize=(8, 4))
    plt.hist(ts_np, bins=40, color="#4a90e2", alpha=0.85, edgecolor="white")
    plt.xlabel("timestamp")
    plt.ylabel("count")
    plt.title("Timestamp distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_spectrum_energy(g_specs: Dict[str, torch.Tensor], out_path: Path) -> None:
    # 三路频谱的平均能量分布
    plt.figure(figsize=(9, 5))
    for name, tensor in g_specs.items():
        energy = tensor.mean(dim=1).cpu().numpy()
        plt.hist(energy, bins=40, alpha=0.5, label=name)
    plt.xlabel("mean spectral power")
    plt.ylabel("count")
    plt.title("Gateway spectral energy distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_distance_vs_energy(gt_pos: torch.Tensor, gateway_pos: Dict[str, torch.Tensor], g_specs: Dict[str, torch.Tensor], out_path: Path) -> None:
    # 距离-能量散点：观测能量随 GT 与网关距离的关系
    plt.figure(figsize=(9, 5))
    gt_np = gt_pos.cpu().numpy()
    for name, pos in gateway_pos.items():
        pos_np = pos.cpu().numpy()
        dist = np.linalg.norm(gt_np - pos_np, axis=1)
        energy = g_specs[name.replace("_pos", "_spec")].mean(dim=1).cpu().numpy()
        plt.scatter(dist, energy, s=5, alpha=0.35, label=name)
    plt.xlabel("distance to gateway")
    plt.ylabel("mean spectral power")
    plt.title("Distance vs spectral energy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    # 加载 train/test → 生成 summary.json 与多张诊断图
    parser = argparse.ArgumentParser(description="Explore Bluetooth positioning tensors.")
    parser.add_argument("--train-path", default="train_data-s02-80-20-seq1.pt", help="Path to training tensor")
    parser.add_argument("--test-path", default="test_data-s02-80-20-seq1.pt", help="Path to test tensor")
    parser.add_argument("--output-dir", default="artifacts/data_exploration", help="Directory for reports and figures")
    parser.add_argument("--allow-unsafe", action="store_true", help="Allow torch.load without weights_only for trusted files")
    args = parser.parse_args()

    out_dir = ensure_outdir(Path(args.output_dir))
    fig_dir = ensure_outdir(out_dir / "figs")

    train_tensor = safe_load(Path(args.train_path), allow_unsafe=args.allow_unsafe)
    test_tensor = safe_load(Path(args.test_path), allow_unsafe=args.allow_unsafe)

    train_fields = split_fields(train_tensor)
    test_fields = split_fields(test_tensor)

    summary = {
        "train_shape": list(train_tensor.shape),
        "test_shape": list(test_tensor.shape),
        "train_stats": {k: summarize_numeric(v) for k, v in train_fields.items()},
        "test_stats": {k: summarize_numeric(v) for k, v in test_fields.items()},
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_ground_truth_xy(train_fields["gt_pos"], train_fields["timestamp"], fig_dir / "ground_truth_xy.png")
    plot_timestamp_hist(train_fields["timestamp"], fig_dir / "timestamp_hist.png")
    plot_spectrum_energy(
        {k: train_fields[k] for k in ["g1_spec", "g2_spec", "g3_spec"]},
        fig_dir / "spectrum_energy.png",
    )
    plot_distance_vs_energy(
        train_fields["gt_pos"],
        {k: train_fields[k] for k in ["g1_pos", "g2_pos", "g3_pos"]},
        {k: train_fields[k] for k in ["g1_spec", "g2_spec", "g3_spec"]},
        fig_dir / "distance_vs_energy.png",
    )

    print(f"Saved summary to {out_dir / 'summary.json'}")
    print(f"Saved figures to {fig_dir}")


if __name__ == "__main__":
    main()
