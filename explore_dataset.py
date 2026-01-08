"""数据集探索脚本。

针对项目使用的蓝牙 AoA 空间谱数据，生成统计与可视化图表（训练集与测试集各一份），
供中期/最终报告引用。输出包含 json 统计与 PNG 图表，默认写入 FigData/ExploreDataset。
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


# =====================
# Fixed paths (combined dataset)
# =====================
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
TRAIN_PT_PATH = DATASET_DIR / "train_combined.pt"
TEST_PT_PATH = DATASET_DIR / "test_combined.pt"

# All outputs go under FigData (as required by the report)
OUT_DIR = PROJECT_ROOT / "FigData" / "ExploreDataset" / "combined"


# 数据布局：g1_pos(3) + g2_pos(3) + g3_pos(3) + timestamp(1) + area(1) + gt_pos(3)
#         + g1_spec(324) + g2_spec(324) + g3_spec(324) = 986
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
    """检查 torch.load 是否支持 weights_only 参数。"""

    return "weights_only" in inspect.signature(torch.load).parameters


def safe_load(path: Path, allow_unsafe: bool) -> torch.Tensor:
    """带 weights_only 的安全加载包装。"""

    try:
        if _supports_weights_only():
            return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # 某些 PyTorch 版本不支持 weights_only
        pass
    if not allow_unsafe:
        raise RuntimeError(
            "Current PyTorch does not support weights_only=True; rerun with --allow-unsafe if you trust the file."
        )
    warnings.warn("Loading without weights_only=True; only do this for trusted files.")
    return torch.load(path, map_location="cpu")


def split_fields(tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """拆分 [N, 1, 986] 张量为字段字典。"""

    if tensor.dim() != 3 or tensor.size(1) != 1:
        raise ValueError("Expected tensor shape [N,1,F].")
    flat = tensor[:, 0, :]
    return {name: flat[:, sl[0] : sl[1]] for name, sl in FIELD_SLICES.items()}


def summarize_numeric(data: torch.Tensor) -> Dict[str, float]:
    """生成均值/方差/分位数等基础统计。"""

    np_data = data.detach().cpu().numpy().reshape(len(data), -1)
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
    """创建目录后返回原 Path。"""

    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_ground_truth_xy(split: str, gt_pos: torch.Tensor, timestamp: torch.Tensor, out_path: Path) -> None:
    """绘制 GT 轨迹（按时间上色）。"""

    gt_np = gt_pos.detach().cpu().numpy()
    ts_np = timestamp.detach().cpu().numpy().flatten()
    plt.figure(figsize=(8, 7))
    sc = plt.scatter(gt_np[:, 0], gt_np[:, 1], c=ts_np, cmap="viridis", s=6, alpha=0.7)
    plt.colorbar(sc, label="timestamp")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{split}: Ground-truth trajectory vs timestamp")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_timestamp_hist(split: str, timestamp: torch.Tensor, out_path: Path) -> None:
    """时间戳直方图。"""

    ts_np = timestamp.detach().cpu().numpy().flatten()
    plt.figure(figsize=(8, 4))
    plt.hist(ts_np, bins=40, color="#4a90e2", alpha=0.85, edgecolor="white")
    plt.xlabel("timestamp")
    plt.ylabel("count")
    plt.title(f"{split}: Timestamp distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_area_bar(split: str, area: torch.Tensor, out_path: Path) -> None:
    """区域标签计数柱状图。"""

    values, counts = np.unique(area.detach().cpu().numpy().flatten(), return_counts=True)
    plt.figure(figsize=(6, 4))
    plt.bar(values.astype(str), counts, color="#7b8b6f")
    plt.xlabel("area")
    plt.ylabel("count")
    plt.title(f"{split}: Area label distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_spectrum_energy(split: str, g_specs: Dict[str, torch.Tensor], out_path: Path) -> None:
    """三路频谱的平均能量直方图（同一图内对比）。"""

    plt.figure(figsize=(9, 5))
    for name, tensor in g_specs.items():
        energy = tensor.mean(dim=1).detach().cpu().numpy()
        plt.hist(energy, bins=40, alpha=0.5, label=name)
    plt.xlabel("mean spectral power")
    plt.ylabel("count")
    plt.title(f"{split}: Gateway spectral energy distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_distance_vs_energy(
    split: str,
    gt_pos: torch.Tensor,
    gateway_pos: Dict[str, torch.Tensor],
    g_specs: Dict[str, torch.Tensor],
    out_path: Path,
) -> None:
    """距离-能量散点：观测能量随 GT 与网关距离的关系。"""

    plt.figure(figsize=(9, 5))
    gt_np = gt_pos.detach().cpu().numpy()
    for name, pos in gateway_pos.items():
        pos_np = pos.detach().cpu().numpy()
        dist = np.linalg.norm(gt_np - pos_np, axis=1)
        energy = g_specs[name.replace("_pos", "_spec")].mean(dim=1).detach().cpu().numpy()
        plt.scatter(dist, energy, s=5, alpha=0.35, label=name)
    plt.xlabel("distance to gateway")
    plt.ylabel("mean spectral power")
    plt.title(f"{split}: Distance vs spectral energy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_gt_hist(split: str, gt_pos: torch.Tensor, out_path: Path) -> None:
    """GT x/y/z 的分布直方图。"""

    gt_np = gt_pos.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = ["x", "y", "z"]
    for idx, ax in enumerate(axes):
        ax.hist(gt_np[:, idx], bins=40, color="#9b59b6", alpha=0.7, edgecolor="white")
        ax.set_xlabel(labels[idx])
        ax.set_ylabel("count")
        ax.set_title(f"{split}: gt_{labels[idx]} distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def gather_invalid_counts(fields: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """统计 NaN / Inf 数量，用于质量检查。"""

    invalid = {"nan": 0, "inf": 0}
    for tensor in fields.values():
        invalid["nan"] += int(torch.isnan(tensor).sum().item())
        invalid["inf"] += int(torch.isinf(tensor).sum().item())
    return invalid


def summarize_split(name: str, tensor: torch.Tensor) -> Dict[str, Dict]:
    """生成单个数据切分的统计，并写图。"""

    fields = split_fields(tensor)
    g_specs = {k: fields[k] for k in ["g1_spec", "g2_spec", "g3_spec"]}
    gateway_pos = {k: fields[k] for k in ["g1_pos", "g2_pos", "g3_pos"]}

    # 距离统计
    distance_stats = {}
    gt_np = fields["gt_pos"].detach().cpu().numpy()
    for k, pos in gateway_pos.items():
        dist = np.linalg.norm(gt_np - pos.detach().cpu().numpy(), axis=1)
        distance_stats[k.replace("_pos", "_dist")] = summarize_numeric(torch.from_numpy(dist))

    # 平均谱能量统计
    spectrum_energy_stats = {}
    for k, spec in g_specs.items():
        energy = spec.mean(dim=1)
        spectrum_energy_stats[k + "_mean_energy"] = summarize_numeric(energy)

    # 区域分布
    unique, counts = np.unique(fields["area"].detach().cpu().numpy(), return_counts=True)
    area_counts = {str(int(u)): int(c) for u, c in zip(unique, counts)}

    return {
        "shape": list(tensor.shape),
        "invalid": gather_invalid_counts(fields),
        "field_stats": {k: summarize_numeric(v) for k, v in fields.items()},
        "distance_stats": distance_stats,
        "spectrum_energy_stats": spectrum_energy_stats,
        "area_counts": area_counts,
    }


def render_figures(split: str, fields: Dict[str, torch.Tensor], fig_dir: Path) -> None:
    """针对单个数据切分生成全部图表。"""

    ensure_outdir(fig_dir)
    g_specs = {k: fields[k] for k in ["g1_spec", "g2_spec", "g3_spec"]}
    gateway_pos = {k: fields[k] for k in ["g1_pos", "g2_pos", "g3_pos"]}

    # Put train/test figures into separate folders and use consistent filenames.
    plot_ground_truth_xy(split, fields["gt_pos"], fields["timestamp"], fig_dir / "ground_truth_xy.png")
    plot_timestamp_hist(split, fields["timestamp"], fig_dir / "timestamp_hist.png")
    plot_area_bar(split, fields["area"], fig_dir / "area_bar.png")
    plot_gt_hist(split, fields["gt_pos"], fig_dir / "gt_hist.png")
    plot_spectrum_energy(split, g_specs, fig_dir / "spectrum_energy.png")
    plot_distance_vs_energy(split, fields["gt_pos"], gateway_pos, g_specs, fig_dir / "distance_vs_energy.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore Bluetooth spatial spectrum dataset.")
    # Defaults are fixed to the combined dataset paths.
    parser.add_argument("--train-path", default=str(TRAIN_PT_PATH), help="Path to training tensor")
    parser.add_argument("--test-path", default=str(TEST_PT_PATH), help="Path to test tensor")
    parser.add_argument(
        "--output-dir",
        default=OUT_DIR,
        help="Directory for reports and figures (default: FigData/ExploreDataset/combined)",
    )
    parser.add_argument("--allow-unsafe", action="store_true", help="Allow torch.load without weights_only for trusted files")
    args = parser.parse_args()

    out_dir = ensure_outdir(Path(args.output_dir))

    datasets = {"train": Path(args.train_path), "test": Path(args.test_path)}

    summary = {}
    for split, path in datasets.items():
        tensor = safe_load(path, allow_unsafe=args.allow_unsafe)
        fields = split_fields(tensor)
        render_figures(split, fields, out_dir / split)
        summary[split] = summarize_split(split, tensor)

    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved statistics to {out_dir / 'dataset_summary.json'}")
    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
