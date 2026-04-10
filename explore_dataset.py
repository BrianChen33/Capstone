"""数据集探索脚本。

针对项目使用的蓝牙 AoA 空间谱数据，生成统计 JSON 文件（逐场景 + 汇总），
供报告引用及可视化脚本 (visualization.py) 使用。

输出写入 FigData/ExploreDataset/。
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from shared import FIELD_SLICES, safe_load, split_fields, SCENE_IDS, get_scene_paths, PROJECT_ROOT

# =====================
# Output directory
# =====================
DATASET_DIR = PROJECT_ROOT / "Dataset"
OUT_DIR = PROJECT_ROOT / "FigData" / "ExploreDataset"


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


def gather_invalid_counts(fields: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """统计 NaN / Inf 数量，用于质量检查。"""
    invalid = {"nan": 0, "inf": 0}
    for tensor in fields.values():
        invalid["nan"] += int(torch.isnan(tensor).sum().item())
        invalid["inf"] += int(torch.isinf(tensor).sum().item())
    return invalid


def summarize_split(name: str, tensor: torch.Tensor) -> Dict:
    """生成单个数据切分的统计。"""
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
        "num_samples": int(tensor.shape[0]),
        "invalid": gather_invalid_counts(fields),
        "field_stats": {k: summarize_numeric(v) for k, v in fields.items()},
        "distance_stats": distance_stats,
        "spectrum_energy_stats": spectrum_energy_stats,
        "area_counts": area_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore Bluetooth spatial spectrum dataset.")
    parser.add_argument("--scenes", nargs="*", default=None, help="Scene IDs (default: all)")
    parser.add_argument("--allow-unsafe", action="store_true")
    args = parser.parse_args()

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = args.scenes or SCENE_IDS
    summary = {}

    for scene_id in scenes:
        print(f"Exploring scene: {scene_id}")
        train_path, test_path = get_scene_paths(scene_id)
        train_tensor = safe_load(train_path, allow_unsafe=args.allow_unsafe or True)
        test_tensor = safe_load(test_path, allow_unsafe=args.allow_unsafe or True)

        summary[scene_id] = {
            "train": summarize_split(f"{scene_id}/train", train_tensor),
            "test": summarize_split(f"{scene_id}/test", test_tensor),
        }

    # Save per-scene + combined summary
    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved statistics to {out_dir / 'dataset_summary.json'}")
    print("Run visualization.py to generate all dataset figures.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
