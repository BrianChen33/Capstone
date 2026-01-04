"""Preprocessing experiments for Bluetooth positioning dataset.
Compares normalization strategies with quick validation metrics.
"""
import argparse
import inspect
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
    # 是否支持 weights_only（PyTorch 版本相关）
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
    # 拆 [N,1,986] -> 字段 dict
    if tensor.dim() != 3 or tensor.size(1) != 1:
        raise ValueError("Expected tensor shape [N,1,F].")
    flat = tensor[:, 0, :]
    return {name: flat[:, sl[0] : sl[1]] for name, sl in FIELD_SLICES.items()}


def compute_stats(train_fields: Dict[str, torch.Tensor]) -> dict:
    # 计算多种统计（均值/方差/分位）供不同归一化策略使用
    stats: dict = {}
    g_pos = torch.cat([train_fields["g1_pos"], train_fields["g2_pos"], train_fields["g3_pos"]], dim=1)
    stats["g_pos_mean"] = g_pos.mean(dim=0)
    stats["g_pos_std"] = g_pos.std(dim=0).clamp(min=1e-6)
    stats["g_pos_min"] = g_pos.min(dim=0).values
    stats["g_pos_max"] = g_pos.max(dim=0).values

    ts = train_fields["timestamp"]
    stats["ts_mean"] = ts.mean(dim=0)
    stats["ts_std"] = ts.std(dim=0).clamp(min=1e-6)
    stats["ts_min"] = ts.min(dim=0).values
    stats["ts_max"] = ts.max(dim=0).values

    for i, key in enumerate(["g1_spec", "g2_spec", "g3_spec"], start=1):
        s = train_fields[key]
        stats[f"spec{i}_mean"] = s.mean(dim=0)
        stats[f"spec{i}_std"] = s.std(dim=0).clamp(min=1e-6)
        stats[f"spec{i}_median"] = s.median(dim=0).values
        q1 = s.quantile(0.25, dim=0)
        q3 = s.quantile(0.75, dim=0)
        stats[f"spec{i}_iqr"] = (q3 - q1).clamp(min=1e-6)
    return stats


def build_features(fields: Dict[str, torch.Tensor], stats: dict, strategy: str) -> torch.Tensor:
    # 根据策略生成输入特征：原始 / 分块 z-score / robust（IQR）
    g_pos = torch.cat([fields["g1_pos"], fields["g2_pos"], fields["g3_pos"]], dim=1)
    ts = fields["timestamp"]
    specs = [fields["g1_spec"], fields["g2_spec"], fields["g3_spec"]]

    if strategy == "raw":
        return torch.cat([g_pos, ts] + specs, dim=1)

    if strategy == "block_zscore":
        g_pos = (g_pos - stats["g_pos_mean"]) / stats["g_pos_std"]
        ts = (ts - stats["ts_mean"]) / stats["ts_std"]
        norm_specs = []
        for i, s in enumerate(specs):
            mean = stats[f"spec{i+1}_mean"]
            std = stats[f"spec{i+1}_std"]
            norm_specs.append((s - mean) / std)
        return torch.cat([g_pos, ts] + norm_specs, dim=1)

    if strategy == "robust":
        g_pos = torch.clamp(g_pos, 0.0, 1.0)
        g_pos = (g_pos - stats["g_pos_min"]) / (stats["g_pos_max"] - stats["g_pos_min"] + 1e-6)
        ts = (ts - stats["ts_min"]) / (stats["ts_max"] - stats["ts_min"] + 1e-6)
        norm_specs = []
        for i, s in enumerate(specs):
            med = stats[f"spec{i+1}_median"]
            iqr = stats[f"spec{i+1}_iqr"]
            x = (s - med) / (iqr + 1e-6)
            x = torch.clamp(x, -3.0, 3.0)
            norm_specs.append(x)
        return torch.cat([g_pos, ts] + norm_specs, dim=1)

    raise ValueError(f"Unknown strategy {strategy}")


class PairDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx]


def make_loaders(features: torch.Tensor, targets: torch.Tensor, batch_size: int = 64):
    # 固定种子划分 8:2 训练/验证
    N = features.size(0)
    val_size = int(0.2 * N)
    idx = torch.randperm(N, generator=torch.Generator().manual_seed(42))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    train_ds = PairDataset(features[train_idx], targets[train_idx])
    val_ds = PairDataset(features[val_idx], targets[val_idx])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


class SimpleRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_one(features: torch.Tensor, targets: torch.Tensor, epochs: int = 5, lr: float = 1e-3, batch_size: int = 64):
    # 用简单 MLP 快速验证不同预处理的效果，返回验证集 MSE/MAE
    loader_train, loader_val = make_loaders(features, targets, batch_size=batch_size)
    model = SimpleRegressor(features.size(1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = math.inf

    for _ in range(epochs):
        model.train()
        for x, y in loader_train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
        model.eval()
        mse_sum = 0.0
        mae_sum = 0.0
        n = 0
        with torch.no_grad():
            for x, y in loader_val:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                mse_sum += loss_fn(pred, y).item() * x.size(0)
                mae_sum += (pred - y).abs().mean().item() * x.size(0)
                n += x.size(0)
        val_mse = mse_sum / n
        if val_mse < best_val:
            best_val = val_mse
            best_state = model.state_dict()
    model.load_state_dict(best_state)
    # final metrics
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader_val:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            mse_sum += loss_fn(pred, y).item() * x.size(0)
            mae_sum += (pred - y).abs().mean().item() * x.size(0)
            n += x.size(0)
    return {"val_mse": mse_sum / n, "val_mae": mae_sum / n}


def run_experiments(train_path: Path, test_path: Path, allow_unsafe: bool, out_dir: Path):
    # 跑三种预处理策略，比较验证 MAE，并保存柱状图与指标 JSON
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(exist_ok=True)

    train_tensor = safe_load(train_path, allow_unsafe=allow_unsafe)
    _ = safe_load(test_path, allow_unsafe=allow_unsafe)  # kept for symmetry
    train_fields = split_fields(train_tensor)
    targets = train_fields["gt_pos"][:, :2]

    stats = compute_stats(train_fields)
    strategies = ["raw", "block_zscore", "robust"]
    results = {}

    for name in strategies:
        feats = build_features(train_fields, stats, strategy=name)
        metrics = train_one(feats, targets, epochs=5)
        results[name] = metrics
        print(f"{name}: val_mse={metrics['val_mse']:.4f}, val_mae={metrics['val_mae']:.4f}")

    labels = list(results.keys())
    mae_vals = [results[k]["val_mae"] for k in labels]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, mae_vals, color=["#4a90e2", "#50e3c2", "#f5a623"])
    plt.ylabel("Validation MAE")
    plt.title("Preprocessing strategy comparison")
    plt.tight_layout()
    plt.savefig(fig_dir / "val_mae_comparison.png")
    plt.close()

    with open(out_dir / "preprocess_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results, fig_dir / "val_mae_comparison.png"


def main():
    ap = argparse.ArgumentParser(description="Run preprocessing experiments.")
    ap.add_argument("--train-path", default="train_data-s02-80-20-seq1.pt")
    ap.add_argument("--test-path", default="test_data-s02-80-20-seq1.pt")
    ap.add_argument("--output-dir", default="artifacts/preprocess")
    ap.add_argument("--allow-unsafe", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    run_experiments(Path(args.train_path), Path(args.test_path), args.allow_unsafe, out_dir)


if __name__ == "__main__":
    main()
