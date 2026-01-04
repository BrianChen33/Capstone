"""Model comparison experiments (MLP vs CNN vs LSTM vs Transformer) on Bluetooth positioning.
Uses block-wise z-score preprocessing as default best practice.
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
    # 检查 torch.load 是否支持 weights_only 参数
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
    # 拆分 [N,1,986] → 字段 dict
    if tensor.dim() != 3 or tensor.size(1) != 1:
        raise ValueError("Expected tensor shape [N,1,F].")
    flat = tensor[:, 0, :]
    return {name: flat[:, sl[0] : sl[1]] for name, sl in FIELD_SLICES.items()}


def compute_stats(train_fields: Dict[str, torch.Tensor]) -> dict:
    # 仅用训练集计算各块均值/方差
    stats: dict = {}
    g_pos = torch.cat([train_fields["g1_pos"], train_fields["g2_pos"], train_fields["g3_pos"]], dim=1)
    stats["g_pos_mean"] = g_pos.mean(dim=0)
    stats["g_pos_std"] = g_pos.std(dim=0).clamp(min=1e-6)

    ts = train_fields["timestamp"]
    stats["ts_mean"] = ts.mean(dim=0)
    stats["ts_std"] = ts.std(dim=0).clamp(min=1e-6)

    for i, key in enumerate(["g1_spec", "g2_spec", "g3_spec"], start=1):
        s = train_fields[key]
        stats[f"spec{i}_mean"] = s.mean(dim=0)
        stats[f"spec{i}_std"] = s.std(dim=0).clamp(min=1e-6)
    return stats


def preprocess_block_zscore(fields: Dict[str, torch.Tensor], stats: dict):
    # 位置合并 z-score，时间戳单独，三路频谱各自
    g_pos = torch.cat([fields["g1_pos"], fields["g2_pos"], fields["g3_pos"]], dim=1)
    ts = fields["timestamp"]
    specs = [fields["g1_spec"], fields["g2_spec"], fields["g3_spec"]]

    g_pos = (g_pos - stats["g_pos_mean"]) / stats["g_pos_std"]
    ts = (ts - stats["ts_mean"]) / stats["ts_std"]
    norm_specs = []
    for i, s in enumerate(specs):
        mean = stats[f"spec{i+1}_mean"]
        std = stats[f"spec{i+1}_std"]
        norm_specs.append((s - mean) / std)

    spec_seq = torch.stack(norm_specs, dim=-1)  # (N, 324, 3)
    flat_feats = torch.cat([g_pos, ts] + norm_specs, dim=1)
    meta = torch.cat([g_pos, ts], dim=1)
    return flat_feats, spec_seq, meta


class SpecSequenceDataset(Dataset):
    def __init__(self, flat_feats: torch.Tensor, spec_seq: torch.Tensor, meta: torch.Tensor, targets: torch.Tensor):
        self.flat_feats = flat_feats
        self.spec_seq = spec_seq
        self.meta = meta
        self.targets = targets

    def __len__(self):
        return self.targets.size(0)

    def __getitem__(self, idx: int):
        return (
            self.flat_feats[idx],
            self.spec_seq[idx],
            self.meta[idx],
            self.targets[idx],
        )


def make_loaders(dataset: SpecSequenceDataset, batch_size: int = 32):
    # 固定随机种子划分 8:2 训练/验证
    N = len(dataset)
    val_size = int(0.2 * N)
    idx = torch.randperm(N, generator=torch.Generator().manual_seed(42))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    def subset(idxs):
        return torch.utils.data.Subset(dataset, idxs)

    train_ds = subset(train_idx)
    val_ds = subset(val_idx)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def forward(self, flat_feats, spec_seq, meta):
        return self.net(flat_feats)


class CNNRegressor(nn.Module):
    def __init__(self, meta_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        conv_out = 64 * 16
        self.head = nn.Sequential(
            nn.Linear(conv_out + meta_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def forward(self, flat_feats, spec_seq, meta):
        x = spec_seq.permute(0, 2, 1)  # (B,3,324)
        x = self.conv(x)
        x = x.flatten(1)
        x = torch.cat([x, meta], dim=1)
        return self.head(x)


class LSTMRegressor(nn.Module):
    def __init__(self, meta_dim: int, hidden: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + meta_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, flat_feats, spec_seq, meta):
        out, _ = self.lstm(spec_seq)
        h = out[:, -1, :]
        x = torch.cat([h, meta], dim=1)
        return self.head(x)


class PositionalEncoding(nn.Module):
    """Learnable positional embedding for sequences."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos[:, : x.size(1), :]


class TransformerRegressor(nn.Module):
    def __init__(self, meta_dim: int, d_model: int = 32, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(3, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model + meta_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, flat_feats, spec_seq, meta):
        x = self.input_proj(spec_seq)
        x = self.pos(x)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        x = torch.cat([pooled, meta], dim=1)
        return self.head(x)


class ConvTransformerRegressor(nn.Module):
    """CNN stem to shorten spectrum sequence, Transformer with CLS + meta token."""

    def __init__(self, meta_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(3, d_model, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.meta_proj = nn.Linear(meta_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model + meta_dim),
            nn.Linear(d_model + meta_dim, 192),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(192, 2),
        )

    def forward(self, flat_feats, spec_seq, meta):
        x = spec_seq.permute(0, 2, 1)  # (B, 3, 324)
        x = self.stem(x)  # (B, d_model, L') with L' ~ 162
        x = x.permute(0, 2, 1)
        cls_token = self.cls.expand(x.size(0), 1, -1)
        meta_token = self.meta_proj(meta).unsqueeze(1)
        x = torch.cat([cls_token, meta_token, x], dim=1)
        x = self.pos(x)
        x = self.encoder(x)
        pooled = x[:, 0, :]
        x = torch.cat([pooled, meta], dim=1)
        return self.head(x)


def train_model(model, loaders, epochs: int = 6, lr: float = 1e-3, weight_decay: float = 1e-4):
    # 训练/验证循环，取最佳验证 MSE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    train_loader, val_loader = loaders
    best_state = None
    best_val = math.inf

    for _ in range(epochs):
        model.train()
        for flat_feats, spec_seq, meta, y in train_loader:
            flat_feats, spec_seq, meta, y = flat_feats.to(device), spec_seq.to(device), meta.to(device), y.to(device)
            opt.zero_grad()
            pred = model(flat_feats, spec_seq, meta)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
        model.eval()
        mse_sum = 0.0
        mae_sum = 0.0
        n = 0
        with torch.no_grad():
            for flat_feats, spec_seq, meta, y in val_loader:
                flat_feats, spec_seq, meta, y = flat_feats.to(device), spec_seq.to(device), meta.to(device), y.to(device)
                pred = model(flat_feats, spec_seq, meta)
                mse_sum += loss_fn(pred, y).item() * y.size(0)
                mae_sum += (pred - y).abs().mean().item() * y.size(0)
                n += y.size(0)
        val_mse = mse_sum / n
        if val_mse < best_val:
            best_val = val_mse
            best_state = model.state_dict()
    model.load_state_dict(best_state)
    # final eval
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    n = 0
    with torch.no_grad():
        for flat_feats, spec_seq, meta, y in val_loader:
            flat_feats, spec_seq, meta, y = flat_feats.to(device), spec_seq.to(device), meta.to(device), y.to(device)
            pred = model(flat_feats, spec_seq, meta)
            mse_sum += loss_fn(pred, y).item() * y.size(0)
            mae_sum += (pred - y).abs().mean().item() * y.size(0)
            n += y.size(0)
    return {"val_mse": mse_sum / n, "val_mae": mae_sum / n}


def run_compare(train_path: Path, allow_unsafe: bool, out_dir: Path, epochs: int = 10):
    # 跑全模型对比：训练集 → 预处理 → 多模型 → 保存指标和柱状图
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(exist_ok=True)

    train_tensor = safe_load(train_path, allow_unsafe=allow_unsafe)
    fields = split_fields(train_tensor)
    targets = fields["gt_pos"][:, :2]
    stats = compute_stats(fields)
    flat_feats, spec_seq, meta = preprocess_block_zscore(fields, stats)

    dataset = SpecSequenceDataset(flat_feats, spec_seq, meta, targets)
    loaders = make_loaders(dataset, batch_size=32)

    meta_dim = meta.size(1)
    # Note: Transformer variants remain below the MLP on this dataset; keep for reference only.
    models = {
        "mlp": (MLPRegressor(flat_feats.size(1)), 1e-3),
        "cnn": (CNNRegressor(meta_dim=meta_dim), 1e-3),
        "lstm": (LSTMRegressor(meta_dim=meta_dim), 1e-3),
        "transformer_base": (
            TransformerRegressor(meta_dim=meta_dim, d_model=32, nhead=4, num_layers=2, dropout=0.1),
            1e-3,
        ),
        "transformer_hybrid": (ConvTransformerRegressor(meta_dim=meta_dim, d_model=64, nhead=4, num_layers=4, dropout=0.1), 5e-4),
    }

    results = {}
    for name, (model, lr) in models.items():
        print(f"\nTraining {name}")
        metrics = train_model(model, loaders, epochs=epochs, lr=lr)
        results[name] = metrics
        print(f"{name}: val_mse={metrics['val_mse']:.4f}, val_mae={metrics['val_mae']:.4f}")

    labels = list(results.keys())
    mae_vals = [results[k]["val_mae"] for k in labels]
    colors = ["#4a90e2", "#50e3c2", "#f5a623", "#bd10e0", "#7ed321"]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, mae_vals, color=colors[: len(labels)])
    plt.ylabel("Validation MAE")
    plt.title("Model comparison")
    plt.tight_layout()
    plt.savefig(fig_dir / "model_val_mae.png")
    plt.close()

    with open(out_dir / "model_compare_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results, fig_dir / "model_val_mae.png"


def main():
    ap = argparse.ArgumentParser(description="Compare model architectures for positioning.")
    ap.add_argument("--train-path", default="train_data-s02-80-20-seq1.pt")
    ap.add_argument("--output-dir", default="artifacts/model_compare")
    ap.add_argument("--allow-unsafe", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    run_compare(Path(args.train_path), args.allow_unsafe, out_dir, epochs=args.epochs)


if __name__ == "__main__":
    main()
