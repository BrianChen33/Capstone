"""使用逐块 Z-score 预处理的训练脚本。
使用 MLP 主干网络（模型对比中的推荐胜出者）。
"""
import argparse
import inspect
import json
import os
import warnings
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# 字段布局
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
    # 检查当前 torch.load 是否支持 weights_only 参数（PyTorch 2.0+ 才有）
    return "weights_only" in inspect.signature(torch.load).parameters


def safe_load(path: str, allow_unsafe: bool) -> torch.Tensor:
    try:
        if _supports_weights_only():
            return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        pass
    if not allow_unsafe:
        raise RuntimeError(
            "This PyTorch version does not support weights_only=True; rerun with --allow-unsafe-load if you trust the file."
        )
    warnings.warn("Loading without weights_only=True; only do this for trusted files.")
    return torch.load(path, map_location="cpu")


def split_fields(tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    # 将形状 [N,1,986] 的张量按字段切片成 dict，便于后续预处理
    if tensor.dim() != 3 or tensor.size(1) != 1:
        raise ValueError("Expected tensor shape [N,1,F].")
    flat = tensor[:, 0, :]
    return {name: flat[:, sl[0] : sl[1]] for name, sl in FIELD_SLICES.items()}


def compute_stats(train_fields: Dict[str, torch.Tensor]) -> dict:
    # 只用训练集估计均值/方差，推理和验证共用
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
    stats["num_samples"] = train_fields["g1_pos"].size(0)
    return stats


def preprocess_block_zscore(fields: Dict[str, torch.Tensor], stats: dict):
    # 分块 z-score：位置合并归一化，时间戳单独归一化，三路频谱各自归一化
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
    targets = fields["gt_pos"][:, :2]
    return flat_feats, spec_seq, meta, targets


class PositionDataset(Dataset):
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


def make_loaders(dataset: Dataset, val_ratio: float, batch_size: int):
    # 固定种子划分训练/验证，确保可复现
    N = len(dataset)
    val_size = int(val_ratio * N)
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


class PositionalEncoding(nn.Module):
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
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
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


def build_model(kind: str, flat_dim: int, meta_dim: int) -> nn.Module:
    kind = kind.lower()
    if kind == "mlp":
        return MLPRegressor(flat_dim)
    if kind == "cnn":
        return CNNRegressor(meta_dim=meta_dim)
    if kind == "transformer":
        return TransformerRegressor(meta_dim=meta_dim)
    raise ValueError(f"Unknown model kind: {kind}")


def train_model(model: nn.Module, loaders, device: torch.device, epochs: int, lr: float):
    # 训练+验证，记录最佳验证 MSE 并恢复
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader, val_loader = loaders
    best_state = None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for flat_feats, spec_seq, meta, y in train_loader:
            flat_feats, spec_seq, meta, y = (
                flat_feats.to(device),
                spec_seq.to(device),
                meta.to(device),
                y.to(device),
            )
            opt.zero_grad()
            pred = model(flat_feats, spec_seq, meta)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        model.eval()
        mse_sum = 0.0
        mae_sum = 0.0
        n = 0
        with torch.no_grad():
            for flat_feats, spec_seq, meta, y in val_loader:
                flat_feats, spec_seq, meta, y = (
                    flat_feats.to(device),
                    spec_seq.to(device),
                    meta.to(device),
                    y.to(device),
                )
                pred = model(flat_feats, spec_seq, meta)
                mse_sum += loss_fn(pred, y).item() * y.size(0)
                mae_sum += (pred - y).abs().mean().item() * y.size(0)
                n += y.size(0)
        val_mse = mse_sum / n
        val_mae = mae_sum / n
        print(
            f"Epoch {epoch}/{epochs} - train_mse: {train_loss/len(train_loader):.4f} val_mse: {val_mse:.4f} val_mae: {val_mae:.4f}"
        )
        if val_mse < best_val:
            best_val = val_mse
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_mse": best_val}


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    loss_fn = nn.MSELoss()
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    n = 0
    with torch.no_grad():
        for flat_feats, spec_seq, meta, y in loader:
            flat_feats, spec_seq, meta, y = (
                flat_feats.to(device),
                spec_seq.to(device),
                meta.to(device),
                y.to(device),
            )
            pred = model(flat_feats, spec_seq, meta)
            mse_sum += loss_fn(pred, y).item() * y.size(0)
            mae_sum += (pred - y).abs().mean().item() * y.size(0)
            n += y.size(0)
    return {"mse": mse_sum / n, "mae": mae_sum / n}


def save_stats(stats: dict, path: str):
    # 统计包含 tensor，需转 list 才能 JSON 序列化
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 将 tensor 转换为 list
    serializable = {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in stats.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def save_report(path: str, content: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)


def main():
    # 加载数据 → 预处理 → 划分 → 训练 → 测试 → 落盘模型与统计
    parser = argparse.ArgumentParser(description="Train positioning models with block-wise z-score.")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--model", default="mlp", choices=["mlp", "cnn", "transformer"], help="Backbone")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--allow-unsafe-load", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tensor = safe_load(args.train_path, allow_unsafe=args.allow_unsafe_load)
    test_tensor = safe_load(args.test_path, allow_unsafe=args.allow_unsafe_load)

    train_fields = split_fields(train_tensor)
    test_fields = split_fields(test_tensor)

    stats = compute_stats(train_fields)
    flat_train, seq_train, meta_train, y_train = preprocess_block_zscore(train_fields, stats)
    flat_test, seq_test, meta_test, y_test = preprocess_block_zscore(test_fields, stats)

    dataset = PositionDataset(flat_train, seq_train, meta_train, y_train)
    loaders = make_loaders(dataset, val_ratio=args.val_ratio, batch_size=args.batch_size)

    model = build_model(args.model, flat_dim=flat_train.size(1), meta_dim=meta_train.size(1)).to(device)
    train_model(model, loaders, device=device, epochs=args.epochs, lr=args.lr)

    # 测试指标
    test_loader = DataLoader(PositionDataset(flat_test, seq_test, meta_test, y_test), batch_size=args.batch_size)
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test mse: {test_metrics['mse']:.4f}, test mae: {test_metrics['mae']:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
    save_stats(stats, os.path.join(args.output_dir, "dataset_stats.json"))
    save_report(
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
