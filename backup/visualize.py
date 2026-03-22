"""用于使用保存的统计数据和模型元数据可视化预测值与 GT 的辅助工具。"""
import argparse
import json
import os
import sys
import warnings
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn

sys.path.append(os.getcwd())


def _supports_weights_only() -> bool:
    import inspect
    # 检查 torch.load 是否支持 weights_only
    return "weights_only" in inspect.signature(torch.load).parameters


def safe_load(path):
    try:
        if _supports_weights_only():
            return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        pass
    warnings.warn(f"Loading {path} without weights_only=True")
    return torch.load(path, map_location="cpu")


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


def split_fields(tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    # 拆 [N,1,986] -> 字段 dict
    if tensor.dim() != 3 or tensor.size(1) != 1:
        raise ValueError("Expected tensor shape [N,1,F].")
    flat = tensor[:, 0, :]
    return {name: flat[:, sl[0] : sl[1]] for name, sl in FIELD_SLICES.items()}


def preprocess(fields: Dict[str, torch.Tensor], stats: dict):
    # 按训练时的统计做分块 z-score，并返回模型输入与标签
    g_pos = torch.cat([fields["g1_pos"], fields["g2_pos"], fields["g3_pos"]], dim=1)
    ts = fields["timestamp"]
    specs = [fields["g1_spec"], fields["g2_spec"], fields["g3_spec"]]

    g_pos = (g_pos - torch.tensor(stats["g_pos_mean"])) / torch.tensor(stats["g_pos_std"])
    ts = (ts - torch.tensor(stats["ts_mean"])) / torch.tensor(stats["ts_std"])
    norm_specs = []
    for i, s in enumerate(specs):
        mean = torch.tensor(stats[f"spec{i+1}_mean"])
        std = torch.tensor(stats[f"spec{i+1}_std"])
        norm_specs.append((s - mean) / std)

    spec_seq = torch.stack(norm_specs, dim=-1)
    flat_feats = torch.cat([g_pos, ts] + norm_specs, dim=1)
    meta = torch.cat([g_pos, ts], dim=1)
    targets = fields["gt_pos"][:, :2]
    return flat_feats, spec_seq, meta, targets


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
        self.head = nn.Sequential(
            nn.Linear(64 * 16 + meta_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )
    def forward(self, flat_feats, spec_seq, meta):
        x = spec_seq.permute(0, 2, 1)
        x = self.conv(x)
        x = x.flatten(1)
        x = torch.cat([x, meta], dim=1)
        return self.head(x)


class TransformerRegressor(nn.Module):
    def __init__(self, meta_dim: int, d_model: int = 32, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(3, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model + meta_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
    def forward(self, flat_feats, spec_seq, meta):
        x = self.input_proj(spec_seq)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        x = torch.cat([pooled, meta], dim=1)
        return self.head(x)


def build_model(kind: str, flat_dim: int, meta_dim: int):
    # 根据报告或命令行指定的类型构建对应模型
    kind = kind.lower()
    if kind == "mlp":
        return MLPRegressor(flat_dim)
    if kind == "cnn":
        return CNNRegressor(meta_dim)
    if kind == "transformer":
        return TransformerRegressor(meta_dim)
    raise ValueError(f"Unknown model kind: {kind}")

def main():
    # 加载数据与统计 → 预处理 → 可选加载模型并推理 → 绘制 GT vs 预测
    parser = argparse.ArgumentParser(description="Visualize PT data and model predictions")
    parser.add_argument("data_path", help="Path to the .pt data file (e.g., test_data-s02-80-20-seq1.pt)")
    parser.add_argument("--model-path", required=True, help="Path to the trained model file")
    parser.add_argument("--stats-path", required=True, help="Path to dataset_stats.json")
    parser.add_argument("--report-path", help="Path to training_report.json (to read model type)")
    parser.add_argument(
        "--model-type",
        default="mlp",
        choices=["mlp", "cnn", "transformer"],
        help="Override model type if report not provided",
    )
    parser.add_argument("--output", default="visualization.png", help="Output image path")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        return

    # 加载数据
    print(f"Loading data from {args.data_path}...")
    data = safe_load(args.data_path)
    
    if data.dim() != 3 or data.size(1) != 1:
        print(f"Unexpected data shape: {data.shape}. Expected [N, 1, F].")
        return

    # 加载统计信息
    with open(args.stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    fields = split_fields(data)
    flat_feats, spec_seq, meta, targets = preprocess(fields, stats)

    # 图表设置
    plt.figure(figsize=(10, 8))
    
    # 绘制 Ground Truth
    plt.scatter(targets[:, 0].numpy(), targets[:, 1].numpy(), c='blue', label='Ground Truth', alpha=0.5, s=2)

    if args.model_path:
        if not os.path.exists(args.model_path):
            print(f"Error: Model file not found at {args.model_path}")
            return

        model_type = args.model_type
        if args.report_path and os.path.exists(args.report_path):
            with open(args.report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            model_type = report.get("model", model_type)

        model = build_model(model_type, flat_dim=flat_feats.size(1), meta_dim=meta.size(1))
        state_dict = safe_load(args.model_path)
        model.load_state_dict(state_dict)
        model.eval()

        print(f"Running inference with {model_type}...")
        with torch.no_grad():
            preds = model(flat_feats, spec_seq, meta)

        plt.scatter(preds[:, 0].numpy(), preds[:, 1].numpy(), c='red', label='Prediction', alpha=0.5, s=2)
        plt.legend()
        plt.title(f"Ground Truth vs Prediction\nModel: {os.path.basename(args.model_path)} ({model_type})")
    else:
        plt.title(f"Ground Truth Trajectory\nData: {os.path.basename(args.data_path)}")
        
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Visualization saved to {args.output}")

if __name__ == "__main__":
    main()
