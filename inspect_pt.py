# inspect_pt.py
import inspect
import json
import warnings
from pathlib import Path
import argparse

import torch
import numpy as np

def supports_weights_only():
    # 检查 torch.load 是否支持 weights_only 参数
    return "weights_only" in inspect.signature(torch.load).parameters

def safe_load(path: str, allow_unsafe: bool):
    if supports_weights_only():
        return torch.load(path, map_location="cpu", weights_only=True)
    if not allow_unsafe:
        raise RuntimeError("当前 PyTorch 不支持 weights_only=True；若信任文件，可用 --allow-unsafe")
    warnings.warn("使用不安全加载（仅在信任文件时使用）")
    return torch.load(path, map_location="cpu")

def summarize_tensor(t: torch.Tensor):
    # 输出张量基本统计，避免打印完整内容
    info = {
        "type": type(t).__name__,
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "min": float(t.min().item()) if t.numel()>0 else None,
        "max": float(t.max().item()) if t.numel()>0 else None,
        "mean": float(t.float().mean().item()) if t.numel()>0 else None,
    }
    return info

def main():
    # CLI：查看 .pt 结构，可选保存为 npz 并展示前若干行
    p = argparse.ArgumentParser()
    p.add_argument("path", help=".pt 文件路径")
    p.add_argument("--allow-unsafe", action="store_true", help="如果必须且信任文件，允许不安全加载")
    p.add_argument("--save-npz", help="将所有 tensor 导出为 .npz（提供路径）")
    p.add_argument("--show-samples", type=int, default=2, help="如果是矩阵/张量，展示前几个样本（默认2）")
    args = p.parse_args()

    obj = safe_load(args.path, allow_unsafe=args.allow_unsafe)
    out = {"file": str(args.path), "top_type": type(obj).__name__}

    if isinstance(obj, torch.Tensor):
        out["content"] = summarize_tensor(obj)
        print(json.dumps(out, indent=2))
        # show a few rows if shape[0] exists
        if obj.dim() >= 1:
            n = min(args.show_samples, obj.shape[0])
            print(f"\nFirst {n} rows (as numpy):\n", obj[:n].numpy())
    elif isinstance(obj, dict):
        out["keys"] = {}
        for k, v in obj.items():
            if isinstance(v, torch.Tensor):
                out["keys"][k] = summarize_tensor(v)
            else:
                out["keys"][k] = {"type": type(v).__name__}
        print(json.dumps(out, indent=2))
        # optionally save tensor dict to npz
        if args.save_npz:
            npz = {}
            for k, v in obj.items():
                if isinstance(v, torch.Tensor):
                    npz[k] = v.cpu().numpy()
            np.savez(args.save_npz, **npz)
            print(f"Saved tensors to {args.save_npz}")
    else:
        # fallback: print repr summary
        out["repr_len"] = len(repr(obj))
        print(json.dumps(out, indent=2))
        print("Top-level repr (truncated):")
        print(repr(obj)[:1000])

if __name__ == "__main__":
    main()