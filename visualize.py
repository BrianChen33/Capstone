import argparse
import torch
import matplotlib.pyplot as plt
import os
import sys
import json
import warnings

# Add current directory to path to import from train.py
sys.path.append(os.getcwd())
try:
    from train import SimpleRegressor
except ImportError:
    # Fallback if train.py is not found or fails to import
    from torch import nn
    class SimpleRegressor(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
            )
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

def safe_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        warnings.warn(f"Loading {path} without weights_only=True")
        return torch.load(path, map_location="cpu")

def main():
    parser = argparse.ArgumentParser(description="Visualize PT data and model predictions")
    parser.add_argument("data_path", help="Path to the .pt data file (e.g., test_data-s02-80-20-seq1.pt)")
    parser.add_argument("--model-path", help="Path to the .pt model file (e.g., artifacts/best_model.pt)")
    parser.add_argument("--stats-path", help="Path to dataset_stats.json (optional, for correct normalization)")
    parser.add_argument("--output", default="visualization.png", help="Output image path")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        return

    # Load Data
    print(f"Loading data from {args.data_path}...")
    data = safe_load(args.data_path)
    
    # Handle [N, 1, F] shape
    if data.dim() == 3 and data.size(1) == 1:
        features = data[:, 0, :-2]
        targets = data[:, 0, -2:]
    else:
        print(f"Unexpected data shape: {data.shape}. Expected [N, 1, F].")
        return

    # Plot Setup
    plt.figure(figsize=(10, 8))
    
    # Plot Ground Truth
    plt.scatter(targets[:, 0].numpy(), targets[:, 1].numpy(), c='blue', label='Ground Truth', alpha=0.5, s=2)

    if args.model_path:
        if not os.path.exists(args.model_path):
            print(f"Error: Model file not found at {args.model_path}")
            return

        print(f"Loading model from {args.model_path}...")
        # Determine input dim
        input_dim = features.shape[1]
        model = SimpleRegressor(input_dim)
        state_dict = safe_load(args.model_path)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Normalization
        mean = 0.0
        std = 1.0
        
        # Try to find stats
        stats_file = args.stats_path
        if not stats_file:
            # Look in artifacts folder or model folder
            candidates = [
                os.path.join(os.path.dirname(args.model_path), "dataset_stats.json"),
                "artifacts/dataset_stats.json"
            ]
            for c in candidates:
                if os.path.exists(c):
                    stats_file = c
                    break
        
        if stats_file and os.path.exists(stats_file):
            print(f"Using statistics from {stats_file}")
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            mean = stats.get('feature_mean', 0.0)
            std = stats.get('feature_std', 1.0)
        else:
            print("Warning: dataset_stats.json not found. Using stats from current file for normalization (may be inaccurate if this is test data).")
            mean = features.mean().item()
            std = features.std().item()
            
        print(f"Normalizing features (mean={mean:.4f}, std={std:.4f})...")
        norm_features = (features - mean) / std
        
        print("Running inference...")
        with torch.no_grad():
            preds = model(norm_features)
            
        plt.scatter(preds[:, 0].numpy(), preds[:, 1].numpy(), c='red', label='Prediction', alpha=0.5, s=2)
        plt.legend()
        plt.title(f"Ground Truth vs Prediction\nModel: {os.path.basename(args.model_path)}")
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
