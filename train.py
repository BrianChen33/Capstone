import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass
class DatasetStats:
    num_samples: int
    feature_dim: int
    feature_mean: float
    feature_std: float
    feature_min: float
    feature_max: float
    coord_min: Tuple[float, float]
    coord_max: Tuple[float, float]


def analyze_dataset(tensor: torch.Tensor) -> DatasetStats:
    """Compute simple descriptive statistics for the raw dataset tensor."""
    if tensor.dim() != 3 or tensor.size(1) != 1 or tensor.size(2) < 3:
        raise ValueError("Expected tensor shape [N, 1, F] with F>=3.")

    features = tensor[:, 0, :-2]
    coords = tensor[:, 0, -2:]

    return DatasetStats(
        num_samples=tensor.size(0),
        feature_dim=features.size(1),
        feature_mean=features.mean().item(),
        feature_std=features.std().item(),
        feature_min=features.min().item(),
        feature_max=features.max().item(),
        coord_min=(coords[:, 0].min().item(), coords[:, 1].min().item()),
        coord_max=(coords[:, 0].max().item(), coords[:, 1].max().item()),
    )


class BluetoothPositioningDataset(Dataset):
    """Dataset wrapper that separates features and 2D coordinate targets."""

    def __init__(
        self,
        tensor: torch.Tensor,
        feature_mean: torch.Tensor = None,
        feature_std: torch.Tensor = None,
    ) -> None:
        if tensor.dim() != 3 or tensor.size(1) != 1 or tensor.size(2) < 3:
            raise ValueError("Expected tensor shape [N, 1, F] with F>=3.")

        raw_features = tensor[:, 0, :-2]
        targets = tensor[:, 0, -2:]

        if feature_mean is None or feature_std is None:
            feature_mean = raw_features.mean(dim=0)
            feature_std = raw_features.std(dim=0).clamp(min=1e-6)

        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.features = (raw_features - self.feature_mean) / self.feature_std
        self.targets = targets

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class SimpleRegressor(nn.Module):
    """Small MLP baseline for 2D coordinate regression."""

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


def save_stats(stats: DatasetStats, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats.__dict__, f, indent=2)


def load_tensors(train_path: str, test_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    train_tensor = torch.load(train_path)
    test_tensor = torch.load(test_path)
    return train_tensor, test_tensor


def make_datasets(
    train_tensor: torch.Tensor, test_tensor: torch.Tensor, val_ratio: float = 0.2
) -> Tuple[BluetoothPositioningDataset, BluetoothPositioningDataset, BluetoothPositioningDataset]:
    generator = torch.Generator().manual_seed(42)
    full_train = BluetoothPositioningDataset(train_tensor)
    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    train_subset, val_subset = random_split(full_train, [train_size, val_size], generator=generator)

    # Use the training normalization stats for validation and test to avoid leakage.
    test_dataset = BluetoothPositioningDataset(
        test_tensor, feature_mean=full_train.feature_mean, feature_std=full_train.feature_std
    )
    return train_subset, val_subset, test_dataset


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.MSELoss()
    mae_sum = 0.0
    mse_sum = 0.0
    num_batches = 0

    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            preds = model(features)
            mse = criterion(preds, targets)
            mae = (preds - targets).abs().mean()
            mse_sum += mse.item()
            mae_sum += mae.item()
            num_batches += 1

    return {
        "mse": mse_sum / max(num_batches, 1),
        "mae": mae_sum / max(num_batches, 1),
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Dict[str, float]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(features)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{epochs} - train_mse: {avg_loss:.4f} "
            f"val_mse: {val_metrics['mse']:.4f} val_mae: {val_metrics['mae']:.4f}"
        )

        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_mse": best_val}


def main() -> None:
    parser = argparse.ArgumentParser(description="Indoor positioning baseline training.")
    parser.add_argument("--train-path", default="train_data-s02-80-20-seq1.pt", help="Path to training tensor")
    parser.add_argument("--test-path", default="test_data-s02-80-20-seq1.pt", help="Path to test tensor")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--output-dir", default="artifacts", help="Directory to store outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tensor, test_tensor = load_tensors(args.train_path, args.test_path)
    feature_dim = train_tensor.size(2) - 2
    stats = analyze_dataset(train_tensor)
    os.makedirs(args.output_dir, exist_ok=True)
    save_stats(stats, os.path.join(args.output_dir, "dataset_stats.json"))
    print(f"Loaded training data: {stats.num_samples} samples, feature_dim={stats.feature_dim}")
    print(
        f"Feature range [{stats.feature_min:.4f}, {stats.feature_max:.4f}], "
        f"mean={stats.feature_mean:.4f}, std={stats.feature_std:.4f}"
    )
    print(
        f"Coordinate range x[{stats.coord_min[0]:.4f}, {stats.coord_max[0]:.4f}] "
        f"y[{stats.coord_min[1]:.4f}, {stats.coord_max[1]:.4f}]"
    )

    train_dataset, val_dataset, test_dataset = make_datasets(train_tensor, test_tensor, val_ratio=args.val_ratio)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = SimpleRegressor(input_dim=feature_dim).to(device)
    train(model, train_loader, val_loader, device=device, epochs=args.epochs, lr=args.lr)

    test_metrics = evaluate(model, test_loader, device)
    print(f"Test mse: {test_metrics['mse']:.4f}, test mae: {test_metrics['mae']:.4f}")

    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
