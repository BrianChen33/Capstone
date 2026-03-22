#!/usr/bin/env python3
"""
Ablation Study Script for COMP4913 Capstone Project.

This script performs ablation experiments to evaluate the contribution
of different input components to model performance.

Dataset Specification:
    - 986-dimensional features:
        - Gateway positions (9D)
        - Timestamp (1D)
        - Area ID (1D)
        - Ground truth position (3D)
        - 3 × 324D spatial spectrum = 972D
    - Preprocessing: Block-wise Z-Score normalization

Usage:
    python ablation_study.py --model MLP --ablation timestamp
    python ablation_study.py --model CNN --ablation gateway --gateway-id 0
    python ablation_study.py --model Transformer --ablation all

Author: COMP4913 Capstone Project Team
Date: 2025
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Constants
DATASET_DIR = './Dataset/'
FIGDATA_DIR = './FigData/'
OUTPUT_DIR = './output/ablation/'
RANDOM_SEED = 42

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Ablation Study for Indoor Positioning Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ablation on timestamp feature
  python ablation_study.py --model MLP --ablation timestamp

  # Ablation on specific gateway
  python ablation_study.py --model CNN --ablation gateway --gateway-id 2

  # Test all ablation configurations
  python ablation_study.py --model Transformer --ablation all

  # Custom epochs and batch size
  python ablation_study.py --model MLP --ablation spectrum --epochs 50 --batch-size 64
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        choices=['MLP', 'CNN', 'Transformer', 'ALL'],
        help='Model architecture to test (MLP, CNN, Transformer, or ALL)'
    )

    parser.add_argument(
        '--ablation', '-a',
        type=str,
        required=True,
        choices=['timestamp', 'gateway', 'spectrum', 'area', 'position', 'all'],
        help='Type of ablation study to perform'
    )

    parser.add_argument(
        '--gateway-id', '-g',
        type=int,
        default=None,
        help='Gateway ID to remove (0-2, required for gateway ablation)'
    )

    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=30,
        help='Number of training epochs (default: 30)'
    )

    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )

    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=OUTPUT_DIR,
        help=f'Output directory for results (default: {OUTPUT_DIR})'
    )

    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=RANDOM_SEED,
        help=f'Random seed for reproducibility (default: {RANDOM_SEED})'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(data_dir: str = DATASET_DIR) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset from specified directory.

    Args:
        data_dir: Directory containing dataset files

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print(f"Loading data from {data_dir}...")

    train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    # Extract features and labels
    # Assuming last 3 columns are ground truth (x, y, z)
    feature_cols = [c for c in train_data.columns if c not in ['x', 'y', 'z']]
    X_train = train_data[feature_cols].values
    y_train = train_data[['x', 'y', 'z']].values
    X_test = test_data[feature_cols].values
    y_test = test_data[['x', 'y', 'z']].values

    print(f"Data loaded: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test


def block_wise_zscore_normalize(
    X_train: np.ndarray,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Apply block-wise Z-Score normalization.

    The 986 features are divided into blocks:
    - Gateway positions: 9D
    - Timestamp: 1D
    - Area: 1D
    - Spectrum blocks: 3 × 324D

    Args:
        X_train: Training features
        X_test: Testing features

    Returns:
        Tuple of (normalized train, normalized test, normalization stats)
    """
    # Define block structure
    blocks = [
        (0, 9, 'gateway_pos'),      # Gateway positions
        (9, 10, 'timestamp'),        # Timestamp
        (10, 11, 'area'),            # Area ID
        (11, 335, 'spectrum_0'),     # Spectrum from gateway 0
        (335, 659, 'spectrum_1'),    # Spectrum from gateway 1
        (659, 983, 'spectrum_2'),    # Spectrum from gateway 2
    ]

    X_train_norm = np.zeros_like(X_train)
    X_test_norm = np.zeros_like(X_test)
    stats = {}

    for start, end, name in blocks:
        block_train = X_train[:, start:end]
        block_test = X_test[:, start:end]

        mean = np.mean(block_train, axis=0)
        std = np.std(block_train, axis=0)
        std[std == 0] = 1  # Avoid division by zero

        X_train_norm[:, start:end] = (block_train - mean) / std
        X_test_norm[:, start:end] = (block_test - mean) / std

        stats[name] = {'mean': mean, 'std': std}

    return X_train_norm, X_test_norm, stats


class MLPModel(nn.Module):
    """Multi-Layer Perceptron for indoor positioning."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128]):
        super(MLPModel, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 3))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CNNModel(nn.Module):
    """Convolutional Neural Network for indoor positioning."""

    def __init__(self, input_dim: int):
        super(CNNModel, self).__init__()
        # Reshape input to 1D signal for convolution
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(64)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class TransformerModel(nn.Module):
    """Transformer model for indoor positioning."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4
    ):
        super(TransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.output_proj(x)
        return x


def create_model(model_name: str, input_dim: int) -> nn.Module:
    """
    Create model instance based on name.

    Args:
        model_name: Name of the model (MLP, CNN, Transformer)
        input_dim: Input feature dimension

    Returns:
        Model instance
    """
    if model_name == 'MLP':
        return MLPModel(input_dim)
    elif model_name == 'CNN':
        return CNNModel(input_dim)
    elif model_name == 'Transformer':
        return TransformerModel(input_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def remove_timestamp(X: np.ndarray) -> np.ndarray:
    """
    Remove timestamp feature (index 9).

    Args:
        X: Input features

    Returns:
        Features without timestamp
    """
    return np.delete(X, 9, axis=1)


def remove_gateway(X: np.ndarray, gateway_id: int) -> np.ndarray:
    """
    Remove data from specific gateway.

    Args:
        X: Input features
        gateway_id: Gateway ID (0, 1, or 2)

    Returns:
        Features without specified gateway data
    """
    # Gateway positions: indices 0-8 (3 per gateway)
    # Spectrum blocks: indices 11-983 (324 per gateway)
    gateway_pos_start = gateway_id * 3
    gateway_pos_end = gateway_pos_start + 3

    spectrum_start = 11 + gateway_id * 324
    spectrum_end = spectrum_start + 324

    # Remove gateway positions and spectrum
    cols_to_remove = list(range(gateway_pos_start, gateway_pos_end)) + \
                     list(range(spectrum_start, spectrum_end))

    return np.delete(X, cols_to_remove, axis=1)


def remove_spectrum(X: np.ndarray) -> np.ndarray:
    """
    Remove all spectrum features, keep only metadata.

    Args:
        X: Input features

    Returns:
        Features with only metadata (gateway pos, timestamp, area)
    """
    return X[:, :11]  # Keep only first 11 features


def remove_area(X: np.ndarray) -> np.ndarray:
    """
    Remove area ID feature (index 10).

    Args:
        X: Input features

    Returns:
        Features without area ID
    """
    return np.delete(X, 10, axis=1)


def remove_position(X: np.ndarray) -> np.ndarray:
    """
    Remove gateway position features (indices 0-8).

    Args:
        X: Input features

    Returns:
        Features without gateway positions
    """
    return X[:, 9:]  # Remove first 9 features


def apply_ablation(
    X_train: np.ndarray,
    X_test: np.ndarray,
    ablation_type: str,
    gateway_id: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Apply ablation to input features.

    Args:
        X_train: Training features
        X_test: Testing features
        ablation_type: Type of ablation
        gateway_id: Gateway ID for gateway ablation

    Returns:
        Tuple of (ablated train, ablated test, ablation description)
    """
    ablation_funcs = {
        'timestamp': (remove_timestamp, "Remove timestamp"),
        'gateway': (lambda x: remove_gateway(x, gateway_id), f"Remove gateway {gateway_id}"),
        'spectrum': (remove_spectrum, "Remove all spectrum"),
        'area': (remove_area, "Remove area ID"),
        'position': (remove_position, "Remove gateway positions"),
    }

    if ablation_type not in ablation_funcs:
        raise ValueError(f"Unknown ablation type: {ablation_type}")

    func, desc = ablation_funcs[ablation_type]
    X_train_abl = func(X_train)
    X_test_abl = func(X_test)

    return X_train_abl, X_test_abl, desc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    verbose: bool = False
) -> Dict:
    """
    Train model and track performance.

    Args:
        model: Model to train
        train_loader: Training data loader
        test_loader: Testing data loader
        epochs: Number of epochs
        learning_rate: Learning rate
        verbose: Whether to print progress

    Returns:
        Training history dictionary
    """
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    history = {'train_loss': [], 'test_loss': [], 'test_mde': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Evaluation
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()

                all_preds.append(outputs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        test_loss /= len(test_loader)

        # Calculate Mean Distance Error (MDE)
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        distances = np.sqrt(np.sum((all_preds - all_targets) ** 2, axis=1))
        test_mde = np.mean(distances)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_mde'].append(test_mde)

        scheduler.step(test_loss)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, "
                  f"Test Loss={test_loss:.4f}, "
                  f"MDE={test_mde:.4f}m")

    return history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader
) -> Dict:
    """
    Evaluate model performance.

    Args:
        model: Trained model
        test_loader: Testing data loader

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate metrics
    errors = all_preds - all_targets
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))

    # Mean Distance Error (Euclidean)
    distances = np.sqrt(np.sum(errors ** 2, axis=1))
    mde = np.mean(distances)

    # Percentiles
    p50 = np.percentile(distances, 50)
    p75 = np.percentile(distances, 75)
    p90 = np.percentile(distances, 90)
    p95 = np.percentile(distances, 95)

    # Per-axis errors
    mae_x = np.mean(np.abs(errors[:, 0]))
    mae_y = np.mean(np.abs(errors[:, 1]))
    mae_z = np.mean(np.abs(errors[:, 2]))

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mde': mde,
        'p50': p50,
        'p75': p75,
        'p90': p90,
        'p95': p95,
        'mae_x': mae_x,
        'mae_y': mae_y,
        'mae_z': mae_z,
        'predictions': all_preds,
        'targets': all_targets
    }


def run_ablation_study(
    model_name: str,
    ablation_type: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace
) -> Dict:
    """
    Run single ablation experiment.

    Args:
        model_name: Model architecture name
        ablation_type: Type of ablation
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        args: Command line arguments

    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Running: {model_name} with ablation '{ablation_type}'")
    print(f"{'='*60}")

    # Apply ablation
    if ablation_type == 'gateway':
        if args.gateway_id is None:
            raise ValueError("--gateway-id must be specified for gateway ablation")
        X_train_abl, X_test_abl, desc = apply_ablation(
            X_train, X_test, ablation_type, args.gateway_id
        )
    else:
        X_train_abl, X_test_abl, desc = apply_ablation(
            X_train, X_test, ablation_type
        )

    print(f"Ablation: {desc}")
    print(f"Input dim: {X_train.shape[1]} -> {X_train_abl.shape[1]}")

    # Normalize
    X_train_norm, X_test_norm, _ = block_wise_zscore_normalize(X_train_abl, X_test_abl)

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_norm),
        torch.FloatTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_norm),
        torch.FloatTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create and train model
    model = create_model(model_name, X_train_norm.shape[1])

    if args.verbose:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = train_model(
        model, train_loader, test_loader,
        args.epochs, args.learning_rate, args.verbose
    )

    # Evaluate
    results = evaluate_model(model, test_loader)
    results['history'] = history
    results['model'] = model_name
    results['ablation'] = ablation_type
    results['description'] = desc
    results['input_dim'] = X_train_norm.shape[1]

    print(f"\nResults:")
    print(f"  MDE: {results['mde']:.4f}m")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  P90: {results['p90']:.4f}m")

    return results


def run_all_ablations(
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace
) -> List[Dict]:
    """
    Run all ablation experiments for a model.

    Args:
        model_name: Model architecture name
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        args: Command line arguments

    Returns:
        List of results dictionaries
    """
    results = []

    # Baseline (no ablation)
    print("\n" + "="*60)
    print("BASELINE (No Ablation)")
    print("="*60)
    baseline_result = run_ablation_study(
        model_name, 'none', X_train, X_test, y_train, y_test, args
    )
    baseline_result['ablation'] = 'baseline'
    baseline_result['description'] = 'No ablation (full features)'
    results.append(baseline_result)

    # Individual ablations
    ablation_types = ['timestamp', 'area', 'spectrum', 'position']

    for abl_type in ablation_types:
        result = run_ablation_study(
            model_name, abl_type, X_train, X_test, y_train, y_test, args
        )
        results.append(result)

    # Gateway ablations (for each gateway)
    for gw_id in range(3):
        args.gateway_id = gw_id
        result = run_ablation_study(
            model_name, 'gateway', X_train, X_test, y_train, y_test, args
        )
        results.append(result)

    return results


def save_results(results: List[Dict], output_dir: str) -> None:
    """
    Save ablation study results.

    Args:
        results: List of results dictionaries
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save summary table
    summary_data = []
    for r in results:
        summary_data.append({
            'Model': r['model'],
            'Ablation': r['ablation'],
            'Description': r['description'],
            'Input Dim': r['input_dim'],
            'MDE (m)': f"{r['mde']:.4f}",
            'RMSE': f"{r['rmse']:.4f}",
            'MAE (m)': f"{r['mae']:.4f}",
            'P50 (m)': f"{r['p50']:.4f}",
            'P90 (m)': f"{r['p90']:.4f}",
            'P95 (m)': f"{r['p95']:.4f}",
            'MAE X (m)': f"{r['mae_x']:.4f}",
            'MAE Y (m)': f"{r['mae_y']:.4f}",
            'MAE Z (m)': f"{r['mae_z']:.4f}",
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'ablation_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Save detailed results as JSON
    json_results = []
    for r in results:
        json_result = {k: v for k, v in r.items() if k not in ['predictions', 'targets', 'history']}
        json_results.append(json_result)

    json_path = os.path.join(output_dir, 'ablation_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Results saved to: {json_path}")

    # Print summary table
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))


def main():
    """Main function."""
    args = parse_arguments()
    set_random_seed(args.seed)

    print("="*60)
    print("ABLATION STUDY FOR INDOOR POSITIONING")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Random seed: {args.seed}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Determine models to test
    models_to_test = ['MLP', 'CNN', 'Transformer'] if args.model == 'ALL' else [args.model]

    all_results = []

    for model_name in models_to_test:
        if args.ablation == 'all':
            results = run_all_ablations(
                model_name, X_train, X_test, y_train, y_test, args
            )
        else:
            result = run_ablation_study(
                model_name, args.ablation, X_train, X_test, y_train, y_test, args
            )
            results = [result]

        all_results.extend(results)

    # Save results
    model_str = args.model if args.model != 'ALL' else 'all_models'
    ablation_str = args.ablation if args.ablation != 'all' else 'all_ablations'
    output_dir = os.path.join(args.output_dir, f"{model_str}_{ablation_str}")
    save_results(all_results, output_dir)

    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()
