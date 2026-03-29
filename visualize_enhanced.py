#!/usr/bin/env python3
"""
Enhanced Visualization Script for COMP4913 Capstone Project.

This script provides comprehensive visualization capabilities for indoor
positioning results, including 3D trajectory visualization, error histograms,
CDF plots, and spatial heatmaps.

Dataset Specification:
    - 986-dimensional features
    - Ground truth: (x, y, z) coordinates
    - Predictions: Model output coordinates

Visualization Types:
    1. 3D trajectory visualization
    2. Error histograms
    3. CDF (Cumulative Distribution Function) plots
    4. Spatial error distribution heatmaps
    5. Confusion matrices for area classification
    6. Time-series error plots

Usage:
    python visualize_enhanced.py --predictions preds.npy --targets targets.npy
    python visualize_enhanced.py --npz-file results.npz --viz-type all
    python visualize_enhanced.py --csv results.csv --viz-type 3d_trajectory

E.g.
c:\Users\chenb\Desktop\个人资料\nextjs-dashboard\Capstone\.venv\Scripts\python.exe c:/Users/chenb/Desktop/个人资料/nextjs-dashboard/Capstone/visualize_enhanced.py -p artifacts/test_predictions.npy -t artifacts/test_targets.npy --viz-type all
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 100

# Constants
DATASET_DIR = './Dataset/'
FIGDATA_DIR = './FigData/'
OUTPUT_DIR = './FigData/Visualization/'
RANDOM_SEED = 42


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Enhanced Visualization for Indoor Positioning Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all visualizations
  python visualize_enhanced.py --predictions preds.npy --targets targets.npy --viz-type all

  # 3D trajectory only
  python visualize_enhanced.py --npz-file results.npz --viz-type 3d_trajectory

  # Error analysis plots
  python visualize_enhanced.py --csv results.csv --viz-type histogram cdf heatmap

  # With custom output
  python visualize_enhanced.py -p preds.npy -t targets.npy -o ./my_figures/ --format pdf
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--predictions', '-p',
        type=str,
        help='Path to predictions .npy file (N x 3 array)'
    )
    input_group.add_argument(
        '--npz-file', '-npz',
        type=str,
        help='Path to .npz file containing predictions and targets'
    )
    input_group.add_argument(
        '--csv', '-c',
        type=str,
        help='Path to CSV file with predictions and targets'
    )

    parser.add_argument(
        '--targets', '-t',
        type=str,
        help='Path to targets .npy file (required with --predictions)'
    )

    parser.add_argument(
        '--pred-cols',
        type=str,
        nargs=3,
        default=['pred_x', 'pred_y', 'pred_z'],
        help='Column names for predictions in CSV'
    )

    parser.add_argument(
        '--target-cols',
        type=str,
        nargs=3,
        default=['true_x', 'true_y', 'true_z'],
        help='Column names for targets in CSV'
    )

    parser.add_argument(
        '--viz-type', '-v',
        type=str,
        nargs='+',
        default=['all'],
        choices=['all', '3d_trajectory', 'histogram', 'cdf', 'heatmap',
                 'error_vector', 'scatter', 'timeseries', 'boxplot', 'violin'],
        help='Types of visualizations to generate'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=OUTPUT_DIR,
        help=f'Output directory for figures (default: {OUTPUT_DIR})'
    )

    parser.add_argument(
        '--format', '-f',
        type=str,
        default='png',
        choices=['png', 'pdf', 'svg', 'jpg'],
        help='Output figure format (default: png)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for output figures (default: 300)'
    )

    parser.add_argument(
        '--figsize',
        type=int,
        nargs=2,
        default=[10, 8],
        help='Figure size in inches (default: 10 8)'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to plot (for large datasets)'
    )

    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='Custom title for figures'
    )

    parser.add_argument(
        '--compare',
        type=str,
        nargs='+',
        help='Additional prediction files for comparison'
    )

    parser.add_argument(
        '--compare-labels',
        type=str,
        nargs='+',
        help='Labels for comparison models'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def load_data(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    """
    Load predictions and targets from various formats.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (predictions, targets, optional metadata DataFrame)
    """
    metadata = None

    if args.predictions:
        if not args.targets:
            raise ValueError("--targets must be specified with --predictions")
        predictions = np.load(args.predictions)
        targets = np.load(args.targets)

    elif args.npz_file:
        data = np.load(args.npz_file)
        predictions = data.get('predictions', data.get('pred', None))
        targets = data.get('targets', data.get('target', None))

        if predictions is None or targets is None:
            raise ValueError("NPZ file must contain 'predictions' and 'targets' arrays")

    elif args.csv:
        df = pd.read_csv(args.csv)
        predictions = df[args.pred_cols].values
        targets = df[args.target_cols].values
        metadata = df

    else:
        raise ValueError("No input data specified")

    # Subsample if needed
    if args.max_samples and len(predictions) > args.max_samples:
        indices = np.random.choice(len(predictions), args.max_samples, replace=False)
        predictions = predictions[indices]
        targets = targets[indices]
        if metadata is not None:
            metadata = metadata.iloc[indices]

    print(f"Loaded {len(predictions)} samples")
    return predictions, targets, metadata


def calculate_errors(predictions: np.ndarray, targets: np.ndarray) -> Dict:
    """
    Calculate various error metrics.

    Args:
        predictions: Predicted coordinates (N x 3)
        targets: Ground truth coordinates (N x 3)

    Returns:
        Dictionary of error metrics
    """
    errors = predictions - targets
    distances = np.sqrt(np.sum(errors ** 2, axis=1))

    return {
        'errors': errors,
        'distances': distances,
        'error_x': errors[:, 0],
        'error_y': errors[:, 1],
        'error_z': errors[:, 2],
        'mae': np.mean(np.abs(errors), axis=0),
        'rmse': np.sqrt(np.mean(errors ** 2, axis=0)),
        'mde': np.mean(distances),
        'std': np.std(distances)
    }


def plot_3d_trajectory(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: str,
    args: argparse.Namespace
) -> None:
    """
    Create 3D trajectory visualization.

    Args:
        predictions: Predicted coordinates
        targets: Ground truth coordinates
        output_path: Output file path
        args: Command line arguments
    """
    print("Generating 3D trajectory plot...")

    fig = plt.figure(figsize=(args.figsize[0], args.figsize[1]))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectories
    ax.plot(targets[:, 0], targets[:, 1], targets[:, 2],
            'b-', linewidth=1.5, alpha=0.7, label='Ground Truth')
    ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2],
            'r--', linewidth=1.5, alpha=0.7, label='Predicted')

    # Plot start and end points
    ax.scatter(*targets[0], c='green', s=100, marker='o', label='Start', edgecolors='black')
    ax.scatter(*targets[-1], c='red', s=100, marker='s', label='End', edgecolors='black')

    # Draw error vectors for subset of points
    step = max(1, len(predictions) // 50)
    for i in range(0, len(predictions), step):
        ax.plot([targets[i, 0], predictions[i, 0]],
                [targets[i, 1], predictions[i, 1]],
                [targets[i, 2], predictions[i, 2]],
                'gray', alpha=0.3, linewidth=0.5)

    # Labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    title = args.title or '3D Trajectory Comparison'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')

    # Add statistics text
    error_stats = calculate_errors(predictions, targets)
    stats_text = f"MDE: {error_stats['mde']:.3f}m\nRMSE: {np.mean(error_stats['rmse']):.3f}m"
    ax.text2D(0.02, 0.02, stats_text, transform=ax.transAxes,
              fontsize=9, verticalalignment='bottom',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_error_histogram(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: str,
    args: argparse.Namespace
) -> None:
    """
    Create error histogram with fitted distributions.

    Args:
        predictions: Predicted coordinates
        targets: Ground truth coordinates
        output_path: Output file path
        args: Command line arguments
    """
    print("Generating error histogram...")

    fig, axes = plt.subplots(2, 2, figsize=(args.figsize[0], args.figsize[1]))

    error_stats = calculate_errors(predictions, targets)
    distances = error_stats['distances']

    # Overall distance error histogram
    ax1 = axes[0, 0]
    n, bins, patches = ax1.hist(distances, bins=50, density=True,
                                 color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(distances), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(distances):.3f}m')
    ax1.axvline(np.median(distances), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(distances):.3f}m')
    ax1.set_xlabel('Distance Error (m)')
    ax1.set_ylabel('Density')
    ax1.set_title('Distance Error Distribution', fontweight='bold')
    ax1.legend()

    # Per-axis error histograms
    axes_list = [axes[0, 1], axes[1, 0], axes[1, 1]]
    axis_names = ['X', 'Y', 'Z']
    colors = ['coral', 'lightgreen', 'plum']

    for i, (ax, name, color) in enumerate(zip(axes_list, axis_names, colors)):
        errors = error_stats[f'error_{name.lower()}']
        ax.hist(errors, bins=50, density=True, color=color, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.axvline(np.mean(errors), color='blue', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(errors):.3f}m')
        ax.set_xlabel(f'{name} Error (m)')
        ax.set_ylabel('Density')
        ax.set_title(f'{name} Axis Error Distribution', fontweight='bold')
        ax.legend()

    title = args.title or 'Error Histograms'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_cdf(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: str,
    args: argparse.Namespace,
    comparison_preds: Optional[List[np.ndarray]] = None,
    comparison_labels: Optional[List[str]] = None
) -> None:
    """
    Create CDF (Cumulative Distribution Function) plot.

    Args:
        predictions: Predicted coordinates
        targets: Ground truth coordinates
        output_path: Output file path
        args: Command line arguments
        comparison_preds: Optional list of predictions for comparison
        comparison_labels: Labels for comparison models
    """
    print("Generating CDF plot...")

    fig, axes = plt.subplots(1, 2, figsize=(args.figsize[0], args.figsize[1] // 2 + 2))

    # Distance error CDF
    ax1 = axes[0]

    # Main model
    error_stats = calculate_errors(predictions, targets)
    distances = error_stats['distances']
    sorted_dist = np.sort(distances)
    cdf = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist)
    ax1.plot(sorted_dist, cdf, linewidth=2.5, label='Main Model')

    # Comparison models
    if comparison_preds:
        for pred, label in zip(comparison_preds, comparison_labels or []):
            comp_dist = calculate_errors(pred, targets)['distances']
            sorted_comp = np.sort(comp_dist)
            cdf_comp = np.arange(1, len(sorted_comp) + 1) / len(sorted_comp)
            ax1.plot(sorted_comp, cdf_comp, linewidth=2, linestyle='--', label=label)

    ax1.set_xlabel('Distance Error (m)')
    ax1.set_ylabel('CDF')
    ax1.set_title('Distance Error CDF', fontweight='bold')
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
    ax1.axhline(y=0.9, color='gray', linestyle=':', alpha=0.7)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Per-axis CDF
    ax2 = axes[1]
    axis_names = ['X', 'Y', 'Z']
    colors = ['#e74c3c', '#2ecc71', '#9b59b6']

    for name, color in zip(axis_names, colors):
        errors = np.abs(error_stats[f'error_{name.lower()}'])
        sorted_err = np.sort(errors)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        ax2.plot(sorted_err, cdf, linewidth=2, label=f'{name} Axis', color=color)

    ax2.set_xlabel('Absolute Error (m)')
    ax2.set_ylabel('CDF')
    ax2.set_title('Per-Axis Error CDF', fontweight='bold')
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
    ax2.axhline(y=0.9, color='gray', linestyle=':', alpha=0.7)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    title = args.title or 'Cumulative Distribution Functions'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_heatmap(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: str,
    args: argparse.Namespace
) -> None:
    """
    Create spatial error distribution heatmap.

    Args:
        predictions: Predicted coordinates
        targets: Ground truth coordinates
        output_path: Output file path
        args: Command line arguments
    """
    print("Generating spatial heatmap...")

    fig, axes = plt.subplots(2, 2, figsize=(args.figsize[0], args.figsize[1]))

    error_stats = calculate_errors(predictions, targets)
    distances = error_stats['distances']

    # XY plane heatmap
    ax1 = axes[0, 0]
    hb1 = ax1.hexbin(targets[:, 0], targets[:, 1], C=distances, gridsize=30,
                     cmap='hot', reduce_C_function=np.mean)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('XY Plane Error Heatmap', fontweight='bold')
    cb1 = plt.colorbar(hb1, ax=ax1)
    cb1.set_label('Mean Error (m)')

    # XZ plane heatmap
    ax2 = axes[0, 1]
    hb2 = ax2.hexbin(targets[:, 0], targets[:, 2], C=distances, gridsize=30,
                     cmap='hot', reduce_C_function=np.mean)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('XZ Plane Error Heatmap', fontweight='bold')
    cb2 = plt.colorbar(hb2, ax=ax2)
    cb2.set_label('Mean Error (m)')

    # YZ plane heatmap
    ax3 = axes[1, 0]
    hb3 = ax3.hexbin(targets[:, 1], targets[:, 2], C=distances, gridsize=30,
                     cmap='hot', reduce_C_function=np.mean)
    ax3.set_xlabel('Y (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('YZ Plane Error Heatmap', fontweight='bold')
    cb3 = plt.colorbar(hb3, ax=ax3)
    cb3.set_label('Mean Error (m)')

    # 2D histogram of errors
    ax4 = axes[1, 1]
    h, xedges, yedges = np.histogram2d(targets[:, 0], targets[:, 1],
                                        bins=30, weights=distances)
    counts, _, _ = np.histogram2d(targets[:, 0], targets[:, 1], bins=30)
    h = np.divide(h, counts, out=np.zeros_like(h), where=counts!=0)

    im = ax4.imshow(h.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    cmap='hot', aspect='auto')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title('2D Error Heatmap (XY)', fontweight='bold')
    cb4 = plt.colorbar(im, ax=ax4)
    cb4.set_label('Mean Error (m)')

    title = args.title or 'Spatial Error Distribution Heatmaps'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_error_vector(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: str,
    args: argparse.Namespace
) -> None:
    """
    Create error vector visualization in 2D.

    Args:
        predictions: Predicted coordinates
        targets: Ground truth coordinates
        output_path: Output file path
        args: Command line arguments
    """
    print("Generating error vector plot...")

    fig, axes = plt.subplots(1, 2, figsize=(args.figsize[0], args.figsize[1] // 2))

    errors = predictions - targets
    distances = np.sqrt(np.sum(errors ** 2, axis=1))

    # Subsample for clarity
    step = max(1, len(predictions) // 200)
    idx = slice(None, None, step)

    # XY plane error vectors
    ax1 = axes[0]
    scatter = ax1.scatter(targets[idx, 0], targets[idx, 1], c=distances[idx],
                          cmap='hot', s=20, alpha=0.6)
    ax1.quiver(targets[idx, 0], targets[idx, 1],
               errors[idx, 0], errors[idx, 1],
               color='blue', alpha=0.5, scale=10, width=0.003)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Error Vectors (XY Plane)', fontweight='bold')
    plt.colorbar(scatter, ax=ax1, label='Error Magnitude (m)')

    # XZ plane error vectors
    ax2 = axes[1]
    scatter2 = ax2.scatter(targets[idx, 0], targets[idx, 2], c=distances[idx],
                           cmap='hot', s=20, alpha=0.6)
    ax2.quiver(targets[idx, 0], targets[idx, 2],
               errors[idx, 0], errors[idx, 2],
               color='blue', alpha=0.5, scale=10, width=0.003)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Error Vectors (XZ Plane)', fontweight='bold')
    plt.colorbar(scatter2, ax=ax2, label='Error Magnitude (m)')

    title = args.title or 'Error Vector Visualization'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_scatter_comparison(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: str,
    args: argparse.Namespace
) -> None:
    """
    Create scatter plots comparing predicted vs true values.

    Args:
        predictions: Predicted coordinates
        targets: Ground truth coordinates
        output_path: Output file path
        args: Command line arguments
    """
    print("Generating scatter comparison plot...")

    fig, axes = plt.subplots(2, 3, figsize=(args.figsize[0], args.figsize[1]))

    axis_names = ['X', 'Y', 'Z']

    # Predicted vs True scatter plots
    for i, name in enumerate(axis_names):
        ax = axes[0, i]
        ax.scatter(targets[:, i], predictions[:, i], alpha=0.3, s=5)

        # Perfect prediction line
        min_val = min(targets[:, i].min(), predictions[:, i].min())
        max_val = max(targets[:, i].max(), predictions[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

        ax.set_xlabel(f'True {name} (m)')
        ax.set_ylabel(f'Predicted {name} (m)')
        ax.set_title(f'{name} Axis: Predicted vs True', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add R^2
        r2 = 1 - np.sum((predictions[:, i] - targets[:, i])**2) / np.sum((targets[:, i] - targets[:, i].mean())**2)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Error scatter plots
    errors = predictions - targets
    for i, name in enumerate(axis_names):
        ax = axes[1, i]
        ax.scatter(targets[:, i], errors[:, i], alpha=0.3, s=5)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel(f'True {name} (m)')
        ax.set_ylabel(f'{name} Error (m)')
        ax.set_title(f'{name} Axis Error vs True Value', fontweight='bold')
        ax.grid(True, alpha=0.3)

    title = args.title or 'Scatter Comparison Plots'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_boxplot(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: str,
    args: argparse.Namespace,
    comparison_preds: Optional[List[np.ndarray]] = None,
    comparison_labels: Optional[List[str]] = None
) -> None:
    """
    Create box plots for error comparison.

    Args:
        predictions: Predicted coordinates
        targets: Ground truth coordinates
        output_path: Output file path
        args: Command line arguments
        comparison_preds: Optional list of predictions for comparison
        comparison_labels: Labels for comparison models
    """
    print("Generating box plot...")

    fig, axes = plt.subplots(1, 2, figsize=(args.figsize[0], args.figsize[1] // 2 + 2))

    # Collect all distances
    all_distances = [calculate_errors(predictions, targets)['distances']]
    labels = ['Main Model']

    if comparison_preds:
        for pred, label in zip(comparison_preds, comparison_labels or []):
            all_distances.append(calculate_errors(pred, targets)['distances'])
            labels.append(label)

    # Box plot of distance errors
    ax1 = axes[0]
    bp = ax1.boxplot(all_distances, labels=labels, patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_distances)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax1.set_ylabel('Distance Error (m)')
    ax1.set_title('Distance Error Distribution', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Per-axis box plots for main model
    ax2 = axes[1]
    errors = predictions - targets
    data_to_plot = [errors[:, i] for i in range(3)]
    bp2 = ax2.boxplot(data_to_plot, labels=['X', 'Y', 'Z'], patch_artist=True)
    colors2 = ['coral', 'lightgreen', 'plum']
    for patch, color in zip(bp2['boxes'], colors2):
        patch.set_facecolor(color)
    ax2.set_ylabel('Error (m)')
    ax2.set_title('Per-Axis Error Distribution', fontweight='bold')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.grid(True, alpha=0.3, axis='y')

    title = args.title or 'Error Box Plots'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_violin(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: str,
    args: argparse.Namespace
) -> None:
    """
    Create violin plots for error distribution.

    Args:
        predictions: Predicted coordinates
        targets: Ground truth coordinates
        output_path: Output file path
        args: Command line arguments
    """
    print("Generating violin plot...")

    fig, axes = plt.subplots(1, 2, figsize=(args.figsize[0], args.figsize[1] // 2 + 2))

    # Distance error violin
    ax1 = axes[0]
    distances = calculate_errors(predictions, targets)['distances']
    parts = ax1.violinplot([distances], positions=[1], showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor('steelblue')
    parts['bodies'][0].set_alpha(0.7)
    ax1.set_xticks([1])
    ax1.set_xticklabels(['Distance Error'])
    ax1.set_ylabel('Error (m)')
    ax1.set_title('Distance Error Distribution', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Per-axis violin plots
    ax2 = axes[1]
    errors = predictions - targets
    data_to_plot = [errors[:, i] for i in range(3)]
    parts2 = ax2.violinplot(data_to_plot, positions=[1, 2, 3], showmeans=True, showmedians=True)
    colors = ['coral', 'lightgreen', 'plum']
    for pc, color in zip(parts2['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(['X', 'Y', 'Z'])
    ax2.set_ylabel('Error (m)')
    ax2.set_title('Per-Axis Error Distribution', fontweight='bold')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.grid(True, alpha=0.3, axis='y')

    title = args.title or 'Error Violin Plots'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main function."""
    args = parse_arguments()

    print("="*60)
    print("ENHANCED VISUALIZATION FOR INDOOR POSITIONING")
    print("="*60)

    # Load data
    predictions, targets, metadata = load_data(args)

    # Load comparison data if provided
    comparison_preds = None
    comparison_labels = None
    if args.compare:
        comparison_preds = [np.load(f) for f in args.compare]
        comparison_labels = args.compare_labels or [f"Model {i+1}" for i in range(len(comparison_preds))]
        print(f"Loaded {len(comparison_preds)} comparison models")

    # Determine visualization types
    if 'all' in args.viz_type:
        viz_types = ['3d_trajectory', 'histogram', 'cdf', 'heatmap',
                     'error_vector', 'scatter', 'boxplot', 'violin']
    else:
        viz_types = args.viz_type

    # Create output directory
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"viz_{timestamp_str}")
    os.makedirs(output_dir, exist_ok=True)

    # Generate visualizations
    viz_functions = {
        '3d_trajectory': plot_3d_trajectory,
        'histogram': plot_error_histogram,
        'cdf': lambda p, t, o, a: plot_cdf(p, t, o, a, comparison_preds, comparison_labels),
        'heatmap': plot_heatmap,
        'error_vector': plot_error_vector,
        'scatter': plot_scatter_comparison,
        'boxplot': lambda p, t, o, a: plot_boxplot(p, t, o, a, comparison_preds, comparison_labels),
        'violin': plot_violin,
    }

    for viz_type in viz_types:
        if viz_type in viz_functions:
            output_path = os.path.join(output_dir, f"{viz_type}.{args.format}")
            try:
                viz_functions[viz_type](predictions, targets, output_path, args)
            except Exception as e:
                print(f"Error generating {viz_type}: {e}")

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETED")
    print(f"Figures saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
