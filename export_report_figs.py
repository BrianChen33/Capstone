#!/usr/bin/env python3
"""
Export Report Figures Script for COMP4913 Capstone Project.

This script provides one-click generation of all figures required for the
final project report. It orchestrates multiple visualization scripts and
outputs publication-ready figures to a specified directory.

Generated Figures:
    1. Model architecture diagrams
    2. Training curves (loss, learning rate)
    3. Error analysis plots (histograms, CDFs)
    4. Spatial visualization (3D trajectories, heatmaps)
    5. Comparison plots (model vs model)
    6. Ablation study results
    7. Statistical summary charts

Usage:
    python export_report_figs.py --all
    python export_report_figs.py --predictions preds.npy --targets targets.npy
    python export_report_figs.py --model-dir ./models/ --output-dir ./report_figs/

Author: COMP4913 Capstone Project Team
Date: 2024
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
import subprocess
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 150

# Constants
DATASET_DIR = './Dataset/'
FIGDATA_DIR = './FigData/'
OUTPUT_DIR = './output/report_figures/'
RANDOM_SEED = 42

# Figure specifications for report
FIGURE_SPECS = {
    'single_column': (3.5, 2.5),      # Single column figure (inches)
    'double_column': (7.0, 4.0),      # Double column figure (inches)
    'full_page': (7.0, 8.0),          # Full page figure (inches)
}


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Export All Report Figures for Indoor Positioning Project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all report figures
  python export_report_figs.py --all --predictions preds.npy --targets targets.npy

  # Generate specific figure types
  python export_report_figs.py --error-analysis --comparison --spatial

  # From model directory
  python export_report_figs.py --model-dir ./models/ --output-dir ./report_figs/

  # High-resolution output
  python export_report_figs.py --all --dpi 600 --format pdf
        """
    )

    # Main options
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Generate all report figures'
    )

    # Input options
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        help='Path to predictions .npy file'
    )

    parser.add_argument(
        '--targets', '-t',
        type=str,
        help='Path to targets .npy file'
    )

    parser.add_argument(
        '--npz-file', '-npz',
        type=str,
        help='Path to .npz file with predictions and targets'
    )

    parser.add_argument(
        '--model-dir',
        type=str,
        help='Directory containing model results'
    )

    parser.add_argument(
        '--training-history',
        type=str,
        nargs='+',
        help='Training history JSON files for learning curves'
    )

    # Figure type options
    parser.add_argument(
        '--error-analysis',
        action='store_true',
        help='Generate error analysis figures'
    )

    parser.add_argument(
        '--comparison',
        action='store_true',
        help='Generate model comparison figures'
    )

    parser.add_argument(
        '--spatial',
        action='store_true',
        help='Generate spatial visualization figures'
    )

    parser.add_argument(
        '--training-curves',
        action='store_true',
        help='Generate training curve figures'
    )

    parser.add_argument(
        '--ablation',
        action='store_true',
        help='Generate ablation study figures'
    )

    parser.add_argument(
        '--statistics',
        action='store_true',
        help='Generate statistical summary figures'
    )

    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=OUTPUT_DIR,
        help=f'Output directory (default: {OUTPUT_DIR})'
    )

    parser.add_argument(
        '--format', '-f',
        type=str,
        default='png',
        choices=['png', 'pdf', 'svg', 'jpg', 'eps'],
        help='Output figure format (default: png)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for output figures (default: 300)'
    )

    parser.add_argument(
        '--quality',
        type=str,
        default='high',
        choices=['low', 'medium', 'high', 'publication'],
        help='Figure quality preset (default: high)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def setup_output_directory(output_dir: str) -> Dict[str, str]:
    """
    Create organized output directory structure.

    Args:
        output_dir: Base output directory

    Returns:
        Dictionary of subdirectory paths
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join(output_dir, f"report_figures_{timestamp}")

    subdirs = {
        'base': base_dir,
        'error_analysis': os.path.join(base_dir, '01_error_analysis'),
        'comparison': os.path.join(base_dir, '02_model_comparison'),
        'spatial': os.path.join(base_dir, '03_spatial_viz'),
        'training': os.path.join(base_dir, '04_training_curves'),
        'ablation': os.path.join(base_dir, '05_ablation_study'),
        'statistics': os.path.join(base_dir, '06_statistics'),
        'combined': os.path.join(base_dir, '07_combined_figures'),
    }

    for path in subdirs.values():
        os.makedirs(path, exist_ok=True)

    return subdirs


def load_predictions_targets(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load predictions and targets from various sources.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (predictions, targets)
    """
    if args.npz_file:
        data = np.load(args.npz_file)
        predictions = data.get('predictions', data.get('pred', None))
        targets = data.get('targets', data.get('target', None))
    elif args.predictions and args.targets:
        predictions = np.load(args.predictions)
        targets = np.load(args.targets)
    else:
        raise ValueError("Must provide --npz-file or both --predictions and --targets")

    return predictions, targets


def calculate_errors(predictions: np.ndarray, targets: np.ndarray) -> Dict:
    """Calculate error metrics."""
    errors = predictions - targets
    distances = np.sqrt(np.sum(errors ** 2, axis=1))

    return {
        'errors': errors,
        'distances': distances,
        'error_x': errors[:, 0],
        'error_y': errors[:, 1],
        'error_z': errors[:, 2],
        'mde': np.mean(distances),
        'rmse': np.sqrt(np.mean(distances**2)),
        'mae': np.mean(distances),
        'std': np.std(distances),
        'p50': np.percentile(distances, 50),
        'p90': np.percentile(distances, 90),
        'p95': np.percentile(distances, 95),
    }


def generate_error_analysis_figs(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_dir: str,
    args: argparse.Namespace
) -> List[str]:
    """
    Generate error analysis figures.

    Args:
        predictions: Predicted coordinates
        targets: Ground truth coordinates
        output_dir: Output directory
        args: Command line arguments

    Returns:
        List of generated figure paths
    """
    print("\n" + "="*60)
    print("Generating Error Analysis Figures")
    print("="*60)

    error_stats = calculate_errors(predictions, targets)
    distances = error_stats['distances']
    generated_files = []

    # Figure 1: Error Histogram
    fig, ax = plt.subplots(figsize=FIGURE_SPECS['single_column'])
    ax.hist(distances, bins=50, density=True, color='steelblue',
            edgecolor='black', alpha=0.7)
    ax.axvline(error_stats['mde'], color='red', linestyle='--', linewidth=2,
               label=f"MDE = {error_stats['mde']:.3f}m")
    ax.axvline(error_stats['p90'], color='orange', linestyle='--', linewidth=2,
               label=f"P90 = {error_stats['p90']:.3f}m")
    ax.set_xlabel('Distance Error (m)')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution Histogram')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(output_dir, f'01_error_histogram.{args.format}')
    plt.savefig(fig_path, dpi=args.dpi, bbox_inches='tight')
    generated_files.append(fig_path)
    plt.close()
    print(f"Generated: {fig_path}")

    # Figure 2: CDF Plot
    fig, ax = plt.subplots(figsize=FIGURE_SPECS['single_column'])
    sorted_dist = np.sort(distances)
    cdf = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist)
    ax.plot(sorted_dist, cdf, linewidth=2, color='darkblue')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
    ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(error_stats['p50'], color='green', linestyle='--', alpha=0.7,
               label=f"P50 = {error_stats['p50']:.3f}m")
    ax.axvline(error_stats['p90'], color='orange', linestyle='--', alpha=0.7,
               label=f"P90 = {error_stats['p90']:.3f}m")
    ax.set_xlabel('Distance Error (m)')
    ax.set_ylabel('CDF')
    ax.set_title('Cumulative Distribution Function')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(output_dir, f'02_error_cdf.{args.format}')
    plt.savefig(fig_path, dpi=args.dpi, bbox_inches='tight')
    generated_files.append(fig_path)
    plt.close()
    print(f"Generated: {fig_path}")

    # Figure 3: Per-Axis Error Distribution
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SPECS['double_column'])
    axes_names = ['X', 'Y', 'Z']
    colors = ['#e74c3c', '#2ecc71', '#9b59b6']

    for i, (ax, name, color) in enumerate(zip(axes, axes_names, colors)):
        errors = error_stats[f'error_{name.lower()}']
        ax.hist(errors, bins=50, density=True, color=color,
                edgecolor='black', alpha=0.7)
        ax.axvline(0, color='black', linestyle='--', linewidth=2)
        ax.axvline(np.mean(errors), color='red', linestyle='--', linewidth=1.5,
                   label=f"Mean = {np.mean(errors):.3f}m")
        ax.set_xlabel(f'{name} Error (m)')
        ax.set_ylabel('Density')
        ax.set_title(f'{name} Axis Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig_path = os.path.join(output_dir, f'03_per_axis_errors.{args.format}')
    plt.savefig(fig_path, dpi=args.dpi, bbox_inches='tight')
    generated_files.append(fig_path)
    plt.close()
    print(f"Generated: {fig_path}")

    # Figure 4: Error Box Plot
    fig, ax = plt.subplots(figsize=FIGURE_SPECS['single_column'])
    errors = predictions - targets
    bp = ax.boxplot([distances, np.abs(errors[:, 0]), np.abs(errors[:, 1]), np.abs(errors[:, 2])],
                    labels=['Distance', '|X|', '|Y|', '|Z|'],
                    patch_artist=True)
    colors = ['steelblue', 'coral', 'lightgreen', 'plum']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('Error (m)')
    ax.set_title('Error Distribution Box Plot')
    ax.grid(True, alpha=0.3, axis='y')

    fig_path = os.path.join(output_dir, f'04_error_boxplot.{args.format}')
    plt.savefig(fig_path, dpi=args.dpi, bbox_inches='tight')
    generated_files.append(fig_path)
    plt.close()
    print(f"Generated: {fig_path}")

    return generated_files


def generate_comparison_figs(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_dir: str,
    args: argparse.Namespace
) -> List[str]:
    """
    Generate model comparison figures.

    Args:
        predictions: Predicted coordinates
        targets: Ground truth coordinates
        output_dir: Output directory
        args: Command line arguments

    Returns:
        List of generated figure paths
    """
    print("\n" + "="*60)
    print("Generating Comparison Figures")
    print("="*60)

    generated_files = []

    # Figure: Predicted vs True Scatter
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SPECS['double_column'])
    axis_names = ['X', 'Y', 'Z']

    for i, (ax, name) in enumerate(zip(axes, axis_names)):
        ax.scatter(targets[:, i], predictions[:, i], alpha=0.3, s=3, c='steelblue')
        min_val = min(targets[:, i].min(), predictions[:, i].min())
        max_val = max(targets[:, i].max(), predictions[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel(f'True {name} (m)')
        ax.set_ylabel(f'Predicted {name} (m)')
        ax.set_title(f'{name} Axis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # R^2
        ss_res = np.sum((predictions[:, i] - targets[:, i])**2)
        ss_tot = np.sum((targets[:, i] - targets[:, i].mean())**2)
        r2 = 1 - ss_res / ss_tot
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig_path = os.path.join(output_dir, f'01_predicted_vs_true.{args.format}')
    plt.savefig(fig_path, dpi=args.dpi, bbox_inches='tight')
    generated_files.append(fig_path)
    plt.close()
    print(f"Generated: {fig_path}")

    return generated_files


def generate_spatial_figs(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_dir: str,
    args: argparse.Namespace
) -> List[str]:
    """
    Generate spatial visualization figures.

    Args:
        predictions: Predicted coordinates
        targets: Ground truth coordinates
        output_dir: Output directory
        args: Command line arguments

    Returns:
        List of generated figure paths
    """
    print("\n" + "="*60)
    print("Generating Spatial Visualization Figures")
    print("="*60)

    error_stats = calculate_errors(predictions, targets)
    distances = error_stats['distances']
    generated_files = []

    # Figure 1: 3D Trajectory
    fig = plt.figure(figsize=FIGURE_SPECS['double_column'])
    ax = fig.add_subplot(111, projection='3d')

    # Subsample for clarity
    step = max(1, len(predictions) // 500)
    idx = slice(None, None, step)

    ax.plot(targets[idx, 0], targets[idx, 1], targets[idx, 2],
            'b-', linewidth=1.5, alpha=0.7, label='Ground Truth')
    ax.plot(predictions[idx, 0], predictions[idx, 1], predictions[idx, 2],
            'r--', linewidth=1.5, alpha=0.7, label='Predicted')
    ax.scatter(*targets[0], c='green', s=50, marker='o', label='Start')
    ax.scatter(*targets[-1], c='red', s=50, marker='s', label='End')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory Comparison')
    ax.legend()

    fig_path = os.path.join(output_dir, f'01_3d_trajectory.{args.format}')
    plt.savefig(fig_path, dpi=args.dpi, bbox_inches='tight')
    generated_files.append(fig_path)
    plt.close()
    print(f"Generated: {fig_path}")

    # Figure 2: Spatial Heatmap (XY)
    fig, ax = plt.subplots(figsize=FIGURE_SPECS['single_column'])
    hb = ax.hexbin(targets[:, 0], targets[:, 1], C=distances, gridsize=25,
                   cmap='hot', reduce_C_function=np.mean)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Spatial Error Heatmap (XY Plane)')
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label('Mean Error (m)')

    fig_path = os.path.join(output_dir, f'02_spatial_heatmap_xy.{args.format}')
    plt.savefig(fig_path, dpi=args.dpi, bbox_inches='tight')
    generated_files.append(fig_path)
    plt.close()
    print(f"Generated: {fig_path}")

    # Figure 3: Error Vectors
    fig, ax = plt.subplots(figsize=FIGURE_SPECS['single_column'])
    errors = predictions - targets

    scatter = ax.scatter(targets[idx, 0], targets[idx, 1], c=distances[idx],
                         cmap='hot', s=15, alpha=0.6)
    ax.quiver(targets[idx, 0], targets[idx, 1],
              errors[idx, 0], errors[idx, 1],
              color='blue', alpha=0.4, scale=15, width=0.003)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Error Vectors (XY Plane)')
    plt.colorbar(scatter, ax=ax, label='Error (m)')

    fig_path = os.path.join(output_dir, f'03_error_vectors.{args.format}')
    plt.savefig(fig_path, dpi=args.dpi, bbox_inches='tight')
    generated_files.append(fig_path)
    plt.close()
    print(f"Generated: {fig_path}")

    return generated_files


def generate_training_curve_figs(
    history_files: List[str],
    output_dir: str,
    args: argparse.Namespace
) -> List[str]:
    """
    Generate training curve figures.

    Args:
        history_files: List of training history JSON files
        output_dir: Output directory
        args: Command line arguments

    Returns:
        List of generated figure paths
    """
    print("\n" + "="*60)
    print("Generating Training Curve Figures")
    print("="*60)

    generated_files = []

    if not history_files:
        print("No training history files provided, skipping...")
        return generated_files

    # Load all histories
    histories = []
    labels = []
    for file in history_files:
        with open(file, 'r') as f:
            histories.append(json.load(f))
        labels.append(os.path.basename(file).replace('.json', ''))

    # Figure 1: Loss Curves
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SPECS['double_column'])

    ax1 = axes[0]
    for hist, label in zip(histories, labels):
        if 'train_loss' in hist:
            ax1.plot(hist['train_loss'], label=f'{label} (train)', linewidth=1.5)
        if 'test_loss' in hist:
            ax1.plot(hist['test_loss'], label=f'{label} (val)', linewidth=1.5, linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for hist, label in zip(histories, labels):
        if 'test_mde' in hist:
            ax2.plot(hist['test_mde'], label=label, linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MDE (m)')
    ax2.set_title('Mean Distance Error Over Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig_path = os.path.join(output_dir, f'01_training_curves.{args.format}')
    plt.savefig(fig_path, dpi=args.dpi, bbox_inches='tight')
    generated_files.append(fig_path)
    plt.close()
    print(f"Generated: {fig_path}")

    return generated_files


def generate_statistics_figs(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_dir: str,
    args: argparse.Namespace
) -> List[str]:
    """
    Generate statistical summary figures.

    Args:
        predictions: Predicted coordinates
        targets: Ground truth coordinates
        output_dir: Output directory
        args: Command line arguments

    Returns:
        List of generated figure paths
    """
    print("\n" + "="*60)
    print("Generating Statistics Summary Figures")
    print("="*60)

    error_stats = calculate_errors(predictions, targets)
    generated_files = []

    # Figure: Statistics Summary Table
    fig, ax = plt.subplots(figsize=FIGURE_SPECS['single_column'])
    ax.axis('off')

    stats_data = [
        ['Metric', 'Value'],
        ['Mean Distance Error (MDE)', f"{error_stats['mde']:.4f} m"],
        ['Root Mean Square Error (RMSE)', f"{error_stats['rmse']:.4f} m"],
        ['Mean Absolute Error (MAE)', f"{error_stats['mae']:.4f} m"],
        ['Standard Deviation', f"{error_stats['std']:.4f} m"],
        ['P50 (Median)', f"{error_stats['p50']:.4f} m"],
        ['P90', f"{error_stats['p90']:.4f} m"],
        ['P95', f"{error_stats['p95']:.4f} m"],
    ]

    table = ax.table(cellText=stats_data, cellLoc='left', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Error Statistics Summary', fontsize=14, fontweight='bold', pad=20)

    fig_path = os.path.join(output_dir, f'01_statistics_table.{args.format}')
    plt.savefig(fig_path, dpi=args.dpi, bbox_inches='tight')
    generated_files.append(fig_path)
    plt.close()
    print(f"Generated: {fig_path}")

    return generated_files


def generate_combined_figure(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_dir: str,
    args: argparse.Namespace
) -> List[str]:
    """
    Generate a combined figure with multiple subplots for the report.

    Args:
        predictions: Predicted coordinates
        targets: Ground truth coordinates
        output_dir: Output directory
        args: Command line arguments

    Returns:
        List of generated figure paths
    """
    print("\n" + "="*60)
    print("Generating Combined Figure")
    print("="*60)

    error_stats = calculate_errors(predictions, targets)
    distances = error_stats['distances']
    errors = predictions - targets
    generated_files = []

    # Create a large combined figure
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Error Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(distances, bins=40, density=True, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(error_stats['mde'], color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Distance Error (m)')
    ax1.set_ylabel('Density')
    ax1.set_title('(a) Error Distribution')
    ax1.grid(True, alpha=0.3)

    # 2. CDF
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_dist = np.sort(distances)
    cdf = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist)
    ax2.plot(sorted_dist, cdf, linewidth=2, color='darkblue')
    ax2.axhline(y=0.9, color='gray', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Distance Error (m)')
    ax2.set_ylabel('CDF')
    ax2.set_title('(b) Cumulative Distribution')
    ax2.grid(True, alpha=0.3)

    # 3. 3D Trajectory (2D projection)
    ax3 = fig.add_subplot(gs[0, 2])
    step = max(1, len(predictions) // 300)
    idx = slice(None, None, step)
    ax3.plot(targets[idx, 0], targets[idx, 1], 'b-', linewidth=1, alpha=0.7, label='True')
    ax3.plot(predictions[idx, 0], predictions[idx, 1], 'r--', linewidth=1, alpha=0.7, label='Pred')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('(c) Trajectory (XY)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4-6. Per-axis errors
    for i, (name, color) in enumerate(zip(['X', 'Y', 'Z'], ['#e74c3c', '#2ecc71', '#9b59b6'])):
        ax = fig.add_subplot(gs[1, i])
        ax.hist(errors[:, i], bins=40, density=True, color=color, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='black', linestyle='--', linewidth=2)
        ax.set_xlabel(f'{name} Error (m)')
        ax.set_ylabel('Density')
        ax.set_title(f'({chr(100+i)}) {name} Axis Error')
        ax.grid(True, alpha=0.3)

    # 7. Spatial Heatmap
    ax7 = fig.add_subplot(gs[2, 0])
    hb = ax7.hexbin(targets[:, 0], targets[:, 1], C=distances, gridsize=20,
                    cmap='hot', reduce_C_function=np.mean)
    ax7.set_xlabel('X (m)')
    ax7.set_ylabel('Y (m)')
    ax7.set_title('(g) Spatial Error Heatmap')

    # 8. Box Plot
    ax8 = fig.add_subplot(gs[2, 1])
    bp = ax8.boxplot([distances, np.abs(errors[:, 0]), np.abs(errors[:, 1]), np.abs(errors[:, 2])],
                     labels=['Dist', '|X|', '|Y|', '|Z|'], patch_artist=True)
    colors = ['steelblue', 'coral', 'lightgreen', 'plum']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax8.set_ylabel('Error (m)')
    ax8.set_title('(h) Error Box Plot')
    ax8.grid(True, alpha=0.3, axis='y')

    # 9. Statistics Text
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    stats_text = f"""
    Performance Summary:

    MDE: {error_stats['mde']:.3f} m
    RMSE: {error_stats['rmse']:.3f} m
    MAE: {error_stats['mae']:.3f} m

    Percentiles:
    P50: {error_stats['p50']:.3f} m
    P90: {error_stats['p90']:.3f} m
    P95: {error_stats['p95']:.3f} m

    Samples: {len(predictions)}
    """
    ax9.text(0.1, 0.5, stats_text, transform=ax9.transAxes,
             fontsize=10, verticalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax9.set_title('(i) Statistics Summary')

    fig.suptitle('Indoor Positioning Performance Analysis', fontsize=14, fontweight='bold')

    fig_path = os.path.join(output_dir, f'combined_analysis.{args.format}')
    plt.savefig(fig_path, dpi=args.dpi, bbox_inches='tight')
    generated_files.append(fig_path)
    plt.close()
    print(f"Generated: {fig_path}")

    return generated_files


def create_index_file(
    output_dirs: Dict[str, str],
    all_files: Dict[str, List[str]],
    args: argparse.Namespace
) -> str:
    """
    Create an HTML index file for easy navigation of generated figures.

    Args:
        output_dirs: Dictionary of output directories
        all_files: Dictionary of generated files by category
        args: Command line arguments

    Returns:
        Path to index file
    """
    index_path = os.path.join(output_dirs['base'], 'index.html')

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>COMP4913 Report Figures</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #4CAF50; margin-top: 30px; }}
        .section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .figure-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
        .figure-item {{ text-align: center; }}
        .figure-item img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
        .figure-item p {{ margin-top: 10px; color: #666; }}
        .info {{ background: #e3f2fd; padding: 15px; border-radius: 4px; margin-bottom: 20px; }}
        .timestamp {{ color: #999; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>COMP4913 Capstone Project - Report Figures</h1>
    <div class="info">
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Format:</strong> {args.format.upper()}</p>
        <p><strong>DPI:</strong> {args.dpi}</p>
    </div>
"""

    for section_name, files in all_files.items():
        if files:
            html_content += f"""
    <div class="section">
        <h2>{section_name.replace('_', ' ').title()}</h2>
        <div class="figure-grid">
"""
            for file_path in files:
                rel_path = os.path.relpath(file_path, output_dirs['base'])
                file_name = os.path.basename(file_path)
                html_content += f"""
            <div class="figure-item">
                <img src="{rel_path}" alt="{file_name}">
                <p>{file_name}</p>
            </div>
"""
            html_content += """
        </div>
    </div>
"""

    html_content += """
</body>
</html>
"""

    with open(index_path, 'w') as f:
        f.write(html_content)

    return index_path


def main():
    """Main function."""
    args = parse_arguments()

    print("="*60)
    print("EXPORT REPORT FIGURES")
    print("="*60)
    print(f"Output format: {args.format.upper()}")
    print(f"DPI: {args.dpi}")

    # Determine which figures to generate
    if args.all:
        args.error_analysis = True
        args.comparison = True
        args.spatial = True
        args.training_curves = True
        args.ablation = True
        args.statistics = True

    # Setup output directories
    output_dirs = setup_output_directory(args.output_dir)
    print(f"\nOutput directory: {output_dirs['base']}")

    # Load data if provided
    predictions = None
    targets = None
    if args.predictions or args.npz_file:
        predictions, targets = load_predictions_targets(args)

    # Generate figures
    all_files = {}

    if args.error_analysis and predictions is not None:
        all_files['Error Analysis'] = generate_error_analysis_figs(
            predictions, targets, output_dirs['error_analysis'], args
        )

    if args.comparison and predictions is not None:
        all_files['Model Comparison'] = generate_comparison_figs(
            predictions, targets, output_dirs['comparison'], args
        )

    if args.spatial and predictions is not None:
        all_files['Spatial Visualization'] = generate_spatial_figs(
            predictions, targets, output_dirs['spatial'], args
        )

    if args.training_curves and args.training_history:
        all_files['Training Curves'] = generate_training_curve_figs(
            args.training_history, output_dirs['training'], args
        )

    if args.statistics and predictions is not None:
        all_files['Statistics'] = generate_statistics_figs(
            predictions, targets, output_dirs['statistics'], args
        )

    # Generate combined figure
    if predictions is not None:
        all_files['Combined Analysis'] = generate_combined_figure(
            predictions, targets, output_dirs['combined'], args
        )

    # Create index file
    index_path = create_index_file(output_dirs, all_files, args)
    print(f"\nIndex file created: {index_path}")

    # Summary
    total_files = sum(len(files) for files in all_files.values())
    print("\n" + "="*60)
    print("REPORT FIGURES GENERATION COMPLETED")
    print("="*60)
    print(f"Total figures generated: {total_files}")
    print(f"Output directory: {output_dirs['base']}")
    print("="*60)


if __name__ == '__main__':
    main()
