#!/usr/bin/env python3
"""
Error Analysis Script for COMP4913 Capstone Project.

This script performs stratified error analysis on indoor positioning results,
including distance-based bucketing, temporal analysis, and spatial analysis.

Dataset Specification:
    - 986-dimensional features
    - Ground truth: (x, y, z) coordinates
    - Predictions: Model output coordinates

Analysis Types:
    1. Distance bucketing: 0-1m, 1-2m, 2-3m, >3m
    2. Temporal analysis: Error distribution over time
    3. Spatial analysis: Error by area/region
    4. Per-axis error analysis

Usage:
    python error_analysis.py --predictions preds.npy --targets targets.npy
    python error_analysis.py --model-output output.npz --analysis all
    python error_analysis.py --csv results.csv --timestamp-col time

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
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Constants
DATASET_DIR = './Dataset/'
FIGDATA_DIR = './FigData/'
OUTPUT_DIR = './output/error_analysis/'
RANDOM_SEED = 42

# Distance buckets for error analysis
DISTANCE_BUCKETS = [0, 1, 2, 3, 5, float('inf')]
BUCKET_LABELS = ['0-1m', '1-2m', '2-3m', '3-5m', '>5m']


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Stratified Error Analysis for Indoor Positioning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze from numpy arrays
  python error_analysis.py --predictions preds.npy --targets targets.npy

  # Analyze from NPZ file
  python error_analysis.py --npz-file results.npz --analysis all

  # Analyze from CSV with timestamps
  python error_analysis.py --csv results.csv --timestamp-col timestamp

  # Generate all analysis types
  python error_analysis.py --predictions preds.npy --targets targets.npy --analysis all
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
        help='Column names for predictions in CSV (default: pred_x pred_y pred_z)'
    )

    parser.add_argument(
        '--target-cols',
        type=str,
        nargs=3,
        default=['true_x', 'true_y', 'true_z'],
        help='Column names for targets in CSV (default: true_x true_y true_z)'
    )

    parser.add_argument(
        '--timestamp-col',
        type=str,
        help='Column name for timestamp in CSV'
    )

    parser.add_argument(
        '--area-col',
        type=str,
        help='Column name for area/region in CSV'
    )

    parser.add_argument(
        '--analysis', '-a',
        type=str,
        nargs='+',
        default=['distance', 'axis', 'statistics'],
        choices=['distance', 'temporal', 'spatial', 'axis', 'statistics', 'all'],
        help='Types of analysis to perform'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=OUTPUT_DIR,
        help=f'Output directory for results (default: {OUTPUT_DIR})'
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
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def load_data(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load predictions and targets from various formats.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (predictions, targets, timestamps, areas)
    """
    timestamps = None
    areas = None

    if args.predictions:
        if not args.targets:
            raise ValueError("--targets must be specified with --predictions")
        predictions = np.load(args.predictions)
        targets = np.load(args.targets)

    elif args.npz_file:
        data = np.load(args.npz_file)
        predictions = data.get('predictions', data.get('pred', None))
        targets = data.get('targets', data.get('target', None))
        timestamps = data.get('timestamps', None)
        areas = data.get('areas', None)

        if predictions is None or targets is None:
            raise ValueError("NPZ file must contain 'predictions' and 'targets' arrays")

    elif args.csv:
        df = pd.read_csv(args.csv)
        predictions = df[args.pred_cols].values
        targets = df[args.target_cols].values

        if args.timestamp_col and args.timestamp_col in df.columns:
            timestamps = pd.to_datetime(df[args.timestamp_col]).values
        if args.area_col and args.area_col in df.columns:
            areas = df[args.area_col].values

    else:
        raise ValueError("No input data specified")

    # Validate shapes
    assert predictions.shape == targets.shape, \
        f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
    assert predictions.shape[1] == 3, \
        f"Expected 3D coordinates, got {predictions.shape[1]}D"

    print(f"Loaded {len(predictions)} samples")
    return predictions, targets, timestamps, areas


def calculate_errors(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict:
    """
    Calculate various error metrics.

    Args:
        predictions: Predicted coordinates (N x 3)
        targets: Ground truth coordinates (N x 3)

    Returns:
        Dictionary of error metrics
    """
    # Point-wise errors
    errors = predictions - targets

    # Euclidean distance errors
    distances = np.sqrt(np.sum(errors ** 2, axis=1))

    # Per-axis errors
    error_x = errors[:, 0]
    error_y = errors[:, 1]
    error_z = errors[:, 2]

    return {
        'errors': errors,
        'distances': distances,
        'error_x': error_x,
        'error_y': error_y,
        'error_z': error_z
    }


def analyze_distance_buckets(
    distances: np.ndarray,
    output_dir: str,
    fmt: str = 'png',
    dpi: int = 300
) -> pd.DataFrame:
    """
    Analyze errors by distance buckets.

    Args:
        distances: Array of distance errors
        output_dir: Output directory
        fmt: Figure format
        dpi: Figure DPI

    Returns:
        DataFrame with bucket statistics
    """
    print("\n" + "="*60)
    print("DISTANCE BUCKET ANALYSIS")
    print("="*60)

    # Assign samples to buckets
    bucket_indices = np.digitize(distances, DISTANCE_BUCKETS) - 1
    bucket_indices = np.clip(bucket_indices, 0, len(BUCKET_LABELS) - 1)

    # Calculate statistics per bucket
    stats = []
    for i, label in enumerate(BUCKET_LABELS):
        mask = bucket_indices == i
        count = np.sum(mask)
        percentage = count / len(distances) * 100

        if count > 0:
            bucket_distances = distances[mask]
            stats.append({
                'Bucket': label,
                'Count': count,
                'Percentage': f"{percentage:.2f}%",
                'Mean Error (m)': f"{np.mean(bucket_distances):.4f}",
                'Std Error (m)': f"{np.std(bucket_distances):.4f}",
                'Min Error (m)': f"{np.min(bucket_distances):.4f}",
                'Max Error (m)': f"{np.max(bucket_distances):.4f}",
                'Median Error (m)': f"{np.median(bucket_distances):.4f}"
            })
        else:
            stats.append({
                'Bucket': label,
                'Count': 0,
                'Percentage': "0.00%",
                'Mean Error (m)': "N/A",
                'Std Error (m)': "N/A",
                'Min Error (m)': "N/A",
                'Max Error (m)': "N/A",
                'Median Error (m)': "N/A"
            })

    df = pd.DataFrame(stats)
    print(df.to_string(index=False))

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot of counts
    ax1 = axes[0]
    counts = [int(s['Count']) for s in stats]
    colors = sns.color_palette("RdYlGn_r", len(BUCKET_LABELS))
    bars = ax1.bar(BUCKET_LABELS, counts, color=colors, edgecolor='black')
    ax1.set_xlabel('Error Bucket', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Sample Distribution by Error Bucket', fontsize=14, fontweight='bold')

    # Add percentage labels
    for bar, stat in zip(bars, stats):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f"{stat['Percentage']}",
                ha='center', va='bottom', fontsize=10)

    # Cumulative distribution
    ax2 = axes[1]
    cumulative_pct = np.cumsum([int(s['Count']) for s in stats]) / len(distances) * 100
    ax2.plot(range(len(BUCKET_LABELS)), cumulative_pct, 'o-', linewidth=2, markersize=8)
    ax2.set_xticks(range(len(BUCKET_LABELS)))
    ax2.set_xticklabels(BUCKET_LABELS)
    ax2.set_xlabel('Error Bucket', fontsize=12)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax2.set_title('Cumulative Error Distribution', fontsize=14, fontweight='bold')
    ax2.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='80% threshold')
    ax2.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f'distance_buckets.{fmt}')
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")
    plt.close()

    return df


def analyze_temporal_errors(
    predictions: np.ndarray,
    targets: np.ndarray,
    timestamps: Optional[np.ndarray],
    output_dir: str,
    fmt: str = 'png',
    dpi: int = 300
) -> Optional[pd.DataFrame]:
    """
    Analyze errors over time.

    Args:
        predictions: Predicted coordinates
        targets: Ground truth coordinates
        timestamps: Array of timestamps
        output_dir: Output directory
        fmt: Figure format
        dpi: Figure DPI

    Returns:
        DataFrame with temporal statistics or None if no timestamps
    """
    if timestamps is None:
        print("\nSkipping temporal analysis (no timestamps provided)")
        return None

    print("\n" + "="*60)
    print("TEMPORAL ERROR ANALYSIS")
    print("="*60)

    errors = predictions - targets
    distances = np.sqrt(np.sum(errors ** 2, axis=1))

    # Convert timestamps to datetime if needed
    if isinstance(timestamps[0], (int, float, np.number)):
        timestamps = pd.to_datetime(timestamps, unit='s')
    else:
        timestamps = pd.to_datetime(timestamps)

    # Create DataFrame for analysis
    df = pd.DataFrame({
        'timestamp': timestamps,
        'error': distances,
        'error_x': errors[:, 0],
        'error_y': errors[:, 1],
        'error_z': errors[:, 2]
    })

    # Hourly analysis
    df['hour'] = df['timestamp'].dt.hour
    hourly_stats = df.groupby('hour')['error'].agg(['mean', 'std', 'median', 'count']).reset_index()
    hourly_stats.columns = ['Hour', 'Mean Error', 'Std Error', 'Median Error', 'Count']

    print("\nHourly Error Statistics:")
    print(hourly_stats.to_string(index=False))

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Error over time
    ax1 = axes[0, 0]
    ax1.plot(df['timestamp'], df['error'], alpha=0.5, linewidth=0.5)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Error (m)', fontsize=12)
    ax1.set_title('Error Over Time', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    # Hourly average
    ax2 = axes[0, 1]
    ax2.bar(hourly_stats['Hour'], hourly_stats['Mean Error'], color='steelblue', edgecolor='black')
    ax2.set_xlabel('Hour of Day', fontsize=12)
    ax2.set_ylabel('Mean Error (m)', fontsize=12)
    ax2.set_title('Mean Error by Hour', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(0, 24, 2))

    # Rolling average
    ax3 = axes[1, 0]
    window = min(100, len(df) // 10)
    rolling_mean = df['error'].rolling(window=window).mean()
    ax3.plot(df['timestamp'], rolling_mean, color='red', linewidth=1.5)
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel(f'Rolling Mean Error (m, window={window})', fontsize=12)
    ax3.set_title('Rolling Average Error', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)

    # Error distribution by hour (box plot)
    ax4 = axes[1, 1]
    hours_to_plot = list(range(0, 24, 4))
    data_to_plot = [df[df['hour'] == h]['error'].values for h in hours_to_plot]
    ax4.boxplot(data_to_plot, labels=hours_to_plot)
    ax4.set_xlabel('Hour of Day', fontsize=12)
    ax4.set_ylabel('Error (m)', fontsize=12)
    ax4.set_title('Error Distribution by Hour', fontsize=14, fontweight='bold')

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f'temporal_analysis.{fmt}')
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")
    plt.close()

    return hourly_stats


def analyze_spatial_errors(
    predictions: np.ndarray,
    targets: np.ndarray,
    areas: Optional[np.ndarray],
    output_dir: str,
    fmt: str = 'png',
    dpi: int = 300
) -> Optional[pd.DataFrame]:
    """
    Analyze errors by spatial regions.

    Args:
        predictions: Predicted coordinates
        targets: Ground truth coordinates
        areas: Array of area labels
        output_dir: Output directory
        fmt: Figure format
        dpi: Figure DPI

    Returns:
        DataFrame with spatial statistics or None if no areas
    """
    if areas is None:
        print("\nSkipping spatial analysis (no area labels provided)")
        print("Performing coordinate-based spatial analysis instead...")

        # Use coordinate-based analysis
        errors = predictions - targets
        distances = np.sqrt(np.sum(errors ** 2, axis=1))

        # Divide space into regions based on x coordinate
        x_min, x_max = targets[:, 0].min(), targets[:, 0].max()
        n_regions = 5
        x_bins = np.linspace(x_min, x_max, n_regions + 1)
        region_labels = [f"X:{x_bins[i]:.1f}-{x_bins[i+1]:.1f}" for i in range(n_regions)]
        region_indices = np.digitize(targets[:, 0], x_bins) - 1
        region_indices = np.clip(region_indices, 0, n_regions - 1)
        areas = [region_labels[i] for i in region_indices]

    print("\n" + "="*60)
    print("SPATIAL ERROR ANALYSIS")
    print("="*60)

    errors = predictions - targets
    distances = np.sqrt(np.sum(errors ** 2, axis=1))

    # Create DataFrame
    df = pd.DataFrame({
        'area': areas,
        'error': distances,
        'error_x': errors[:, 0],
        'error_y': errors[:, 1],
        'error_z': errors[:, 2],
        'true_x': targets[:, 0],
        'true_y': targets[:, 1],
        'true_z': targets[:, 2]
    })

    # Statistics by area
    area_stats = df.groupby('area')['error'].agg(['count', 'mean', 'std', 'median', 'min', 'max']).reset_index()
    area_stats.columns = ['Area', 'Count', 'Mean Error', 'Std Error', 'Median Error', 'Min Error', 'Max Error']
    area_stats = area_stats.sort_values('Mean Error', ascending=False)

    print("\nError Statistics by Area:")
    print(area_stats.to_string(index=False))

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Bar plot of mean error by area
    ax1 = axes[0, 0]
    ax1.barh(area_stats['Area'], area_stats['Mean Error'], color='coral', edgecolor='black')
    ax1.set_xlabel('Mean Error (m)', fontsize=12)
    ax1.set_ylabel('Area', fontsize=12)
    ax1.set_title('Mean Error by Area', fontsize=14, fontweight='bold')

    # Box plot by area
    ax2 = axes[0, 1]
    areas_to_plot = area_stats['Area'].values[:min(8, len(area_stats))]
    data_to_plot = [df[df['area'] == a]['error'].values for a in areas_to_plot]
    bp = ax2.boxplot(data_to_plot, labels=areas_to_plot, vert=False)
    ax2.set_xlabel('Error (m)', fontsize=12)
    ax2.set_ylabel('Area', fontsize=12)
    ax2.set_title('Error Distribution by Area', fontsize=14, fontweight='bold')

    # 2D spatial error heatmap (XY plane)
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df['true_x'], df['true_y'], c=df['error'], cmap='hot', s=10, alpha=0.6)
    ax3.set_xlabel('X (m)', fontsize=12)
    ax3.set_ylabel('Y (m)', fontsize=12)
    ax3.set_title('Error Distribution in XY Plane', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax3, label='Error (m)')

    # Error vs distance from center
    ax4 = axes[1, 1]
    center = np.mean(targets, axis=0)
    dist_from_center = np.sqrt(np.sum((targets - center) ** 2, axis=1))
    ax4.scatter(dist_from_center, distances, alpha=0.3, s=5)
    ax4.set_xlabel('Distance from Center (m)', fontsize=12)
    ax4.set_ylabel('Error (m)', fontsize=12)
    ax4.set_title('Error vs Distance from Center', fontsize=14, fontweight='bold')

    # Add trend line
    z = np.polyfit(dist_from_center, distances, 1)
    p = np.poly1d(z)
    ax4.plot(sorted(dist_from_center), p(sorted(dist_from_center)), "r--", linewidth=2, label='Trend')
    ax4.legend()

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f'spatial_analysis.{fmt}')
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")
    plt.close()

    return area_stats


def analyze_per_axis_errors(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_dir: str,
    fmt: str = 'png',
    dpi: int = 300
) -> pd.DataFrame:
    """
    Analyze errors per axis (X, Y, Z).

    Args:
        predictions: Predicted coordinates
        targets: Ground truth coordinates
        output_dir: Output directory
        fmt: Figure format
        dpi: Figure DPI

    Returns:
        DataFrame with per-axis statistics
    """
    print("\n" + "="*60)
    print("PER-AXIS ERROR ANALYSIS")
    print("="*60)

    errors = predictions - targets
    axes = ['X', 'Y', 'Z']

    stats = []
    for i, axis in enumerate(axes):
        axis_error = errors[:, i]
        stats.append({
            'Axis': axis,
            'MAE (m)': np.mean(np.abs(axis_error)),
            'RMSE (m)': np.sqrt(np.mean(axis_error ** 2)),
            'Mean Error (m)': np.mean(axis_error),
            'Std Error (m)': np.std(axis_error),
            'Min Error (m)': np.min(axis_error),
            'Max Error (m)': np.max(axis_error),
            'P50 (m)': np.percentile(np.abs(axis_error), 50),
            'P90 (m)': np.percentile(np.abs(axis_error), 90),
            'P95 (m)': np.percentile(np.abs(axis_error), 95),
        })

    df = pd.DataFrame(stats)
    for col in df.columns:
        if col != 'Axis':
            df[col] = df[col].apply(lambda x: f"{x:.4f}")

    print(df.to_string(index=False))

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for i, axis in enumerate(['X', 'Y', 'Z']):
        axis_error = errors[:, i]

        # Histogram
        ax1 = axes[0, i]
        ax1.hist(axis_error, bins=50, color=f'C{i}', edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel(f'{axis} Error (m)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title(f'{axis} Axis Error Distribution', fontsize=12, fontweight='bold')

        # CDF
        ax2 = axes[1, i]
        sorted_errors = np.sort(np.abs(axis_error))
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax2.plot(sorted_errors, cdf, linewidth=2, color=f'C{i}')
        ax2.set_xlabel(f'|{axis}| Error (m)', fontsize=11)
        ax2.set_ylabel('CDF', fontsize=11)
        ax2.set_title(f'{axis} Axis Error CDF', fontsize=12, fontweight='bold')
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f'per_axis_errors.{fmt}')
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")
    plt.close()

    return df


def generate_statistics_summary(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_dir: str
) -> pd.DataFrame:
    """
    Generate comprehensive statistics summary.

    Args:
        predictions: Predicted coordinates
        targets: Ground truth coordinates
        output_dir: Output directory

    Returns:
        DataFrame with summary statistics
    """
    print("\n" + "="*60)
    print("STATISTICS SUMMARY")
    print("="*60)

    errors = predictions - targets
    distances = np.sqrt(np.sum(errors ** 2, axis=1))

    # Overall statistics
    stats = {
        'Metric': [
            'Mean Distance Error (MDE)',
            'Root Mean Square Error (RMSE)',
            'Mean Absolute Error (MAE)',
            'Median Distance Error',
            'Standard Deviation',
            'Minimum Error',
            'Maximum Error',
            'P50 (Median)',
            'P75',
            'P90',
            'P95',
            'P99',
            'MAE X',
            'MAE Y',
            'MAE Z',
            'Total Samples'
        ],
        'Value': [
            f"{np.mean(distances):.4f} m",
            f"{np.sqrt(np.mean(distances**2)):.4f} m",
            f"{np.mean(distances):.4f} m",
            f"{np.median(distances):.4f} m",
            f"{np.std(distances):.4f} m",
            f"{np.min(distances):.4f} m",
            f"{np.max(distances):.4f} m",
            f"{np.percentile(distances, 50):.4f} m",
            f"{np.percentile(distances, 75):.4f} m",
            f"{np.percentile(distances, 90):.4f} m",
            f"{np.percentile(distances, 95):.4f} m",
            f"{np.percentile(distances, 99):.4f} m",
            f"{np.mean(np.abs(errors[:, 0])):.4f} m",
            f"{np.mean(np.abs(errors[:, 1])):.4f} m",
            f"{np.mean(np.abs(errors[:, 2])):.4f} m",
            f"{len(predictions)}"
        ]
    }

    df = pd.DataFrame(stats)
    print(df.to_string(index=False))

    # Save to CSV
    csv_path = os.path.join(output_dir, 'statistics_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nStatistics saved to: {csv_path}")

    return df


def save_all_results(
    results: Dict,
    output_dir: str
) -> None:
    """
    Save all analysis results.

    Args:
        results: Dictionary of analysis results
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save as JSON
    json_path = os.path.join(output_dir, 'error_analysis_results.json')
    json_results = {}
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            json_results[key] = value.to_dict(orient='records')
        elif value is not None:
            json_results[key] = value

    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    print(f"\nAll results saved to: {json_path}")


def main():
    """Main function."""
    args = parse_arguments()

    print("="*60)
    print("ERROR ANALYSIS FOR INDOOR POSITIONING")
    print("="*60)

    # Load data
    predictions, targets, timestamps, areas = load_data(args)

    # Determine analysis types
    if 'all' in args.analysis:
        analysis_types = ['distance', 'temporal', 'spatial', 'axis', 'statistics']
    else:
        analysis_types = args.analysis

    # Create output directory
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"analysis_{timestamp_str}")
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # Run analyses
    if 'distance' in analysis_types:
        results['distance_buckets'] = analyze_distance_buckets(
            calculate_errors(predictions, targets)['distances'],
            output_dir, args.format, args.dpi
        )

    if 'temporal' in analysis_types:
        results['temporal'] = analyze_temporal_errors(
            predictions, targets, timestamps,
            output_dir, args.format, args.dpi
        )

    if 'spatial' in analysis_types:
        results['spatial'] = analyze_spatial_errors(
            predictions, targets, areas,
            output_dir, args.format, args.dpi
        )

    if 'axis' in analysis_types:
        results['per_axis'] = analyze_per_axis_errors(
            predictions, targets,
            output_dir, args.format, args.dpi
        )

    if 'statistics' in analysis_types:
        results['statistics'] = generate_statistics_summary(
            predictions, targets, output_dir
        )

    # Save all results
    save_all_results(results, output_dir)

    print("\n" + "="*60)
    print("ERROR ANALYSIS COMPLETED")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
