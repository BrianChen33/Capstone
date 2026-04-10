#!/usr/bin/env python3
"""
Consolidated Visualization Script for COMP4913 Capstone Project.

Generates ALL figures for the final report:
  Part 0 — Dataset exploration figures       -> FigData/ExploreDataset/
  Part 1 — Preprocessing & model comparison  -> FigData/PreprocessExperiments/ & FigData/ModelCompare/
  Part 2 — Per-scene 4-in-1 prediction figs  -> FigData/Visualization/
  Part 3 — Aggregate multi-scene figures     -> FigData/Aggregate/

Usage:
    python visualization.py                   # generate all figures
    python visualization.py --only dataset
    python visualization.py --only per-scene
    python visualization.py --only aggregate
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import warnings

from shared import (
    DEVICE,
    FIELD_SLICES,
    PROJECT_ROOT,
    SCENE_IDS,
    get_scene_paths,
    safe_load,
    split_fields,
)

warnings.filterwarnings("ignore")

# ── Style — enlarged fonts for report readability ────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.dpi": 150,
    "figure.titlesize": 20,
})

DPI = 150

# ── Output directories ──────────────────────────────────────────────────
EXPLORE_DIR = PROJECT_ROOT / "FigData" / "ExploreDataset"
PP_DIR = PROJECT_ROOT / "FigData" / "PreprocessExperiments"
MC_DIR = PROJECT_ROOT / "FigData" / "ModelCompare"
VIS_DIR = PROJECT_ROOT / "FigData" / "Visualization"
AGG_DIR = PROJECT_ROOT / "FigData" / "Aggregate"

for _d in [EXPLORE_DIR, PP_DIR, MC_DIR, VIS_DIR, AGG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

SCENE_COLORS = {"s13": "#e74c3c", "s20": "#3498db", "s27": "#2ecc71", "s34": "#f39c12"}


# ═════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════
def _load_scene_results(sid: str):
    """Return (predictions, targets) npy arrays for a scene, or (None, None)."""
    pred_p = PROJECT_ROOT / "artifacts" / sid / "test_predictions.npy"
    tgt_p = PROJECT_ROOT / "artifacts" / sid / "test_targets.npy"
    if not pred_p.exists() or not tgt_p.exists():
        return None, None
    return np.load(pred_p), np.load(tgt_p)


# ═════════════════════════════════════════════════════════════════════════
# Part 0 — Dataset Exploration Figures  (FigData/ExploreDataset/)
# ═════════════════════════════════════════════════════════════════════════

def plot_dataset_overview():
    """Combined 4-scene GT scatter (train+test) + sample count bar."""
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    train_counts, test_counts = [], []
    for idx, sid in enumerate(SCENE_IDS):
        ax = axes[idx // 2, idx % 2]
        tp, ep = get_scene_paths(sid)
        train_t = safe_load(tp)
        test_t = safe_load(ep)
        train_counts.append(len(train_t))
        test_counts.append(len(test_t))
        for tag, tensor, marker, color in [
            ("Train", train_t, "o", "#3498db"),
            ("Test", test_t, "x", "#e74c3c"),
        ]:
            fields = split_fields(tensor)
            gt = fields["gt_pos"].cpu().numpy()
            ax.scatter(gt[:, 0], gt[:, 1], s=6, alpha=0.4,
                       marker=marker, color=color, label=tag)
        ax.set_title(f"Scene {sid}  (train={len(train_t)}, test={len(test_t)})")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend()
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
    # Sample count bar chart
    ax_bar = axes[1, 2]
    x = np.arange(len(SCENE_IDS))
    w = 0.35
    ax_bar.bar(x - w / 2, train_counts, w, label="Train", color="#3498db")
    ax_bar.bar(x + w / 2, test_counts, w, label="Test", color="#e74c3c")
    for i, (tr, te) in enumerate(zip(train_counts, test_counts)):
        ax_bar.text(i - w / 2, tr + 50, str(tr), ha="center", fontsize=11)
        ax_bar.text(i + w / 2, te + 50, str(te), ha="center", fontsize=11)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(SCENE_IDS)
    ax_bar.set_ylabel("Sample Count")
    ax_bar.set_title("Dataset Sample Counts per Scene")
    ax_bar.legend()
    axes[0, 2].axis("off")
    fig.suptitle("Dataset Overview: Ground-Truth Spatial Distributions and Sample Counts")
    plt.tight_layout()
    plt.savefig(EXPLORE_DIR / "dataset_overview.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  [D1] dataset_overview.png")


def plot_spectrum_range_combined():
    """Spectrum value range for all scenes combined."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    spec_names = ["g1_spec", "g2_spec", "g3_spec"]
    gw_labels = ["Gateway 1", "Gateway 2", "Gateway 3"]
    for ax, sn, gl in zip(axes, spec_names, gw_labels):
        all_vals = []
        for sid in SCENE_IDS:
            tp, _ = get_scene_paths(sid)
            fields = split_fields(safe_load(tp))
            all_vals.append(fields[sn].detach().cpu().numpy().flatten())
        combined = np.concatenate(all_vals)
        ax.hist(combined, bins=60, color="#2980b9", alpha=0.75, edgecolor="white")
        ax.axvline(0, color="red", ls="--", lw=1.5, label="0")
        ax.axvline(1, color="red", ls="--", lw=1.5, label="1")
        ax.set_title(f"{gl} Spectrum Range")
        ax.set_xlabel("Spectrum Value")
        ax.set_ylabel("Count")
        ax.legend()
        ax.text(0.95, 0.95,
                f"min={combined.min():.4f}\nmax={combined.max():.4f}\nmean={combined.mean():.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=11,
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))
    fig.suptitle("Spatial Spectrum Value Range (All Scenes Combined)")
    plt.tight_layout()
    plt.savefig(EXPLORE_DIR / "spectrum_range_combined.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  [D2] spectrum_range_combined.png")


def plot_timestamp_dist_combined():
    """Timestamp distribution for all scenes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, sid in zip(axes.flat, SCENE_IDS):
        tp, ep = get_scene_paths(sid)
        for tag, path, color in [("Train", tp, "#3498db"), ("Test", ep, "#e74c3c")]:
            fields = split_fields(safe_load(path))
            ts = fields["timestamp"].cpu().numpy().flatten()
            ax.hist(ts, bins=40, alpha=0.6, color=color, edgecolor="white", label=tag)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Count")
        ax.set_title(f"Scene {sid}")
        ax.legend()
    fig.suptitle("Timestamp Distribution per Scene")
    plt.tight_layout()
    plt.savefig(EXPLORE_DIR / "timestamp_dist.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  [D3] timestamp_dist.png")


def plot_spectrum_energy_combined():
    """Gateway spectral energy distribution for all scenes combined."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, sid in zip(axes.flat, SCENE_IDS):
        tp, _ = get_scene_paths(sid)
        fields = split_fields(safe_load(tp))
        for sn, lbl in [("g1_spec", "G1"), ("g2_spec", "G2"), ("g3_spec", "G3")]:
            energy = fields[sn].mean(dim=1).detach().cpu().numpy()
            ax.hist(energy, bins=40, alpha=0.5, label=lbl)
        ax.set_xlabel("Mean Spectral Power")
        ax.set_ylabel("Count")
        ax.set_title(f"Scene {sid}")
        ax.legend()
    fig.suptitle("Gateway Spectral Energy Distribution per Scene")
    plt.tight_layout()
    plt.savefig(EXPLORE_DIR / "spectrum_energy.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  [D4] spectrum_energy.png")


def plot_gt_coordinate_hist():
    """GT x/y distribution for all scenes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, sid in zip(axes.flat, SCENE_IDS):
        tp, _ = get_scene_paths(sid)
        fields = split_fields(safe_load(tp))
        gt = fields["gt_pos"].cpu().numpy()
        ax.hist(gt[:, 0], bins=40, alpha=0.6, label="X", color="#e74c3c")
        ax.hist(gt[:, 1], bins=40, alpha=0.6, label="Y", color="#3498db")
        ax.set_xlabel("Coordinate (m)")
        ax.set_ylabel("Count")
        ax.set_title(f"Scene {sid}")
        ax.legend()
    fig.suptitle("Ground-Truth Coordinate Distribution per Scene")
    plt.tight_layout()
    plt.savefig(EXPLORE_DIR / "gt_coordinate_hist.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  [D5] gt_coordinate_hist.png")


# ═════════════════════════════════════════════════════════════════════════
# Part 1 — Experiment Comparison Figures
# ═════════════════════════════════════════════════════════════════════════

def plot_preprocess_comparison():
    """Bar chart from preprocess_metrics.json (cross-scene combined)."""
    pp_path = PP_DIR / "preprocess_metrics.json"
    if not pp_path.exists():
        print("  [PP] SKIP (preprocess_metrics.json not found)")
        return
    data = json.loads(pp_path.read_text())
    strategies = ["fully_raw", "ts_only", "block_zscore"]
    labels = ["Fully Raw", "Timestamp-Only", "Block Z-Score"]
    colours = ["#95a5a6", "#3498db", "#2ecc71"]
    for split in ("val", "test"):
        vals = [data[s][f"{split}_mae_mean"] for s in strategies]
        errs = [data[s][f"{split}_mae_std"] for s in strategies]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(labels, vals, yerr=errs, capsize=5, color=colours)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.4f}", ha="center", fontsize=12)
        ax.set_ylabel(f"{split.title()} MAE (m)")
        ax.set_title(f"Preprocessing Strategy Comparison ({split.title()}, Cross-Scene)")
        plt.tight_layout()
        plt.savefig(PP_DIR / f"{split}_mae_comparison.png", dpi=DPI)
        plt.close()
    print("  [PP] preprocess comparison plots")


def plot_model_comparison():
    """Bar chart from model_compare_metrics.json (cross-scene combined)."""
    mc_path = MC_DIR / "model_compare_metrics.json"
    if not mc_path.exists():
        print("  [MC] SKIP (model_compare_metrics.json not found)")
        return
    data = json.loads(mc_path.read_text())
    models = ["mlp", "cnn", "transformer"]
    labels = ["MLP", "CNN", "Transformer"]
    colours = ["#e74c3c", "#f39c12", "#9b59b6"]
    vals = [data[m]["val_mae"] for m in models]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vals, color=colours)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.4f}", ha="center", fontsize=12)
    ax.set_ylabel("Validation MAE (m)")
    ax.set_title("Model Architecture Comparison (6-epoch budget, Cross-Scene)")
    plt.tight_layout()
    plt.savefig(MC_DIR / "model_comparison.png", dpi=DPI)
    plt.close()
    print("  [MC] model_comparison.png")


# ═════════════════════════════════════════════════════════════════════════
# Part 2 — Per-scene 4-in-1 combined figures  (FigData/Visualization/)
# ═════════════════════════════════════════════════════════════════════════

def plot_4in1_trajectory():
    """2x2: predicted vs GT scatter (XY) for each scene."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    for ax, sid in zip(axes.flat, SCENE_IDS):
        preds, tgts = _load_scene_results(sid)
        if preds is None:
            ax.set_title(f"{sid} — no data"); continue
        n = len(preds)
        step = max(1, n // 500)
        idx = np.arange(0, n, step)
        ax.scatter(tgts[idx, 0], tgts[idx, 1], s=6, alpha=0.5,
                   label="Ground Truth", color="#3498db")
        ax.scatter(preds[idx, 0], preds[idx, 1], s=6, alpha=0.5,
                   label="Predicted", color="#e74c3c", marker="x")
        # error lines
        line_step = max(1, len(idx) // 50)
        for i in idx[::line_step]:
            ax.plot([tgts[i, 0], preds[i, 0]], [tgts[i, 1], preds[i, 1]],
                    "gray", alpha=0.2, lw=0.5)
        dists = np.linalg.norm(preds - tgts, axis=1)
        ax.set_title(f"{sid}  (n={n}, MAE={np.mean(dists):.3f} m)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(loc="upper right")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Predicted vs Ground-Truth Trajectory — All Scenes")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "trajectory_4scenes.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  [V1/8] trajectory_4scenes.png")


def plot_4in1_scatter():
    """2 rows (X, Y axis) x 4 cols (scenes): pred vs true scatter."""
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    for col, sid in enumerate(SCENE_IDS):
        preds, tgts = _load_scene_results(sid)
        if preds is None:
            continue
        for row, (ai, aname) in enumerate([(0, "X"), (1, "Y")]):
            ax = axes[row, col]
            ax.scatter(tgts[:, ai], preds[:, ai], s=3, alpha=0.3,
                       color=SCENE_COLORS[sid])
            lims = [min(tgts[:, ai].min(), preds[:, ai].min()) - 0.2,
                    max(tgts[:, ai].max(), preds[:, ai].max()) + 0.2]
            ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ss_res = np.sum((preds[:, ai] - tgts[:, ai]) ** 2)
            ss_tot = np.sum((tgts[:, ai] - tgts[:, ai].mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            ax.set_title(f"{sid} — {aname}  (R²={r2:.3f})")
            ax.set_xlabel(f"True {aname} (m)")
            ax.set_ylabel(f"Pred {aname} (m)")
            ax.set_aspect("equal")
    fig.suptitle("Scatter: Predicted vs Ground Truth — All Scenes")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "scatter_4scenes.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  [V2/8] scatter_4scenes.png")


def plot_4in1_histogram():
    """2x2: Euclidean-error histograms per scene."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    for ax, sid in zip(axes.flat, SCENE_IDS):
        preds, tgts = _load_scene_results(sid)
        if preds is None:
            ax.set_title(f"{sid} — no data"); continue
        dists = np.linalg.norm(preds - tgts, axis=1)
        ax.hist(dists, bins=50, density=True, color=SCENE_COLORS[sid],
                edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(dists), color="red", ls="--", lw=2,
                   label=f"Mean: {np.mean(dists):.3f} m")
        ax.axvline(np.median(dists), color="green", ls="--", lw=2,
                   label=f"Median: {np.median(dists):.3f} m")
        ax.set_xlabel("Euclidean Error (m)")
        ax.set_ylabel("Density")
        ax.set_title(f"{sid}  (n={len(dists)})")
        ax.legend()
    fig.suptitle("Error Histogram — All Scenes")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "histogram_4scenes.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  [V3/8] histogram_4scenes.png")


def plot_4in1_cdf():
    """2x2: CDF of Euclidean error per scene, with per-axis overlays."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    for ax, sid in zip(axes.flat, SCENE_IDS):
        preds, tgts = _load_scene_results(sid)
        if preds is None:
            ax.set_title(f"{sid} — no data"); continue
        dists = np.linalg.norm(preds - tgts, axis=1)
        sorted_e = np.sort(dists)
        cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        ax.plot(sorted_e, cdf, color=SCENE_COLORS[sid], lw=2, label="Euclidean")
        # per-axis overlays
        for ai, aname, ac in [(0, "X", "#e74c3c"), (1, "Y", "#2ecc71")]:
            ae = np.sort(np.abs(preds[:, ai] - tgts[:, ai]))
            ax.plot(ae, np.arange(1, len(ae) + 1) / len(ae),
                    ls=":", lw=1.2, color=ac, alpha=0.7, label=f"{aname}-axis")
        ax.axhline(0.5, ls="--", color="gray", alpha=0.5)
        ax.axhline(0.9, ls="--", color="gray", alpha=0.3)
        ax.set_xlabel("Error (m)")
        ax.set_ylabel("CDF")
        ax.set_title(f"{sid}  (n={len(dists)})")
        ax.set_xlim(0, max(5, sorted_e[-1]))
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("CDF of Euclidean Error — All Scenes")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "cdf_4scenes.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  [V4/8] cdf_4scenes.png")


def plot_4in1_heatmap():
    """2x2: spatial error heatmap (XY) per scene."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    for ax, sid in zip(axes.flat, SCENE_IDS):
        preds, tgts = _load_scene_results(sid)
        if preds is None:
            ax.set_title(f"{sid} — no data"); continue
        dists = np.linalg.norm(preds - tgts, axis=1)
        hb = ax.hexbin(tgts[:, 0], tgts[:, 1], C=dists, gridsize=25,
                       cmap="hot", reduce_C_function=np.mean)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"{sid}")
        plt.colorbar(hb, ax=ax, label="Mean Error (m)")
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Spatial Error Heatmap (XY) — All Scenes")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "heatmap_4scenes.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  [V5/8] heatmap_4scenes.png")


def plot_4in1_error_vector():
    """2x2: error vector arrows from true -> pred per scene."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    for ax, sid in zip(axes.flat, SCENE_IDS):
        preds, tgts = _load_scene_results(sid)
        if preds is None:
            ax.set_title(f"{sid} — no data"); continue
        errs = preds - tgts
        dists = np.linalg.norm(errs, axis=1)
        step = max(1, len(preds) // 200)
        idx = slice(None, None, step)
        sc = ax.scatter(tgts[idx, 0], tgts[idx, 1], c=dists[idx],
                        cmap="hot", s=12, alpha=0.6)
        ax.quiver(tgts[idx, 0], tgts[idx, 1], errs[idx, 0], errs[idx, 1],
                  color="blue", alpha=0.4, scale=10, width=0.003)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"{sid}")
        plt.colorbar(sc, ax=ax, label="Error (m)")
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Error Vectors (XY Plane) — All Scenes")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "error_vector_4scenes.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  [V6/8] error_vector_4scenes.png")


def plot_4in1_boxplot():
    """2x2: box plots of Euclidean + per-axis errors per scene."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    for ax, sid in zip(axes.flat, SCENE_IDS):
        preds, tgts = _load_scene_results(sid)
        if preds is None:
            ax.set_title(f"{sid} — no data"); continue
        errs = preds - tgts
        dists = np.linalg.norm(errs, axis=1)
        data = [dists, errs[:, 0], errs[:, 1]]
        labels_bp = ["Euclidean", "X-axis", "Y-axis"]
        bp = ax.boxplot(data, labels=labels_bp, patch_artist=True)
        colors = ["steelblue", "coral", "lightgreen"]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        ax.set_ylabel("Error (m)")
        ax.set_title(f"{sid}  (median={np.median(dists):.3f} m)")
        ax.axhline(0, color="red", ls="--", lw=1)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Error Box Plots — All Scenes")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "boxplot_4scenes.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  [V7/8] boxplot_4scenes.png")


def plot_4in1_violin():
    """2x2: violin plots of error distributions per scene."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    for ax, sid in zip(axes.flat, SCENE_IDS):
        preds, tgts = _load_scene_results(sid)
        if preds is None:
            ax.set_title(f"{sid} — no data"); continue
        errs = preds - tgts
        dists = np.linalg.norm(errs, axis=1)
        data = [dists, errs[:, 0], errs[:, 1]]
        positions = [1, 2, 3]
        xlabels = ["Euclidean", "X", "Y"]
        parts = ax.violinplot(data, positions=positions,
                              showmeans=True, showmedians=True)
        colors = ["steelblue", "coral", "lightgreen"]
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.7)
        ax.set_xticks(positions); ax.set_xticklabels(xlabels)
        ax.set_ylabel("Error (m)")
        ax.set_title(f"{sid}")
        ax.axhline(0, color="red", ls="--", lw=1)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Error Violin Plots — All Scenes")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "violin_4scenes.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  [V8/8] violin_4scenes.png")


# ═════════════════════════════════════════════════════════════════════════
# Part 3 — Aggregate multi-scene comparison figures  (FigData/Aggregate/)
# ═════════════════════════════════════════════════════════════════════════

def plot_final_metrics():
    """Bar chart: MAE, RMSE, Median for each scene."""
    metrics_path = PROJECT_ROOT / "artifacts" / "all_scenes_metrics.json"
    if not metrics_path.exists():
        print("  [A1] SKIP final metrics (JSON not found)")
        return
    data = json.loads(metrics_path.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metric_keys = [("mae_euclidean", "MAE (m)"),
                   ("rmse", "RMSE (m)"),
                   ("median_error", "Median Error (m)")]
    for ax, (key, ylabel) in zip(axes, metric_keys):
        vals = [data[sid][key] for sid in SCENE_IDS]
        bars = ax.bar(SCENE_IDS, vals, color=[SCENE_COLORS[s] for s in SCENE_IDS])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{v:.3f}", ha="center", fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
    fig.suptitle("Unified Transformer Model — Test Set Metrics per Scene")
    plt.tight_layout()
    plt.savefig(AGG_DIR / "final_metrics_summary.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  [A1] final_metrics_summary.png")


def plot_combined_cdf():
    """Overlay CDF curves for all scenes on one plot."""
    fig, ax = plt.subplots(figsize=(10, 7))
    for sid in SCENE_IDS:
        preds, tgts = _load_scene_results(sid)
        if preds is None:
            continue
        errors = np.linalg.norm(preds - tgts, axis=1)
        sorted_e = np.sort(errors)
        cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        ax.plot(sorted_e, cdf, label=f"{sid} (n={len(errors)})",
                color=SCENE_COLORS[sid], linewidth=2.5)
    ax.axhline(0.5, ls="--", color="gray", alpha=0.5, label="50%")
    ax.axhline(0.9, ls="--", color="gray", alpha=0.3, label="90%")
    ax.axvline(0.5, ls=":", color="gray", alpha=0.4)
    ax.axvline(1.0, ls=":", color="gray", alpha=0.4)
    ax.set_xlabel("Euclidean Error (m)")
    ax.set_ylabel("CDF")
    ax.set_title("Cumulative Error Distribution — All Scenes (Unified Model)")
    ax.legend()
    ax.set_xlim(0, 5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(AGG_DIR / "combined_cdf.png", dpi=DPI)
    plt.close()
    print("  [A2] combined_cdf.png")


def plot_combined_scatter():
    """X-axis and Y-axis pred-vs-gt scatter for all scenes (2x4 panel)."""
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    for col, sid in enumerate(SCENE_IDS):
        preds, tgts = _load_scene_results(sid)
        if preds is None:
            continue
        for row, (ai, aname) in enumerate([(0, "X"), (1, "Y")]):
            ax = axes[row, col]
            ax.scatter(tgts[:, ai], preds[:, ai], s=3, alpha=0.3,
                       color=SCENE_COLORS[sid])
            lims = [min(tgts[:, ai].min(), preds[:, ai].min()) - 0.2,
                    max(tgts[:, ai].max(), preds[:, ai].max()) + 0.2]
            ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_title(f"{sid} — {aname}-axis")
            ax.set_xlabel(f"True {aname}")
            ax.set_ylabel(f"Pred {aname}")
            ax.set_aspect("equal")
    fig.suptitle("Prediction vs Ground Truth — All Scenes (Unified Model)")
    plt.tight_layout()
    plt.savefig(AGG_DIR / "combined_scatter.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  [A3] combined_scatter.png")


def plot_ablation_results():
    """Horizontal bar chart of delta-MAE (%) for each ablation condition."""
    abl_path = PROJECT_ROOT / "artifacts" / "ablation_results.json"
    if not abl_path.exists():
        print("  [A4] SKIP ablation (JSON not found)")
        return
    data = json.loads(abl_path.read_text())
    conditions = ["no_timestamp", "no_gateway_pos", "no_spectrum_g1",
                   "no_spectrum_g2", "no_spectrum_g3", "no_spectrum_all"]
    labels = ["No Timestamp", "No Gateway Pos", "No Spec G1",
              "No Spec G2", "No Spec G3", "No All Spectra"]
    bl = data["baseline"]["test_mae"]
    deltas = [(data[c]["test_mae"] - bl) / bl * 100 for c in conditions]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if d > 0 else "#2ecc71" for d in deltas]
    bars = ax.barh(labels, deltas, color=colors)
    for bar, v in zip(bars, deltas):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v:+.1f}%", va="center", fontsize=12)
    ax.set_xlabel("Δ Test MAE (%)")
    ax.set_title("Feature Ablation Study — MAE Change (Cross-Scene)")
    ax.axvline(0, color="black", lw=1)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(AGG_DIR / "ablation_results.png", dpi=DPI)
    plt.close()
    print("  [A4] ablation_results.png")


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Consolidated visualization for Capstone Project")
    parser.add_argument("--only", choices=["dataset", "per-scene", "aggregate"],
                        default=None, help="Generate only one category of figures")
    args = parser.parse_args()

    if args.only is None or args.only == "dataset":
        print("=" * 60)
        print("Generating dataset exploration figures -> FigData/ExploreDataset/")
        print("=" * 60)
        plot_dataset_overview()
        plot_spectrum_range_combined()
        plot_timestamp_dist_combined()
        plot_spectrum_energy_combined()
        plot_gt_coordinate_hist()
        print("\nGenerating experiment comparison plots...")
        plot_preprocess_comparison()
        plot_model_comparison()

    if args.only is None or args.only == "per-scene":
        print("=" * 60)
        print("Generating per-scene 4-in-1 figures -> FigData/Visualization/")
        print("=" * 60)
        plot_4in1_trajectory()
        plot_4in1_scatter()
        plot_4in1_histogram()
        plot_4in1_cdf()
        plot_4in1_heatmap()
        plot_4in1_error_vector()
        plot_4in1_boxplot()
        plot_4in1_violin()

    if args.only is None or args.only == "aggregate":
        print("=" * 60)
        print("Generating aggregate figures -> FigData/Aggregate/")
        print("=" * 60)
        plot_final_metrics()
        plot_combined_cdf()
        plot_combined_scatter()
        plot_ablation_results()

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
