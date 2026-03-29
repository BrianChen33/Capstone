# Bluetooth AoA Indoor Localization Capstone

This project applies deep learning to indoor localization using Bluetooth AoA (Angle of Arrival) spatial spectrum data. It covers the full pipeline: dataset exploration, preprocessing experiments, model comparison, training, ablation study, error analysis, and result visualization.

## Requirements

- Python 3.8+
- PyTorch 2.2.2 (CUDA recommended)

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Scripts

### 1. Dataset Exploration — `explore_dataset.py`
Generates dataset statistics and visualizations (trajectories, timestamp distribution, signal strength, etc.).
- **Output**: `FigData/ExploreDataset/combined/` (PNG charts + `dataset_summary.json`)
```bash
python explore_dataset.py
```

### 2. `.pt` File Inspector — `inspect_pt.py`
Inspects the internal structure, shapes, and sample data of `.pt` files. Supports exporting tensors to `.npz`.
```bash
python inspect_pt.py Dataset/train_combined.pt
```

### 3. Preprocessing Experiments — `preprocess_experiments.py`
Compares different preprocessing strategies (Raw, Block Z-score, Robust) and their impact on model performance.
- **Output**: `FigData/PreprocessExperiments/combined/`
```bash
python preprocess_experiments.py
```

### 4. Model Comparison — `model_compare.py`
Benchmarks MLP, CNN, and Transformer architectures under a fixed compute budget (6 epochs).
- **Output**: `FigData/ModelCompare/combined/`
```bash
python model_compare.py
```

### 5. Model Training — `train.py`
Trains the selected best architecture (Transformer) with full training and validation, then saves model weights.
- **Output**: `artifacts/best_model.pt`, `artifacts/training_report.json`, `artifacts/dataset_stats.json`
```bash
python train.py --train-path Dataset/train_combined.pt --test-path Dataset/test_combined.pt --epochs 50
```

### 6. Prediction & Detailed Metrics — `generate_predictions.py`
Loads the trained model and runs inference on the test set, outputting predicted coordinates and detailed error metrics.
- **Output**: `artifacts/test_predictions.npy`, `artifacts/test_targets.npy`, `artifacts/test_timestamps.npy`, `artifacts/detailed_metrics.json`
```bash
python generate_predictions.py
```

### 7. Visualization — `visualize_enhanced.py`
Generates a variety of visualization charts from predictions and ground truth (3D trajectories, error heatmaps, CDF, histograms, box plots, etc.).
- **Output**: Timestamped subdirectory under `FigData/Visualization/`
```bash
python visualize_enhanced.py
```

### 8. Error Analysis — `error_analysis.py`
Performs in-depth error analysis including quantile statistics and spatial error distributions.
```bash
python error_analysis.py
```

### 9. Ablation Study — `run_ablation.py`
Evaluates the contribution of each feature group by zeroing out input feature blocks one at a time.
- **Output**: `artifacts/ablation_results.json`
```bash
python run_ablation.py
```

### Shared Modules
- **`models.py`** — Unified model definitions (MLPRegressor, CNNRegressor, TransformerRegressor).
- **`shared.py`** — Common data loading, field parsing, preprocessing, and dataset utilities.

## Recommended Workflow

1. **Explore data**: `python explore_dataset.py` — Understand data distributions and basic statistics.
2. **Select preprocessing**: `python preprocess_experiments.py` — Compare normalization methods; Block Z-score performs best.
3. **Compare models**: `python model_compare.py` — Benchmark MLP / CNN / Transformer; select best architecture.
4. **Train**: `python train.py --epochs 50` — Full training with Transformer.
5. **Generate predictions**: `python generate_predictions.py` — Produce test-set predictions and detailed metrics.
6. **Visualize**: `python visualize_enhanced.py` — Generate error analysis charts.
7. **Ablation study**: `python run_ablation.py` — Evaluate feature group contributions.
8. **Error analysis**: `python error_analysis.py` — In-depth error distribution analysis.

## Directory Structure

```
Capstone/
├── Dataset/          # .pt data files (train/test, per-scenario and combined)
├── FigData/          # Experiment-generated figures and report data
│   ├── ExploreDataset/
│   ├── ModelCompare/
│   ├── PreprocessExperiments/
│   └── Visualization/
├── artifacts/        # Training outputs (model weights, stats, predictions)
├── backup/           # Historical code and report backups
├── models.py         # Model architectures
├── shared.py         # Shared utilities
└── *.py              # Experiment scripts
```

## Key Results

| Metric | Value |
|--------|-------|
| Best Model | Transformer |
| Test Euclidean MAE | 0.373 m |
| Test Median Error | 0.172 m |
| Within 0.5 m | 76.6% |
| Within 1.0 m | 90.8% |
| X-axis MAE | 0.350 m |
| Y-axis MAE | 0.080 m |
