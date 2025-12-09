# Capstone – Indoor Bluetooth Positioning Prototype

This repository contains a minimal, reproducible pipeline for analyzing the provided Bluetooth spatial-spectrum tensors, preprocessing them, selecting a baseline deep learning architecture, and performing preliminary modeling.

## Repository contents
- `train_data-s02-80-20-seq1.pt`, `test_data-s02-80-20-seq1.pt`: training and evaluation tensors shaped `[N, 1, 986]`. The last two elements per sample represent normalized 2D coordinates; the preceding 984 values are spatial-spectrum features.
- `train.py`: end-to-end script that performs dataset analysis, preprocessing (normalization), model training, validation, and testing.
- `.gitignore`: ignores generated artifacts such as saved models.

## Setup
Install CPU-only PyTorch:

```bash
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu --progress-bar off
```

## Stage outcomes
1) **Dataset analysis**: `train.py` reports sample count, feature dimensionality, global feature range/mean/std, and coordinate bounds and saves them to `artifacts/dataset_stats.json`. On the provided data, the script detects `5960` training samples with `984` features and coordinate ranges roughly in `[0,1]`.

2) **Preprocessing analysis**: The pipeline standardizes features using training-set mean/std (computed once and reused for validation/test to avoid leakage) and performs an 80/20 train/validation split with a fixed seed.

3) **Network architecture selection**: A compact MLP (`SimpleRegressor`) is used for 2D coordinate regression: `984 -> 256 -> 128 -> 2` with ReLU activations and a light dropout layer.

4) **Preliminary modeling**: The training loop uses Adam + MSE loss, tracks validation MSE/MAE each epoch, retains the best weights, and evaluates on the held-out test set. Metrics are stored in `artifacts/metrics.json`, and the model checkpoint is saved to `artifacts/best_model.pt`.

## How to run and train
Run a quick smoke training (adjust epochs/batch size as needed):

```bash
python train.py --epochs 5 --batch-size 64 --lr 1e-3 \
  --train-path train_data-s02-80-20-seq1.pt \
  --test-path test_data-s02-80-20-seq1.pt \
  --output-dir artifacts
```
If your PyTorch build predates `weights_only=True`, add `--allow-unsafe-load` (only for trusted `.pt` files).

Key CLI options:
- `--epochs`: training epochs (default 5).
- `--batch-size`: batch size (default 64).
- `--lr`: learning rate (default 1e-3).
- `--val-ratio`: validation fraction from the training tensor (default 0.2).

The script prints dataset stats, per-epoch validation metrics, final test MSE/MAE, and writes artifacts under the chosen output directory.

## Model training, tuning, and experimental design
- **Splits**: Use the built-in 80/20 train/val split for early stopping; keep the provided test tensor for final evaluation only.
- **Hyperparameters to tune**: learning rate (1e-4–5e-3), batch size (32–256), dropout (0.0–0.3), hidden widths (128–512), and epochs (with early stopping on validation MSE).
- **Feature handling**: Keep the last two tensor values as regression targets; standardize only the feature columns using training statistics.
- **Metrics**: Track MSE (optimization target) and MAE (interpretability). Compare validation curves to detect overfitting.
- **Extensions**: Swap `SimpleRegressor` with deeper MLPs, 1D CNNs over the 984-length spectrum, or lightweight Transformer encoders for spatial-spectral modeling if capacity is needed.
