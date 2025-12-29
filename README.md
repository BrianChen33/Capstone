# Capstone – Indoor Bluetooth Positioning Prototype

## 1. Background and Problem Statement
Indoor positioning technologies have become pivotal for applications such as smart buildings, asset tracking, unmanned retail, and security. Unlike outdoor environments where GPS is available, indoor spaces are rife with reflections, occlusions, and complex structures, resulting in highly dynamic channels and imposing stricter requirements on positioning accuracy, stability, and real-time performance. Among available technologies, Bluetooth has emerged as an important choice in both industry and academia due to its low cost, low power consumption, and ease of deployment. However, traditional Bluetooth positioning methods are constrained by large fluctuations in signal strength, making it difficult to achieve stable sub-meter or better accuracy in complex scenarios.

In recent years, array techniques focusing on phase and Angle of Arrival (AoA) and deep learning have pushed the performance frontier of indoor positioning. The iArk platform achieves protocol-agnostic phase estimation and spatial construction via "massive antenna arrays + side channels", enabling high-precision AoA and triangulation across protocols, significantly mitigating multipath interference and providing an engineering-feasible hardware and middleware solution (An et al., 2020). At the algorithmic level, Zhao et al. (2024) further introduced Transformers into the positioning task, using A-Subnetworks learn AoA, T-Subnetwork learns triangulation and motion context, and the pre-trained model LocGPT supports cross-scenario transfer, substantially reducing data requirements and deployment costs. These two lines of work, from systems and algorithms respectively, validate the superiority of “spatial spectrum + deep learning” in complex indoor environments.

Building on these advances, this project focuses on the following core question: under multi-gateway Bluetooth AoA acquisition, how can deep learning models effectively represent high-dimensional spatial spectra and perform multi-view fusion to achieve stable, reproducible, and real-time high-precision localization in complex multipath environments? To address this question, the project must systematically tackle four challenges: (1)observation aliasing caused by multipath interference and distance attenuation; (2)representation learning of high-dimensional spatial spectra and temporal context modeling; (3)the impact of multi-gateway spatial geometry on localization observability and robustness; and (4)the trade-off between cross-scenario generalization and engineering real-time performance.

The significance of this project lies in proposing and validating an end-to-end “spatial spectrum–driven” deep learning solution grounded in Bluetooth’s cost and power advantages. It will incorporate iArk’s spatial spectrum construction philosophy and introduce Transformer-based multi-view fusion and context modeling, with the potential to significantly improve accuracy, robustness, and deployability of indoor positioning without requiring expensive ultra-wideband (UWB) or complex hardware modifications.

## 2. Objectives and Outcomes
This project aims to achieve the following objectives:
1.	**Data and mechanism understanding**: Systematically analyze the statistical properties of Bluetooth spatial spectra, visualize multipath and its relationship with accuracy, and quantify the effects of gateway count, placement, and directivity on observability.
2.	**Model design and implementation**: Build and compare multiple deep learning localization models; at a controlled scale, draw on the division of labor in A-Subnetworks/T-Subnetwork to form an "AoA first, triangulation next” hierarchical architecture.
3.	**End-to-end evaluation**: Train and test on a given dataset; evaluate localization error (mean/median/90th percentile), inference latency, and throughput.
4.	**Methodological comparison and recommendations**: Compare with traditional geometric methods and CNN/MLP baselines; provide recommended configurations and deployment guidelines for engineering practice.

Expected outcomes include: (1)a reproducible deep learning–based Bluetooth indoor localization prototype (including data processing, training/inference, and visualization toolchain); and (2)a quantitative report and visualizations, offering configuration recommendations and experiential summaries for different scenarios.

## 3. Implementation Status
This section details the current implementation of the core components, corresponding to the initial phase of the project methodology.

### 3.1 Dataset Analysis
**Goal**: Understand the statistical properties of the raw data.
**Implementation**:
- **Script**: `train.py` -> `analyze_dataset` function.
- **Method**: The script loads the raw tensor `[N, 1, 986]` and computes global statistics for the feature channels (first 984 columns) and the target coordinates (last 2 columns).
- **Metrics**: Sample count, feature dimensionality, mean, standard deviation, min/max values for features, and bounding box for coordinates.
- **Output**: Results are saved to `artifacts/dataset_stats.json`.

### 3.2 Preprocessing Dataset Analysis
**Goal**: Prepare data for stable model training.
**Implementation**:
- **Script**: `train.py` -> `BluetoothPositioningDataset` class and `make_datasets` function.
- **Method**:
    - **Normalization**: Z-score standardization is applied to the features: $x' = \frac{x - \mu}{\sigma}$. The mean ($\mu$) and standard deviation ($\sigma$) are computed from the training set only and applied to validation/test sets to prevent data leakage.
    - **Splitting**: The `make_datasets` function performs a random 80/20 split of the training tensor into training and validation sets using a fixed random seed (42) for reproducibility.

### 3.3 Network Architecture Selection
**Goal**: Establish a baseline model for coordinate regression.
**Implementation**:
- **Script**: `train.py` -> `SimpleRegressor` class.
- **Architecture**: A Multi-Layer Perceptron (MLP) baseline.
    - **Input**: 984-dimensional spatial spectrum features.
    - **Hidden Layers**:
        - Linear (984 -> 256) + ReLU
        - Dropout (0.1) for regularization
        - Linear (256 -> 128) + ReLU
    - **Output**: Linear (128 -> 2) representing (x, y) coordinates.
- **Rationale**: MLP serves as a robust baseline to validate the pipeline before exploring complex architectures like CNNs or Transformers.

### 3.4 Preliminary Modeling
**Goal**: Train the baseline model and evaluate performance.
**Implementation**:
- **Script**: `train.py` -> `train` and `evaluate` functions.
- **Training Loop**:
    - **Optimizer**: Adam (`lr=1e-3`).
    - **Loss Function**: Mean Squared Error (MSE).
    - **Procedure**: Iterates for a fixed number of epochs (default 5). Tracks training loss and validation MSE/MAE.
    - **Checkpointing**: Saves the model state with the lowest validation MSE to `artifacts/best_model.pt`.
- **Evaluation**: Final evaluation is performed on the held-out test set (`test_data-s02-80-20-seq1.pt`), with metrics saved to `artifacts/metrics.json`.

## 4. Future Plan

### 4.1 Model Training and Tuning
**Plan**:
- **Hyperparameter Optimization**: Systematically tune learning rate, batch size, and dropout rates using grid search or Bayesian optimization.
- **Advanced Architectures**:
    - **CNN**: Implement 1D CNNs to capture local correlations in the spatial spectrum.
    - **Transformer**: Implement the "AoA first, triangulation next" architecture using Transformer encoders to model the global context of the spatial spectrum.
- **Regularization**: Experiment with weight decay and different dropout strategies to improve generalization.

### 4.2 Experimental Design
**Plan**:
- **Cross-Validation**: Implement k-fold cross-validation to ensure robustness of the results.
- **Ablation Studies**: Analyze the impact of different feature subsets (e.g., specific frequency bands or antennas).
- **Comparison**: Compare the deep learning approach against traditional geometric triangulation methods.
- **Visualization**: Develop tools to visualize the predicted trajectories vs. ground truth to identify error patterns (e.g., corner cases, multipath-heavy regions).

## 5. Setup and Usage
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
