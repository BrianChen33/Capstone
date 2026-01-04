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

### 3.5 Model Comparison Outcome (current dataset)
- **Script**: `model_compare.py` (block z-score preprocessing, 80/20 split, batch size 32).
- **Configs tested**: MLP, CNN, LSTM, Transformer (base), Transformer (tuned, conv+CLS).
- **Validation MAE (20 epochs)**: MLP ≈ 0.0900, CNN ≈ 0.1077, Transformer_base ≈ 0.1206, Transformer_tuned ≈ 0.1248, LSTM ≈ 0.1660.
- **Conclusion**: MLP remains the best performer and is the recommended production backbone for this dataset size; Transformer variants are kept in the repo for reference only.

## 4. 模型对比与选型（已改为首选 MLP）

### 4.1 目标与原则
统一数据划分、预处理（block-zscore）、损失（MSE）与优化器（Adam，初始 lr=1e-3），以验证集 MAE/MSE 作为主要依据。控制快速迭代的训练轮数为 6，以便对小样本数据集进行公平、可重复的比较。

### 4.2 实验设置与可复现性
- 脚本：`model_compare.py`（仍保留 CNN/LSTM/Transformer 作为参考基线）。
- 输入：三路频谱堆叠为 `[N, 324, 3]`，meta 包含 9 维网关位置 + 1 维时间戳。
- 预处理：block-zscore（仅用训练集统计量）。
- 划分：80/20 随机划分，seed=42。
- 训练：6 轮（快速对比），Adam，lr=1e-3，batch size=32。
- 产物：`artifacts/model_compare/model_compare_metrics.json` 与 `figs/model_val_mae.png`。
- 复现命令：
    ```powershell
    C:/Users/chenb/Desktop/个人资料/nextjs-dashboard/Capstone/.venv/Scripts/python.exe model_compare.py --epochs 6
    ```

### 4.3 模型设计要点（对比理由）
- **MLP（首选）**：展平后直接回归，参数高效，训练稳定，对小数据集最友好。
- **1D CNN（参考）**：可抓局部尖峰/平滑模式，但长程依赖与跨网关对齐不足。
- **LSTM（参考）**：顺序建模对应频率轴，优势有限，当前数据规模下易欠拟合。
- **Transformer（参考）**：具备全局注意力与扩展潜力，但在本数据量与短轮次下欠收敛，指标落后。

### 4.4 实验结果
验证集表现（6 轮快速训练，block-zscore，batch 32）：

| 模型 | val_MAE | val_MSE |
| --- | --- | --- |
| MLP | ~0.10 | ~0.033 |
| CNN | ~0.20 | ~0.13 |
| LSTM | ~0.24 | ~0.19 |
| Transformer | ~0.25 | ~0.17 |

图表：`artifacts/model_compare/figs/model_val_mae.png`。

### 4.5 选用 MLP 的理由
1) **小样本稳定性**：在当前数据量与 6 轮限制下，MLP 收敛最快、误差最低；
2) **推理高效**：参数/算力开销最小，便于部署；
3) **可扩展性足够**：可通过加宽隐藏层、调节 dropout、权重衰减等方式继续提升，而无需引入复杂归纳偏置；
4) **风险更低**：相较欠收敛的 Transformer，MLP 在现有数据分布上的泛化更可控。

### 4.6 后续：若扩展数据再评估高级模型
保留 CNN/LSTM/Transformer 代码仅作参考。若未来有更大数据量或多时刻、多网关场景，可重新开启比较，重点在：更长训练、学习率调度、改进位置编码与通道注意力等。

## 5. 初步训练与代码（已切换为 MLP 默认）

### 5.1 训练脚本
- 文件：`train.py`
- 主干：固定为 MLP（`build_model` 返回 MLP）
- 预处理：block-zscore（训练统计量），标签为 `gt_pos` 前两维。
- 默认超参：epochs=6，batch_size=64，lr=1e-3，val_ratio=0.2。
- 产物：`best_model.pt`、`dataset_stats.json`、`training_report.json`。

运行示例（保持 MLP，无需指定模型）：

```powershell
C:/Users/chenb/Desktop/个人资料/nextjs-dashboard/Capstone/.venv/Scripts/python.exe train.py `
    --epochs 6 --batch-size 64 --lr 1e-3 `
    --train-path train_data-s02-80-20-seq1.pt `
    --test-path test_data-s02-80-20-seq1.pt `
    --output-dir artifacts
```
如遇旧版 PyTorch 不支持 `weights_only=True`，可加 `--allow-unsafe-load`（仅限可信文件）。

### 5.2 调优建议（围绕 MLP）
- 学习率与权重衰减：lr 1e-4–3e-3，wd 0–1e-3。
- 隐层宽度与深度：隐藏层 128–512，可加一层以提升表示力。
- 正则化：dropout 0.0–0.3；必要时早停或更长轮次配合 cos 调度。
- 评估：以验证 MAE/MSE 选模；保持测试集仅用于最终报告。
