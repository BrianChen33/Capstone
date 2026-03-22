# 蓝牙定位项目 (Bluetooth AoA Indoor Localization Capstone)

本项目旨在基于蓝牙 AoA（Angle of Arrival）空间谱数据，利用深度学习进行室内定位。项目包含数据探索、预处理实验、模型对比、模型训练、消融实验、误差分析以及结果可视化等完整流程。

## 环境要求

- Python 3.8+
- PyTorch 2.2.2（含 CUDA 支持，推荐）

通过以下命令安装所有依赖：
```bash
pip install -r requirements.txt
```

## 文件结构与说明

### 1. 数据集探索
- **文件**: `explore_dataset.py`
- **功能**: 生成数据集的统计信息与可视化图表（轨迹图、时间戳分布、信号强度分布等）。
- **输出**: `FigData/ExploreDataset/combined` 目录，包含 PNG 图表和 `dataset_summary.json`。
- **运行**:
  ```bash
  python explore_dataset.py
  ```

### 2. .pt 文件检查工具
- **文件**: `inspect_pt.py`
- **功能**: 快速查看 `.pt` 文件的内部结构、形状和部分数据内容，支持将张量保存为 `.npz`。
- **运行**:
  ```bash
  python inspect_pt.py Dataset/train_combined.pt
  ```

### 3. 预处理实验
- **文件**: `preprocess_experiments.py`
- **功能**: 比较不同预处理策略（Raw、Block Z-score、Robust）对模型训练效果的影响。
- **输出**: 结果保存在 `FigData/PreprocessExperiments/combined`。
- **运行**:
  ```bash
  python preprocess_experiments.py
  ```

### 4. 模型对比
- **文件**: `model_compare.py`
- **功能**: 在固定计算预算（6 epochs）下对比 MLP、CNN 和 Transformer 三种模型的性能。
- **输出**: 比较结果与图表保存在 `FigData/ModelCompare/combined`。
- **运行**:
  ```bash
  python model_compare.py
  ```

### 5. 模型训练
- **文件**: `train.py`
- **功能**: 使用选定的最佳模型架构（Transformer）进行完整的训练和验证，并保存模型权重。
- **输出**: 模型权重 `artifacts/best_model.pt`，训练报告 `artifacts/training_report.json`，数据统计 `artifacts/dataset_stats.json`。
- **运行**:
  ```bash
  python train.py --train-path Dataset/train_combined.pt --test-path Dataset/test_combined.pt --epochs 10
  ```

### 6. 预测生成与详细指标
- **文件**: `generate_predictions.py`
- **功能**: 加载训练好的模型，对测试集进行推理，输出预测坐标和详细误差指标。
- **输出**: `artifacts/test_predictions.npy`、`artifacts/test_targets.npy`、`artifacts/test_timestamps.npy`、`artifacts/detailed_metrics.json`。
- **运行**:
  ```bash
  python generate_predictions.py
  ```

### 7. 结果可视化
- **文件**: `visualize_enhanced.py`
- **功能**: 加载预测结果和真实坐标，绘制多种可视化图表（3D 轨迹、误差热力图、CDF、直方图、箱线图等）。
- **输出**: 图表保存在 `FigData/FinalReport/` 下的时间戳子目录。
- **运行**:
  ```bash
  python visualize_enhanced.py
  ```

### 8. 误差分析
- **文件**: `error_analysis.py`
- **功能**: 对预测结果进行深入误差分析，包括分位数统计、空间误差分布等。
- **运行**:
  ```bash
  python error_analysis.py
  ```

### 9. 消融实验
- **文件**: `run_ablation.py`
- **功能**: 通过逐一屏蔽（零化）输入特征组，评估各特征对模型性能的贡献。
- **输出**: `artifacts/ablation_results.json`。
- **运行**:
  ```bash
  python run_ablation.py
  ```
- 注：`ablation_study.py` 为原始版本（依赖 CSV），`run_ablation.py` 直接使用 `.pt` 数据。

### 10. 报告图表导出
- **文件**: `export_report_figs.py`
- **功能**: 将实验图表整理并导出为报告格式。
- **运行**:
  ```bash
  python export_report_figs.py
  ```

## 推荐运行流程

1. **探索数据**: `python explore_dataset.py` — 了解数据分布与基本统计信息。
2. **确定预处理**: `python preprocess_experiments.py` — 比较归一化方案，确认 Block Z-score 为最佳。
3. **模型选型**: `python model_compare.py` — 对比 MLP/CNN/Transformer，选择最佳架构。
4. **正式训练**: `python train.py --epochs 10` — 使用 Transformer 进行完整训练。
5. **生成预测**: `python generate_predictions.py` — 输出测试集预测和详细指标。
6. **可视化**: `python visualize_enhanced.py` — 生成各类误差分析图表。
7. **消融实验**: `python run_ablation.py` — 评估各特征组的贡献。
8. **误差分析**: `python error_analysis.py` — 深入分析误差分布。

## 目录结构说明

```
Capstone/
├── Dataset/          # .pt 数据文件（train/test，含单场景和合并版）
├── FigData/          # 实验生成的图表和报告数据
│   ├── ExploreDataset/
│   ├── ModelCompare/
│   ├── PreprocessExperiments/
│   └── FinalReport/
├── artifacts/        # 训练产物（模型权重、统计信息、预测结果）
├── backup/           # 历史代码和报告备份
└── *.py              # 各功能脚本
```

## 主要实验结果

| 指标 | 数值 |
|------|------|
| 最佳模型 | Transformer |
| 验证集 MAE（6 epoch 对比） | 0.2533 m |
| 测试集欧氏距离 MAE | 0.468 m |
| 测试集中位误差 | 0.294 m |
| 0.5 m 内占比 | 69.0% |
| 1.0 m 内占比 | 87.6% |
