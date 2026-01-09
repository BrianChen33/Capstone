# 蓝牙定位项目 (Bluetooth Positioning Capstone)

本项目旨在基于蓝牙 AoA 空间谱数据进行室内定位。项目包含数据探索、预处理实验、模型对比、模型训练以及结果可视化等完整流程。

## 环境要求

请确保安装了 Python 3.8+ 及以下依赖：
- torch
- numpy
- matplotlib

可以通过以下命令安装：
```bash
pip install -r requirements.txt
```
*(如果没有 `requirements.txt`，请参考代码中的 import 自行安装)*

## 文件结构与说明

### 1. 数据集探索
- **文件**: `explore_dataset.py`
- **功能**: 生成数据集的统计信息与可视化图表（如轨迹图、时间戳分布、信号强度分布等）。
- **输出**: 默认输出到 `FigData/ExploreDataset/combined` 目录，包含 PNG 图表和 `dataset_summary.json`。
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

### 3. 数据校验与导出
- **文件**: `verify_and_export.py`
- **功能**: 
    1. 校验不同网关（g2/g3）数据是否一致。
    2. 将 `.pt` 数据集导出为 CSV 格式以便其他工具分析。
- **运行**:
  ```bash
  # 校验 g2/g3
  python verify_and_export.py --train-path Dataset/train_combined.pt --test-path Dataset/test_combined.pt
  
  # 导出为 CSV
  python verify_and_export.py --pt-path Dataset/train_combined.pt --csv-out artifacts/train.csv
  ```

### 4. 预处理实验
- **文件**: `preprocess_experiments.py`
- **功能**: 比较不同预处理策略（Raw, Block Z-score, Robust）对模型训练效果的影响。
- **输出**: 结果保存在 `FigData/PreprocessExperiments/combined`。
- **运行**:
  ```bash
  python preprocess_experiments.py
  ```

### 5. 模型对比
- **文件**: `model_compare.py`
- **功能**: 在固定预算下对比 MLP、CNN 和 Transformer 三种模型的性能。
- **输出**: 比较结果与图表保存在 `FigData/ModelCompare/combined`。
- **运行**:
  ```bash
  python model_compare.py
  ```

### 6. 模型训练
- **文件**: `train.py`
- **功能**: 使用选定的最佳模型架构（默认为 MLP）进行完整的训练和验证，并保存模型权重。
- **输出**: 模型权重保存在 `artifacts/best_model.pt`，训练报告在 `artifacts/training_report.json`。
- **运行**:
  ```bash
  python train.py --train-path Dataset/train_combined.pt --test-path Dataset/test_combined.pt --epochs 10
  ```

### 7. 结果可视化
- **文件**: `visualize.py`
- **功能**: 加载训练好的模型和数据集，绘制预测坐标与真实坐标 (Ground Truth) 的对比图。
- **运行**:
  ```bash
  python visualize.py Dataset/test_combined.pt --model-path artifacts/best_model.pt --stats-path artifacts/dataset_stats.json
  ```

## 运行流程建议

1. **探索数据**: 运行 `explore_dataset.py` 了解数据分布。
2. **确定预处理**: 运行 `preprocess_experiments.py` 确认最佳归一化方案（通常是 Block Z-score）。
3. **选型**: 运行 `model_compare.py` 选择表现最好的模型架构。
4. **训练**: 运行 `train.py` 进行正式训练。
5. **验证**: 运行 `visualize.py` 查看可视化结果。

## 目录结构说明

- `Dataset/`: 存放 `.pt` 数据文件。
- `FigData/`: 存放各类实验生成的图表和报告数据。
- `artifacts/`: 存放训练好的模型权重和训练过程统计信息。
