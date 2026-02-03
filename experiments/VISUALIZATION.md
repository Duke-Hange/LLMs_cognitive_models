# 实验可视化说明

主流程下应看的图与可选分析图如下。

## 主流程：最终应看的图

| 图 | 路径 | 说明 |
|----|------|------|
| **跨模型比较图** | `04_comparison/output/comparison_r2_{timestamp}.png` | 运行 `run_comparison.py` 后自动生成；符号 vs 神经各模型在三种划分上的 Test R²、Test MSE；报告中会引用该图。 |

## 训练阶段默认图

| 图 | 路径 | 说明 |
|----|------|------|
| 符号均值曲线 | `01_symbolic_models_enhanced/results/enhanced_training/curves/mean_curve_*_{划分}_{模型}.png` | 默认 `--curves mean_only`；用于检查收敛。 |
| 神经均值曲线 | `02_neural_models/results/curves/mean_curve_*_{划分}_{模型}.png` | 默认 `--curves mean_only`；用于检查收敛。 |

- 需要每种子单次曲线时：`--curves all`。
- 不需要任何曲线图时：`--curves none`（仍保存曲线 JSON）。

## 可选分析/调试图（非主流程）

| 模块 | 产出图 | 用途 |
|------|--------|------|
| `02_neural_models/run_learning_curve.py` | data_quantity_curve_*_{split}.png | 训练集比例 vs 测试 MSE。 |
| `01_symbolic_models_enhanced/run_learning_curve.py` | data_quantity_curve_symbolic_* | 同上（符号模型）。 |
| `01_symbolic_models_enhanced/analysis/` | enhanced_performance_comparison.png、error_analysis、parameter_distribution、calibration_curves | 符号内部或按划分的深入分析。 |
| `01_symbolic_models_enhanced/diagnostic_r2_check.py` | diagnostic_r2_pred_vs_actual.png | 调试 R² 用。 |
| `01_symbolic_models_enhanced/test_enhanced_integration.py` | enhanced_*_predictions.png | 测试用。 |
| `00_data_preparation/` | bRate_distribution、feedback、ev_diff、feature_correlation 等 | 数据探索；01/02/04 不依赖 00。 |
