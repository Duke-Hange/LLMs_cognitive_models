# R² 快速检查报告

## 1. 代码检查：R² 是否在测试集上正确计算？

**结论：是。**

- `train_enhanced_models.py` 中：
  - 测试指标：`y_test = y[test_idx]`，`y_test_pred = adapter.predict_from_standardized(test_data)`，`test_metrics = self.evaluate_model(y_test, y_test_pred)`，**确为测试集**。
  - R² 公式：`ss_res = sum((y_true - y_pred)^2)`，`ss_tot = sum((y_true - mean(y_true))^2)`，`r2 = 1 - ss_res/ss_tot`，与标准定义一致。

## 2. 训练 R² vs 测试 R²（来自最新结果 CSV）

| 划分 | 模型 | train_r2 | test_r2 |
|------|------|----------|---------|
| problem | ev | -0.303 | -0.305 |
| problem | eu | -0.019 | -0.022 |
| problem | pt3 | -0.015 | -0.019 |
| problem | pt5 | -0.010 | -0.013 |
| parameter_amb | ev | -0.302 | -0.361 |
| parameter_amb | eu | -0.012 | -0.090 |
| parameter_amb | pt3 | -0.010 | -0.078 |
| parameter_amb | pt5 | -0.006 | -0.069 |
| parameter_ev_extreme | ev | -0.724 | -0.414 |
| parameter_ev_extreme | eu | -0.037 | -0.013 |
| parameter_ev_extreme | pt3 | -0.027 | -0.014 |
| parameter_ev_extreme | pt5 | -0.017 | -0.008 |

**发现：**

- **训练集 R² 也多为负**（ev 最差，pt5 最接近 0）。说明不是单纯的“过拟合导致测试差”，而是**在训练集上符号模型也没有超过“预测均值”这条基线**。
- 训练/测试 R² 接近（如 problem 上 pt5：train -0.010 vs test -0.013），**泛化断裂不明显**，更像是**模型/链接函数或目标与数据不匹配**。

## 3. 预测 vs 真实散点图

已添加脚本 `diagnostic_r2_check.py`。在项目根目录执行：

```bash
python experiments/01_symbolic_models_enhanced/diagnostic_r2_check.py
```

会：

- 在终端打印上述 train_r2 / test_r2 表；
- 对 problem_split + pt5 复算测试集 R² 与相关系数；
- 生成 `results/diagnostic_r2_pred_vs_actual.png`（y_test vs y_pred 散点图）。

若散点大致沿 y=x、相关系数为正，说明预测与真实对齐、负 R² 主要来自方差/尺度；若散点乱或反相关，需排查实现或数据。

## 4. 小结

- **R² 计算与使用的数据集合无误**（测试集、标准公式）。
- **训练集 R² 也为负**，更支持“模型/设定与当前数据匹配度有限”，而非单纯泛化问题。
- 建议本地运行 `diagnostic_r2_check.py` 查看散点图，再决定是否进一步查实现或数据。
