# 02_neural_models（experiments：CSV + 1-bRate）

## 与 experiments2/02 的主要差异

**本目录直接读取 `experiments/data/data.csv`，标签固定为 `y = 1 - bRate`，并使用单一 `train_test_split`。**

- 数据源：`experiments/data/data.csv`（列 `bRate,A,B`，`A/B` 为 `[[p,x], ...]`）。
- 标签：`TARGET_MODE = "one_minus_bRate"`，即 `y = 1 - bRate`。
- 划分：单一 `train_test_split`；学习曲线在同一训练集上按比例抽样。
- 训练时**不对** `enc_A` / `enc_B` / `enc_full` 做 sklearn `StandardScaler`。

其余（模型结构、早停、数据量曲线、绘图）保持与原实现一致风格。

## 如何运行

确保已存在 `experiments/data/data.csv`（已从参考代码复制）。

```bash
cd experiments/02_neural_models
python run_learning_curve.py
```

快速验证（更短）：

```bash
python run_learning_curve.py --n-fractions 5 --seeds 42
```
