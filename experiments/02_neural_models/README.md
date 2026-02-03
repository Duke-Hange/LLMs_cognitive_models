# 02_neural_models — 神经网络实验

## 目的

在 Choices13k 上实现并评估**两类**神经网络（Value-Based、Context-Dependent），与项目方案及 Peterson et al. (2021) 参考一致。**主输入为每个赌局的完整结果分布**（所有结果–概率对），不做摘要压缩。

## 数据来源

- **标准化数据**：与 `01_symbolic_models_enhanced` 一致，使用其产出的 `c13k_enhanced_standardized.json`（或通过 `EnhancedChoices13kStandardizer` 生成）。
- **划分**：使用 01 的 `create_enhanced_splits(standardized_data, split_type)`，三种类型为 `problem`、`parameter_amb`、`parameter_ev_extreme`。

## 输入：完整分布编码

- 每条样本从 `context['gamble_a']['distribution']` 与 `context['gamble_b']['distribution']` 读取 `[[p,x],...]`。
- 按 `max_outcomes`（默认 9）做 padding，不足补 0，得到固定长度向量。
- **Value-Based**：enc_A、enc_B 各 (n, 2×max_outcomes)，共享网络 f 得 V(A)、V(B)，再 Softmax(η) 得 P(B)。
- **Context-Dependent**：输入 (enc_A | enc_B)，维度 4×max_outcomes，网络直接输出 bRate。

## 模型定义

| 模型 | 输入 | 结构 |
|------|------|------|
| Value-Based | 单赌局分布编码 enc_A / enc_B | 1 隐藏层 64 单元 Sigmoid，Softmax(η) |
| Context-Dependent | 两赌局编码拼接 | 2 层各 32 单元 Sigmoid，Sigmoid 输出 |

## 依赖

需安装：`torch`、`numpy`、`scikit-learn`。在本目录下执行：

```bash
pip install -r requirements.txt
```

## 如何运行

**推荐：使用 conda 环境 `yh311_G`**（已有所需包时）

在本地进入本目录后任选一种方式：

```bash
# 方式一：激活环境后运行
conda activate yh311_G
python train.py
```

```bash
# 方式二：使用脚本（Windows 可双击 run_train.bat）
conda run -n yh311_G python train.py
```

或从项目根目录：

```bash
conda activate yh311_G
cd experiments/02_neural_models
python train.py
```

## 结果目录

- `results/` 下按时间戳存放当次运行的详细 JSON、summary CSV 与可选 Excel。
- 可与 `01_symbolic_models_enhanced/results/enhanced_training/enhanced_models_summary_*.csv` 按 split_type 对比符号模型与两类神经网络（Value-Based、Context-Dependent）的 test R²、MSE、Correlation。

## 结果与报告

- **训练结束后**：控制台会打印「神经网络实验结果汇总」（按划分、按模型的测试 MSE、R²、相关性）；结果文件在 `results/` 下（`neural_models_results_*.json`、`neural_models_summary_*.csv`，可选 `neural_models_summary_*.xlsx`）。
- **生成报告**：在本目录运行 `python report_results.py`，将加载 `results/` 下**最新**的 summary CSV，在控制台再次打印汇总，并生成 Markdown 报告到 `results/reports/neural_models_report_YYYYMMDD_HHMMSS.md`（实验概述、数据与方法、结果表格、关键发现）。
- **与符号模型对比**：若需在报告中加入「与符号模型对比」一节，请传入 01 的增强 summary CSV 路径，例如：
  ```bash
  python report_results.py --symbolic ../01_symbolic_models_enhanced/results/enhanced_training/enhanced_models_summary_20260203_150249.csv
  ```
  将额外生成 `results/reports/neural_vs_symbolic_comparison.csv` 与 `comparison_summary.json`，并在 Markdown 报告中增加对比表与小结。
- **指定 summary 或报告输出路径**：`python report_results.py --summary results/neural_models_summary_20260203_161613.csv --out results/reports/my_report.md`
