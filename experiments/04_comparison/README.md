# 04_comparison — 跨模型比较

## 目的

集中进行**不同模型**（符号模型增强、神经网络、LLM）的跨模型比较：加载各实验的 summary CSV，按划分对齐，生成统一比较表、按划分小结与带**可比性说明**与**比较设计**的 Markdown 报告。

## 输入

- **01 符号模型增强**：`01_symbolic_models_enhanced/results/enhanced_training/enhanced_models_summary_*.csv`，需含列：`split_type`（或 `split`）、`model`、`test_mse`、`test_r2`、`test_correlation`。
- **02 神经网络**：`02_neural_models/results/neural_models_summary_*.csv`，需含列：`model_type`、`split_type`、`test_mse`、`test_r2`、`test_correlation`。
- **03 LLM（可选）**：`03_LLM/results/llm_models_summary_*.csv`，需含列：`model`、`split_type`、`test_mse`、`test_r2`、`test_correlation`（多种子可用 `*_mean/*_std`）。

默认使用 01/02 各自目录下**最新**的 summary CSV（按文件修改时间），并自动尝试加载 03 的最新 LLM summary（若存在）。

## 输出

运行后在本目录 `output/` 下生成（时间戳为运行时刻）：

- **comparison_table_YYYYMMDD_HHMMSS.csv**：长表，列 `split_type`, `family`（symbolic/neural/llm）, `model`, `test_mse`, `test_r2`, `test_correlation`。
- **comparison_summary_YYYYMMDD_HHMMSS.json**：按 split_type 的汇总（符号/神经最佳 R²、平均 R²、最佳模型名）及整体平均。
- **comparison_report_YYYYMMDD_HHMMSS.md**：含比较目的、比较设计、可比性说明、结果表、按划分小结、局限与复现信息。

## 可比性说明

- **固定**：数据源（01 标准化数据或与 `data/data.csv` 对齐的 CSV 轨）、划分（`problem` / `parameter_amb` / `parameter_ev_extreme` 或 `train_test`）、评估指标（test MSE, R², correlation）、评估方式（仅测试集）。
- **标签空间**：01 符号与 03 LLM 主口径多为 **bRate**；02 神经在 CSV 轨默认监督 **y = 1 − bRate**。MSE 在 `p` 与 `1−p` 变换下数值一致；报告 R² / correlation 时请确认各 summary 是否在**同一标量空间**（详见 `run_comparison.py` 生成报告中的「标签空间」一节）。
- **变化**：模型族（符号 vs 神经 vs LLM）、输入表示（符号用 50+ 维增强特征，神经用完整分布编码，LLM 用自然语言）。
- 结论限于「同一划分、同一指标公式；表示与目标参数化可能不同」，差异为**模型族、表示与目标定义的联合效应**。

## 如何运行

在本目录或项目根下：

```bash
# 使用默认最新 summary
python run_comparison.py

# 指定符号/神经/LLM summary 与输出目录
python run_comparison.py --symbolic ../01_symbolic_models_enhanced/results/enhanced_training/enhanced_models_summary_20260203_150249.csv --neural ../02_neural_models/results/neural_models_summary_20260203_161613.csv --llm ../03_LLM/results/llm_models_summary_20260324_210000.csv --out-dir output
```

## 后续扩展

- 加入更多模型时：在 `run_comparison.py` 中增加对应 summary 的加载逻辑（列名对齐为 `split_type`, `model`, test_mse/test_r2/test_correlation），将新模型并入长表并增加 `family`；在 `build_summary_by_split` 与报告中增加对新族的按划分小结即可。
