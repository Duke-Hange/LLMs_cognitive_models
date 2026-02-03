# 实验运行顺序说明

按以下顺序运行可复现「数据 → 符号模型 → 神经模型 → 跨模型比较」全流程。每一步的输入依赖已标明。

---

## 0. 前置条件

- **原始数据**：项目根目录下 `数据集/choices13k-main/` 中需存在：
  - `c13k_selections.csv`
  - `c13k_problems.json`
- **环境**：`experiments/env_verification.py` 可检查依赖；神经实验需安装 `02_neural_models/requirements.txt`。

---

## 1. 生成标准化数据（02 依赖此文件）

**目的**：在 `01_symbolic_models_enhanced` 目录下生成 `c13k_enhanced_standardized.json`，供 **02 神经模型** 读取。

**操作**：在项目根或 `experiments` 下执行（确保能导入 `01` 的模块）：

```bash
cd experiments/01_symbolic_models_enhanced
python enhanced_data_standardization.py
```

- 会从 `数据集/choices13k-main/` 读入原始数据，标准化后写入 `01_symbolic_models_enhanced/c13k_enhanced_standardized.json`。
- **只需运行一次**（或数据/标准化逻辑变更后重跑）。若该 JSON 已存在且未改数据，可跳过。

---

## 2. 符号模型训练（01）

**目的**：在三种划分上训练增强符号模型（ev / eu / pt3 / pt5），得到 summary CSV，供 04 比较使用。

**操作**：

```bash
cd experiments/01_symbolic_models_enhanced
python train_enhanced_models.py
```

- **多种子**（可选）：  
  `python train_enhanced_models.py --seeds 42 43 44 45 46`
- **输出**：
  - `results/enhanced_training/enhanced_models_summary_*.csv`（04 会取「最新」）
  - `results/enhanced_training/enhanced_models_results_*.json`

**依赖**：仅依赖原始数据（`数据集/choices13k-main/`），不依赖 00。

---

## 3. 神经模型训练（02）

**目的**：在三种划分上训练 value_based / context_dependent，得到 summary CSV，供 04 比较使用。

**操作**：

```bash
cd experiments/02_neural_models
python train.py
```

- **多种子**（可选）：  
  `python train.py --seeds 42 43 44 45 46`
- **输出**：
  - `results/neural_models_summary_*.csv`（04 会取「最新」）
  - `results/neural_models_results_*_seed*.json`

**依赖**：需要 `01_symbolic_models_enhanced/c13k_enhanced_standardized.json` 存在（由**步骤 1** 生成）。

---

## 4. 跨模型比较（04）

**目的**：对齐 01 与 02 的 summary，生成比较表、汇总 JSON 和 Markdown 报告。

**操作**：

```bash
cd experiments/04_comparison
python run_comparison.py
```

- 默认使用 **01 与 02 各自 results 下最新的 summary CSV**。
- 可选指定文件：  
  `python run_comparison.py --symbolic path/to/enhanced_models_summary_xxx.csv --neural path/to/neural_models_summary_xxx.csv`
- **输出**：  
  `output/comparison_table_*.csv`、`comparison_summary_*.json`、`comparison_report_*.md`、**`output/comparison_r2_*.png`**（跨模型比较图，报告中会引用）

**依赖**：需已运行**步骤 2** 和**步骤 3**，且至少各有一份 summary CSV。

---

## 可视化说明

- **主流程下最终结果图**：`04_comparison/output/comparison_r2_{timestamp}.png`。运行 `run_comparison.py` 后自动生成，报告 Markdown 中会嵌入该图（符号 vs 神经各模型在三种划分上的 Test R²、Test MSE）。
- **训练阶段默认图**：01 与 02 默认只生成**均值训练曲线**（`mean_curve_{timestamp}_{划分}_{模型}.png`），用于检查收敛；不再为每个 seed 生成单次曲线 PNG。若需要每种子曲线，可加 `--curves all`；若不需要任何曲线图，可加 `--curves none`（仍会保存曲线 JSON）。
  - 01：`python train_enhanced_models.py [--curves all|mean_only|none]`，默认 `mean_only`。
  - 02：`python train.py [--curves all|mean_only|none]`，默认 `mean_only`。
- **可选图**：数据量–性能曲线（`run_learning_curve.py`）、01 分析脚本（`analysis/`）、诊断/测试图等，均不在主复现链上，按需运行。详见 `experiments/VISUALIZATION.md`。

---

## 推荐顺序小结

| 顺序 | 步骤 | 命令（在对应目录下） | 必须/可选 |
|------|------|----------------------|------------|
| 0 | 准备原始数据 | 将 c13k 数据放到 `数据集/choices13k-main/` | 必须 |
| 1 | 生成标准化 JSON | `python enhanced_data_standardization.py`（在 01 目录） | 02 依赖，建议先跑一次 |
| 2 | 符号模型训练 | `python train_enhanced_models.py` [或 `--seeds 42 43 ...`] | 必须（04 依赖） |
| 3 | 神经模型训练 | `python train.py` [或 `--seeds 42 43 ...`] | 必须（04 依赖） |
| 4 | 跨模型比较 | `python run_comparison.py` | 必须 |

**最小复现**：先执行步骤 1，再依次执行 2 → 3 → 4。

---

## 关于 00_data_preparation

- `00_data_preparation` 使用自己的脚本和输出路径（如 `outputs/problem_split/`、`parameter_amb_split/` 等），其产出的 `c13k_enhanced_standardized.json` 与 01 的**不是同一路径**。
- **01 / 02 / 04 不依赖 00**：01 直接从 `数据集/choices13k-main/` 读原始数据；02 只读 01 目录下的 `c13k_enhanced_standardized.json`。
- 若你做 00 的探索或其它分析，可单独跑 00；跑 01→02→04 时无需先跑 00。
- **Parameter-Amb 划分验证**：验证脚本位于 `experiments/00_data_preparation/scripts/validate_parameter_amb_split.py`，输出在 `experiments/00_data_preparation/outputs/parameter_amb_split_validation_report.json`。
