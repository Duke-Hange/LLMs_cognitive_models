# 05 方案跑完后的图表总览（基于当前配置）

本文基于当前 `experiments/05_program` 代码与配置梳理：当你完整执行 `run_all.py` 后，会生成哪些图、各图横纵轴是什么、图中线条各代表什么，以及可以从图中读出什么信息。

---

## 1) 当前配置下会参与出图的模型与指标

- 目标指标（primary metrics）：`cross_entropy`、`mae`
- 采样规模（横轴候选 N）：`[0, 4, 16, 64, 256, 1024]`
- seeds：`[41, 42, 43]`
- 模型家族与模型：
  - symbolic：`EV`、`EU`、`PT3`、`PT5`
  - neural：`value-based`、`context-dependent(s)`
  - llm：`qwen2.5:7b-instruct`
  - qlora：仅在提供到 `test_metrics.json`（或 `--run-qlora` 成功）时出现

说明：
- 对于 LLM，当前协议是 **`k_shot = N`**。
- 当 `N=0` 时，LLM 会以 **水平虚线（ICL0）** 展示，不接入 log2 折线段。

---

## 2) 跑完后会生成哪些图（文件名）

输出目录：`run_时间戳/figures/`

### A. 经典学习曲线与主图

- `learning_curve_cross_entropy.png`
- `learning_curve_mae.png`
- `main_plot_cross_entropy.png`
- `main_plot_mae.png`
- `figure_all_models_cross_entropy.png`（`main_plot_cross_entropy.png` 的别名拷贝）
- `figure_all_models_mae.png`（`main_plot_mae.png` 的别名拷贝）

### B. 家族内比较图

- `family_comparison_symbolic_cross_entropy.png`
- `family_comparison_symbolic_mae.png`
- `family_comparison_neural_cross_entropy.png`
- `family_comparison_neural_mae.png`
- `family_comparison_llm_cross_entropy.png`
- `family_comparison_llm_mae.png`

### C. 论文叙事 Step1-6 图（charter bundle）

- `plot_01_baselineVsN_cross_entropy.png`
- `plot_01_baselineVsN_mae.png`
- `plot_02_symbolic_cross_entropy.png`
- `plot_02_symbolic_mae.png`
- `plot_03_neural_cross_entropy.png`
- `plot_03_neural_mae.png`
- `plot_04_llm_icl_cross_entropy.png`
- `plot_04_llm_icl_mae.png`
- `plot_05_qlora_cross_entropy.png`（可选，可能被跳过）
- `plot_05_qlora_mae.png`（可选，可能被跳过）
- `main_plot_all_families_cross_entropy.png`

### D. 与图相关但不是图片的文件

- `baselines_legend.txt`（记录基线数值）
- `reports/charter_plots_skipped.json`（记录哪些 charter 图被跳过及原因）
- `reports/family_comparison_skipped.json`（记录 family 图是否有跳过）

---

## 3) 每类图的自变量、因变量、线条含义、可读信息

## 3.1 经典学习曲线：`learning_curve_*.png`

- 自变量（X）：训练样本量 `N`（`N>0` 段为 log2 轴）
- 因变量（Y）：
  - `learning_curve_cross_entropy.png` -> `cross_entropy`
  - `learning_curve_mae.png` -> `mae`
- 线条代表：
  - 彩色折线：每个模型（symbolic / neural / llm / 可选 qlora）的均值曲线（跨 seed 聚合）
  - 阴影带：该模型在该 N 处的 bootstrap 置信区间（`ci_low` ~ `ci_high`）
  - LLM 的 `N=0`：`模型名 · ICL0 (N=0)` 水平虚线
  - 基线水平线：
    - `Train-mean constant`
    - `Constant 0.5`
    - `Neural full-train level`
    - `Neural train-on-0.5 level`
- 能得到的信息：
  - 数据效率：随着 N 增加，误差是否单调下降、下降快慢如何
  - 稳定性：置信区间宽窄（越窄说明 seed 间更稳定）
  - 与基线差距：是否“显著优于常数预测/接近上限参考线”
  - LLM 零样本与少样本增益：ICL0 线与 `N>0` 曲线的差值

## 3.2 全家族主图：`main_plot_*.png` 与 `figure_all_models_*.png`

- 自变量（X）：训练样本量 `N`（log2）
- 因变量（Y）：对应主指标（CE 或 MAE）
- 线条代表：
  - 每条线是一个具体模型，颜色按家族（symbolic/neural/llm）主色，家族内再用线型/marker区分
  - LLM `N=0` 仍是单独水平虚线
  - `Train-mean constant`、`Neural full-train level`、`Neural train-on-0.5 level` 作为参考水平线
- 能得到的信息：
  - 跨家族直接对比（谁在同 N 下最好）
  - 误差-样本量的整体排序是否在 N 变大后发生“反转”
  - 哪类方法更早达到“可用区间”（例如 MAE 某阈值）

## 3.3 家族内比较图：`family_comparison_<family>_<metric>.png`

- 自变量（X）：训练样本量 `N`（log2）
- 因变量（Y）：`cross_entropy` 或 `mae`
- 线条代表：
  - 在单一家族内，不同模型各一条曲线
  - 同样可出现基线水平线（用于参照）
- 能得到的信息：
  - 家族内部架构优劣与样本效率差异
  - 哪个模型在小样本区间更强、哪个在大样本区间更优
  - 是否存在“初期好、后期被反超”的模型

## 3.4 Step1-6 图（charter）

### Step1：`plot_01_baselineVsN_*.png`
- X：`N`
- Y：目标指标（CE 或 MAE）
- 线条：仅基线水平线（`Train-mean constant` + 可选 `Constant 0.5`）
- 信息：提供“最低参照门槛”，后续模型曲线需明显低于该线才有意义。

### Step2：`plot_02_symbolic_*.png`
- X：`N`
- Y：CE/MAE
- 线条：symbolic 家族各模型（EV/EU/PT3/PT5）
- 信息：符号模型随数据规模扩张的收益、内部最优模型是谁。

### Step3：`plot_03_neural_*.png`
- X：`N`
- Y：CE/MAE
- 线条：neural 家族（value-based、context-dependent(s)）
- 信息：神经模型在不同 N 的收益曲线、是否比 symbolic 更快逼近低误差。

### Step4：`plot_04_llm_icl_*.png`
- X：`N`（且 `k_shot=N`）
- Y：CE/MAE
- 线条：LLM 曲线 + ICL0 水平线
- 信息：LLM 的零样本基线与 few-shot 增益幅度；增益是否随 N 递减。

### Step5：`plot_05_qlora_*.png`（可选）
- X：`N`
- Y：CE/MAE
- 线条：QLoRA 模型曲线（来自 06 的 `test_metrics.json` 聚合）
- 信息：微调路线相对 ICL/传统模型在小样本下是否有优势。
- 可能跳过的原因：没有 qlora 输入数据、glob 未配置、文件缺失。

### Step6：`main_plot_all_families_cross_entropy.png`
- X：`N`（log2）
- Y：`cross_entropy`（默认主图用 CE）
- 线条：symbolic + neural + llm + 可选 qlora，外加基线线
- 信息：给出最终“全方法总对比”叙事图，可直接用于论文主结果页面。

---

## 4) 如何通过这些图回答核心研究问题

- **样本效率问题**：看同一目标误差下哪条曲线最早到达（最小 N）。
- **方法稳定性问题**：看置信区间宽度及是否跨越关键比较对象。
- **零样本与小样本收益**：看 LLM 的 ICL0 线到小 N 点的下降幅度。
- **跨家族优劣**：看主图在不同 N 的排名是否一致，是否存在交叉点。
- **是否值得引入 QLoRA**：看 Step5 与 Step6 中 qlora 曲线相对其他方法的位置。

---

## 5) 读图注意事项（避免误解）

- CE / MAE 都是 **越低越好**。
- `N=0` 是 LLM 特殊点（ICL0），表现为水平虚线，不应当按普通折线点解释。
- 未出现的图通常不是“程序失败”，而是数据为空后被安全跳过；以 `reports/*_skipped.json` 为准。
- 若 `charter_plots.qlora.test_metrics_glob` 为空且未启用 `--run-qlora`，Step5 大概率不会有图。

---

## 6) 一句话总结

完整跑完 05 后，你会得到一套“基线 -> 各家族 -> 全家族汇总”的分层图谱；核心自变量是训练样本量 `N`，核心因变量是 `cross_entropy` 与 `mae`，每条线对应“某模型在不同 N 下的平均表现（含置信区间）”，可直接用于判断样本效率、稳定性、跨方法优劣及是否需要引入 QLoRA。
