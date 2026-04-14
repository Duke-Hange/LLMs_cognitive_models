# experiments：CSV 数据 + 标签约定 + 单一 train_test_split

> **路径说明**：本目录即仓库中的 `experiments/`。历史文档中的 `experiments3/` 与本目录**同义**，命令与相对路径请以当前仓库为准（例如 `experiments/data/data.csv`）。

与 `experiments2` 结构对齐，核心差异：

- **02 神经模型数据源切换为 CSV**：读取 `experiments/data/data.csv`（列：`bRate,A,B`），其中 `A/B` 为字符串化分布列表。
- **划分方式简化**：仅 `train_test_split` + learning curve，不再使用 `problem/parameter_amb/parameter_ev_extreme`（除非在 01 的 JSON 轨上单独启用）。

## 标签空间（写作与比较前必读）

| 模块 | 训练/评估用的标量 | 说明 |
|------|-------------------|------|
| **01 符号** | **bRate**（选 B 的比例） | 拟合与预测均在 bRate 空间。 |
| **02 神经** | **y = 1 − bRate** | 网络 Sigmoid 输出与损失均对该标量；与「认知语义上的 P(B)」对应时需看 `TARGET_MODE`。 |
| **03 LLM** | Prompt 要求 **P_B = bRate**；与 02 同表比较时常转换为 **1 − pred_bRate** | 详见 `03_LLM/README.md`。 |
| **05_program** | 与 `protocol.yaml` 中 `target_mode` 一致（默认 `one_minus_bRate`） | 符号侧将 `pred_bRate` 转为 `1−pred` 再算指标；LLM 侧同理。 |

**MSE** 在 `p` 与 `1−p` 下等价；**Pearson 相关**若一方在 bRate、另一方在 `1−bRate` 空间且未统一变换，符号可能相反，比较前需对齐定义。

## 目录

- `00_data_preparation/` — 数据出处与准备说明（见该目录 README）
- `01_symbolic_models_enhanced/` — 符号模型；默认仍可用 c13k JSON + 三种划分；**`--data-source csv` 时与 02 共用 `data/data.csv`，划分 `train_test`（`random_state=1017+seed`）**
- `02_neural_models/` — 神经模型（CSV 编码，`1-bRate`，单一 train/test 划分）
- `03_LLM/` — 大模型预测 bRate；**勿**与 `build_prompt_en2.py` 的「个人 A/B 选择」任务混为主实验
- `04_comparison/` — 跨模型比较
- `05_program/` — 统一学习曲线与联合协议（含 prompt–CSV 一致性校验脚本）
- `run_full_pipeline.py` — 01→02→04 一键流程
- `run_learning_curves_all.py` — **一键**：依次跑 01+02 数据量曲线，复制到 `results/learning_curves_all/`，并生成 04 **联合 CE 图**（默认 `--symbolic-data csv` 与 02 对齐）
- `setup_directories.py` — 创建缺失目录

## 数据

- 02 读取 `experiments/data/data.csv`（建议保留本地副本，避免跨目录依赖）。
- 01 若要与 02 **同一批样本、同一 train/test 切分**：`python train_enhanced_models.py --data-source csv`（或一键流程 `run_full_pipeline.py --symbolic-data csv`）。
- **Prompt 与 CSV**：`05_program` 可从自然语言 prompt 反解析分布；应用前建议在仓库根目录运行  
  `python experiments/05_program/scripts/validate_prompt_csv_consistency.py --csv experiments/data/data.csv --jsonl <你的 prompts jsonl>`  
  确认与 `data.csv` 一致。

## 运行

### 一键：01 + 02 学习曲线 + 联合可视化（推荐）

在仓库根目录下：

```bash
python experiments/run_learning_curves_all.py --seeds 42
```

默认 **`--symbolic-data csv`**：01 与 02 均基于 `data/data.csv`，划分均为 `train_test`，便于在同一张 **Fig. S1 风格 Test Cross-Entropy** 图中对比符号模型（EV/EU/PT3/PT5）与神经模型（三条架构）。产出位于 `results/learning_curves_all/<时间戳>/symbolic|neural/`，联合图在 `04_comparison/output/fig_s1_style_ce_*.png`。

- 需要 c13k 上三种划分时：`python experiments/run_learning_curves_all.py --symbolic-data json`（联合图会按 `split_type` 分子图；神经侧仅 `train_test` 一条条件）。
- 跳过联合绘图：`--no-plot-fig-s1`。
- 联合 **MSE** 图（在已有 `run-dir` 后）：`cd experiments/04_comparison && python plot_fig_s1_style_ce.py --metric mse --run-dir ../results/learning_curves_all/<时间戳>`。

说明：`run_full_pipeline.py` 是「单次训练 + 04 表格比较」，**不包含**数据量学习曲线；学习曲线与联合曲线请用本脚本。

### 仅 02 神经（CSV + 1-bRate）

```bash
cd experiments/02_neural_models

# 学习曲线
python run_learning_curve.py

# 快速测试（更短）
python run_learning_curve.py --n-fractions 5 --seeds 42
```
