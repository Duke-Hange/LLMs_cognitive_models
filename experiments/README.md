# experiments 实验目录总说明

本目录是 Choices13k 项目的实验实现区。当前文档策略是：`experiments/README.md` 作为总入口，00-06 每个阶段各保留一个 `README.md`，旧方案文档与运行产物报告不再作为手写维护文档。

## 1. 总体结构

| 阶段 | 目录 | 作用 | 当前状态 |
| --- | --- | --- | --- |
| 00 | `00_data_preparation` | 数据准备与统一 CSV 入口 | 已可用 |
| 01 | `01_symbolic_models_enhanced` | 增强符号模型，EV/EU/PT 系列 | 已可单跑，05 会复用核心逻辑 |
| 02 | `02_neural_models` | 神经网络模型，value-based/context-dependent 系列 | 已可单跑，05 会复用核心逻辑 |
| 03 | `03_LLM` | Prompt 构建与 Ollama LLM ICL 评估 | 已支持 dry-run；真实结果依赖本地模型 |
| 04 | `04_program` | 一阶段比较：symbolic vs neural | 已完成主流程，不含 LLM |
| 05 | `05_program` | 主比较：baseline/symbolic/neural/LLM，可自动触发 06 | 当前主入口 |
| 06 | `06_qlora_text` | QLoRA 文本回归扩展 | 已支持 smoke 与真实训练入口 |

`data` 与 `shared` 是实验共享资源目录，不作为独立研究阶段。

## 2. 数据与目标口径

主数据入口是 `experiments/data/data.csv`。其中每一行对应一个 Choices13k 赌博选择任务，包含 A/B 两个选项的结果、概率与群体选择比例 `bRate`。

当前 04/05/06 的统一目标口径是：

```text
target_mode = one_minus_bRate
y = 1 - bRate
```

也就是说，统一比较中预测的是选择 A 的群体比例。03 阶段的 LLM prompt 通常要求模型输出 `P_B=0.####`，05 会在统一评估时把它转换到同一目标空间。这个设计是为了让符号、神经、LLM、QLoRA 在同一 `y_target` 下比较。

## 3. 主推荐入口

日常开发和最终比较优先使用 05：

```powershell
python experiments/05_program/src/runners/run_all.py --smoke --no-staging
```

需要顺路验证 06 时：

```powershell
python experiments/05_program/src/runners/run_all.py --smoke --no-staging --run-qlora --qlora-smoke --qlora-sample-sizes 8 --qlora-seeds 42
```

正式运行时去掉 `--smoke`，并根据机器情况决定是否保留 `--run-qlora`。当前开发机器没有部署本地 LLM/QLoRA 模型，因此 smoke 的意义是验证数据、划分、指标、绘图、文件输出链路，而不是给出科学结论。

## 4. 04 与 05 的关系

04 是一阶段比较项目，只比较符号模型与神经模型。它适合回答“在不引入 LLM 的情况下，传统符号模型和神经模型的学习曲线如何不同”。

05 是主比较项目，在 04 的基础上加入 LLM，并使用更完整的报告、图表和可选 QLoRA 合并逻辑。05 的可视化风格与 04 保持一致：学习曲线、模型族比较、全模型图、置信区间/误差带等逻辑相同；差异是 05 增加了 LLM family，开启 QLoRA 后还会增加 QLoRA family。

## 5. 输出约定

04 输出默认位于：

```text
experiments/04_program/outputs/run_*/
```

05 输出默认位于：

```text
experiments/05_program/outputs/run_*/
```

常见输出包括：

- `split_manifest.json`：统一训练/测试划分与 `(N, seed)` 样本子集。
- `configs_snapshot/`：本次运行使用的配置快照。
- `metrics/curve_aggregated.csv`：主学习曲线聚合表。
- `metrics/predictions_long.csv`：长表预测结果。
- `metrics/curve_qlora_aggregated.csv`：开启 06 后的 QLoRA 曲线。
- `metrics/curve_aggregated_with_qlora.csv`：05 主曲线与 QLoRA 合并后的曲线。
- `figures/`：论文图和诊断图。

如果 05 使用 `--run-qlora`，06 的输出默认写入当前 05 run 目录下的 `qlora_auto/`，这样可以保证 06 与该次 05 的 manifest 强绑定。

## 6. 可视化原则

当前图表使用论文友好的色盲安全配色。模型族颜色大致为：

- baseline：灰色
- symbolic：蓝色
- neural：橙色
- LLM：绿色
- QLoRA：紫红色

同一模型族内的不同模型会使用同族调色板区分，并通过线型、点型和图例共同表达。05 的图表逻辑位于 `05_program/src/analysis/`，04 会复用其中一部分绘图组件以保持风格一致。

## 7. 文档维护规则

当前手写 Markdown 文档只保留以下层级：

- 根目录：`Choices13k 项目方案.md`
- 实验总入口：`experiments/README.md`
- 阶段入口：`experiments/00_data_preparation/README.md` 到 `experiments/06_qlora_text/README.md`

模型说明、数据集原始说明和共享工具 README 如果位于 `docs`、`shared`、`数据集` 等资料目录中，可以作为参考资料保留；它们不属于本次 00-06 阶段文档压缩范围。
