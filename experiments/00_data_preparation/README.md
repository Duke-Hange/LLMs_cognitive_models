# 00_data_preparation

本阶段说明 **Choices13k 相关数据在本仓库中的位置与用法**，便于复现与审稿材料中的数据来源描述。

## 权威数据路径

| 用途 | 路径 | 说明 |
|------|------|------|
| **神经 / CSV 轨共用的主表** | `experiments/data/data.csv` | 列至少包含 `bRate`, `A`, `B`；`A`/`B` 为字符串形式的 `[[概率, 结果值], ...]` 列表，与 `02_neural_models/data_loader.py` 中 `ast.literal_eval` 一致。 |
| **01 符号 JSON 轨** | `experiments/01_symbolic_models_enhanced/` 下增强标准化产物 | 如 `c13k_enhanced_standardized.json`（具体文件名以训练脚本为准）。 |

## 与 Prompt 的一致性

若使用 `03_LLM` 生成的 jsonl（由 `build_prompt_en.py` 等从 **同一** `data.csv` 构建），建议在仓库根目录运行：

`python experiments/05_program/scripts/validate_prompt_csv_consistency.py --csv experiments/data/data.csv --jsonl <你的 prompts.jsonl>`

以确认从 prompt 文本反解析的 Option A/B 与 CSV 中结构化分布一致（见 `05_program/src/data/prompt_parser.py` 所支持的英文模板）。

## 行顺序与索引

- `row_index` 与 `data.csv` **数据行**（不含表头）从 0 起一一对应；**勿**在未同步重生成 jsonl 的情况下单独重排 CSV。
- 划分随机种子约定见 `02_neural_models/config.py`（`SPLIT_RANDOM_STATE + seed`）。

## 外部文献与原始集

Peterson et al. (2021) 等与 Choices13k 大规模实验相关的引用，请在论文「数据」节写明与公开数据集的对应关系；本目录可后续补充下载脚本或校验和，当前以仓库内 `data.csv` 为执行入口。
