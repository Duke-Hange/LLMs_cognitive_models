# 00_data_preparation

本阶段负责数据准备、格式统一与基础校验。它不是最终比较入口，而是为 01-06 提供可信的数据来源说明。

## 1. 阶段目标

Choices13k 原始任务包含两个赌博选项 A/B、每个选项的概率-结果分布，以及群体选择比例 `bRate`。00 阶段的目标是把这些信息整理成后续模型都能读取的稳定格式，并记录行号、目标变量和 prompt 之间的对应关系。

当前主流程实际使用的权威表是：

```text
experiments/data/data.csv
```

这个 CSV 是 02、03、04、05、06 的共同数据入口。01 的增强符号模型还保留了自己的标准化 JSON 产物，例如：

```text
experiments/01_symbolic_models_enhanced/c13k_enhanced_standardized.json
```

## 2. 关键文件

- `scripts/prepare_data.py`：历史数据标准化脚本，面向早期符号/神经实验产物。
- `scripts/prepare_enhanced_data.py`：增强数据标准化脚本，服务于 01 的增强符号模型。
- `scripts/exploratory_analysis.py`：探索性统计分析脚本。
- `scripts/validate_parameter_amb_split.py`：用于检查参数化划分策略的辅助脚本。
- `experiments/data/data.csv`：当前主比较协议的结构化数据入口。

## 3. 数据字段约定

主 CSV 至少需要包含：

- `bRate`：选择 B 的群体比例。
- `A`、`B`：两个选项的概率-结果分布，通常是字符串形式的列表结构。
- 可选的任务元信息列：用于追踪原始问题、条件或分组。

04/05/06 的主协议使用：

```text
y = 1 - bRate
```

因此统一比较中的 `y_target` 表示选择 A 的群体比例。03 阶段 prompt 让 LLM 输出 `P_B`，05 会在评估时转换到统一目标空间。

## 4. 行号与 prompt 对齐

`row_index` 是跨阶段对齐的关键。它表示 `data.csv` 中的数据行号，通常从 0 开始，不包含表头。生成 prompt JSONL 后，05 会依靠 `row_index` 把文本 prompt、结构化 A/B 分布与标签重新对齐。

如果重新生成 `data.csv`，必须同步重新生成：

```text
experiments/03_LLM/results/prompts_en_v2.jsonl
```

并运行一致性检查：

```powershell
python experiments/05_program/scripts/validate_prompt_csv_consistency.py --csv experiments/data/data.csv --jsonl experiments/03_LLM/results/prompts_en_v2.jsonl
```

## 5. 当前风险

00 中部分历史脚本来自早期数据处理流程，代码注释在当前 Windows 终端中可能出现编码显示异常。这不影响当前主实验直接读取 `experiments/data/data.csv`，但如果将来要重新从原始 Choices13k 文件构建 CSV，建议先单独清理这些脚本的编码与路径假设。

当前科学结论应以 04/05 输出中保存的 `split_manifest.json`、`configs_snapshot/` 和主 CSV 哈希/路径记录为准。
