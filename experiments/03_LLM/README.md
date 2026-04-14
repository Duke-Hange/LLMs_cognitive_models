# 03_LLM 本地模型实验

本目录用于完成 Choices13k 的本地 LLM 端到端实验，目标是预测每个问题的人群 `bRate`（即 `P(Option B)`），并与 `01_symbolic_models_enhanced`、`02_neural_models` 在同一口径下比较。

**与 02 神经默认标签的关系**：`02_neural_models` 在 CSV 轨上训练目标为 **`y = 1 − bRate`**。本目录主结果仍为 **`pred_bRate`**；若要与神经 summary **同一标量空间**对比或接入 `05_program`（`target_mode: one_minus_bRate`），应对预测做 **1 − pred_bRate** 或在表格中双列标注。全局说明见仓库 `experiments/README.md` 中的「标签空间」表。

**勿混用任务**：`build_prompt_en2.py`（`en2_preference`）为**个人** Option A/B 选择表述，与主线的**人群比例**回归构造不同；主表请用 `en_v2` / `en_v3` 等聚合指令版本。

## 实验合同（固定协议）

- **主任务**：回归 `bRate`，预测值记为 `pred_bRate`，范围 `[0,1]`。
- **主 Prompt 版本**：`en_v2`（由 `build_prompt_en.py --version v2` 生成）。
- **主输出格式**：`P_B=0.6273`（严格单行，4 位小数推荐）。
- **解析失败策略**：重试 1 次；仍失败写 `NaN`，禁止静默填充 `0.5`。
- **主划分**：`train_test`，与 `02_neural_models` 对齐：
  - `TEST_SIZE = 0.2`
  - `random_state = 1017 + seed`
  - 在 `np.arange(N)` 上划分索引
- **k-shot 硬规则**：
  - 示例仅来自 `train_idx`
  - 推理样本仅来自 `test_idx`
  - 默认固定示例池策略（全测试集共享同一组示例）

## 目录约定

- `build_prompt_en.py` / `build_prompt_en2.py`: Prompt 构建
- `configs/default.yaml`: 默认运行参数
- `src/llm_client.py`: Ollama 调用与缓存
- `src/parse_output.py`: 输出解析与重试提示
- `src/splits.py`: 与 02 对齐的划分
- `src/kshot.py`: k-shot 示例池与拼接
- `src/run_eval.py`: 主评测入口（推理、指标、导出、稳健性）
- `src/finetune_qlora.py`: QLoRA 微调入口
- `results/`: Prompt、预测、summary、稳健性输出

## 结果文件约定

### 逐样本预测

`results/llm_predictions_<split_type>_<model>_<timestamp>.csv` 最低包含：

- `row_index`
- `split_type`
- `model`
- `prompt_version`
- `k`
- `pool_seed`
- `example_row_indices`
- `prompt_hash`
- `prompt_full_hash`
- `bRate`
- `pred_bRate`
- `raw_completion`
- `parse_ok`
- `retry_count`
- `is_swap`

### 模型汇总

`results/llm_models_summary_<timestamp>.csv` 最低包含：

- `model`
- `split_type`
- `prompt_version`
- `k`
- `seed`
- `test_mse`
- `test_r2`
- `test_correlation`
- `test_cross_entropy`
- `test_rmse`
- `test_mae`
- `n_test`
- `n_valid`
- `parse_success_rate`
- `swap_pass_rate`（若未运行稳健性则为 `NaN`）

## 快速开始

1. 构建 Prompt（若尚未生成）：

```bash
python build_prompt_en.py --version v2 --out results/prompts_en_v2.jsonl
```

2. 零样本（k=0）评测：

```bash
python src/run_eval.py --model qwen2.5:7b-instruct --k 0 --seed 42
```

3. k-shot 主实验：

```bash
python src/run_eval.py --model qwen2.5:7b-instruct --k 0,1,2,5,10 --seed 42 --pool-seed 42010
```

4. 启用 A/B 对调稳健性：

```bash
python src/run_eval.py --model llama3.1 --k 0,2,5 --seed 42 --run-swap-robustness
```

5. QLoRA 微调（示例）：

```bash
python src/finetune_qlora.py --base-model Qwen/Qwen2.5-7B-Instruct --seed 42
```

## 复现最小清单

- 固定 `seed`、`split_random_state`、`pool_seed`
- 固定模型 tag（如 `llama3.1`）
- 固定 `prompt_version` 与 `k` 配置
- 记录 `prompt_hash` / `prompt_full_hash`
- 保存原始 completion（用于审计与复查）
