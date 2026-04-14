# 05_program

配置驱动的极小样本学习曲线实验框架，用于比较三类模型：

- Symbolic: `EV`, `EU`, `PT3`, `PT5`
- Neural: `value-based`, `context-dependent(s)`
- LLM (ICL-only): `llama3.1`, `qwen2.5:7b-instruct`

## 实验口径

- 统一监督目标：`y = 1 - bRate`
- 固定测试集：所有模型共享同一 `test_holdout`
- 学习曲线横轴：训练样本量 `N`
- 主图：每个主指标一张图（`cross_entropy`、`mae`）
- 每个点统计：多 seed 的均值 + 95% CI

## 目录

- `configs/`: 实验配置（协议与模型清单）
- `src/data/`: 数据读取、prompt 分布解析、split 与抽样
- `src/runners/`: Symbolic / Neural / LLM 三类运行器
- `src/analysis/`: 指标聚合、汇总表、学习曲线绘图
- `outputs/`: 运行产物（预测、指标、图表、报告）

## 运行

1. 先修改 `configs/protocol.yaml`（尤其 `sample_sizes`、`seeds`、`llm.dry_run`）。
2. **机械盘 + LLM 大量写盘**：可设置 `paths.staging_output_root` 为 SSD 上的绝对路径（如 `C:/数据中转站/choices13k_05_runs`）。运行期 `run_*` 只写该目录，**整次成功结束后**再复制到 `output_root` 并删除 SSD 上的本次目录；失败或中断则结果仍留在 staging，需自行处理。另可将 `llm.cache_dir` 指到同一块 SSD，减轻缓存随机写。
3. 执行：

```bash
python experiments/05_program/src/runners/run_all.py
```

## 说明

- LLM 默认支持 `dry_run`（不调用 Ollama）以便快速打通流程。
- 若要真实调用，请在 `protocol.yaml` 将 `llm.dry_run` 设为 `false`，并确保本地 Ollama 服务可用。
- **Prompt 与 CSV**：在仓库根目录运行  
  `python experiments/05_program/scripts/validate_prompt_csv_consistency.py --csv experiments/data/data.csv --jsonl <protocol 中的 prompts_path>`  
  校验从 prompt 解析的赌局是否与 `data.csv` 一致（主实验建议使用 `prompts_en_v2.jsonl`，与聚合 `P_B` 任务对齐）。
- **解析失败**：`evaluation.llm_parse_fail_fill: null` 时，主指标仅在解析成功且预测为有限值的样本上计算（不静默填 0.5）；若需与旧实验对齐可改为数值（如 `0.5`）。
