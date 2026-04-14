# en_v2 上 k-shot（In-Context Learning）实施方案

本文档约定：在 **`results/prompts_en_v2.jsonl`** 基础上做 k-shot，并与 **`experiments/02_neural_models`** 的 **train/test 划分**一致，便于与神经/符号结果对照。

---

## 1. 目标

- **输入**：测试题的 en_v2 题干（jsonl 中的 `prompt`）+ 前置 **k 道示范**（仅来自训练集）。
- **输出**：模型给出 **`P_B=...`**（对 bRate 的点估计）。
- **对比**：k = 0（零样本，即直接用现有 `prompt`）vs k ∈ {1, 2, 5, 10, …}，看 test 上 **MSE / Cross-Entropy** 等是否随 k 变化。

---

## 2. 数据与索引对齐

| 对象 | 说明 |
|------|------|
| `prompts_en_v2.jsonl` | 每行 `row_index` 与 `experiments/data/data.csv` **数据行 0…N-1** 一一对应（不含表头）。 |
| 标签 | 人类 **bRate** = jsonl 同行 `bRate`；**不得**写入发给模型的字符串（除示范中的训练标签）。 |

生成 jsonl 后若 **`data.csv` 行顺序或内容变更**，需 **重新生成 jsonl**，否则 `row_index` 与划分会错位。

---

## 3. Train / Test 划分（与 02 一致）

与 `02_neural_models/train.py` 一致：

- `sklearn.model_selection.train_test_split`
- `test_size = 0.2`
- `shuffle = True`
- **`random_state = SPLIT_RANDOM_STATE + seed`**，其中 `SPLIT_RANDOM_STATE = 1017`（见 `02_neural_models/config.py`），**`seed` 与神经实验相同**（如 `42` → `random_state=1059`）。
- 在 **索引数组 `np.arange(N)`** 上划分，得到 `train_idx`、`test_idx`（均为 `row_index` 集合）。

**k-shot 硬性规则**：

- 示范题的 `row_index` **必须 ∈ train_idx**。
- 当前推理题的 `row_index` **必须 ∈ test_idx**。
- **禁止**用测试题或其 `bRate` 做示范（泄漏）。

---

## 4. 单条「发给 API」的字符串怎么拼（推荐结构）

### 4.1 版本命名

建议将「带 k 个示范」的完整字符串记为 **`en_v2_kshot_v1`**（与纯 `en_v2` 区分），便于写论文与缓存。

### 4.2 推荐模板（逻辑结构）

1. **总说明（仅出现一次）**  
   - 说明接下来有 **k 个例题**，格式与最后一题相同；每道例题末尾一行给出 **人类数据中选 Option B 的比例**（`P_B=`）。  
   - 说明最后一题 **只输出一行 `P_B=...`**，不要解释。

2. **例题 i = 1…k**（每道结构一致）  
   - 从 jsonl 取 `row_index = e_i` 的 **`prompt` 全文**（与 0-shot 完全一致）。  
   - **紧接着一行**：`P_B={bRate}`，`bRate` 取该训练样本的 **`bRate`**，小数位 **固定规则**（建议 **4 位小数**，与 `round(x, 4)` 一致，全文档统一）。

3. **分隔**  
   - 用固定分隔行，例如：  
     `---`  
     或  
     `Now answer the following problem. Output only one line P_B=...`

4. **当前测试题**  
   - 使用测试样本的 **`prompt` 全文**（与 k=0 时相同）。  
   - **不要**在测试题后再附真值。

**注意**：k=0 时 **不要**加总说明块，直接等于 jsonl 中的 `prompt`。

### 4.3 备选（省 token）

若 k 较大导致超长：

- 例题可改为 **缩略版**：只保留 `Q:` + `Option A/B` 两行描述 + `P_B=`，**删掉**例题里重复的「Your task is to estimate…」长段（**仅对例题压缩**，测试题仍用完整 `prompt`）。  
- 一旦采用缩略版，须固定为 **`en_v2_kshot_v1_compact`**，**不可与完整版混比**。

---

## 5. 示范样本怎么选（k 个从哪来）

至少做一种主方案 + 可选对照：

| 策略 | 做法 | 复现 |
|------|------|------|
| **固定池（推荐主表）** | `rng = np.random.default_rng(pool_seed)`，从 `train_idx` 无放回抽 **k** 个 `row_index`，**全测试集共用同一组 k 题** | 记录 `pool_seed`、`k`、排序后的 `example_row_indices` |
| **每题重抽（可选）** | 对每个 `test_idx` 单独抽 k 个示范（仍仅从 train） | 方差更大，需报告 `pool_seed` 或多次重复取均值 |

**主表建议**：**固定池**，`pool_seed` 可与 `seed` 相同或记为 `pool_seed = seed * 1000 + k` 避免碰撞。

**进阶（若有特征）**：从 `train_idx` 按 **bRate 分箱**或 **EV 差**分层抽 k，使示范覆盖不同难度（需在文档中说明分层规则）。

---

## 6. 超参数与解码

| 项 | 建议 |
|----|------|
| **k** | 0, 1, 2, 5, 10（**k ≤ |train_idx|**；若 train 约 7864，10 没问题） |
| **Temperature** | 主结果 **0**（或可重复采样） |
| **模型** | 固定 checkpoint / API 版本号 |
| **解析** | 正则提取 `P_B=([0-9.]+)`，失败则重试 1 次或记 `NaN` |

---

## 7. 评估

- **仅在 test_idx 上**计算 `y_true = bRate`，`y_pred` 为解析值（clip 到 `(1e-15, 1-1e-15)` 再算 CE）。  
- 指标与 `experiments/evaluation_metrics.py` 一致：**MSE, R², Correlation, Cross-Entropy**。  
- 报告 **k × 指标** 表或小图（k 为横轴）。

---

## 8. 记录与可复现清单（实验日志最少应含）

- `prompt_version`：`en_v2` / `en_v2_kshot_v1`  
- `split_random_state`（如 1059）、`test_size`（0.2）  
- `k`、`pool_seed`、**k 个 `example_row_indices`**（排序后列表）  
- 模型名、温度、API 日期  
- 输出：`results/llm_kshot_k{k}_seed{seed}.jsonl`（每行含 `row_index`、`prompt_full_hash`、`raw_completion`、`pred_p_b`、`bRate`）

---

## 9. 与 0-shot 的公平性

- **测试集相同**、**划分相同**、**解析相同**。  
- 唯一变化是 **前缀是否含 k 个带标签例题**。  
- 费用：k-shot 每题 token 数上升，全量 9831×test 比例 约 **0.2×9831 ≈ 1966** 题 × (k+1) 量级长度，需做 **成本预估**。

---

## 10. 实施顺序建议

1. 用 `seed=42` 复现 `train_idx` / `test_idx`（与 02 同脚本或单独 `train_test_split` 一行代码验证）。  
2. 实现 **字符串拼接** + **PRINT** 打印 1 条 k=2 的完整 prompt 人工检查。  
3. 小样本（如 test 前 20 条）跑通 API + 解析。  
4. 全 test 跑 k=0,1,2,5,10，汇总 CSV。

---

*与 `03_LLM_实验方案.md` §3.3、§4 一致；划分参数以 `experiments/02_neural_models/config.py` 为准。*
