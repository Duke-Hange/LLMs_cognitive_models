# Choices13k · 层级 3：大语言模型（03_LLM）实验方案

本文档与《Choices13k 项目方案》第 5.3 节及 `experiments` 现有管道对齐，规定 **03_LLM** 的目标、数据接口、方法学约束、评估指标与交付物，便于后续实现并与 `04_comparison` 统一比较。

---

## 1. 定位与科学问题

### 1.1 在项目中的角色

| 层级 | 代表 | 先验 |
|------|------|------|
| 01 符号模型 | EU / PT 等 | 显式认知理论 |
| 02 神经网络 | Value-Based / Context-Dependent | 无语言先验的函数逼近 |
| **03 LLM** | API 或微调后的生成模型 | **自然语言与世界知识先验** |

**核心科学问题**（可写入论文）：

1. 在**群体选择频率**（bRate）预测任务上，LLM 能否达到或超过符号/神经基准？
2. LLM 的误差模式是否在 **Problem-Split / Parameter-Amb / Parameter-EV-Extreme**（或当前 `experiments` 采用的 **train_test CSV**）下与 01、02 **一致或可解释**（例如泛化到模糊、极端 EV 差时是否崩溃）？
3. **语言先验**是帮助还是误导（例如对「确定性」「损失框架」的刻板表述敏感）？需通过 **提示扰动** 与 **对照基线** 论证，而非单次 Prompt 的准确率。

### 1.2 任务定义（与全项目一致）

- **输入**：两个赌局 **Gamble A / Gamble B** 的**完整结果–概率分布**（可多结果，与 01 增强特征、02 分布编码口径一致），以及 **context** 中可用字段（如 `feedback`、模糊性相关元数据，若写入标准化 JSON）。
- **输出**：标量 **P(选择 Gamble B)**，记为 **bRate 的预测值** \(\hat{p} \in [0,1]\)，与符号模型阶段标签一致。
- **真值**：人类聚合 **bRate**（0–1）。

### 1.3 与 `experiments` 的标签约定（必须统一）

`experiments/README.md` 约定：**02 神经网络训练标签为 `y = 1 - bRate`**，而 **01 符号默认拟合 bRate**。接入 04 时：

- **03 方案推荐**：内部统一以 **bRate** 为预测目标（与 01、Choices13k 原始语义一致），在导出给需与 02 逐样本对比的脚本时 **显式转换或双列记录**（`pred_bRate` / `pred_y_neural_space`），并在表格脚注中说明，避免静默混用。

---

## 2. 数据与划分（与 01 / 02 对齐）

### 2.1 数据源（两轨，与现有代码一致）

| 轨 | 路径 / 入口 | 划分 | 用途 |
|----|----------------|------|------|
| **A. JSON 增强标准化** | `experiments/01_symbolic_models_enhanced/` 产出的 `c13k_enhanced_standardized.json`（或等价物） | Problem-Split / Parameter-Amb / Parameter-EV-Extreme | 与主方案、论文叙事一致，**OOD 能力**可解释 |
| **B. CSV（与 02 同批）** | `experiments/data/data.csv`（列含 `bRate`, `A`, `B` 等） | `train_test_split`，`random_state=1017+seed` | 与 **02、学习曲线、Fig. S1 风格 CE 图** 严格同分布 |

**实施建议**：先完成 **轨 B**（低成本对齐 04 与学习曲线），再扩展 **轨 A**（三种参数划分写进同一份 `llm_models_summary_*.csv` 的 `split_type` 列）。

### 2.2 Prompt 中的赌局表述

- **必须**枚举每个赌局的全部 \((概率, 结果)\) 对，顺序规则写死（例如按结果数值升序，或按数据文件原始顺序），并在文档中记录，保证可复现。
- **禁止**仅使用 Ha/pHa/La 等过度压缩特征作为唯一输入（除非单独做「消融：仅摘要特征」子实验）。
- **可选字段**：是否在 Prompt 中写明「反馈条件」等与数据集一致的语境，应作为 **消融维度**（有 / 无 / 误导性措辞）。

---

## 3. 方法设计（严谨流程）

以下融合认知心理学式 LLM 评测惯例（Binz & Schulz, PNAS 2023；Binz et al., Nature 2025 中 Psych-101 / Centaur 思路）与本项目**聚合标签**特点。

### 3.1 范式：单试次「从描述决策」+ 非序列

Choices13k 为**问题独立、无试次历史**，对应文献中的 **decisions from description**：每条样本一条 Prompt，**无**多轮状态转移；与 Centaur 的「整段被试会话转写」不同，实现上更简单，但 **不得** 虚构不存在的「上一试次结果」。

### 3.2 提示工程（固定模板 + 版本号）

**系统 / 用户角色**（建议固定其一并版本化，如 `prompt_v1`）：

- 说明任务：在 A、B 两个赌局间预测**一大群人类被试**选 B 的比例（或「估计选 B 的概率」），避免模型回答「我会选 B」与个人决策混淆。
- 要求输出：**仅**一个 \([0,1]\) 内数值，或严格格式如 `P_B=0.63`，便于解析（改进 `数据集/choices13k-main/models.py` 中 `LLMModel.parse_output` 的脆弱正则）。

**推荐解析策略**：

- 主路径：约束格式 + 单次解析；
- 失败时：重试 1 次并附「仅输出数字」；
- 仍失败：`NaN` 记录 + 不计入均值或单独报表，**禁止**静默用 0.5 混入主指标（当前占位实现需改掉）。

### 3.3 路径 A：上下文学习（ICL / Few-shot）

- **零样本**：无示例，仅指令 + 当前题。
- **Few-shot**：从 **训练集** 中分层抽样 \(k\) 个例题（\(k \in \{0, 2, 5\}\)），输入为「题干 + 真值 bRate」；**禁止**从测试集取例题。
- **解码**：主表用 **temperature = 0**（或 API 等价 deterministic 设置）保证可复现；若需估计认知偏差式概率，可另开 **T>0 多次采样** 子实验（与 PNAS 前景理论对比类似，本项目可选）。

**数据污染说明**：Choices13k 与 Peterson et al. 大规模赌博问题有渊源，需在论文 **Limitation** 中承认「训练语料可能见过相似风险决策文本」；缓解方式包括：**轨 B 上仅报告 held-out test**、**改写数字与标签但保持分布** 的鲁棒性子集（可选）。

### 3.4 路径 B：监督微调（QLoRA / LoRA）

**目标**：在训练集上拟合 \((文本描述) \mapsto bRate\)（回归或分箱分类后再校准）。

- **基座**：与团队算力匹配的开源指令模型（如 Llama 3.x、Qwen 等），**记录 checkpoint id 与 commit**。
- **技术**：QLoRA/LoRA，冻结大部分权重；**仅监督「目标数值 token」或专用回归头**（若用隐藏状态 + 小 MLP 头，需在方案中单独命名为「LLM-encoder + head」以免与纯生成混谈）。
- **损失**：与 02 对齐时可采用 **MSE** 或 **Bernoulli 负对数似然**（用 `n_subjects` 加权则接近方案中的 Beta / 分层 NLL，若 JSON 中有被试数）。
- **验证**：与 01/02 相同划分上的 early stopping；报告 **校准**（分箱后预测均值 vs 真实均值）。

**与 Centaur 的差异**：Centaur 在试次级行为上微调；本项目是 **聚合 bRate**，标签噪声更大，**不宜**声称「复制 Centaur」，应表述为「受 Psych-101 范式启发的**聚合标签**微调」。

### 3.5 稳健性与「是否背题」（必做 subset）

至少完成一类定量扰动（与 PNAS 一致思想）：

- **标签轮换**：A/B 名称对调或选项顺序对调，看 \(\hat{p}\) 是否变为 \(1-\hat{p}\)（允许误差阈值，如 \(| \hat{p}_\text{swap} - (1-\hat{p}) | < 0.05\) 的比例）。
- **措辞**：美元 / 抽象「点数」、请求式 vs 疑问式指令。
- **数值微扰**：在保持期望或随机占优关系不变的前提下做小扰动（需程序化生成，记录规则）。

报告：**主指标**（无扰动 test）+ **稳健性分数**（扰动下通过准则的比例）。

---

## 4. 评估指标（与 `evaluation_metrics.py` 一致）

对测试集真值 \(y_i =\) bRate，预测 \(\hat{p}_i\)（裁剪到 \([\epsilon, 1-\epsilon]\)）：

| 指标 | 说明 |
|------|------|
| **MSE / RMSE / MAE** | 与 01/02 可直接对比 |
| **R²** | 解释方差 |
| **Correlation** | Pearson |
| **Cross-Entropy** | \(-\frac{1}{N}\sum_i [y_i\log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]\)，与 `compute_cross_entropy` 一致 |

**可选（方案级已有）**：

- **Beta NLL** 或按 `n_subjects` 加权的二项对数似然（更贴近「多次二元选择聚合」的生成假设，需 JSON 提供每题被试数）。
- **分箱校准图**（predicted vs actual）。

**禁止**：仅用「选 B / 不选 B」的硬分类准确率作为主要指标（聚合任务信息损失大）；若报告，需说明阈值（如 \(\hat{p}>0.5\) 与 \(y>0.5\) 一致率）。

---

## 5. 工程结构与交付物

### 5.1 建议目录（`experiments/03_LLM/`）

```
03_LLM/
├── 03_LLM_实验方案.md          # 本文件
├── README.md                   # 运行说明、环境变量、依赖版本
├── prompts/
│   └── templates.yaml          # 模板版本与 changelog
├── src/
│   ├── build_prompt.py         # JSON/CSV → 文本
│   ├── llm_client.py           # OpenAI / vLLM / 本地 API 统一接口
│   ├── parse_output.py         # 结构化解析 + 重试策略
│   ├── run_eval.py             # 按 split 批量推理 + 写结果
│   └── finetune_qlora.py       # 可选
├── results/
│   ├── llm_models_summary_<timestamp>.csv
│   └── llm_predictions_<split>_<timestamp>.csv
└── configs/
    └── default.yaml            # 模型名、温度、批大小、路径
```

### 5.2 对接 `04_comparison` 的 CSV 约定

在 `llm_models_summary_*.csv` 中至少包含（与神经多种子表对齐时可加 `_mean/_std`）：

- `model`：如 `llm_gpt4o_zero`, `llm_qwen_qlora`
- `split_type`：`train_test` / `problem` / `parameter_amb` / `parameter_ev_extreme`
- `test_mse`, `test_r2`, `test_correlation`, `test_cross_entropy`（或与现有列名完全一致）

扩展 `04_comparison/run_comparison.py` 时增加 `load_llm()` 与合并逻辑（本方案不改代码，仅约定列名）。

### 5.3 复现与合规

- **环境变量**：API Key 不入库；`config.local.yaml` gitignore。
- **缓存**：按 `problem_id` + `prompt_hash` 缓存原始 completion，避免重复计费。
- **随机种子**：数据划分、ICL 例题抽样、子采样评测均记录 `seed`。
- **成本**：全量 13k × 多模型前做 **pilot**（如 500 题）估计费用与延迟。

---

## 6. 可选扩展（非里程碑必做）

1. **RSA / 表征**：用 LLM 最后一层 hidden state（需同一批问题的固定长度 pooling）与 01 的 EU_diff、PT_diff 等做相关，与 02 已有 RSA 设计对齐。
2. **与符号偏差假设对齐**：在子集上构造 **框架效应 / 确定性效应** 式问题对（需从 Choices13k 元数据中筛选或合成），检验 \(\Delta \hat{p}\) 是否与人类 \(\Delta y\) 同号（探索性分析）。
3. **科学遗憾最小化**（Nature 案例）：用强 LLM 或 02 作 reference，找出 01 理论模型系统偏差的题类（高阶、长期）。

---

## 7. 里程碑与时间顺序（建议）

| 阶段 | 内容 | 产出 |
|------|------|------|
| M0 | 锁定 `prompt_v1` + 解析器 + 50 题 pilot | 可解析率 ≥ 95% |
| M1 | 轨 B 全量零样本 + 一种 API 模型 | `llm_models_summary_*.csv`，接入 04 草表 |
| M2 | ICL \(k=2,5\) 与标签轮换稳健性 | 子表 + 简短分析段落 |
| M3 | （可选）QLoRA 与轨 A 三划分 | 与 01/02 同表完整比较 |
| M4 | 论文级附录：完整 Prompt、参数、缓存策略 | 可复现包或 Zenodo |

---

## 8. 参考文献（方法学依据）

1. Binz, M., & Schulz, E. Using cognitive psychology to understand GPT-3. *PNAS*, 2023.（程序化任务、提示扰动、机制分析）
2. Binz, M., et al. A foundation model to predict and capture human cognition (Centaur / Psych-101). *Nature*, 2025.（自然语言转录行为、微调、OOD、开环检验思路；本项目聚合任务为简化版）
3. Peterson, J. C., et al. Using large-scale experiments and machine learning to discover theories of human decision-making. *Science*, 2021.（Choices13k 相关方法论背景）

---

## 9. 与主方案文档的交叉引用

- 总目标与三层对比：《Choices13k 项目方案》§1、§5、§9。
- 划分详解：`数据集/choices13k-main/Choices13k 划分策略详解.md`。
- `experiments` 运行顺序：`experiments/README.md`；全仓库旧版顺序见 `experiments/RUN_ORDER.md`。
- 占位接口：`数据集/choices13k-main/models.py` 中 `LLMModel`（实现时应迁移或封装至 `03_LLM/src/`，避免与数据集仓库强耦合）。

---

*文档版本：v1.0 | 与 `Choices13k 项目方案.md` §5.3 及 `experiments` 管道对齐。*
