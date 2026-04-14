# Choices13k：05_program 项目说明（系统化学术表达版）

## 1. 研究定位与核心问题

`05_program` 是一个**统一实验协议（unified protocol）**下的小样本学习曲线研究。  
研究目标不是只比较“谁的最高分更高”，而是系统回答以下问题：

- 在小样本区间（低 `N`）中，哪类模型更具样本效率（sample efficiency）？
- 随训练样本规模增加，模型性能提升速度是否一致？
- 不同模型在随机采样扰动下是否稳定（robustness across seeds）？

从研究方法上看，这是一种**控制变量的横向比较设计**：数据源、任务定义、评估协议一致，仅改变模型家族和训练样本规模。


## 2. 实验对象与比较维度

### 2.1 模型家族

- `symbolic`: EV, EU, PT3, PT5
- `neural`: value-based, context-dependent(s)
- `llm` (ICL): llama3.1, qwen2.5:7b-instruct

### 2.2 主要比较维度

- **模型类型**：symbolic vs neural vs llm
- **样本规模**：`N = [4, 8, 16, 32, 64, 128]`
- **随机重复**：`seed = [42, 43]`

因此每个模型至少会在 12 个组合点（6 个 N × 2 个 seed）上被评估。


## 3. 数据与任务定义

### 3.1 数据来源

- 数据文件：`experiments/03_LLM/results/prompts_en_v2.jsonl`

### 3.2 预测目标

- `target_mode: one_minus_bRate`

可理解为：模型输出一个连续概率值，用于拟合行为比例型标签。  
这属于**概率回归/概率估计任务**，而非传统的硬标签分类。

### 3.3 划分策略

- `test_size: 0.2`
- `master_seed: 42`
- `split_random_state_base: 1017`

设计要点：所有模型共享同一测试集，保证可比性；固定随机状态，保证可复现性。


## 4. 方法流程（Pipeline）

主入口：`src/runners/run_all.py`。流程可分为五个阶段：

1. **配置加载阶段**  
   读取 `protocol.yaml` 与模型清单，建立当次 run 的实验协议。

2. **数据与采样阶段**  
   加载数据集，构建主 train/test 划分，再从 train 中按 `(N, seed)` 采样子集。

3. **模型执行阶段**  
   依次执行 symbolic、neural、llm 三个家族，记录逐样本预测结果。

4. **评估与聚合阶段**  
   统一计算各项指标，按模型与样本规模聚合，生成学习曲线统计量与置信区间。

5. **产物与复现阶段**  
   写出图表、summary、复现清单（`split_manifest`）和配置快照（`configs_snapshot`）。

这种流程可以理解为：**先冻结实验条件，再并行比较方法，最后统一统计解释**。


## 5. 关键参数及方法学含义

以下参数来自 `experiments/05_program/configs/protocol.yaml`。

### 5.1 指标体系（evaluation endpoints）

- 主指标：`mae`, `cross_entropy`
- 次指标：`rmse`, `mse`, `r2`, `correlation`, `parse_success_rate`

说明：

- `MAE` 便于解释“平均偏差”，适合主文展示；
- `Cross-Entropy` 对概率校准偏差更敏感；
- 次指标用于补充刻画误差结构与拟合相关性。

### 5.2 统计不确定性（uncertainty quantification）

- `ci_level: 0.95`
- `bootstrap_samples: 1000`

说明：

- 通过 bootstrap 估计 95% 置信区间，不只报告点估计；
- 有助于判断“看起来更好”是否在统计上有稳健支撑。

### 5.3 数值与解析策略

- `ce_eps: 1e-15`：避免 `log(0)` 数值不稳定；
- `llm_parse_fail_fill: null`：LLM 解析失败不做固定值填补，主指标仅在有效解析样本上计算。

方法学含义：

- 该策略提升了指标解释的“有效性纯度”，但会牺牲部分样本覆盖率；
- 因此建议同时报告 `parse_success_rate` 作为配套解释指标。

### 5.4 LLM 推理协议

- `temperature: 0.0`（确定性优先）
- `timeout_seconds: 120`
- `num_predict: 32`
- `max_retries: 1`
- `dedup_icl: true`
- `kshot_decimals: 4`
- `cache_dir` 启用缓存

说明：

- 整体策略偏向“稳定与可复现”，而非“探索式多样输出”；
- 与本项目的比较目标一致（公平、可重复、可解释）。

### 5.5 工程层输出策略

- `staging_output_root` 指向中转盘（SSD）
- 成功后归档到 `output_root`

说明：

- 这是典型的“先快盘写入、再归档”策略，可降低长任务 I/O 风险。


## 6. 产物结构与用途映射

一次 `run_xxx` 目录中，核心文件可按“证据链”理解：

- `predictions/predictions_long.csv`：微观层证据（逐样本预测）
- `metrics/metrics_by_model_n_seed.csv`：中观层证据（模型-样本规模-随机种子指标）
- `metrics/curve_aggregated.csv`：宏观层证据（学习曲线聚合统计）
- `figures/*.png`：可视化呈现
- `tables/summary_by_model.csv`：汇报与论文表格入口
- `split_manifest.json`：复现锚点（数据划分与采样映射）
- `configs_snapshot/*.yaml`：参数追溯锚点
- `reports/methods_and_reproducibility.md`：方法说明文档

这套结构支持“从图回到数据、从数据回到配置”的完整追踪。


## 7. 可视化设计建议（结合当前实验）

### 7.1 主图建议

建议主图采用：**按 family 分面的 MAE 学习曲线（均值 + 95% CI）**。

理由：

- MAE更直观，便于跨学科沟通；
- 分面可避免模型过多导致同图拥挤；
- CI 阴影可呈现稳定性而非只看均值。

### 7.2 补充图建议

交叉熵建议使用“聚焦纵轴”版本（例如 `0.62-1.05`）并保留一张全范围图。

理由：

- 全范围图保证完整性；
- 聚焦图提高主趋势可读性，避免少量离群点压缩主体变化。


## 8. 当前方案的优势与边界

### 8.1 优势

- 协议统一，比较公平；
- 小样本学习曲线能展示“性能-数据规模”关系；
- 复现材料完整（manifest + snapshot）；
- 可与后续 `06_qlora_text` 自然对接。

### 8.2 边界与解释注意

- 不同家族的输入机制并非完全同构，结论应表述为“协议下比较”；
- LLM 解析失败处理会影响有效样本集合；
- Cross-Entropy 对极端概率敏感，需配套可视化尺度说明。


## 9. 可直接引用的方法描述（简版）

本研究在统一实验协议下，比较了 symbolic、neural 与 LLM-ICL 三类模型在小样本场景中的学习曲线行为。我们固定数据划分与随机种子策略，在多个训练样本规模（N=4,8,16,32,64,128）上进行重复评估，并以 MAE 和 Cross-Entropy 作为主指标。为刻画估计不确定性，采用 bootstrap 估计 95% 置信区间。实验同时输出逐样本预测、分组指标、聚合曲线、配置快照与划分清单，以保证分析可追溯、结果可复现。


## 10. 建议的后续工作

- 在 summary 中加入“达到目标 MAE 所需最小 N”（sample complexity 指标）；
- 将 `06_qlora_text` 结果按同一 `split_manifest` 合并到统一图表；
- 固化“主图 + 补图 + 表格”的报告模板，形成可重复产出流程。


## 附录A：参数字典（全量解释版）

本节覆盖三类参数来源：

- `configs/protocol.yaml`
- `configs/models_symbolic.yaml`、`configs/models_neural.yaml`、`configs/models_llm.yaml`
- `src/runners/run_all.py` 的命令行参数

### A.1 `protocol.yaml` 参数说明

#### A.1.1 `experiment` 组

- `experiment.name`  
  **含义**：实验名称标识。  
  **当前值**：`05_program_small_sample_curve`。  
  **作用**：用于文档/报告语义标注，不直接改变训练逻辑。

- `experiment.target_mode`  
  **含义**：预测目标模式。  
  **当前值**：`one_minus_bRate`。  
  **作用**：决定数据集中取哪一列作为监督信号；影响所有模型训练与评估的标签定义。

- `experiment.primary_metrics`  
  **含义**：主指标列表。  
  **当前值**：`["cross_entropy", "mae"]`。  
  **作用**：用于主结论与主图展示的核心评价维度。

- `experiment.secondary_metrics`  
  **含义**：次指标列表。  
  **当前值**：`["rmse", "mse", "r2", "correlation", "parse_success_rate"]`。  
  **作用**：用于补充解释误差结构、相关性和解析成功率。

- `experiment.research_notes`  
  **含义**：研究备忘录条目（非执行参数）。  
  **当前值**：4条说明（上下文长度、QLoRA策略、目标量级、训练预算）。  
  **作用**：指导后续实验设计与写作，不直接参与计算图。

#### A.1.2 `data` 组

- `data.prompts_path`  
  **含义**：输入数据文件路径。  
  **当前值**：`experiments/03_LLM/results/prompts_en_v2.jsonl`。  
  **作用**：决定实验样本来源；改变该值会直接改变结果分布与任务语义。

- `data.test_size`  
  **含义**：测试集比例。  
  **当前值**：`0.2`。  
  **作用**：控制 train/test 分配；测试比例越大，评估方差通常更低但训练数据更少。

- `data.split_random_state_base`  
  **含义**：划分随机状态基值。  
  **当前值**：`1017`。  
  **作用**：与 `master_seed` 一起确定可复现的数据划分。

- `data.master_seed`  
  **含义**：全局主种子。  
  **当前值**：`42`。  
  **作用**：用于构造主划分和相关随机过程的复现锚点。

#### A.1.3 `sampling` 组

- `sampling.sample_sizes`  
  **含义**：训练样本规模网格（学习曲线横轴）。  
  **当前值**：`[4, 8, 16, 32, 64, 128]`。  
  **作用**：决定曲线分辨率与计算成本；点越多，趋势越细，但运行更慢。

- `sampling.seeds`  
  **含义**：重复采样的随机种子集合。  
  **当前值**：`[42, 43]`。  
  **作用**：用于估计稳定性；seed 越多，不确定性估计越稳但成本更高。

#### A.1.4 `llm` 组

- `llm.base_url`  
  **含义**：LLM 推理服务地址。  
  **当前值**：`http://127.0.0.1:11434`。  
  **作用**：决定请求发送到哪个本地/远程推理后端。

- `llm.temperature`  
  **含义**：采样温度。  
  **当前值**：`0.0`。  
  **作用**：控制随机性；越低越确定，越高越多样。当前配置偏可复现。

- `llm.timeout_seconds`  
  **含义**：单请求超时秒数。  
  **当前值**：`120`。  
  **作用**：避免慢请求长期阻塞。

- `llm.num_predict`  
  **含义**：单次生成最大 token 数。  
  **当前值**：`32`。  
  **作用**：控制输出长度上界；过小可能截断，过大增加延迟和异常文本风险。

- `llm.max_retries`  
  **含义**：失败重试次数上限。  
  **当前值**：`1`。  
  **作用**：平衡鲁棒性与总耗时。

- `llm.dedup_icl`  
  **含义**：是否使用去重的 ICL 模板。  
  **当前值**：`true`。  
  **作用**：减少冗余上下文，降低长度压力与重复提示噪声。

- `llm.kshot_decimals`  
  **含义**：few-shot 示例中数值保留小数位。  
  **当前值**：`4`。  
  **作用**：影响示例数值精度与提示文本长度。

- `llm.cache_dir`  
  **含义**：LLM 调用缓存目录。  
  **当前值**：`experiments/03_LLM/results/cache`。  
  **作用**：减少重复请求，降低成本并提升复现实验速度。

- `llm.dry_run`  
  **含义**：是否仅做演练而不执行完整真实推理。  
  **当前值**：`false`。  
  **作用**：`true` 常用于冒烟测试；`false` 才是正式结果。

#### A.1.5 `evaluation` 组

- `evaluation.ce_eps`  
  **含义**：Cross-Entropy 数值稳定项。  
  **当前值**：`1.0e-15`。  
  **作用**：防止 `log(0)` 导致数值溢出。

- `evaluation.llm_parse_fail_fill`  
  **含义**：LLM 解析失败时是否填充值。  
  **当前值**：`null`。  
  **作用**：`null` 表示主指标仅在可解析样本上计算；若设数值则会对失败样本插补。

- `evaluation.ci_level`  
  **含义**：置信区间置信度。  
  **当前值**：`0.95`。  
  **作用**：控制区间宽度；置信度越高，区间通常越宽。

- `evaluation.bootstrap_samples`  
  **含义**：bootstrap 重采样次数。  
  **当前值**：`1000`。  
  **作用**：次数越高，CI 估计更稳定，但计算更慢。

- `evaluation.n_target_metric`  
  **含义**：样本规模目标判定所依据的指标。  
  **当前值**：`"mae"`。  
  **作用**：用于 summary 中“达到目标性能所需 N”的计算基准。

- `evaluation.n_target_value`  
  **含义**：目标指标阈值。  
  **当前值**：`0.18`。  
  **作用**：与 `n_target_metric` 联合定义“达标”条件。

#### A.1.6 `paths` 组

- `paths.output_root`  
  **含义**：归档输出根目录。  
  **当前值**：`experiments/05_program/outputs`。  
  **作用**：最终结果落盘位置。

- `paths.staging_output_root`  
  **含义**：中转输出目录（通常是 SSD）。  
  **当前值**：`C:/数据中转站/choices13k_05_runs`。  
  **作用**：先快盘写入再归档，提高长任务 I/O 稳定性；设 `null` 可关闭中转。

### A.2 模型配置参数说明（`models_*.yaml`）

这三份模型配置共享同一结构。

- `family`  
  **含义**：模型家族名。  
  **当前值**：`symbolic` / `neural` / `llm`。  
  **作用**：用于流程分派、结果分组与图表分面。

- `models`  
  **含义**：模型条目列表。  
  **作用**：定义该家族要运行的具体模型集合。

- `models[].model_id`  
  **含义**：外部展示名称（报告/图例/表格中出现）。  
  **作用**：决定结果中的模型标签可读性。

- `models[].source_name`  
  **含义**：内部实现映射名（代码中用于定位具体模型实现或后端名称）。  
  **作用**：连接配置与执行逻辑；写错会导致无法找到模型或调用错误后端。

### A.3 运行入口参数说明（`run_all.py`）

命令行参数定义于 `src/runners/run_all.py`：

- `--smoke`（布尔开关）  
  **含义**：开启冒烟模式。  
  **作用**：代码中会自动把 `sample_sizes` 缩为 `[8, 32]`，并强制 `llm.dry_run=True`，用于快速检查流程通路。

- `--neural-epochs`（整数，默认 `1000`）  
  **含义**：神经网络训练最大 epoch。  
  **作用**：上限越高，潜在收敛更充分，但时间更长、过拟合风险更高。

- `--neural-patience`（整数，默认 `100`）  
  **含义**：神经网络早停耐心值。  
  **作用**：控制“验证集不提升时还能等多少轮”；越大越保守，越小越激进。

- `--quiet`（布尔开关）  
  **含义**：静默模式。  
  **作用**：关闭大部分进度日志输出，适合批量脚本或减少终端噪声。

### A.4 运行时自动生成的“派生参数”（非手填但重要）

这些不是配置文件直接填写的字段，但对解释结果很关键：

- `run_id`：按时间戳生成，每次运行唯一标识。  
- `split_manifest.prompts_sha256`：输入数据哈希，用于确认数据冻结一致性。  
- `split_manifest.train_indices / test_indices`：真实划分索引。  
- `split_manifest.sample_map`：每个 `(N, seed)` 的训练子集索引映射。  

这些字段构成复现闭环：**同数据哈希 + 同划分索引 + 同参数配置 = 可重复结果**。

