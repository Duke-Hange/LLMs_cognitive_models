# 01_symbolic_models_enhanced

本阶段实现增强符号模型。它对应项目中的“符号认知模型”部分，核心作用是用可解释的低参数模型预测 Choices13k 的群体选择比例，并与神经模型、LLM 模型进行样本效率比较。

## 1. 阶段目标

符号模型提供强理论先验：模型结构直接表达期望值、效用函数、概率加权或前景理论等心理学假设。与神经模型相比，它的容量更小、解释更直接，因此适合检验“少样本条件下，理论结构是否比纯数据拟合更稳定”。

本阶段可单独运行，也可由 04/05 统一比较框架调用。正式比较优先看 04/05，因为它们使用统一 split、统一指标和统一图表。

## 2. 主要模型

当前代码中的学习曲线模型列表为：

- `ev`：Expected Value，期望值模型。
- `eu`：Expected Utility，期望效用模型。
- `pt3`：三参数前景理论模型。
- `pt5`：五参数前景理论模型。

模型实现位于：

```text
enhanced_symbolic_models.py
```

数据标准化与 CSV 适配位于：

```text
enhanced_data_standardization.py
csv_data.py
```

## 3. 运行方式

单独训练增强符号模型：

```powershell
python experiments/01_symbolic_models_enhanced/train_enhanced_models.py --seeds 42
```

运行符号模型学习曲线：

```powershell
python experiments/01_symbolic_models_enhanced/run_learning_curve.py --seeds 42 43 44 --data-source csv --csv-path experiments/data/data.csv
```

快速连通性检查：

```powershell
python experiments/01_symbolic_models_enhanced/quick_smoke_test.py
```

## 4. 输出

单跑输出通常写入 01 自己的 `results/` 或相关输出目录。04/05 调用符号模型时，会把统一结果写入各自的 `outputs/run_*/metrics/` 与 `outputs/run_*/figures/`。

正式论文或报告建议优先引用 04/05 的输出，因为那些结果与神经模型、LLM 使用同一测试集和同一指标。

## 5. 与其他阶段的关系

01 是符号模型的源头。04 使用它与 02 做一阶段比较；05 在此基础上加入 LLM；06 不直接调用 01，但 06 的结果会回到 05 的同一学习曲线中，与符号模型同图比较。

解释结果时，符号模型的优势应表述为“理论先验和低参数结构带来的样本效率或稳定性”，而不是简单等同于人类真实决策机制。
