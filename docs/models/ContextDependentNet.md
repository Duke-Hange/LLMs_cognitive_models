# 上下文依赖神经网络 (Context-Dependent Neural Network)

## 1. 模型概述

**ContextDependentNet** 是项目中用于预测风险选择概率的**上下文依赖**神经网络。与 ValueBasedNet 不同，它不单独为每个赌局估计价值再比较，而是将两个赌局的完整信息拼接后，由单一网络**直接**输出选择概率。该设计允许“上下文效应”——选项 A 的呈现方式或选项 B 的存在可以影响对 A 的评估，对应决策理论中**违反独立性**的假设。

- **定位**：无语言先验的纯统计学习，强调拟合能力与对上下文效应的建模。
- **参考**：架构与数学形式参考 Peterson et al. (2021, *Science*)。
- **代码位置**：`experiments/02_neural_models/models.py` 中的 `ContextDependentNet` 类。

---

## 2. 核心假设与数学形式

### 2.1 上下文依赖假设

- 选择概率由**两个赌局的全部信息**共同决定：
  $$P(\text{choose B}) = g(\text{enc}_A \| \text{enc}_B)$$
- $g(\cdot)$ 是单一前馈网络，输入为 A、B 编码的拼接，输出为标量概率。
- 不假设“每个赌局有独立价值再比较”，因此可以刻画对比效应、框架效应等上下文现象。

### 2.2 输出

- 网络直接输出 $P(\text{choose B})$（即 bRate），取值 $[0, 1]$。
- 最后一层使用 **Sigmoid**，保证输出在 $(0, 1)$ 内。

---

## 3. 网络架构

### 3.1 整体结构

| 层级       | 类型    | 输入维度 | 输出维度 | 说明                    |
|------------|---------|----------|----------|-------------------------|
| 输入层     | Linear  | 36       | 32       | 两赌局拼接编码，默认 36 |
| 隐藏层 1   | Sigmoid | 32       | 32       | 非线性                  |
| 隐藏层 2   | Sigmoid | 32       | 32       | 非线性                  |
| 输出层     | Linear + Sigmoid | 32 | 1        | 选择 B 的概率           |

- **默认输入维度**：`input_dim = 4 * max_outcomes`，`max_outcomes=9` 时为 **36 维**（enc_A 18 维 + enc_B 18 维）。
- **隐藏层**：2 层，每层 **32** 单元，激活为 Sigmoid。

### 3.2 前向计算

- 输入：`enc_full = [enc_A | enc_B]`，形状 `(batch_size, 36)`。
- 输出：`(batch_size,)`，即每个样本一个 $P(\text{choose B})$。

```python
def forward(self, enc_full: torch.Tensor) -> torch.Tensor:
    return self.net(enc_full).squeeze(-1)
```

---

## 4. 输入与编码

### 4.1 拼接编码 enc_full

- **enc_A**：单赌局 A 的分布编码，长度 `2 * max_outcomes`（默认 18），顺序为 $p_1, x_1, p_2, x_2, \ldots$。
- **enc_B**：单赌局 B 的分布编码，格式同 enc_A。
- **enc_full**：`[enc_A | enc_B]`，长度 `4 * max_outcomes`（默认 36）。
- 编码由 `data_loader.py` 的 `build_distribution_encodings` 生成；训练前对整个 `enc_full` 做 **StandardScaler** 标准化（与 ValueBasedNet 对 enc_A、enc_B 分别标准化不同，此处只用一个 scaler）。

### 4.2 输入张量形状

- `enc_full`：`(batch_size, input_dim)`，如 `(N, 36)`。
- 输出：`(batch_size,)`。

---

## 5. 训练与评估

### 5.1 训练目标

- **任务**：回归群体选择频率 **bRate**。
- **损失函数**：MSE，`nn.MSELoss()`。
- **优化器**：Adam，超参与 ValueBasedNet 共用（见 `train.py`）。

### 5.2 训练流程要点

- 与 ValueBasedNet 使用相同的数据划分（problem / parameter_amb / parameter_ev_extreme）和多种子设置。
- 验证集早停，保存验证 MSE 最优的模型用于测试。
- 结果写入 `neural_models_summary_*.csv` 等，并参与 `04_comparison` 的跨模型比较。

### 5.3 评估指标

MSE、R²、Pearson 相关系数等，与符号模型、ValueBasedNet 一致。

---

## 6. 使用示例

```python
from models import ContextDependentNet, get_encoding_dims
import torch

max_outcomes = 9
_, full_dim = get_encoding_dims(max_outcomes)
model = ContextDependentNet(input_dim=full_dim, hidden_dim=32)

# 假设 enc_full 已标准化，形状 (batch, 36)
enc_full = torch.randn(32, full_dim)
p_B = model(enc_full)  # (32,)
```

---

## 7. 与 ValueBasedNet 的对比

| 对比项     | ValueBasedNet           | ContextDependentNet        |
|------------|-------------------------|-----------------------------|
| 理论基础   | 广义效用 / 独立性       | 上下文交互 / 违反独立性    |
| 数学形式   | $V(A), V(B) \to \mathrm{Softmax} \to P$ | 直接 $g(\text{enc}_A \| \text{enc}_B) \to P$ |
| 输入       | 单赌局编码分别输入      | 两赌局拼接一次输入         |
| 隐藏层     | 1 层，64 单元           | 2 层，各 32 单元           |
| 输出       | Softmax 得到 P(B)       | Sigmoid 直接得到 P(B)      |
| 可解释性   | 较高（有价值比较结构）  | 较低（黑盒）               |

在 Peterson et al. 及本项目实验中，Context-Dependent 模型在预测上常优于 Value-Based 模型，提示人类风险决策会受备选项（context）影响；ContextDependentNet 用于捕捉这类效应。

---

## 8. 参考文献与相关文档

- Peterson, J. C., Bourgin, D. D., Agrawal, M., Reichman, D., & Griffiths, T. L. (2021). Using large-scale experiments and machine learning to discover theories of human decision-making. *Science*.
- 项目方案：`Choices13k 项目方案.md` § 5.2（神经网络部分）。
- 实现与数据：`experiments/02_neural_models/models.py`、`data_loader.py`、`train.py`。
