# 基于价值的神经网络 (Value-Based Neural Network)

## 1. 模型概述

**ValueBasedNet** 是项目中用于预测风险选择概率的**基于价值**的神经网络。其核心思想是：先为每个赌局估计一个标量“价值”，再通过 Softmax 将两个价值的比较转化为选择概率。该设计对应决策理论中的**独立性假设**——赌局 A 的价值仅依赖 A 的自身属性，与备选赌局 B 无关。

- **定位**：无语言先验的纯统计学习，强调“价值比较”的可解释结构。
- **参考**：架构与数学形式参考 Peterson et al. (2021, *Science*)。
- **代码位置**：`experiments/02_neural_models/models.py` 中的 `ValueBasedNet` 类。

---

## 2. 核心假设与数学形式

### 2.1 独立性假设

- 赌局 A 的价值：$V(A) = f(\text{enc}_A)$，仅依赖 A 的编码。
- 赌局 B 的价值：$V(B) = f(\text{enc}_B)$，仅依赖 B 的编码。
- $f(\cdot)$ 为**共享**子网络，对 A、B 使用同一套参数，保证“同一赌局在不同题目中出现时价值一致”。

### 2.2 从价值到选择概率

选择 Gamble B 的概率由两价值的 Softmax 得到：

$$
P(\text{choose B}) = \frac{e^{\eta \cdot V(B)}}{e^{\eta \cdot V(A)} + e^{\eta \cdot V(B)}}
$$

- **$\eta$ (eta)**：可学习标量，作用类似逆温度（inverse temperature），控制决策的确定性程度。
- $\eta$ 在实现中被约束在 $[10^{-6}, 10^2]$ 内，避免数值不稳定。

---

## 3. 网络架构

### 3.1 共享子网络 $f$

| 层级       | 类型    | 输入维度        | 输出维度 | 说明           |
|------------|---------|-----------------|----------|----------------|
| 输入层     | Linear  | `input_dim_per_gamble` | 64  | 单赌局编码维度，默认 18 |
| 隐藏层     | Sigmoid | 64              | 64       | 非线性         |
| 输出层     | Linear  | 64              | 1        | 标量价值 $V$   |

- **默认单赌局编码维度**：`input_dim_per_gamble = 2 * max_outcomes`，`max_outcomes=9` 时为 **18 维**。
- **隐藏层单元数**：默认 64。

### 3.2 可学习参数

- 子网络 $f$ 的权重与偏置。
- **$\eta$**：`nn.Parameter`，初始值为 1.0，训练中更新。

### 3.3 前向计算

1. `v_A = f(enc_A)`，`v_B = f(enc_B)`，得到两个标量价值。
2. `logits = η * [v_A, v_B]`。
3. `P(B) = softmax(logits)[1]`，即选择 B 的概率。

---

## 4. 输入与编码

### 4.1 单赌局编码 (enc_A / enc_B)

- 每个赌局由其**完整结果分布**表示：若干 $(p_j, x_j)$ 对。
- 编码向量长度固定为 **2 × max_outcomes**（默认 18）：
  - 顺序：$p_1, x_1, p_2, x_2, \ldots$，不足用 0 填充。
- 编码由 `data_loader.py` 中的 `encode_distribution` 与 `build_distribution_encodings` 生成；训练前会对 `enc_A`、`enc_B` 分别做 **StandardScaler** 标准化。

### 4.2 输入张量形状

- `enc_A`, `enc_B`：`(batch_size, input_dim_per_gamble)`，如 `(N, 18)`。
- 输出：`(batch_size,)`，即每个样本一个选择 B 的概率。

---

## 5. 训练与评估

### 5.1 训练目标

- **任务**：回归群体选择频率 **bRate**（选择 Gamble B 的比例）。
- **损失函数**：MSE，即 `nn.MSELoss()`。
- **优化器**：Adam，学习率等超参见 `experiments/02_neural_models/train.py`（如 `LR`、`EPOCHS`、`BATCH_SIZE`）。

### 5.2 训练流程要点

- 对每个划分（problem / parameter_amb / parameter_ev_extreme），将数据拆成 train/val；在训练集上拟合，在验证集上早停（early stopping）。
- 保存验证集 MSE 最优的模型状态用于测试集评估。
- 支持多种子运行，汇总得到 test MSE / R² / correlation 的均值与标准差。

### 5.3 评估指标

与项目其他模型一致：**MSE**、**R²**、**Pearson 相关系数**等，用于与符号模型、Context-Dependent 模型比较（见 `experiments/04_comparison`）。

---

## 6. 使用示例

```python
from models import ValueBasedNet, get_encoding_dims
import torch

max_outcomes = 9
dim_per_gamble, _ = get_encoding_dims(max_outcomes)
model = ValueBasedNet(input_dim_per_gamble=dim_per_gamble, hidden_dim=64)

# 假设 enc_A, enc_B 已标准化，形状 (batch, 18)
enc_A = torch.randn(32, dim_per_gamble)
enc_B = torch.randn(32, dim_per_gamble)
p_B = model(enc_A, enc_B)  # (32,)
```

---

## 7. 与其它模型的关系

| 对比项     | ValueBasedNet     | ContextDependentNet |
|------------|-------------------|----------------------|
| 输入       | 单赌局编码分别输入 | 两赌局拼接后一次输入 |
| 输出方式   | 先价值再 Softmax   | 直接输出 P(B)        |
| 可解释性   | 较高（保留价值比较） | 较低（黑盒）         |
| 参数量     | 较少（共享 f + η） | 较多（全连接堆叠）   |

在文献与项目中，上下文依赖模型在预测上常优于基于价值的模型，提示人类风险决策可能受备选项（context）影响；ValueBasedNet 则提供可解释性更强的基线。

---

## 8. 参考文献与相关文档

- Peterson, J. C., Bourgin, D. D., Agrawal, M., Reichman, D., & Griffiths, T. L. (2021). Using large-scale experiments and machine learning to discover theories of human decision-making. *Science*.
- 项目方案中的“神经网络”部分：`Choices13k 项目方案.md` § 5.2。
- 数据编码与划分：`experiments/02_neural_models/data_loader.py`、`config.py`。
