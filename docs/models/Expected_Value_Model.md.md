# 期望值模型 (Expected Value Model) 详解

## 1. 模型概述
期望值模型 (Expected Value, EV) 是决策理论中最经典、最基础的**规范性模型 (Normative Model)**。

它代表了**完全理性 (Rationality)** 的决策视角。该模型假设决策者是“风险中性”的计算机，旨在通过计算概率加权平均值来最大化长期收益。在认知神经科学和行为经济学研究中，EV 模型通常被用作**基准模型 (Baseline Model)** 或**零假设**，用来检测受试者是否表现出非理性的偏差（如风险厌恶）。

### 核心假设
1.  **价值线性 (Linear Value):** 客观金额等于主观价值 ($100元 = 100单位效用$)。
2.  **概率线性 (Linear Probability):** 客观概率等于决策权重 ($1\% = 0.01$)。
3.  **风险中性 (Risk Neutrality):** 决策者对风险无偏好，只关心数学期望的最大化。

---

## 2. 数学公式详解

### 2.1 期望值计算 (Valuation)
对于一个选项 $i$，假设其包含 $n$ 个可能的结果，每个结果的金额为 $x_{j}$，发生的客观概率为 $p_{j}$。
期望值模型不需要参数转换，直接计算数学期望：

$$
EV^{(i)} = \sum_{j=1}^{n} p_{j} \cdot x_{j}
$$

* **对比 PT 模型：** 这里隐含了 $v(x) = x$ 以及 $w(p) = p$。

### 2.2 决策规则 (Choice Rule)
在纯粹的经济学定义中，决策者会确定性地选择 $EV$ 最高的选项。
但在**计算认知建模 (Computational Modeling)** 中，为了解释人类行为中的噪音（即人有时会因为疲劳或注意力不集中而犯错），通常也会引入 **Softmax** 函数或 **Logistic** 函数将价值转化为选择概率。

选择选项 $i$ 的概率为：

$$
P(\text{choose } i) = \frac{\exp(\tau \cdot EV^{(i)})}{\sum_{k} \exp(\tau \cdot EV^{(k)})}
$$

或者对于二选一 (Option A vs Option B) 的情况，公式常简写为：

$$
P(\text{choose A}) = \frac{1}{1 + \exp(-\tau \cdot (EV_A - EV_B))}
$$

* **$\tau$ (Inverse Temperature):** 即使是 EV 模型，在拟合行为数据时通常也包含这就这一个自由参数 $\tau$，用于捕捉决策的**随机性/噪声**水平。

---

## 3. 模型参数对比

与 Prospect Theory (PT) 相比，Expected Value (EV) 极其精简。

| 模型 | 自由参数数量 | 参数含义 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **Expected Value (EV)** | **0 或 1** | 无 (纯理性) 或 仅含 $\tau$ (噪声) | 理想化理性人、算法交易、小额高频决策 |
| **Prospect Theory (PT)** | **3 - 5** | $\alpha, \beta, \lambda, \gamma, \tau$ | 真实人类行为、赌博任务、包含情绪的决策 |

---

## 4. Python 代码实现示例

以下代码展示了如何在 Python 中实现 EV 模型。为了方便与之前的 PT 模型进行 AIC/BIC 模型比较，这里保留了 Softmax 结构。

```python
import numpy as np

def calculate_ev_choice_prob(value_left, prob_left, value_right, prob_right, tau=1.0):
    """
    计算基于期望值(EV)模型的选择概率
    
    Args:
        value_left/right: 选项金额 (可以是数组)
        prob_left/right:  选项获胜概率 (0-1之间)
        tau:              逆温度参数 (控制决策噪声), 默认为1.0
        
    Returns:
        P_left: 选择左侧选项的概率
    """
    
    # 1. 计算期望值 (Expected Value)
    # 核心公式: EV = p * x
    EV_left  = prob_left * value_left
    EV_right = prob_right * value_right
    
    # 2. 计算效用差值
    ev_diff = EV_left - EV_right
    
    # 3. Softmax / Logistic 转换
    # 将价值差转化为 0-1 之间的概率
    # 如果 tau 趋近于无穷大，函数变成阶跃函数(Step Function)，即完全确定的选择
    p_choose_left = 1 / (1 + np.exp(-tau * ev_diff))
    
    return p_choose_left

# --- 使用示例 ---
# 场景: 
# Option A: 50% 概率赢 100元 (EV = 50)
# Option B: 100% 概率赢 40元  (EV = 40)

# 即使 A 的 EV (50) 高于 B (40)，真实人类常因为"风险厌恶"选B。
# 但 EV 模型会预测受试者倾向于选 A。

tau_estimate = 0.5 # 假设的噪声水平
p_choice = calculate_ev_choice_prob(100, 0.5, 40, 1.0, tau_estimate)

print(f"EV模型预测选择 Option A 的概率: {p_choice:.4f}")
# 结果应 > 0.5，因为 EV_A > EV_B