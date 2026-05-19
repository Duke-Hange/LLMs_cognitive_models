# 前景理论 (Prospect Theory) 计算模型详解

## 1. 模型概述
前景理论 (Prospect Theory, PT) 是行为经济学和认知神经科学中描述**风险决策 (Decision under Risk)** 的核心模型。与传统的期望效用理论 (Expected Utility Theory) 不同，PT 认为人类的决策行为存在非理性偏差。

根据本文档提供的模型设定，该模型包含三个核心假设：
1.  **敏感度递减 (Diminishing Sensitivity)：** 对数值变化的感知是非线性的。
2.  **损失厌恶 (Loss Aversion)：** 损失带来的心理效用绝对值大于同等收益。
3.  **概率非线性加权 (Nonlinear Probability Weighting)：** 小概率被高估，大概率被低估。

---

## 2. 数学公式详解

### 2.1 主观价值函数 (Subjective Value Function)
该函数将客观的金额结果 $x$ 转化为心理上的主观价值 $v(x)$。

$$
v(x) = \begin{cases} 
x^\alpha & \text{if } x \ge 0 \\
-\lambda(-x)^\beta & \text{if } x < 0 
\end{cases}
$$

* **$x \ge 0$ (收益域):** 呈现**凹函数 (Concave)** 形态，体现收益的边际效用递减。
* **$x < 0$ (损失域):** 呈现**凸函数 (Convex)** 形态，体现损失的边际敏感度递减。
* **$\lambda$ (Loss Aversion):** 损失厌恶系数。通常 $\lambda > 1$，使得损失曲线比收益曲线更陡峭。

### 2.2 概率加权函数 (Probability Weighting Function)
该函数将客观概率 $p$ 转化为决策权重 $w(p)$。此处使用的是单参数形式：

$$
w(p) = \frac{p^\gamma}{(p^\gamma + (1-p)^\gamma)^{1/\gamma}}
$$

* 函数形态通常呈 **倒S形 (Inverse S-shape)**。
* **低概率区间：** 曲线位于对角线上方（高估，Overweighting）。
* **高概率区间：** 曲线位于对角线下方（低估，Underweighting）。

### 2.3 总效用计算 (Overall Utility)
对于包含多个结果 $j$ 的选项 $i$，其总前景值（效用）$U^{(i)}$ 为各结果主观价值与其决策权重的乘积之和：

$$
U^{(i)} = \sum_{j} w(p_{ij}) \cdot v(v_{ij})
$$

### 2.4 决策规则 (Choice Rule)
模型假设决策存在一定的随机性（噪声），使用 **Softmax** 函数计算选择选项 $i$ 的概率：

$$
P(\text{choose } i) = \frac{\exp(\tau \cdot U^{(i)})}{\sum_{k=1}^{2} \exp(\tau \cdot U^{(k)})}
$$

---

## 3. 自由参数说明 (Free Parameters)

模型共包含 **5个自由参数**，通常通过极大似然估计 (MLE) 或贝叶斯分层模型 (HBM) 进行拟合。为了优化稳定性，参数通常经过 Sigmoid 变换限制在特定范围内。

| 参数符号 | 含义 | 作用描述 | 典型范围 |
| :--- | :--- | :--- | :--- |
| **$\alpha$** | 收益曲率 | 控制收益域的风险厌恶程度 (Risk Aversion) | $(0, 1)$ |
| **$\beta$** | 损失曲率 | 控制损失域的风险寻求程度 (Risk Seeking) | $(0, 1)$ |
| **$\lambda$** | 损失厌恶 | 衡量对损失的敏感程度相对于收益的倍数 | $(0, 10)$ |
| **$\gamma$** | 概率扭曲 | 控制概率加权函数的弯曲程度 | $(0, 1)$ |
| **$\tau$** | 逆温度 | 控制决策的一致性 (Inverse Temperature) | $(0, 100)$ |

> **注：** > * $\tau \to 0$: 决策趋于随机 (Random)。
> * $\tau \to \infty$: 决策趋于完全确定 (Deterministic)，只选效用最高的选项。

---

## 4. Python 代码实现示例

以下是基于上述公式的 Python `numpy` 实现，可用于模型拟合或模拟。

```python
import numpy as np

def calculate_choice_prob(value_left, prob_left, value_right, prob_right, params):
    """
    计算选择左侧选项的概率
    
    Args:
        value_left/right: 选项金额 (可以是数组)
        prob_left/right:  选项获胜概率 (0-1之间)
        params: 列表或数组 [alpha, beta, lambda_, gamma, tau]
        
    Returns:
        P_left: 选择左侧选项的概率
    """
    alpha, beta, lambda_, gamma, tau = params
    
    # 1. 主观价值函数 v(x)
    def value_func(x):
        # 使用 np.where 处理数组输入
        return np.where(x >= 0, 
                        np.power(x, alpha), 
                        -lambda_ * np.power(np.abs(x), beta))
    
    # 2. 概率加权函数 w(p)
    def weight_func(p):
        denom = np.power(np.power(p, gamma) + np.power(1-p, gamma), 1/gamma)
        return np.power(p, gamma) / denom
    
    # 3. 计算总效用 U
    # 假设每个选项只有一个非零结果 (Gambles vs 0)
    # 如果是复杂赌局(如: 50%赢10块, 50%输5块), 需分别计算再求和
    U_left  = weight_func(prob_left)  * value_func(value_left)
    U_right = weight_func(prob_right) * value_func(value_right)
    
    # 4. Softmax 计算选择概率
    # P(Left) = exp(tau * U_L) / (exp(tau * U_L) + exp(tau * U_R))
    # 为防止溢出，通常使用: 1 / (1 + exp(-tau * (U_L - U_R)))
    
    utility_diff = U_left - U_right
    p_choose_left = 1 / (1 + np.exp(-tau * utility_diff))
    
    return p_choose_left

# --- 使用示例 ---
# 设定一组参数
params_example = [0.88, 0.88, 2.25, 0.65, 1.5] 

# 模拟情境: 
# 选项A: 50% 概率赢 100元
# 选项B: 100% 概率赢 40元 (即 value=40, prob=1.0)
p_choice = calculate_choice_prob(100, 0.5, 40, 1.0, params_example)

print(f"选择选项A (风险选项) 的概率: {p_choice:.4f}")
