# Choices13k 数据集项目方案

## 一、项目目标

### 1.1 核心目标

- **聚焦风险决策**：对比三种模型（符号模型、神经网络、大语言模型）在解释人类风险决策行为时的异同
- **机制探索**：探索LLM是否拥有类似人类的决策机制（如前景理论、损失厌恶等）
- **任务范围**：专注于风险选择任务（Risky Choice），通过问题参数空间的多样性保持泛化性测试

### 1.2 科学问题

| 科学问题 | 说明 |
|---------|------|
| 模型能否跨问题参数空间泛化？ | 测试模型在不同概率、收益、模糊性条件下的表现 |
| 模型能否捕捉反馈对决策的影响？ | 测试模型是否理解反馈机制（经验决策 vs 描述决策） |
| 模型能否预测群体选择频率分布？ | 预测人类群体的选择模式，而非个体选择 |

## 二、数据标准化协议

### 2.1 统一格式定义

Choices13k 提供**聚合数据**（bRate），数据格式如下：

```python
{
    "context": {
        # 自然语言描述
        "description": "选择 Gamble A 或 Gamble B",
        "gamble_a": {
            "outcomes": [[pHa, Ha], [1-pHa, La]],
            "expected_value": EV_A
        },
        "gamble_b": {
            "outcomes": [[pHb, Hb], [1-pHb, Lb]],
            "expected_value": EV_B,
            "lottery_shape": LotShapeB,
            "lottery_num": LotNumB
        },
        # 问题特征向量
        "features": {
            "Ha": int, "pHa": float, "La": int,
            "Hb": int, "pHb": float, "Lb": float,
            "LotShapeB": int, "LotNumB": int,
            "Amb": bool, "Corr": int,
            "EV_A": float, "EV_B": float,
            "EV_diff": float,  # EV_B - EV_A
            "risk_A": float,   # 方差或标准差
            "risk_B": float
        },
        # 反馈条件
        "feedback": bool,
        "block": int
    },
    
    "action": {
        # 注意：这是聚合数据，不是个体选择
        "bRate": float,  # 选择 Gamble B 的频率 [0, 1]
        "bRate_std": float,  # 跨被试的标准差
        "n_subjects": int  # 被试数量
    },
    
    "reward": {
        # 计算得到的奖励信息
        "expected_reward_A": float,
        "expected_reward_B": float,
        "feedback_received": bool,
        # 如果 feedback=True，可以计算实际获得的奖励分布
        "reward_distribution_A": [...],
        "reward_distribution_B": [...]
    },
    
    "metadata": {
        "problem_id": int,
        "feedback_condition": bool,
        "block": int,
        "dataset": "choices13k"
    }
}
```

### 2.2 自然语言 Prompt 生成

为 LLM 生成标准化的自然语言描述：

```python
def generate_prompt(context):
    """生成 LLM 可理解的自然语言描述"""
    gamble_a = context['gamble_a']
    gamble_b = context['gamble_b']
    
    prompt = f"""
    你面临一个风险选择问题：
    
    Gamble A: 
    - 以概率 {gamble_a['outcomes'][0][0]:.2f} 获得 {gamble_a['outcomes'][0][1]} 点
    - 以概率 {gamble_a['outcomes'][1][0]:.2f} 获得 {gamble_a['outcomes'][1][1]} 点
    - 期望值: {gamble_a['expected_value']:.2f}
    
    Gamble B:
    - 以概率 {gamble_b['outcomes'][0][0]:.2f} 获得 {gamble_b['outcomes'][0][1]} 点
    - 以概率 {gamble_b['outcomes'][1][0]:.2f} 获得 {gamble_b['outcomes'][1][1]} 点
    - 期望值: {gamble_b['expected_value']:.2f}
    
    """
    
    if context['features']['Amb']:
        prompt += "注意：Gamble B 的概率信息不完全明确（存在模糊性）。\n"
    
    if context['feedback']:
        prompt += "你将获得反馈：选择后会看到实际获得的奖励和错过的奖励。\n"
    else:
        prompt += "你不会获得反馈：选择后不会看到实际结果。\n"
    
    prompt += "\n请选择 Gamble A 或 Gamble B，并给出选择概率（0-1之间）。"
    
    return prompt
```

## 三、划分策略

由于 Choices13k 是聚合数据（没有个体被试数据），采用以下划分方式：

### 策略 1: Problem-Split（问题划分）

```python
# 随机划分问题ID
train_problems, test_problems = train_test_split(
    problem_ids, test_size=0.2, random_state=42
)
```
- **测试目标**: 模型对未见问题的泛化能力
- **适用场景**: 基础泛化测试

### 策略 2: Feedback-Split（反馈条件划分）⭐ **最重要**

```python
# 训练集：只有反馈条件
train = df[df['Feedback'] == True]
# 测试集：只有无反馈条件
test = df[df['Feedback'] == False]
```
- **测试目标**: 模型从经验学习（with feedback）到描述决策（without feedback）的迁移
- **科学意义**: 测试模型是否真正理解了反馈的作用
- **重要性**: 这是区分"真正理解"和"死记硬背"的关键测试

### 策略 3: Parameter-Split（参数空间划分）

```python
# 按问题特征划分
# 例如：训练集使用低风险问题，测试集使用高风险问题
train = df[df['EV_diff'].abs() < threshold]
test = df[df['EV_diff'].abs() >= threshold]

# 或按模糊性划分
train = df[df['Amb'] == False]  # 明确概率
test = df[df['Amb'] == True]    # 模糊概率
```
- **测试目标**: 模型对问题参数空间的抽象能力
- **科学意义**: 测试模型是否捕捉了决策的深层规律

### 策略 4: Block-Split（区块划分）

```python
# 训练集：Block 1-3
train = df[df['Block'].isin([1, 2, 3])]
# 测试集：Block 4-5
test = df[df['Block'].isin([4, 5])]
```
- **测试目标**: 模型对问题呈现顺序的鲁棒性

### 策略 5: Cross-Validation Split（交叉验证）

```python
# 按问题ID进行 K-fold
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kf.split(problem_ids):
    train = df[df['Problem'].isin(problem_ids[train_idx])]
    test = df[df['Problem'].isin(problem_ids[test_idx])]
```

### 推荐划分组合

**主要实验**：
1. **Feedback-Split**（最重要）：测试反馈机制的理解
2. **Parameter-Split**（按 EV_diff 或 Amb）：测试参数空间泛化

**辅助实验**：
3. **Problem-Split**：基础泛化测试
4. **Block-Split**：顺序效应测试

## 四、三层模型实现

### 4.1 层级 1：符号模型（Symbolic - World A）

**定位**：传统认知心理学模型，提供可解释的基准线

#### 模型 1: 期望效用理论 (Expected Utility Theory)

```python
def expected_utility(gamble, utility_func=lambda x: x):
    """计算期望效用"""
    EU = sum(p * utility_func(outcome) 
             for p, outcome in gamble['outcomes'])
    return EU

# 选择概率（使用 softmax）
def choice_probability(EU_A, EU_B, temperature=1.0):
    return 1 / (1 + np.exp(-(EU_B - EU_A) / temperature))
```

#### 模型 2: 前景理论 (Prospect Theory)

```python
def prospect_theory(gamble, alpha=0.88, lambda_param=2.25, gamma=0.61):
    """
    alpha: 收益的风险厌恶系数
    lambda_param: 损失厌恶系数
    gamma: 概率权重函数参数
    """
    def value_function(x):
        if x >= 0:
            return x ** alpha
        else:
            return -lambda_param * ((-x) ** alpha)
    
    def probability_weighting(p):
        return (p ** gamma) / ((p ** gamma + (1 - p) ** (1 - gamma)) ** (1/gamma))
    
    PT_value = sum(probability_weighting(p) * value_function(outcome)
                   for p, outcome in gamble['outcomes'])
    return PT_value
```

#### 模型 3: 累积前景理论 (Cumulative Prospect Theory)
- 更复杂的概率权重函数
- 考虑收益和损失的分离处理

#### 模型 4: 启发式模型
- 最大最小规则（Maximin）
- 期望值最大化（Max EV）
- 概率权重启发式

### 4.2 层级 2：神经模型（Neural - World B）

**定位**：无语言先验的纯统计学习，强调拟合能力

#### 输入特征

```python
# 特征向量（而非序列）
features = [
    Ha, pHa, La,  # Gamble A
    Hb, pHb, Lb,  # Gamble B
    LotShapeB, LotNumB,  # Gamble B 特征
    Amb, Corr,  # 问题特征
    EV_A, EV_B, EV_diff,  # 计算特征
    risk_A, risk_B,  # 风险特征
    feedback, block  # 条件特征
]
```

#### 网络架构

```python
import torch
import torch.nn as nn

class RiskDecisionNet(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出 bRate [0, 1]
        )
    
    def forward(self, x):
        return self.net(x)
```

#### 训练目标
- **回归任务**：预测 bRate（而非分类）
- **损失函数**：MSE 或 Huber Loss
- **考虑不确定性**：可以预测 bRate_std

### 4.3 层级 3：大语言模型（LLMs - World C）

**定位**：拥有语言先验的通用推理能力

#### Prompt 设计

**Few-shot In-Context Learning**：
```python
few_shot_examples = """
示例 1:
Gamble A: 以 0.95 概率获得 26，以 0.05 概率获得 -1
Gamble B: 以 0.95 概率获得 21，以 0.05 概率获得 23
反馈: 是
人类选择 Gamble B 的频率: 0.63

示例 2:
Gamble A: 以 0.5 概率获得 2，以 0.5 概率获得 0
Gamble B: 以 1.0 概率获得 1
反馈: 否
人类选择 Gamble B 的频率: 0.61
"""

prompt = few_shot_examples + "\n" + current_problem_prompt
```

**Fine-tuning 数据格式**：
```json
{
    "instruction": "预测人类选择 Gamble B 的概率",
    "input": "Gamble A: ... Gamble B: ... 反馈: ...",
    "output": "0.63"
}
```

#### 输出处理

LLM 输出需要转换为数值：
```python
def parse_llm_output(response):
    """从 LLM 文本输出中提取概率"""
    # 尝试提取数字
    import re
    numbers = re.findall(r'0\.\d+', response)
    if numbers:
        return float(numbers[0])
    # 如果输出是 "Gamble A" 或 "Gamble B"
    if "Gamble B" in response or "B" in response:
        return 1.0
    if "Gamble A" in response or "A" in response:
        return 0.0
    return 0.5  # 默认值
```

## 五、评估指标

### 5.1 拟合优度（Goodness-of-Fit）

**指标 1: 均方误差 (MSE)**
```python
mse = mean_squared_error(y_true=bRate_true, y_pred=bRate_pred)
```

**指标 2: 负对数似然（针对 Beta 分布）**
```python
# 将 bRate 视为 Beta 分布的均值
# bRate ~ Beta(alpha, beta)，其中 alpha + beta = n_subjects
from scipy.stats import beta

def nll_beta(bRate_true, bRate_pred, n_subjects):
    # 从 bRate 和 n 估计 alpha, beta
    alpha_true = bRate_true * n_subjects
    beta_true = (1 - bRate_true) * n_subjects
    
    alpha_pred = bRate_pred * n_subjects
    beta_pred = (1 - bRate_pred) * n_subjects
    
    # 计算 NLL
    nll = -beta.logpdf(bRate_true, alpha_pred, beta_pred)
    return nll
```

**指标 3: 相关系数**
```python
correlation = np.corrcoef(bRate_true, bRate_pred)[0, 1]
```

**指标 4: R²**
```python
from sklearn.metrics import r2_score
r2 = r2_score(bRate_true, bRate_pred)
```

### 5.2 生成式验证（Generative Checks）

**方法 1: 分布匹配**
```python
# 生成多个问题的 bRate 分布
# 比较真实分布 vs 模型生成分布
from scipy.stats import wasserstein_distance, ks_2samp

def distribution_match(true_bRates, pred_bRates):
    wd = wasserstein_distance(true_bRates, pred_bRates)
    ks_stat, p_value = ks_2samp(true_bRates, pred_bRates)
    return wd, ks_stat, p_value
```

**方法 2: 条件分布检查**
```python
# 按反馈条件、模糊性等分组检查
def conditional_distribution_check(df, model_predictions):
    results = {}
    for condition in ['Feedback', 'Amb', 'Corr']:
        for value in df[condition].unique():
            mask = df[condition] == value
            true_vals = df[mask]['bRate']
            pred_vals = model_predictions[mask]
            results[f"{condition}_{value}"] = {
                'mse': mse(true_vals, pred_vals),
                'corr': np.corrcoef(true_vals, pred_vals)[0, 1]
            }
    return results
```

**方法 3: 参数空间行为模式**
```python
# 检查模型在不同参数空间的行为
# 例如：EV_diff vs bRate 的关系
def parameter_space_analysis(df, model_predictions):
    # 按 EV_diff 分组
    df['EV_diff_bin'] = pd.cut(df['EV_diff'], bins=10)
    
    true_by_bin = df.groupby('EV_diff_bin')['bRate'].mean()
    pred_by_bin = pd.Series(model_predictions).groupby(df['EV_diff_bin']).mean()
    
    return {
        'true_curve': true_by_bin,
        'pred_curve': pred_by_bin,
        'correlation': np.corrcoef(true_by_bin, pred_by_bin)[0, 1]
    }
```

### 5.3 模型恢复（Model Recovery）

由于是聚合数据，模型恢复需要调整：

**方法 1: 问题级别恢复**
```python
# 给定一组问题的 bRate，判断是由哪个模型生成的
from sklearn.ensemble import RandomForestClassifier

# 特征：问题的统计特征（bRate, bRate_std, n_subjects等）
# 标签：生成模型（Symbolic/Neural/LLM）
classifier = RandomForestClassifier()
classifier.fit(X_features, y_model_labels)
```

**方法 2: 参数空间恢复**
```python
# 检查不同模型在不同参数空间的预测模式
# 如果模型机制不同，它们的预测模式应该可区分
```

### 5.4 内部表征对齐（RSA）

**符号模型的潜变量**：
- 期望效用差异 (EU_diff)
- 前景理论价值差异 (PT_diff)
- 风险差异 (Risk_diff)

**神经网络的内部表征**：
```python
# 提取中间层激活
activations = model.get_intermediate_activations(features)
# activations shape: [n_problems, hidden_dim]
```

**LLM 的内部表征**：
```python
# 提取最后一层 hidden states
# 使用平均池化得到问题表征
llm_embeddings = model.get_embeddings(prompts)
# llm_embeddings shape: [n_problems, embedding_dim]
```

**RSA 计算**：
```python
from scipy.stats import spearmanr

def rsa(symbolic_vars, neural_activations, llm_embeddings):
    # 计算距离矩阵
    symbolic_dist = pairwise_distances(symbolic_vars)
    neural_dist = pairwise_distances(neural_activations)
    llm_dist = pairwise_distances(llm_embeddings)
    
    # 计算相关性
    rsa_neural = spearmanr(
        squareform(symbolic_dist),
        squareform(neural_dist)
    )[0]
    
    rsa_llm = spearmanr(
        squareform(symbolic_dist),
        squareform(llm_dist)
    )[0]
    
    return rsa_neural, rsa_llm
```

## 六、实施路线图

### 阶段 1: 数据标准化（1-2周）
- [ ] 实现数据加载和预处理
- [ ] 实现统一格式转换
- [ ] 实现自然语言 Prompt 生成
- [ ] 数据质量检查

### 阶段 2: 划分策略实现（1周）
- [ ] 实现 Problem-Split
- [ ] 实现 Feedback-Split
- [ ] 实现 Parameter-Split
- [ ] 实现 Block-Split

### 阶段 3: 模型实现（3-4周）
- [ ] 符号模型：期望效用、前景理论、累积前景理论
- [ ] 神经网络：特征工程、网络设计、训练
- [ ] LLM：Prompt 设计、Few-shot、Fine-tuning

### 阶段 4: 评估与比较（2-3周）
- [ ] 拟合优度评估
- [ ] 生成式验证
- [ ] 模型恢复实验
- [ ] RSA 分析

### 阶段 5: 结果分析与报告（2周）
- [ ] 结果可视化
- [ ] 机制解释
- [ ] 论文撰写

## 七、关键挑战与解决方案

### 挑战 1: 聚合数据 vs 个体数据
**解决方案**：
- 将预测目标从"个体选择"改为"群体选择频率"
- 使用 Beta 分布建模不确定性
- 在评估时考虑 n_subjects 的影响

### 挑战 2: 缺乏序列信息
**解决方案**：
- 利用 Feedback 条件模拟学习过程
- 使用 Block 信息作为顺序代理
- 在 Prompt 中加入历史问题（如果可用）

### 挑战 3: 任务单一性
**解决方案**：
- 通过参数空间多样性保持泛化测试
- 使用 Parameter-Split 测试抽象能力
- 强调反馈机制的理解（经验 vs 描述决策）

### 挑战 4: LLM 输出格式
**解决方案**：
- 设计结构化输出格式（JSON）
- 使用 Chain-of-Thought 引导数值输出
- 实现 robust 的解析函数

## 八、预期成果

1. **科学发现**：
   - LLM 在风险决策中是否表现出类似人类的偏差（如损失厌恶、概率权重）
   - 三种模型在解释风险决策时的互补性

2. **技术贡献**：
   - 聚合数据的认知建模框架
   - LLM 在风险决策中的 Prompt 设计最佳实践
   - 跨参数空间的泛化评估方法

3. **可发表性**：
   - 聚焦风险决策，符合认知科学期刊的发表范围
   - 方法创新（聚合数据建模）具有方法论价值
