"""
三层模型实现：符号模型、神经网络、大语言模型
针对 Choices13k 数据集调整
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler


# ==================== 层级 1: 符号模型 (Symbolic Models) ====================

class ExpectedUtilityModel:
    """期望效用理论模型"""
    
    def __init__(self, utility_func=None):
        """
        Args:
            utility_func: 效用函数，默认为线性 u(x) = x
        """
        self.utility_func = utility_func if utility_func else lambda x: x
        self.temperature = 1.0  # softmax 温度参数
    
    def expected_utility(self, gamble: List) -> float:
        """计算期望效用"""
        return sum(p * self.utility_func(outcome) for p, outcome in gamble)
    
    def predict(self, context: Dict) -> float:
        """
        预测选择 Gamble B 的概率
        
        Args:
            context: 包含 gamble_a 和 gamble_b 的上下文
            
        Returns:
            bRate (选择 Gamble B 的概率)
        """
        gamble_a = context['gamble_a']['outcomes']
        gamble_b = context['gamble_b']['outcomes']
        
        eu_a = self.expected_utility(gamble_a)
        eu_b = self.expected_utility(gamble_b)
        
        # Softmax 选择
        prob_b = 1 / (1 + np.exp(-(eu_b - eu_a) / self.temperature))
        return prob_b
    
    def fit(self, X: np.ndarray, y: np.ndarray, contexts: List[Dict]):
        """
        拟合模型（优化温度参数）
        
        Args:
            X: 特征矩阵（未使用，保持接口一致性）
            y: 真实 bRate
            contexts: 上下文列表
        """
        def loss(temperature):
            self.temperature = temperature[0]
            predictions = [self.predict(ctx) for ctx in contexts]
            mse = np.mean((np.array(predictions) - y) ** 2)
            return mse
        
        result = minimize(loss, x0=[1.0], bounds=[(0.01, 10.0)])
        self.temperature = result.x[0]
        return self


class ProspectTheoryModel:
    """前景理论模型"""
    
    def __init__(self, alpha=0.88, lambda_param=2.25, gamma=0.61):
        """
        Args:
            alpha: 收益的风险厌恶系数
            lambda_param: 损失厌恶系数
            gamma: 概率权重函数参数
        """
        self.alpha = alpha
        self.lambda_param = lambda_param
        self.gamma = gamma
        self.temperature = 1.0
    
    def value_function(self, x: float) -> float:
        """价值函数"""
        if x >= 0:
            return x ** self.alpha
        else:
            return -self.lambda_param * ((-x) ** self.alpha)
    
    def probability_weighting(self, p: float) -> float:
        """概率权重函数"""
        if p == 0:
            return 0
        if p == 1:
            return 1
        return (p ** self.gamma) / ((p ** self.gamma + (1 - p) ** (1 - self.gamma)) ** (1/self.gamma))
    
    def prospect_value(self, gamble: List) -> float:
        """计算前景理论价值"""
        # 分离收益和损失
        gains = [(p, x) for p, x in gamble if x >= 0]
        losses = [(p, x) for p, x in gamble if x < 0]
        
        # 计算收益部分
        gain_value = sum(self.probability_weighting(p) * self.value_function(x) 
                        for p, x in gains)
        
        # 计算损失部分
        loss_value = sum(self.probability_weighting(p) * self.value_function(x) 
                        for p, x in losses)
        
        return gain_value + loss_value
    
    def predict(self, context: Dict) -> float:
        """预测选择概率"""
        gamble_a = context['gamble_a']['outcomes']
        gamble_b = context['gamble_b']['outcomes']
        
        pt_a = self.prospect_value(gamble_a)
        pt_b = self.prospect_value(gamble_b)
        
        prob_b = 1 / (1 + np.exp(-(pt_b - pt_a) / self.temperature))
        return prob_b
    
    def fit(self, X: np.ndarray, y: np.ndarray, contexts: List[Dict]):
        """拟合模型参数"""
        def loss(params):
            self.alpha, self.lambda_param, self.gamma, self.temperature = params
            predictions = [self.predict(ctx) for ctx in contexts]
            mse = np.mean((np.array(predictions) - y) ** 2)
            return mse
        
        result = minimize(
            loss,
            x0=[0.88, 2.25, 0.61, 1.0],
            bounds=[(0.1, 1.0), (1.0, 5.0), (0.1, 1.0), (0.01, 10.0)]
        )
        self.alpha, self.lambda_param, self.gamma, self.temperature = result.x
        return self


# ==================== 层级 2: 神经网络模型 (Neural Models) ====================

class RiskDecisionNet(nn.Module):
    """风险决策神经网络"""
    
    def __init__(self, input_dim=17, hidden_dim=64, num_layers=2, dropout=0.3):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: 隐藏层数量
            dropout: Dropout 概率
        """
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())  # 输出 [0, 1]
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze()
    
    def get_intermediate_activations(self, x):
        """获取中间层激活（用于 RSA 分析）"""
        activations = []
        for layer in self.net[:-2]:  # 排除最后的 Linear 和 Sigmoid
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                activations.append(x.detach().cpu().numpy())
        return activations


class NeuralModel:
    """神经网络模型包装器"""
    
    def __init__(self, input_dim=17, hidden_dim=64, num_layers=2, dropout=0.3, 
                 device='cpu'):
        self.device = torch.device(device)
        self.model = RiskDecisionNet(input_dim, hidden_dim, num_layers, dropout).to(self.device)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs=100, batch_size=64, 
            lr=0.001, verbose=True):
        """
        训练模型
        
        Args:
            X: 特征矩阵 [n_samples, n_features]
            y: 目标向量 [n_samples]
            epochs: 训练轮数
            batch_size: 批次大小
            lr: 学习率
            verbose: 是否打印训练信息
        """
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 转换为 Tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # 训练
        self.model.train()
        for epoch in range(epochs):
            # 随机打乱
            indices = torch.randperm(len(X_tensor))
            X_shuffled = X_tensor[indices]
            y_shuffled = y_tensor[indices]
            
            epoch_loss = 0
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(X_tensor)*batch_size:.6f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用 fit()")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
    
    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """获取中间层表征（用于 RSA 分析）"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            activations = self.model.get_intermediate_activations(X_tensor)
            # 返回最后一层激活
            return activations[-1] if activations else None


# ==================== 层级 3: 大语言模型 (LLM Models) ====================

class LLMModel:
    """大语言模型包装器（需要根据实际使用的 LLM 调整）"""
    
    def __init__(self, model_name: str = "llama3-8b", use_few_shot: bool = True):
        """
        Args:
            model_name: 模型名称
            use_few_shot: 是否使用 Few-shot 学习
        """
        self.model_name = model_name
        self.use_few_shot = use_few_shot
        self.few_shot_examples = []
        self.is_fitted = False
    
    def add_few_shot_examples(self, examples: List[Dict]):
        """
        添加 Few-shot 示例
        
        Args:
            examples: 示例列表，每个包含 'prompt' 和 'output'
        """
        self.few_shot_examples = examples
    
    def generate_few_shot_prompt(self, prompt: str) -> str:
        """生成包含 Few-shot 示例的 Prompt"""
        if not self.few_shot_examples:
            return prompt
        
        few_shot_text = "以下是一些示例：\n\n"
        for i, example in enumerate(self.few_shot_examples, 1):
            few_shot_text += f"示例 {i}:\n"
            few_shot_text += f"{example['prompt']}\n"
            few_shot_text += f"答案: {example['output']:.3f}\n\n"
        
        return few_shot_text + f"现在请回答以下问题：\n\n{prompt}"
    
    def parse_output(self, response: str) -> float:
        """
        从 LLM 输出中解析概率值
        
        Args:
            response: LLM 的文本响应
            
        Returns:
            解析得到的概率值 [0, 1]
        """
        import re
        
        # 尝试提取 0-1 之间的浮点数
        numbers = re.findall(r'0?\.\d+', response)
        if numbers:
            prob = float(numbers[0])
            return np.clip(prob, 0.0, 1.0)
        
        # 尝试提取百分比
        percentages = re.findall(r'(\d+(?:\.\d+)?)%', response)
        if percentages:
            prob = float(percentages[0]) / 100.0
            return np.clip(prob, 0.0, 1.0)
        
        # 如果提到 "Gamble B" 或 "B"
        if "Gamble B" in response or "选择B" in response or "选项B" in response:
            return 1.0
        if "Gamble A" in response or "选择A" in response or "选项A" in response:
            return 0.0
        
        # 默认返回 0.5
        print(f"警告：无法解析输出 '{response[:100]}...'，返回默认值 0.5")
        return 0.5
    
    def predict_single(self, prompt: str) -> float:
        """
        预测单个问题的概率
        
        注意：这是一个占位实现，需要根据实际使用的 LLM API 调整
        
        Args:
            prompt: 输入 Prompt
            
        Returns:
            预测的概率值
        """
        # TODO: 实现实际的 LLM 调用
        # 这里是一个示例，需要根据实际使用的库（如 transformers, openai 等）调整
        
        if self.use_few_shot:
            full_prompt = self.generate_few_shot_prompt(prompt)
        else:
            full_prompt = prompt
        
        # 示例：使用 transformers 库
        # from transformers import AutoTokenizer, AutoModelForCausalLM
        # tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # model = AutoModelForCausalLM.from_pretrained(self.model_name)
        # inputs = tokenizer(full_prompt, return_tensors="pt")
        # outputs = model.generate(**inputs, max_length=200)
        # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 占位实现
        response = "0.5"  # 需要替换为实际 LLM 输出
        
        return self.parse_output(response)
    
    def predict(self, prompts: List[str]) -> np.ndarray:
        """
        批量预测
        
        Args:
            prompts: Prompt 列表
            
        Returns:
            预测的概率数组
        """
        predictions = []
        for prompt in prompts:
            pred = self.predict_single(prompt)
            predictions.append(pred)
        return np.array(predictions)
    
    def fit(self, prompts: List[str], y: np.ndarray, contexts: List[Dict] = None):
        """
        Fine-tuning（需要根据实际使用的 LLM 调整）
        
        Args:
            prompts: 训练 Prompt 列表
            y: 真实 bRate
            contexts: 上下文列表（可选）
        """
        # TODO: 实现 Fine-tuning
        # 对于 QLoRA 微调，需要准备训练数据并调用相应的训练脚本
        
        # 示例数据格式（用于 QLoRA）：
        # training_data = [
        #     {
        #         "instruction": "预测人类选择 Gamble B 的概率",
        #         "input": prompt,
        #         "output": str(bRate)
        #     }
        #     for prompt, bRate in zip(prompts, y)
        # ]
        
        self.is_fitted = True
        return self
    
    def get_embeddings(self, prompts: List[str]) -> np.ndarray:
        """
        获取 LLM 的内部表征（用于 RSA 分析）
        
        注意：需要根据实际使用的 LLM 调整
        
        Args:
            prompts: Prompt 列表
            
        Returns:
            表征矩阵 [n_samples, embedding_dim]
        """
        # TODO: 实现实际的 embedding 提取
        # 示例：使用 transformers 库
        # from transformers import AutoTokenizer, AutoModel
        # tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # model = AutoModel.from_pretrained(self.model_name)
        # 
        # embeddings = []
        # for prompt in prompts:
        #     inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        #     with torch.no_grad():
        #         outputs = model(**inputs)
        #         # 使用 [CLS] token 或平均池化
        #         embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        #         embeddings.append(embedding)
        # 
        # return np.array(embeddings)
        
        # 占位实现
        return np.random.randn(len(prompts), 768)


# ==================== 模型比较工具 ====================

def compare_models(models: Dict[str, object], X: np.ndarray, y: np.ndarray, 
                  contexts: List[Dict] = None, prompts: List[str] = None) -> Dict:
    """
    比较多个模型的性能
    
    Args:
        models: 模型字典 {name: model}
        X: 特征矩阵
        y: 真实值
        contexts: 上下文列表（符号模型需要）
        prompts: Prompt 列表（LLM 需要）
        
    Returns:
        比较结果字典
    """
    from sklearn.metrics import mean_squared_error, r2_score
    
    results = {}
    
    for name, model in models.items():
        if isinstance(model, (ExpectedUtilityModel, ProspectTheoryModel)):
            # 符号模型
            if contexts is None:
                raise ValueError(f"{name} 需要 contexts 参数")
            predictions = np.array([model.predict(ctx) for ctx in contexts])
            
        elif isinstance(model, NeuralModel):
            # 神经网络
            predictions = model.predict(X)
            
        elif isinstance(model, LLMModel):
            # LLM
            if prompts is None:
                raise ValueError(f"{name} 需要 prompts 参数")
            predictions = model.predict(prompts)
            
        else:
            raise ValueError(f"未知的模型类型: {type(model)}")
        
        # 计算指标
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        correlation = np.corrcoef(y, predictions)[0, 1]
        
        results[name] = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'correlation': correlation,
            'predictions': predictions
        }
    
    return results
