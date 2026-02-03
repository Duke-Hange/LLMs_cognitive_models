"""
增强版符号模型实现
支持多结果分布，直接从JSON原始分布数据计算价值

所有模型都接受分布列表作为输入，输出选择 Gamble B 的概率 (bRate)
分布格式: [[概率1, 结果1], [概率2, 结果2], ...]
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Callable, Union
import warnings
warnings.filterwarnings('ignore')


class EnhancedSymbolicModel:
    """增强符号模型基类 - 支持多结果分布"""
    
    def __init__(self):
        self.parameters = {}
        self.is_fitted = False
    
    def predict_from_distributions(self, gamble_a_dist: List[List[float]], 
                                   gamble_b_dist: List[List[float]]) -> float:
        """
        从分布直接预测选择 Gamble B 的概率
        
        Args:
            gamble_a_dist: Gamble A的分布 [[p1, x1], [p2, x2], ...]
            gamble_b_dist: Gamble B的分布
            
        Returns:
            选择 Gamble B 的概率
        """
        value_a, value_b = self.compute_gamble_values_from_distributions(
            gamble_a_dist, gamble_b_dist
        )
        value_diff = value_b - value_a
        return self.value_to_probability(value_diff)
    
    def predict_batch(self, distributions_a: List[List[List[float]]], 
                     distributions_b: List[List[List[float]]]) -> np.ndarray:
        """
        批量预测
        
        Args:
            distributions_a: Gamble A分布列表
            distributions_b: Gamble B分布列表
            
        Returns:
            选择概率数组
        """
        n_samples = len(distributions_a)
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            predictions[i] = self.predict_from_distributions(
                distributions_a[i], distributions_b[i]
            )
        
        return predictions
    
    def compute_gamble_values_from_distributions(self, 
        gamble_a_dist: List[List[float]], 
        gamble_b_dist: List[List[float]]
    ) -> Tuple[float, float]:
        """
        从分布计算赌博价值（由子类实现）
        
        Args:
            gamble_a_dist: Gamble A分布
            gamble_b_dist: Gamble B分布
            
        Returns:
            (value_a, value_b): Gamble A 和 Gamble B 的价值
        """
        raise NotImplementedError
    
    def value_to_probability(self, value_diff: float) -> float:
        """
        将价值差异转换为选择概率（使用 softmax/logistic 函数）
        
        Args:
            value_diff: 价值差异 (value_b - value_a)
            
        Returns:
            选择 Gamble B 的概率
        """
        # 使用 temperature 参数控制随机性
        temperature = self.parameters.get('temperature', 1.0)
        
        # 避免数值溢出
        z = value_diff / max(temperature, 1e-10)
        z = np.clip(z, -50, 50)  # 防止溢出
        
        return 1.0 / (1.0 + np.exp(-z))
    
    def fit_from_distributions(self, 
        distributions_a: List[List[List[float]]], 
        distributions_b: List[List[List[float]]], 
        y: np.ndarray
    ):
        """
        从分布数据拟合模型参数
        
        Args:
            distributions_a: Gamble A分布列表
            distributions_b: Gamble B分布列表  
            y: 真实 bRate
            
        Returns:
            self
        """
        raise NotImplementedError
    
    def get_parameters(self) -> Dict:
        """获取模型参数"""
        return self.parameters.copy()
    
    def set_parameters(self, params: Dict):
        """设置模型参数"""
        self.parameters.update(params)


class EnhancedExpectedValueModel(EnhancedSymbolicModel):
    """增强期望值模型 (Enhanced Expected Value Model)"""
    
    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: softmax 温度参数，控制决策随机性
        """
        super().__init__()
        self.parameters = {
            'temperature': temperature,
            'model_name': 'Enhanced Expected Value'
        }
    
    def compute_gamble_values_from_distributions(self, 
        gamble_a_dist: List[List[float]], 
        gamble_b_dist: List[List[float]]
    ) -> Tuple[float, float]:
        """
        计算多结果分布的期望值
        
        Args:
            gamble_a_dist: Gamble A分布
            gamble_b_dist: Gamble B分布
            
        Returns:
            (ev_a, ev_b): Gamble A 和 Gamble B 的期望值
        """
        # 计算 Gamble A 的期望值
        ev_a = sum(p * x for p, x in gamble_a_dist)
        
        # 计算 Gamble B 的期望值
        ev_b = sum(p * x for p, x in gamble_b_dist)
        
        return ev_a, ev_b
    
    def fit_from_distributions(self, 
        distributions_a: List[List[List[float]]], 
        distributions_b: List[List[List[float]]], 
        y: np.ndarray
    ):
        """拟合温度参数"""
        
        def loss(params):
            self.parameters['temperature'] = params[0]
            
            # 计算预测
            predictions = np.zeros(len(y))
            for i in range(len(y)):
                predictions[i] = self.predict_from_distributions(
                    distributions_a[i], distributions_b[i]
                )
            
            mse = np.mean((predictions - y) ** 2)
            return mse
        
        # 记录迭代曲线（单次训练曲线）
        history = []
        def callback(xk):
            history.append({"iteration": len(history) + 1, "train_loss": float(loss(xk))})
        result = minimize(
            loss,
            x0=[self.parameters['temperature']],
            bounds=[(0.01, 10.0)],
            method='L-BFGS-B',
            callback=callback
        )
        
        self.parameters['temperature'] = result.x[0]
        self.is_fitted = True
        self.fit_history = history
        return self


class EnhancedExpectedUtilityModel(EnhancedSymbolicModel):
    """增强期望效用模型 (Enhanced Expected Utility Model)"""
    
    def __init__(self, temperature: float = 1.0, alpha: float = 1.0):
        """
        Args:
            temperature: softmax 温度参数
            alpha: 效用函数参数，u(x) = sign(x) * |x|^alpha (alpha=1 时为线性)
        """
        super().__init__()
        self.parameters = {
            'temperature': temperature,
            'alpha': alpha,
            'model_name': 'Enhanced Expected Utility'
        }
    
    def utility_function(self, x: float) -> float:
        """效用函数: u(x) = sign(x) * |x|^alpha"""
        alpha = self.parameters['alpha']
        
        if abs(x) < 1e-10:
            return 0.0
        
        sign = 1.0 if x >= 0 else -1.0
        return sign * (abs(x) ** alpha)
    
    def compute_gamble_values_from_distributions(self, 
        gamble_a_dist: List[List[float]], 
        gamble_b_dist: List[List[float]]
    ) -> Tuple[float, float]:
        """
        计算多结果分布的期望效用
        
        Args:
            gamble_a_dist: Gamble A分布
            gamble_b_dist: Gamble B分布
            
        Returns:
            (eu_a, eu_b): Gamble A 和 Gamble B 的期望效用
        """
        # 计算 Gamble A 的期望效用
        eu_a = sum(p * self.utility_function(x) for p, x in gamble_a_dist)
        
        # 计算 Gamble B 的期望效用
        eu_b = sum(p * self.utility_function(x) for p, x in gamble_b_dist)
        
        return eu_a, eu_b
    
    def fit_from_distributions(self, 
        distributions_a: List[List[List[float]]], 
        distributions_b: List[List[List[float]]], 
        y: np.ndarray
    ):
        """拟合 alpha 和 temperature 参数"""
        
        def loss(params):
            self.parameters['alpha'] = params[0]
            self.parameters['temperature'] = params[1]
            
            # 计算预测
            predictions = np.zeros(len(y))
            for i in range(len(y)):
                predictions[i] = self.predict_from_distributions(
                    distributions_a[i], distributions_b[i]
                )
            
            mse = np.mean((predictions - y) ** 2)
            return mse
        
        # 记录迭代曲线
        history = []
        def callback(xk):
            history.append({"iteration": len(history) + 1, "train_loss": float(loss(xk))})
        result = minimize(
            loss,
            x0=[self.parameters['alpha'], self.parameters['temperature']],
            bounds=[(0.1, 2.0), (0.01, 10.0)],
            method='L-BFGS-B',
            callback=callback
        )
        
        self.parameters['alpha'] = result.x[0]
        self.parameters['temperature'] = result.x[1]
        self.is_fitted = True
        self.fit_history = history
        return self


class EnhancedProspectTheory3PModel(EnhancedSymbolicModel):
    """增强前景理论 3 参数模型 (Enhanced Prospect Theory 3-Parameter Model)"""
    
    def __init__(self, alpha: float = 0.88, lambda_param: float = 2.25, 
                 gamma: float = 0.61, temperature: float = 1.0):
        """
        Args:
            alpha: 价值函数风险厌恶系数
            lambda_param: 损失厌恶系数
            gamma: 概率权重函数参数
            temperature: softmax 温度参数
        """
        super().__init__()
        self.parameters = {
            'alpha': alpha,
            'lambda': lambda_param,
            'gamma': gamma,
            'temperature': temperature,
            'model_name': 'Enhanced Prospect Theory (3P)'
        }
    
    def value_function(self, x: float) -> float:
        """价值函数: v(x) = x^alpha if x >= 0, -lambda * (-x)^alpha if x < 0"""
        alpha = self.parameters['alpha']
        lambda_param = self.parameters['lambda']
        
        if x >= 0:
            return x ** alpha
        else:
            return -lambda_param * ((-x) ** alpha)
    
    def probability_weighting(self, p: float) -> float:
        """概率权重函数: w(p) = p^gamma / (p^gamma + (1-p)^gamma)^(1/gamma)"""
        gamma = self.parameters['gamma']
        
        # 处理边界情况
        if p == 0:
            return 0.0
        if p == 1:
            return 1.0
        
        numerator = p ** gamma
        denominator = (p ** gamma + (1 - p) ** gamma) ** (1 / gamma)
        return numerator / denominator
    
    def compute_gamble_values_from_distributions(self, 
        gamble_a_dist: List[List[float]], 
        gamble_b_dist: List[List[float]]
    ) -> Tuple[float, float]:
        """
        计算多结果分布的前景理论价值
        
        Args:
            gamble_a_dist: Gamble A分布
            gamble_b_dist: Gamble B分布
            
        Returns:
            (pt_a, pt_b): Gamble A 和 Gamble B 的前景理论价值
        """
        # 对每个分布计算前景理论价值
        pt_a = sum(self.probability_weighting(p) * self.value_function(x) 
                  for p, x in gamble_a_dist)
        pt_b = sum(self.probability_weighting(p) * self.value_function(x) 
                  for p, x in gamble_b_dist)
        
        return pt_a, pt_b
    
    def fit_from_distributions(self, 
        distributions_a: List[List[List[float]]], 
        distributions_b: List[List[List[float]]], 
        y: np.ndarray
    ):
        """拟合所有参数"""
        
        def loss(params):
            self.parameters['alpha'] = params[0]
            self.parameters['lambda'] = params[1]
            self.parameters['gamma'] = params[2]
            self.parameters['temperature'] = params[3]
            
            # 计算预测
            predictions = np.zeros(len(y))
            for i in range(len(y)):
                predictions[i] = self.predict_from_distributions(
                    distributions_a[i], distributions_b[i]
                )
            
            mse = np.mean((predictions - y) ** 2)
            return mse
        
        # 记录迭代曲线
        history = []
        def callback(xk):
            history.append({"iteration": len(history) + 1, "train_loss": float(loss(xk))})
        result = minimize(
            loss,
            x0=[
                self.parameters['alpha'],
                self.parameters['lambda'],
                self.parameters['gamma'],
                self.parameters['temperature']
            ],
            bounds=[
                (0.1, 1.0),      # alpha
                (1.0, 5.0),      # lambda
                (0.1, 1.0),      # gamma
                (0.01, 10.0)     # temperature
            ],
            method='L-BFGS-B',
            callback=callback
        )
        
        self.parameters['alpha'] = result.x[0]
        self.parameters['lambda'] = result.x[1]
        self.parameters['gamma'] = result.x[2]
        self.parameters['temperature'] = result.x[3]
        self.is_fitted = True
        self.fit_history = history
        return self


class EnhancedProspectTheory5PModel(EnhancedProspectTheory3PModel):
    """增强前景理论 5 参数模型 (Enhanced Prospect Theory 5-Parameter Model)"""
    
    def __init__(self, alpha_gain: float = 0.88, alpha_loss: float = 0.88,
                 lambda_param: float = 2.25, gamma_gain: float = 0.61,
                 gamma_loss: float = 0.69, temperature: float = 1.0):
        """
        Args:
            alpha_gain: 收益的风险厌恶系数
            alpha_loss: 损失的风险厌恶系数
            lambda_param: 损失厌恶系数
            gamma_gain: 收益的概率权重参数
            gamma_loss: 损失的概率权重参数
            temperature: softmax 温度参数
        """
        super().__init__(alpha=alpha_gain, lambda_param=lambda_param, 
                        gamma=gamma_gain, temperature=temperature)
        
        # 覆盖参数名称
        self.parameters = {
            'alpha_gain': alpha_gain,
            'alpha_loss': alpha_loss,
            'lambda': lambda_param,
            'gamma_gain': gamma_gain,
            'gamma_loss': gamma_loss,
            'temperature': temperature,
            'model_name': 'Enhanced Prospect Theory (5P)'
        }
    
    def value_function(self, x: float) -> float:
        """价值函数: 收益和损失使用不同的 alpha 参数"""
        alpha_gain = self.parameters['alpha_gain']
        alpha_loss = self.parameters['alpha_loss']
        lambda_param = self.parameters['lambda']
        
        if x >= 0:
            return x ** alpha_gain
        else:
            return -lambda_param * ((-x) ** alpha_loss)
    
    def probability_weighting_gain(self, p: float) -> float:
        """收益的概率权重函数"""
        gamma = self.parameters['gamma_gain']
        
        if p == 0:
            return 0.0
        if p == 1:
            return 1.0
        
        numerator = p ** gamma
        denominator = (p ** gamma + (1 - p) ** gamma) ** (1 / gamma)
        return numerator / denominator
    
    def probability_weighting_loss(self, p: float) -> float:
        """损失的概率权重函数"""
        gamma = self.parameters['gamma_loss']
        
        if p == 0:
            return 0.0
        if p == 1:
            return 1.0
        
        numerator = p ** gamma
        denominator = (p ** gamma + (1 - p) ** gamma) ** (1 / gamma)
        return numerator / denominator
    
    def compute_gamble_values_from_distributions(self, 
        gamble_a_dist: List[List[float]], 
        gamble_b_dist: List[List[float]]
    ) -> Tuple[float, float]:
        """
        计算多结果分布的前景理论价值（5 参数版本）
        
        Args:
            gamble_a_dist: Gamble A分布
            gamble_b_dist: Gamble B分布
            
        Returns:
            (pt_a, pt_b): Gamble A 和 Gamble B 的前景理论价值
        """
        # 对于每个结果，根据结果是收益还是损失使用不同的概率权重函数
        pt_a = 0.0
        for p, x in gamble_a_dist:
            if x >= 0:
                pt_a += self.probability_weighting_gain(p) * self.value_function(x)
            else:
                pt_a += self.probability_weighting_loss(p) * self.value_function(x)
        
        pt_b = 0.0
        for p, x in gamble_b_dist:
            if x >= 0:
                pt_b += self.probability_weighting_gain(p) * self.value_function(x)
            else:
                pt_b += self.probability_weighting_loss(p) * self.value_function(x)
        
        return pt_a, pt_b
    
    def fit_from_distributions(self, 
        distributions_a: List[List[List[float]]], 
        distributions_b: List[List[List[float]]], 
        y: np.ndarray
    ):
        """拟合所有 5 个参数"""
        
        def loss(params):
            self.parameters['alpha_gain'] = params[0]
            self.parameters['alpha_loss'] = params[1]
            self.parameters['lambda'] = params[2]
            self.parameters['gamma_gain'] = params[3]
            self.parameters['gamma_loss'] = params[4]
            self.parameters['temperature'] = params[5]
            
            # 计算预测
            predictions = np.zeros(len(y))
            for i in range(len(y)):
                predictions[i] = self.predict_from_distributions(
                    distributions_a[i], distributions_b[i]
                )
            
            mse = np.mean((predictions - y) ** 2)
            return mse
        
        # 记录迭代曲线
        history = []
        def callback(xk):
            history.append({"iteration": len(history) + 1, "train_loss": float(loss(xk))})
        result = minimize(
            loss,
            x0=[
                self.parameters['alpha_gain'],
                self.parameters['alpha_loss'],
                self.parameters['lambda'],
                self.parameters['gamma_gain'],
                self.parameters['gamma_loss'],
                self.parameters['temperature']
            ],
            bounds=[
                (0.1, 1.0),      # alpha_gain
                (0.1, 1.0),      # alpha_loss
                (1.0, 5.0),      # lambda
                (0.1, 1.0),      # gamma_gain
                (0.1, 1.0),      # gamma_loss
                (0.01, 10.0)     # temperature
            ],
            method='L-BFGS-B',
            callback=callback
        )
        
        self.parameters['alpha_gain'] = result.x[0]
        self.parameters['alpha_loss'] = result.x[1]
        self.parameters['lambda'] = result.x[2]
        self.parameters['gamma_gain'] = result.x[3]
        self.parameters['gamma_loss'] = result.x[4]
        self.parameters['temperature'] = result.x[5]
        self.is_fitted = True
        self.fit_history = history
        return self


# 与标准化数据集的适配器类
class EnhancedModelAdapter:
    """增强模型适配器 - 连接标准化数据与增强模型"""
    
    def __init__(self, model: EnhancedSymbolicModel):
        self.model = model
    
    def predict_from_standardized(self, standardized_data: List[Dict]) -> np.ndarray:
        """
        从标准化数据预测
        
        Args:
            standardized_data: 标准化数据列表
            
        Returns:
            预测的 bRate 数组
        """
        distributions_a = []
        distributions_b = []
        
        for item in standardized_data:
            distributions_a.append(item['context']['gamble_a']['distribution'])
            distributions_b.append(item['context']['gamble_b']['distribution'])
        
        return self.model.predict_batch(distributions_a, distributions_b)
    
    def fit_from_standardized(self, 
        standardized_data: List[Dict], 
        y: np.ndarray
    ):
        """
        从标准化数据拟合
        
        Args:
            standardized_data: 标准化数据列表
            y: 真实 bRate
            
        Returns:
            self
        """
        distributions_a = []
        distributions_b = []
        
        for item in standardized_data:
            distributions_a.append(item['context']['gamble_a']['distribution'])
            distributions_b.append(item['context']['gamble_b']['distribution'])
        
        self.model.fit_from_distributions(distributions_a, distributions_b, y)
        return self
    
    def get_model(self) -> EnhancedSymbolicModel:
        """获取底层模型"""
        return self.model


# 模型工厂函数
def create_enhanced_model(model_name: str, **kwargs) -> EnhancedSymbolicModel:
    """
    创建增强符号模型
    
    Args:
        model_name: 模型名称 ('ev', 'eu', 'pt3', 'pt5')
        **kwargs: 模型参数
    
    Returns:
        增强符号模型实例
    """
    model_map = {
        'ev': EnhancedExpectedValueModel,
        'eu': EnhancedExpectedUtilityModel,
        'pt3': EnhancedProspectTheory3PModel,
        'pt5': EnhancedProspectTheory5PModel
    }
    
    if model_name not in model_map:
        raise ValueError(f"未知模型: {model_name}。可用模型: {list(model_map.keys())}")
    
    return model_map[model_name](**kwargs)


def test_enhanced_models():
    """测试增强模型"""
    print("测试增强符号模型...")
    
    # 创建测试分布数据
    np.random.seed(42)
    
    # 测试数据1: 简单二结果分布
    gamble_a_simple = [[0.7, 10.0], [0.3, -5.0]]
    gamble_b_simple = [[0.5, 15.0], [0.5, -10.0]]
    
    # 测试数据2: 多结果分布
    gamble_a_multi = [[0.4, 5.0], [0.3, 10.0], [0.2, 15.0], [0.1, 20.0]]
    gamble_b_multi = [[0.25, -5.0], [0.25, 0.0], [0.25, 5.0], [0.25, 10.0]]
    
    # 测试每个模型
    models = {
        'Enhanced Expected Value': EnhancedExpectedValueModel(),
        'Enhanced Expected Utility': EnhancedExpectedUtilityModel(),
        'Enhanced Prospect Theory (3P)': EnhancedProspectTheory3PModel(),
        'Enhanced Prospect Theory (5P)': EnhancedProspectTheory5PModel()
    }
    
    print("\n1. 测试简单二结果分布:")
    for name, model in models.items():
        prob = model.predict_from_distributions(gamble_a_simple, gamble_b_simple)
        ev_a, ev_b = model.compute_gamble_values_from_distributions(
            gamble_a_simple, gamble_b_simple
        )
        print(f"   {name}:")
        print(f"     价值: A={ev_a:.4f}, B={ev_b:.4f}")
        print(f"     选择B概率: {prob:.4f}")
    
    print("\n2. 测试多结果分布:")
    for name, model in models.items():
        prob = model.predict_from_distributions(gamble_a_multi, gamble_b_multi)
        ev_a, ev_b = model.compute_gamble_values_from_distributions(
            gamble_a_multi, gamble_b_multi
        )
        print(f"   {name}:")
        print(f"     价值: A={ev_a:.4f}, B={ev_b:.4f}")
        print(f"     选择B概率: {prob:.4f}")
    
    # 测试批量预测
    print("\n3. 测试批量预测:")
    distributions_a = [gamble_a_simple, gamble_a_multi]
    distributions_b = [gamble_b_simple, gamble_b_multi]
    
    for name, model in models.items():
        predictions = model.predict_batch(distributions_a, distributions_b)
        print(f"   {name}: {predictions}")
    
    print("\n增强模型测试完成!")


if __name__ == "__main__":
    test_enhanced_models()