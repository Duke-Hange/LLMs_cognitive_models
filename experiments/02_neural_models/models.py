"""
Value-Based 与 Context-Dependent 神经网络
与 Peterson et al. (2021) 参考一致：Sigmoid 隐藏层，完整分布编码输入。

这些模型是神经网络类，但我们将另外创建封装类来实现ModelBase接口
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class ValueBasedNet(nn.Module):
    """
    基于价值的模型：V(A)=f(enc_A), V(B)=f(enc_B)，经 softmax 得到 [0,1] 标量输出。
    共享子网络 f：1 隐藏层 64 单元 Sigmoid；训练目标由 `TARGET_MODE` 决定（CSV 轨默认 **1−bRate**）。
    """

    def __init__(self, input_dim_per_gamble: int = 18, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim_per_gamble
        self.hidden_dim = hidden_dim
        self.f = nn.Sequential(
            nn.Linear(input_dim_per_gamble, hidden_dim),
            nn.Sigmoid(), 
            nn.Linear(hidden_dim, 1),  
        )
        self.eta = nn.Parameter(torch.tensor(1.0))

    def forward(self, enc_A: torch.Tensor, enc_B: torch.Tensor) -> torch.Tensor:
        # enc_A, enc_B: (batch, input_dim_per_gamble)
        v_A = self.f(enc_A).squeeze(-1)  # (batch,)
        v_B = self.f(enc_B).squeeze(-1)  # (batch,)
        eta = torch.clamp(self.eta, min=1e-6, max=1e2)
        logits = eta * torch.stack([v_A, v_B], dim=1)  # (batch, 2)
        p_B = torch.softmax(logits, dim=1)[:, 1]  # 与损失中的 y 同空间（CSV 轨默认 y=1−bRate 时即对该标量建模）
        return p_B


class ContextDependentNet(nn.Module):
    """
    上下文依赖模型：单网络 g(enc_full) 经 Sigmoid 输出 [0,1] 标量。
    在 `data_loader.TARGET_MODE="one_minus_bRate"` 时，该输出与监督目标 **y=1−bRate** 对齐（而非在损失中直接拟合 bRate）。
    2 隐藏层各 32 单元，隐藏层使用 ReLU。输入 4*max_outcomes 维。
    """

    def __init__(self, input_dim: int = 36, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),   ##
        )

    def forward(self, enc_full: torch.Tensor) -> torch.Tensor:
        return self.net(enc_full).squeeze(-1)


class ContextDependentNetSigmoid(nn.Module):
    """
    与 Peterson et al. (2021) 更严格对齐的上下文依赖模型（隐藏层 Sigmoid）：
    输出为 [0,1]；训练时与 `TARGET_MODE` 一致（CSV 轨默认拟合 **1−bRate**）。
    输入为拼接编码 enc_full ∈ R^{4*max_outcomes}。
    """

    def __init__(self, input_dim: int = 36, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, enc_full: torch.Tensor) -> torch.Tensor:
        return self.net(enc_full).squeeze(-1)


def get_encoding_dims(max_outcomes: int = 9) -> Tuple[int, int]:
    """返回 (单赌局编码维, 拼接编码维)"""
    per_gamble = 2 * max_outcomes
    full = 4 * max_outcomes
    return per_gamble, full


# 新增兼容ModelBase接口的封装器
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_BASE_PATH = PROJECT_ROOT
sys.path.append(str(MODEL_BASE_PATH))

from model_base import ModelBase


class ValueBasedNetWrapper(ModelBase):
    """ValueBasedNet的ModelBase封装器"""
    
    def __init__(self, input_dim_per_gamble: int = 18, hidden_dim: int = 64):
        self.model = ValueBasedNet(input_dim_per_gamble, hidden_dim)
        
    def fit(self, X_train, y_train):
        """训练由外部完成，此处仅作占位符"""
        pass
        
    def predict(self, X):
        """预测接口"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, tuple) and len(X) == 2:
                # X = (encoded_A_arrays, encoded_B_arrays)
                enc_A_np, enc_B_np = X
                enc_A_tensor = torch.FloatTensor(enc_A_np)
                enc_B_tensor = torch.FloatTensor(enc_B_np)
                predictions = self.model(enc_A_tensor, enc_B_tensor)
                return predictions.numpy()
            else:
                raise ValueError("ValueBasedNetWrapper需要输入为(enc_A, enc_B)的元组形式")


class ContextDependentNetWrapper(ModelBase):
    """ContextDependentNet的ModelBase封装器"""
    
    def __init__(self, input_dim: int = 36, hidden_dim: int = 32):
        self.model = ContextDependentNet(input_dim, hidden_dim)
        
    def fit(self, X_train, y_train):
        """训练由外部完成，此处仅作占位符"""
        pass
        
    def predict(self, X):
        """预测接口"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.FloatTensor(X)
            else:
                X_tensor = torch.FloatTensor(np.array(X))
            predictions = self.model(X_tensor)
            return predictions.numpy()


class ContextDependentNetSigmoidWrapper(ModelBase):
    """ContextDependentNetSigmoid的ModelBase封装器"""
    
    def __init__(self, input_dim: int = 36, hidden_dim: int = 32):
        self.model = ContextDependentNetSigmoid(input_dim, hidden_dim)
        
    def fit(self, X_train, y_train):
        """训练由外部完成，此处仅作占位符"""
        pass
        
    def predict(self, X):
        """预测接口"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.FloatTensor(X)
            else:
                X_tensor = torch.FloatTensor(np.array(X))
            predictions = self.model(X_tensor)
            return predictions.numpy()