"""
Value-Based 与 Context-Dependent 神经网络
与 Peterson et al. (2021) 参考一致：Sigmoid 隐藏层，完整分布编码输入。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class ValueBasedNet(nn.Module):
    """
    基于价值的模型：V(A)=f(enc_A), V(B)=f(enc_B)，P(B)=softmax(η*V(A), η*V(B)) 取 B 的概率。
    共享子网络 f：1 隐藏层 64 单元 Sigmoid，输入为单赌局分布编码 (2*max_outcomes)。
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

    def forward(
        self,
        enc_A: torch.Tensor,
        enc_B: torch.Tensor,
    ) -> torch.Tensor:
        # enc_A, enc_B: (batch, input_dim_per_gamble)
        v_A = self.f(enc_A).squeeze(-1)  # (batch,)
        v_B = self.f(enc_B).squeeze(-1)  # (batch,)
        eta = torch.clamp(self.eta, min=1e-6, max=1e2)
        logits = eta * torch.stack([v_A, v_B], dim=1)  # (batch, 2)
        p_B = torch.softmax(logits, dim=1)[:, 1]  # P(choose B)
        return p_B


class ContextDependentNet(nn.Module):
    """
    上下文依赖模型：单网络 g(enc_A | enc_B) 直接输出 P(B)=bRate。
    2 隐藏层各 32 单元 Sigmoid，输入 4*max_outcomes 维。
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
