import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class GeneralPowerUtility(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.88))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.lambda_ = nn.Parameter(torch.tensor(2.25))
        self.gamma = nn.Parameter(torch.tensor(0.88))
        self.delta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        results = torch.zeros_like(x)
        gains_mask = x >= 0
        losses_mask = x < 0

        results[gains_mask] = self.beta * torch.pow(x[gains_mask] + 1e-8, self.alpha)
        results[losses_mask] = -self.lambda_ * torch.pow(-self.delta * x[losses_mask], self.gamma)

        return results

class KTWeighting(nn.Module):
    def __init__(self):
        super().__init__()
        self.kt_gamma = nn.Parameter(torch.tensor(0.65))

    def forward(self, p):
        numerator = torch.pow(p, self.kt_gamma)
        denominator = torch.pow(
            torch.pow(p, self.kt_gamma) + torch.pow(1 - p, self.kt_gamma),
            1 / self.kt_gamma
        )
        return numerator / (denominator + 1e-9) # 防止除零

class MixtureNetwork(nn.Module):
    def __init__(self, input_dim=36, hidden_dim=20, num_experts=2):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.utility_head = nn.Linear(hidden_dim, num_experts)
        self.prob_head = nn.Linear(hidden_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        shared_features = self.shared_layers(x)
        utility_weights = self.softmax(self.utility_head(shared_features))
        prob_weights = self.softmax(self.prob_head(shared_features))
        return utility_weights, prob_weights

class MoT(nn.Module):
    def __init__(self, input_dim=36, hidden_dim=20, num_experts=2):
        super().__init__()
        # 初始化专家模块：两个价值函数和两个概率权重函数
        self.experts_uf = nn.ModuleList([GeneralPowerUtility() for _ in range(num_experts)])
        self.experts_pf = nn.ModuleList([KTWeighting() for _ in range(num_experts)])

        # 初始化混合网络
        self.mixture_network = MixtureNetwork(input_dim=input_dim, hidden_dim=hidden_dim, num_experts=num_experts)

        # 其他模型参数
        self.eta = nn.Parameter(torch.tensor(1.0))
        self.p_dominant = nn.Parameter(torch.tensor(-2.0))

    @staticmethod
    def is_dominance(outcomes_batch):
        min_a, max_a = outcomes_batch[:, 0, :].min(dim=1).values, outcomes_batch[:, 0, :].max(dim=1).values
        min_b, max_b = outcomes_batch[:, 1, :].min(dim=1).values, outcomes_batch[:, 1, :].max(dim=1).values
        return (min_a > max_b) | (min_b > max_a)

    def calculate_subjective_value(self, outcomes, probabilities, u_weights, p_weights):
        u_vals = torch.stack([uf(outcomes) for uf in self.experts_uf])
        p_vals = torch.stack([pf(probabilities) for pf in self.experts_pf])

        # 调整权重维度以进行广播
        # (num_experts, batch_size, 1)
        u_weights_exp = u_weights.T.unsqueeze(-1)
        p_weights_exp = p_weights.T.unsqueeze(-1)

        # 计算加权价值
        weighted_u = u_vals * u_weights_exp
        weighted_p = p_vals * p_weights_exp

        # 主观价值 = sum(w_u * u(x) * w_p * p(x))
        mixed_values = (weighted_u * weighted_p).sum(dim=0) # 在专家维度上求和
        return mixed_values.sum(dim=-1) # 在结果维度上求和

    def forward(self, X):
        x_a = X[:, 0, :]
        p_a = X[:, 1, :]
        x_b = X[:, 2, :]
        p_b = X[:, 3, :]

        outcomes = torch.stack([x_a, x_b], dim=1)
        probabilities = torch.stack([p_a, p_b], dim=1)


        input_vector = torch.cat([x_a, p_a, x_b, p_b], dim=1)

        # 获取混合权重
        u_weights_batch, p_weights_batch = self.mixture_network(input_vector)

        # 处理显性优势情况
        
        dominance_mask = self.is_dominance(outcomes)
        pred_dominant = torch.sigmoid(self.p_dominant)
        # 分别计算两个选项的主观价值
        v_a = self.calculate_subjective_value(x_a, p_a, u_weights_batch, p_weights_batch)
        v_b = self.calculate_subjective_value(x_b, p_b, u_weights_batch, p_weights_batch)

        # 计算常规情况下的选择概率
        diff = self.eta * (v_a - v_b)
        pred_normal = torch.sigmoid(diff)

        # 结合显性优势和常规情况，得到最终预测
        predictions = torch.where(dominance_mask, pred_dominant, pred_normal)

        return predictions


class ContextNN(nn.Module):
    def __init__(self, input_dim=36, hidden_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        x_a = X[:, 0, :]
        p_a = X[:, 1, :]
        x_b = X[:, 2, :]
        p_b = X[:, 3, :]

        input_vector = torch.cat([x_a, p_a, x_b, p_b], dim=1)
        y_pred = self.network(input_vector)
        return y_pred.squeeze(-1)
        