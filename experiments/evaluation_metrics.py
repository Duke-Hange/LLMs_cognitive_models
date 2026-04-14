"""
共享的模型评估指标，提供统一的计算接口给神经网络和符号模型
"""
import numpy as np
from typing import Tuple
from model_base import compute_mse, compute_r2, compute_correlation, compute_cross_entropy


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    对模型预测进行全面评估，返回统一指标字典
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        包含多个评估指标的字典
    """
    # 确保输入是 numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 验证长度匹配
    if len(y_true) != len(y_pred):
        raise ValueError(f"长度不匹配: y_true({len(y_true)}) vs y_pred({len(y_pred)})")
    
    # 预测值裁剪以避免数值不稳定
    y_pred = np.clip(y_pred, 1e-15, 1-1e-15)  # 防止 log(0) 
    
    # 计算各种评估指标
    mse = compute_mse(y_true, y_pred)
    r2_score = compute_r2(y_true, y_pred)
    corr = compute_correlation(y_true, y_pred)
    ce_score = compute_cross_entropy(y_true, y_pred)
    
    return {
        'mse': mse,
        'r2': r2_score,
        'correlation': corr,
        'cross_entropy': ce_score,
        'rmse': np.sqrt(mse),
        'mae': np.mean(np.abs(y_true - y_pred))
    }


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算均方根误差"""
    return np.sqrt(compute_mse(y_true, y_pred))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算平均绝对误差"""
    return np.mean(np.abs(y_true - y_pred))


def compute_accuracy_with_tolerance(y_true: np.ndarray, y_pred: np.ndarray, tolerance: float = 0.1) -> float:
    """计算在容忍度内的准确率"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    within_tolerance = np.abs(y_true - y_pred) <= tolerance
    return np.mean(within_tolerance)