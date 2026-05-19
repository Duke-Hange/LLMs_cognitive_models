"""
实验项目专用的可视化工具函数
适用于模型性能比较、学习曲线等多种可视化场景
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Union, Dict, Optional
import os
from . import setup_chinese_font, apply_modern_theme, get_model_color, get_metric_color


# 启用中文显示和现代主题
setup_chinese_font()
apply_modern_theme()


def plot_model_performance_comparison(
    results_dict: Dict[str, Dict[str, float]],
    metric_names: List[str],
    title: str = "Model Performance Comparison",
    subtitle: str = "",
    save_path: Optional[str] = None
):
    """
    绘制模型性能比较图

    Args:
        results_dict: 模型名称 -> 性能指标字典的映射 {model_name: {metric1: value1, ...}}
        metric_names: 要比较的指标列表
        title: 主标题
        subtitle: 副标题
        save_path: 保存路径
    
    Returns:
        fig, ax
    """
    # 准备数据
    models = list(results_dict.keys())
    data_for_plot = []
    
    for model in models:
        row = [results_dict[model].get(metric, 0) for metric in metric_names]
        data_for_plot.append(row)
    
    data = np.array(data_for_plot)
    
    # 创建子图
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 为每个评估指标使用不同宽度和颜色的棒图
    bar_width = 0.6 / len(metric_names)  # 每个指标的条形图宽度
    index = np.arange(len(models))
    
    opacity = 0.8
    bars = []
    
    for i, metric in enumerate(metric_names):
        offset = (i - len(metric_names)/2 + 0.5) * bar_width
        color = get_metric_color(metric)
        
        bars.append(ax.bar(
            index + offset,
            data[:, i],
            bar_width,
            alpha=opacity,
            color=color,
            label=metric.replace('_', ' ').title(),
            edgecolor='black',
            linewidth=0.8
        ))
    
    # 标签和标题
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    
    # 设置x轴标签
    ax.set_xticks(index)
    ax.set_xticklabels(models, rotation=45, ha='right')
    
    # 添加图例
    ax.legend(loc='best', ncol=len(metric_names))
    
    # 添加标题
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=14, fontweight='bold', pad=20)
    else:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_cross_entropy_comparison(
    results_data: Dict[str, Union[List[float], np.ndarray]],
    labels: List[str],
    title: str = "Cross-Entropy Loss Comparison",
    xlabel: str = "Data Fraction",
    ylabel: str = "Cross-Entropy Loss",
    save_path: Optional[str] = None,
    semilogy_scale: bool = True
):
    """
    绘制交叉熵损失比较图

    Args:
        results_data: 结果数据字典 {model_name: [cross_entropy_values]}
        labels: 模型标签列表
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        save_path: 保存路径
        semilogy_scale: 是否使用对数纵坐标
    
    Returns:
        fig, ax
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 绘制每种模型的交叉熵
    for label in labels:
        if label in results_data:
            values = np.array(results_data[label])
            x_axis = np.arange(len(values))
            
            color = get_model_color(label)
            ax.plot(
                x_axis,
                values,
                marker='o',
                linestyle='-',
                linewidth=2,
                alpha=0.8,
                color=color,
                label=label,
                markersize=6
            )
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # 设置坐标轴
    if semilogy_scale:
        ax.set_yscale('log')
    
    # 图例和网格
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_multiple_learning_curves(
    learning_data: Dict[str, Dict[str, List[float]]],
    model_name_key: str = 'fractions',
    metric_names: List[str] = ['mse', 'ce'],
    title: str = "Multiple Learning Curves",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8)
):
    """
    绘制多个模型的学习曲线（每个评价指标一个子图）

    Args:
        learning_data: 学习曲线数据 {"model_name": {metric_name: data_array, ...}, ...}
        model_name_key: 训练比例的键名
        metric_names: 要绘制的指标列表
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小
    
    Returns:
        fig: matplotlib figures
    """
    # 获取所有模型名
    models = list(learning_data.keys())
    
    # 计算子图数量
    n_metrics = len(metric_names)
    
    # 创建子图
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    # 如果只有一个子图则axes不再是数组
    if n_metrics == 1:
        axes = [axes]
    
    # 为每个评估指标画一个图
    for i, metric in enumerate(metric_names):
        ax = axes[i]
        
        # 在当前子图中绘制所有模型的对应指标
        for model in models:
            if metric in learning_data[model]:
                fractions = learning_data[model][model_name_key]
                metric_values = learning_data[model][metric]
                
                if len(fractions) == len(metric_values):
                    color = get_model_color(model)
                    ax.plot(
                        [f * 100 for f in fractions],  # 转换为百分比
                        metric_values,
                        'o-', 
                        linewidth=2,
                        color=color,
                        label=model,
                        markersize=4,
                        alpha=0.8
                    )
        
        ax.set_xlabel('Percent Training Data Used (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Test {metric.upper()}', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.upper()} Learning Curve', fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 只在第一个子图放置图例
        if i == 0:
            ax.legend(loc='best', frameon=True, shadow=True)
    
    # 主标题
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # 调整子图间距
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 保存图片
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_parameter_heatmap(
    param_grid: Dict[str, List],
    validation_scores: np.ndarray,
    param_x: str,
    param_y: str,
    title: str = "Hyperparameter Heatmap",
    save_path: Optional[str] = None
):
    """
    创建超参数网格搜索结果的热力图

    Args:
        param_grid: 参数网格
        validation_scores: 验证分数网格 (2D array)
        param_x: X轴参数名
        param_y: Y轴参数名
        title: 图表标题
        save_path: 保存路径

    Returns:
        fig, ax
    """
    # 创建坐标轴值
    x_vals = param_grid[param_x]
    y_vals = param_grid[param_y]
    
    # 转换为DataFrame以便于可视化
    df = pd.DataFrame(
        validation_scores,
        columns=[f'{param_x}={x}' for x in x_vals],
        index=[f'{param_y}={y}' for y in y_vals]
    )
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(
        validation_scores,
        cmap='viridis',
        aspect='auto',
        origin='lower'
    )
    
    # 设置坐标轴
    ax.set_xticks(range(len(x_vals)))
    ax.set_yticks(range(len(y_vals)))
    ax.set_xticklabels([f'{x:.3f}' for x in x_vals])
    ax.set_yticklabels([f'{y:.3f}' for y in y_vals])
    
    # 添加标签
    ax.set_xlabel(param_x, fontsize=12)
    ax.set_ylabel(param_y, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Validation Score', rotation=270, labelpad=20)
    
    # 添加分数注释
    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            text = ax.text(
                j, i, f'{validation_scores[i, j]:.3f}',
                ha="center", va="center", color="white", fontsize=9
            )
    
    # 旋转x轴标签以更好地适应
    plt.xticks(rotation=45)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


# 实用函数
def plot_residuals(true_vals: np.ndarray, pred_vals: np.ndarray, 
                  title: str = "Residual Plot", save_path: Optional[str] = None):
    """
    绘制残差图
    """
    residuals = true_vals - pred_vals
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(pred_vals, residuals, alpha=0.6, color='steelblue')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax.set_xlabel('Predicted Values', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


if __name__ == "__main__":
    print("实验项目专用可视化工具模块已加载")
    
    # 示例用法
    # results = {
    #     'EV': {'mse': 0.2, 'r2': 0.8, 'correlation': 0.7},
    #     'EU': {'mse': 0.18, 'r2': 0.82, 'correlation': 0.72}
    # }
    # plot_model_performance_comparison(results, ['mse', 'r2', 'correlation'])
    # plt.show()