"""
高级可视化函数库，用于创建常见的图表类型
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, Union, List, Dict
import os
from . import get_model_color, apply_modern_theme, get_metric_color

# 设置默认样式
apply_modern_theme()


def plot_learning_curve(
    fractions: Union[List[float], np.ndarray],
    metrics: List[Union[List[float], np.ndarray]],
    model_names: List[str],
    metric_name: str = "Performance",
    title: str = "Learning Curve",
    xlabel: str = "Percent Training Data Used (%)",
    ylabel: Optional[str] = None,
    save_path: Optional[str] = None,
    show_std: bool = True,
    std_devs: Optional[List[Union[List[float], np.ndarray]]] = None,
    figsize: tuple = (12, 8),
    show_legend: bool = True
):
    """
    绘制学习曲线图 - 展示训练数据量与模型性能之间的关系
    
    Args:
        fractions: 训练数据占比列表（0-1之间）
        metrics: 不同模型的性能指标列表 
        model_names: 模型名称列表
        metric_name: 指标名称（用于显示）
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签（如果为None，则自动生成）
        save_path: 保存路径（如果提供则保存图表）
        show_std: 是否显示标准差/置信区间
        std_devs: 对应的标准差值
        figsize: 图表大小
        show_legend: 是否显示图例
    
    Returns:
        fig, ax: matplotlib图表对象
    """
    if ylabel is None:
        ylabel = f"Test {metric_name}"
        
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # 处理x轴 - 转换为百分比
    x_values = [frac * 100 for frac in fractions]
    
    for i, (metric_values, model_name) in enumerate(zip(metrics, model_names)):
        metric_values = np.array(metric_values)
        color = get_model_color(model_name)
        
        # 绘制中心线
        ax.plot(
            x_values, 
            metric_values, 
            "o-", 
            label=model_name, 
            linewidth=2, 
            alpha=0.8, 
            color=color,
            markersize=6
        )
        
        # 如果需要显示标准差，绘制填充区域
        if show_std and std_devs and i < len(std_devs):
            std_values = np.array(std_devs[i])
            ax.fill_between(
                x_values, 
                metric_values - std_values, 
                metric_values + std_values, 
                alpha=0.2, 
                color=color
            )
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    
    if show_legend:
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # 设置坐标轴范围
    ax.set_xlim(min(x_values), max(x_values))
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 保存图像（如果指定路径）
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
    colormap: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    绘制模型比较棒图
    
    Args:
        metrics_df: 性能数据框，每一行为模型，每一列为性能指标
        title: 图表标题
        save_path: 保存路径
        colormap: 颜色映射
        figsize: 图表大小
        
    Returns:
        fig, ax: matplotlib图表对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 选择最适合的图类型：如果指标过多使用热力图，否则使用棒图
    n_models, n_metrics = metrics_df.shape
    
    if n_metrics == 1:
        # 简单的单指标对比棒图
        metric_col = metrics_df.columns[0]
        metrics_df_sorted = metrics_df.sort_values(by=metric_col, ascending=False)
        
        bars = ax.bar(
            range(len(metrics_df_sorted)),
            metrics_df_sorted[metric_col].values,
            color=[get_model_color(name) for name in metrics_df_sorted.index]
        )
        
        ax.set_xticks(range(len(metrics_df_sorted)))
        ax.set_xticklabels(metrics_df_sorted.index, rotation=45, ha='right')
        ax.set_title(title)
        ax.set_ylabel(metric_col)
        
        # 添加数值标签
        for bar, value in zip(bars, metrics_df_sorted[metric_col].values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., 
                height,
                f'{value:.3f}',
                ha='center', va='bottom'
            )
    else:
        # 指标较多的情况使用热力图
        # 为保持模型间有意义的比较，我们转置矩阵，并根据行（指标）标准化
        im = ax.imshow(metrics_df.values.T.astype(float), 
                       cmap=colormap or 'viridis', 
                       aspect='auto',
                       interpolation='nearest')
        
        # 设置坐标轴标签
        ax.set_xticks(range(len(metrics_df.index)))
        ax.set_yticks(range(len(metrics_df.columns)))
        ax.set_xticklabels(metrics_df.index, rotation=45, ha='right')
        ax.set_yticklabels(metrics_df.columns)
        
        # 添加值文本
        for i in range(len(metrics_df.columns)):
            for j in range(len(metrics_df.index)):
                text = ax.text(j, i, f"{metrics_df.iloc[j, i]:.3f}",
                              ha="center", va="center", color="white", fontsize=8)
        
        ax.set_title(title)
        
        # 添加颜色条
        plt.colorbar(im, ax=ax)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_scatter_with_regression(
    x_data: Union[List[float], np.ndarray],
    y_data: Union[List[float], np.ndarray],
    title: str = "Scatter Plot with Regression Line",
    xlabel: str = "X Values",
    ylabel: str = "Y Values",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    绘制带回归线的散点图
    
    Args:
        x_data: X轴数据
        y_data: Y轴数据
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        save_path: 保存路径
        figsize: 图表大小
        
    Returns:
        fig, ax: matplotlib图表对象
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制散点图
    ax.scatter(x_data, y_data, alpha=0.6, s=50, color='steelblue', edgecolors='white', linewidth=0.5)
    
    # 计算并绘制回归线
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    ax.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2, label=f'Regression Line (R²={np.corrcoef(x_data,y_data)[0,1]**2:.3f})')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_distribution_comparison(
    series_list: List[Union[List[float], np.ndarray]],
    labels: List[str],
    title: str = "Distribution Comparison",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
):
    """
    绘制多组数据分布比较图（直方图 + 箱线图）
    
    Args:
        series_list: 数据序列列表
        labels: 对应的标签列表
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小
        
    Returns:
        fig, ax: matplotlib图表对象
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # 绘制直方图
    for i, (series, label) in enumerate(zip(series_list, labels)):
        series = np.array(series)
        color = get_model_color(label)
        axes[0].hist(series, bins=30, alpha=0.5, label=label, density=True, color=color)
    
    axes[0].set_ylabel('Density')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 绘制箱线图
    data_for_boxplot = [np.array(series) for series in series_list]
    bp = axes[1].boxplot(data_for_boxplot, labels=labels, patch_artist=True)
    
    for patch, label in zip(bp['boxes'], labels):
        color = get_model_color(label)
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        
    axes[1].set_ylabel('Values')
    axes[1].grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def create_correlation_heatmap(
    data: Union[pd.DataFrame, np.ndarray],
    title: str = "Correlation Heatmap",
    annot: bool = True,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
):
    """
    创建相关性热力图
    
    Args:
        data: 输入数据
        title: 标题
        annot: 是否显示数值
        save_path: 保存路径
        figsize: 图表大小
        
    Returns:
        fig, ax: matplotlib图表对象
    """
    if isinstance(data, pd.DataFrame):
        correlation_matrix = data.corr()
    else:
        correlation_matrix = pd.DataFrame(data).corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=annot,
        cmap='coolwarm',
        center=0,
        square=True,
        fmt='.2f',
        cbar_kws={"shrink": .8},
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


if __name__ == "__main__":
    # 测试函数
    import numpy as np
    
    # 测试学习曲线
    print("测试学习曲线绘图...")
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    metric1 = [0.3, 0.35, 0.4, 0.42, 0.45, 0.47, 0.48, 0.49, 0.495, 0.5]
    metric2 = [0.25, 0.32, 0.38, 0.41, 0.43, 0.45, 0.46, 0.47, 0.48, 0.485]
    
    fig, ax = plot_learning_curve(
        fractions=x,
        metrics=[metric1, metric2],
        model_names=["Model A", "Model B"],
        title="Sample Learning Curves"
    )
    
    print("学习曲线绘图完成")
    
    # 显示测试图表
    # plt.show()  # 在生产环境中可能不需要显示