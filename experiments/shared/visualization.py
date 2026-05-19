"""
共享的可视化模块，用于统一项目中的图表设置和样式。
中文字体逻辑委托仓库根目录 `shared/visualization/matplotlib_chinese.py`，避免两套实现漂移。
"""
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt

_CANONICAL_FONT_MODULE = None


def _load_canonical_font_module():
    """从仓库根加载 canonical 字体模块（与数据集侧 evaluation 共用实现）。"""
    global _CANONICAL_FONT_MODULE
    if _CANONICAL_FONT_MODULE is not None:
        return _CANONICAL_FONT_MODULE
    # experiments/shared/visualization.py -> parents[2] = 仓库根
    repo_root = Path(__file__).resolve().parents[2]
    mod_path = repo_root / "shared" / "visualization" / "matplotlib_chinese.py"
    spec = importlib.util.spec_from_file_location("_choices13k_matplotlib_chinese", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load font module from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CANONICAL_FONT_MODULE = mod
    return mod


def setup_chinese_font():
    """设置中文字体与负号显示；实现见仓库根 `shared/visualization/matplotlib_chinese.py`。"""
    mod = _load_canonical_font_module()
    mod.setup_chinese_font()


def apply_modern_theme():
    """
    应用现代可视化主题
    """
    import seaborn as sns

    # 设置seaborn样式
    sns.set_style("whitegrid", {
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    
    # 优化matplotlib参数以获得更好的视觉效果
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'figure.dpi': 150,
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.titlepad': 10,
        'axes.labelpad': 8,
        'grid.alpha': 0.3,
    })


def get_color_palette(family="default"):
    """
    获取统一颜色调色板
    
    Args:
        family: 调色板系列名称
        
    Returns:
        dict或matplotlib色彩映射
    """
    import seaborn as sns

    palettes = {
        "models": {
            "ev": "#1f77b4",  # 蓝色
            "eu": "#ff7f0e",  # 橙色
            "pt3": "#2ca02c",  # 绿色
            "pt5": "#d62728",  # 红色
            "cpt5": "#9467bd", # 紫色
            "value_based": "#8c564b",  # 棕色
            "context_dependent": "#e377c2",  # 粉色
            "context_dependent_sigmoid": "#7f7f7f"  # 灰色
        },
        
        "quality_metrics": {
            "mse": "#1f77b4",
            "r2": "#ff7f0e", 
            "corr": "#2ca02c",
            "cross_entropy": "#d62728",
            "rmse": "#9467bd"
        },
        
        # 返回标准颜色列表
        "standard": sns.color_palette("husl", 10).as_hex()
    }
    
    return palettes.get(family, sns.color_palette("husl"))


def get_model_color(model_name):
    """
    获取指定模型的颜色
    
    Args:
        model_name: 模型名称
        
    Returns:
        hex颜色值
    """
    colors = get_color_palette("models")
    return colors.get(model_name.lower(), "#666666")  # 默认灰色


def get_metric_color(metric_name):
    """
    获取指定评估指标的颜色
    
    Args:
        metric_name: 指标名称
        
    Returns:
        hex颜色值
    """
    colors = get_color_palette("quality_metrics")
    return colors.get(metric_name.lower(), "#333333")  # 默认深灰


if __name__ == "__main__":
    # 测试函数
    setup_chinese_font()
    apply_modern_theme()
    print("可视化模块已准备好")
    
    color = get_model_color("ev")
    print(f"EV模型颜色: {color}")