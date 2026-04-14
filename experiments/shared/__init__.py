"""
共享可视化模块的入口点 - 一次性导入所有可视化相关函数
"""
from .visualization import (
    setup_chinese_font,
    apply_modern_theme,
    get_color_palette,
    get_model_color,
    get_metric_color
)

from .plots import (
    plot_learning_curve,
    plot_model_comparison,
    plot_scatter_with_regression,
    plot_distribution_comparison,
    create_correlation_heatmap
)

from .viz_tools import (
    plot_model_performance_comparison,
    plot_cross_entropy_comparison,
    plot_multiple_learning_curves,
    create_parameter_heatmap,
    plot_residuals
)

__all__ = [
    # 来自 visualization 模块的函数
    'setup_chinese_font',
    'apply_modern_theme', 
    'get_color_palette',
    'get_model_color',
    'get_metric_color',
    
    # 来自 plots 模块的函数
    'plot_learning_curve',
    'plot_model_comparison', 
    'plot_scatter_with_regression',
    'plot_distribution_comparison',
    'create_correlation_heatmap',
    
    # 来自 viz_tools 模块的函数
    'plot_model_performance_comparison',
    'plot_cross_entropy_comparison',
    'plot_multiple_learning_curves',
    'create_parameter_heatmap',
    'plot_residuals'
]

# 设置默认参数
setup_chinese_font()
apply_modern_theme()