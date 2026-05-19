"""
增强Parameter Split深度分析脚本
深入分析增强符号模型在参数划分上的表现
基于JSON原始分布数据和增强模型架构
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Any
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置路径和导入
sys.path.insert(0, str(Path(__file__).parent.parent))

# 项目根路径并设置中文字体（避免中文/负号显示为方框）
_p = Path(__file__).resolve().parent
for _ in range(10):
    _p = _p.parent
    if (_p / "shared").is_dir():
        sys.path.insert(0, str(_p))
        break
from shared.visualization import setup_chinese_font
setup_chinese_font()

from enhanced_data_standardization import (
    EnhancedChoices13kStandardizer,
    create_enhanced_splits
)

sns.set_style("whitegrid")
sns.set_palette("husl")


class EnhancedParameterSplitAnalyzer:
    """增强Parameter Split深度分析器"""
    
    def __init__(self, split_name: str = 'parameter_amb', results_file: Optional[Path] = None, output_dir: Optional[Path] = None):
        """
        Args:
            split_name: 划分名称 ('problem', 'parameter_amb', 'parameter_ev_extreme')
            results_file: 增强训练结果JSON文件路径
            output_dir: 输出目录
        """
        self.split_name = split_name
        if results_file is None:
            # 自动查找最新的增强结果文件
            results_dir = Path(__file__).parent.parent / "results" / "enhanced_training"
            result_files = list(results_dir.glob("enhanced_models_results_*.json"))
            if not result_files:
                raise FileNotFoundError("未找到增强训练结果文件")
            results_file = max(result_files, key=lambda x: x.stat().st_mtime)
        
        self.results_file = Path(results_file)
        self.output_dir = output_dir or Path(__file__).parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据标准化器
        self.standardizer = EnhancedChoices13kStandardizer(
            selections_path=str(Path(__file__).parent.parent.parent.parent / "数据集" / "choices13k-main" / "c13k_selections.csv"),
            problems_path=str(Path(__file__).parent.parent.parent.parent / "数据集" / "choices13k-main" / "c13k_problems.json")
        )
        
        # 加载数据
        self.results = None
        self.standardized_data = None
        self.y = None
        self.train_idx = None
        self.test_idx = None
        
        self._load_data()
    
    def _load_data(self):
        """加载所有必要数据"""
        print("=" * 80)
        print("增强Parameter Split深度分析")
        print("=" * 80)
        
        # 加载训练结果
        print(f"加载增强训练结果: {self.results_file}")
        with open(self.results_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        # 检查是否有指定划分的结果
        if self.split_name not in self.results:
            raise ValueError(f"结果文件中未找到 {self.split_name} 数据")
        
        split_results = self.results[self.split_name]
        print(f"找到 {len(split_results)} 个模型的结果: {list(split_results.keys())}")
        
        # 加载标准化数据
        print("加载标准化数据...")
        df = self.standardizer.load_and_merge_data()
        self.standardized_data = self.standardizer.standardize_all()
        
        # 提取目标变量
        self.y = np.array([item['action']['bRate'] for item in self.standardized_data])
        
        # 获取划分索引
        print(f"创建 {self.split_name} 划分...")
        self.train_idx, self.test_idx, split_info = create_enhanced_splits(
            self.standardized_data, split_type=self.split_name
        )
        
        print(f"数据加载完成:")
        print(f"  总样本数: {len(self.standardized_data)}")
        print(f"  训练集: {len(self.train_idx)} 样本")
        print(f"  测试集: {len(self.test_idx)} 样本")
        print(f"  划分描述: {split_info['description']}")
        
        # 提取真实值
        self.y_train_true = self.y[self.train_idx]
        self.y_test_true = self.y[self.test_idx]
    
    def analyze_prediction_errors(self):
        """分析预测误差"""
        print("\n" + "=" * 80)
        print("预测误差分析")
        print("=" * 80)
        
        split_results = self.results[self.split_name]
        
        # 为每个模型计算误差统计
        error_stats = {}
        
        for model_key, result in split_results.items():
            if result is None:
                continue
            
            model_name = result['model_name']
            y_test_pred = np.array(result['test_predictions'])
            y_train_pred = np.array(result['train_predictions'])
            
            # 确保预测值与真实值长度一致
            if len(y_test_pred) != len(self.y_test_true):
                print(f"警告: {model_name} 测试集预测值长度不匹配 ({len(y_test_pred)} vs {len(self.y_test_true)})")
                continue
            
            if len(y_train_pred) != len(self.y_train_true):
                print(f"警告: {model_name} 训练集预测值长度不匹配 ({len(y_train_pred)} vs {len(self.y_train_true)})")
                continue
            
            # 计算测试集误差
            test_errors = y_test_pred - self.y_test_true
            train_errors = y_train_pred - self.y_train_true
            
            # 误差统计
            stats_dict = {
                'model_name': model_name,
                'test_errors': {
                    'mean': float(np.mean(test_errors)),
                    'std': float(np.std(test_errors)),
                    'mean_abs': float(np.mean(np.abs(test_errors))),
                    'max_abs': float(np.max(np.abs(test_errors))),
                    'skewness': float(stats.skew(test_errors) if len(test_errors) > 2 else 0),
                    'kurtosis': float(stats.kurtosis(test_errors) if len(test_errors) > 3 else 0)
                },
                'train_errors': {
                    'mean': float(np.mean(train_errors)),
                    'std': float(np.std(train_errors)),
                    'mean_abs': float(np.mean(np.abs(train_errors)))
                },
                'test_metrics': result['test_metrics'],
                'train_metrics': result['train_metrics']
            }
            
            error_stats[model_key] = stats_dict
            
            print(f"\n{model_name} 误差分析:")
            print(f"  测试集误差均值: {stats_dict['test_errors']['mean']:.6f}")
            print(f"  测试集误差标准差: {stats_dict['test_errors']['std']:.6f}")
            print(f"  测试集平均绝对误差: {stats_dict['test_errors']['mean_abs']:.6f}")
            print(f"  测试集最大绝对误差: {stats_dict['test_errors']['max_abs']:.6f}")
            print(f"  测试集R^2: {result['test_metrics']['r2']:.4f}")
            print(f"  测试集MSE: {result['test_metrics']['mse']:.6f}")
        
        # 可视化误差分布
        self._plot_error_distributions(error_stats)
        
        return error_stats
    
    def _plot_error_distributions(self, error_stats):
        """绘制误差分布图"""
        print("\n生成误差分布可视化...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # 1. 误差分布直方图
        ax = axes[0]
        for model_key, stats in error_stats.items():
            if 'test_errors' in stats:
                # 从结果中获取预测值
                split_results = self.results[self.split_name]
                y_pred = np.array(split_results[model_key]['test_predictions'])
                errors = y_pred - self.y_test_true
                
                sns.histplot(errors, bins=50, alpha=0.5, label=model_key, ax=ax)
        
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('预测误差 (预测值 - 真实值)')
        ax.set_ylabel('频数')
        ax.set_title('各模型测试集误差分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 绝对误差箱线图
        ax = axes[1]
        abs_errors_data = []
        model_labels = []
        
        for model_key, stats in error_stats.items():
            if 'test_errors' in stats:
                split_results = self.results[self.split_name]
                y_pred = np.array(split_results[model_key]['test_predictions'])
                abs_errors = np.abs(y_pred - self.y_test_true)
                abs_errors_data.append(abs_errors)
                model_labels.append(model_key)
        
        if abs_errors_data:
            ax.boxplot(abs_errors_data, labels=model_labels)
            ax.set_xlabel('模型')
            ax.set_ylabel('绝对误差')
            ax.set_title('各模型绝对误差分布')
            ax.grid(True, alpha=0.3)
        
        # 3. 预测值 vs 真实值散点图
        ax = axes[2]
        colors = plt.cm.get_cmap('tab10')(np.arange(len(error_stats)))
        
        for idx, (model_key, stats) in enumerate(error_stats.items()):
            if 'test_errors' in stats:
                split_results = self.results[self.split_name]
                y_pred = np.array(split_results[model_key]['test_predictions'])
                
                ax.scatter(self.y_test_true, y_pred, alpha=0.6, s=20, 
                          color=colors[idx], label=model_key)
        
        # 添加理想预测线
        min_val = min(self.y_test_true.min(), 0)
        max_val = max(self.y_test_true.max(), 1)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='理想预测')
        
        ax.set_xlabel('真实 bRate')
        ax.set_ylabel('预测 bRate')
        ax.set_title('预测值 vs 真实值')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 模型性能比较（R²和MSE）
        ax = axes[3]
        model_names = []
        r2_values = []
        mse_values = []
        
        for model_key, stats in error_stats.items():
            if 'test_metrics' in stats:
                model_names.append(model_key)
                r2_values.append(stats['test_metrics']['r2'])
                mse_values.append(stats['test_metrics']['mse'])
        
        if model_names:
            x = np.arange(len(model_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, r2_values, width, label='R²', alpha=0.8)
            bars2 = ax.bar(x + width/2, mse_values, width, label='MSE', alpha=0.8)
            
            # 在柱状图上添加数值标签
            for i, (r2, mse) in enumerate(zip(r2_values, mse_values)):
                ax.text(i - width/2, r2, f'{r2:.3f}', ha='center', va='bottom', fontsize=9)
                ax.text(i + width/2, mse, f'{mse:.4f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('模型')
            ax.set_ylabel('性能指标')
            ax.set_title('模型性能比较 (R²和MSE)')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 保存图形
        output_file = self.output_dir / f"enhanced_error_analysis_{self.split_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"误差分析图已保存到: {output_file}")
        plt.close()
    
    def analyze_parameter_distribution(self):
        """分析模型参数分布"""
        print("\n" + "=" * 80)
        print("模型参数分布分析")
        print("=" * 80)
        
        split_results = self.results[self.split_name]
        
        # 收集所有模型参数
        all_params = {}
        
        for model_key, result in split_results.items():
            if result is None:
                continue
            
            model_name = result['model_name']
            params = result['parameters']
            
            print(f"\n{model_name} 参数:")
            for param_name, param_value in params.items():
                if param_name != 'model_name':
                    print(f"  {param_name}: {param_value}")
            
            all_params[model_key] = params
        
        # 可视化参数分布（如果多个模型有相同参数）
        self._plot_parameter_distributions(all_params)
        
        return all_params
    
    def _plot_parameter_distributions(self, all_params):
        """绘制参数分布图"""
        # 找出所有模型共有的参数
        common_params = set()
        param_values = {}
        
        for model_key, params in all_params.items():
            for param_name, param_value in params.items():
                if param_name != 'model_name' and isinstance(param_value, (int, float)):
                    if param_name not in param_values:
                        param_values[param_name] = []
                    param_values[param_name].append((model_key, param_value))
        
        # 只绘制有足够数据的参数
        plot_params = [p for p, values in param_values.items() if len(values) > 1]
        
        if not plot_params:
            print("没有足够的参数数据进行可视化")
            return
        
        n_plots = min(len(plot_params), 4)
        n_rows = int(np.ceil(n_plots / 2))
        
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, param_name in enumerate(plot_params[:n_plots]):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            values = param_values[param_name]
            
            model_keys = [v[0] for v in values]
            param_vals = [v[1] for v in values]
            
            bars = ax.bar(model_keys, param_vals, alpha=0.7)
            
            # 添加数值标签
            for bar, val in zip(bars, param_vals):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('模型')
            ax.set_ylabel('参数值')
            ax.set_title(f'{param_name} 参数分布')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 隐藏多余的子图
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图形
        output_file = self.output_dir / f"enhanced_parameter_distribution_{self.split_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"参数分布图已保存到: {output_file}")
        plt.close()
    
    def analyze_calibration(self):
        """分析模型校准（预测概率的准确性）"""
        print("\n" + "=" * 80)
        print("模型校准分析")
        print("=" * 80)
        
        split_results = self.results[self.split_name]
        
        calibration_results = {}
        
        for model_key, result in split_results.items():
            if result is None:
                continue
            
            model_name = result['model_name']
            y_test_pred = np.array(result['test_predictions'])
            
            # 将预测值分箱
            n_bins = 10
            bins = np.linspace(0, 1, n_bins + 1)
            
            bin_indices = np.digitize(y_test_pred, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            bin_means_pred = []
            bin_means_true = []
            bin_counts = []
            
            for i in range(n_bins):
                mask = bin_indices == i
                if np.sum(mask) > 0:
                    bin_means_pred.append(np.mean(y_test_pred[mask]))
                    bin_means_true.append(np.mean(self.y_test_true[mask]))
                    bin_counts.append(np.sum(mask))
            
            calibration_results[model_key] = {
                'model_name': model_name,
                'bin_means_pred': bin_means_pred,
                'bin_means_true': bin_means_true,
                'bin_counts': bin_counts,
                'n_bins': n_bins
            }
            
            print(f"\n{model_name} 校准分析:")
            print(f"  分箱数量: {n_bins}")
            print(f"  有效分箱: {len(bin_means_pred)}")
        
        # 可视化校准曲线
        self._plot_calibration_curves(calibration_results)
        
        return calibration_results
    
    def _plot_calibration_curves(self, calibration_results):
        """绘制校准曲线"""
        print("\n生成校准曲线可视化...")
        
        n_models = len(calibration_results)
        if n_models == 0:
            print("没有校准数据可绘制")
            return
        
        fig, axes = plt.subplots(1, min(n_models, 3), figsize=(5 * min(n_models, 3), 5))
        if n_models == 1:
            axes = np.array([axes])
        
        for idx, (model_key, calib_data) in enumerate(calibration_results.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            bin_means_pred = calib_data['bin_means_pred']
            bin_means_true = calib_data['bin_means_true']
            bin_counts = calib_data['bin_counts']
            
            # 绘制校准曲线
            ax.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='完美校准')
            ax.scatter(bin_means_pred, bin_means_true, s=np.array(bin_counts)/2, alpha=0.7, label='分箱均值')
            
            # 添加分箱大小信息
            for pred, true, count in zip(bin_means_pred, bin_means_true, bin_counts):
                ax.text(pred, true, f'{count}', fontsize=8, ha='center', va='bottom')
            
            ax.set_xlabel('预测概率均值')
            ax.set_ylabel('真实概率均值')
            ax.set_title(f'{calib_data["model_name"]} 校准曲线')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # 保存图形
        output_file = self.output_dir / f"enhanced_calibration_curves_{self.split_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"校准曲线图已保存到: {output_file}")
        plt.close()
    
    def run_comprehensive_analysis(self, plots: str = "minimal"):
        """运行分析。plots='minimal' 仅保留误差分析图与报告；plots='full' 包含参数分布、校准曲线等全部图。"""
        print("\n" + "=" * 80)
        print("开始分析" + ("（仅误差分析图）" if plots == "minimal" else "（全面）"))
        print("=" * 80)
        
        try:
            # 1. 预测误差分析（必跑，minimal 下唯一出图）
            error_stats = self.analyze_prediction_errors()
            
            # 2. 参数分布分析（仅 full 时出图）
            param_dist = {}
            if plots == "full":
                param_dist = self.analyze_parameter_distribution()
            
            # 3. 校准分析（仅 full 时出图）
            calibration = {}
            if plots == "full":
                calibration = self.analyze_calibration()
            
            # 4. 生成综合报告
            self.generate_comprehensive_report(error_stats, param_dist, calibration)
            
            print("\n" + "=" * 80)
            print("全面分析完成!")
            print("=" * 80)
            
            return True
            
        except Exception as e:
            print(f"\n分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_comprehensive_report(self, error_stats, param_dist, calibration):
        """生成综合报告"""
        report = {
            'split_name': self.split_name,
            'results_file': str(self.results_file),
            'analysis_timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data_summary': {
                'total_samples': len(self.standardized_data),
                'train_samples': len(self.train_idx),
                'test_samples': len(self.test_idx),
                'models_analyzed': len(error_stats)
            },
            'error_analysis': error_stats,
            'parameter_distribution': param_dist,
            'calibration_analysis': calibration,
            'performance_summary': {}
        }
        
        # 生成性能摘要
        for model_key, stats in error_stats.items():
            if 'test_metrics' in stats:
                report['performance_summary'][model_key] = {
                    'model_name': stats['model_name'],
                    'test_r2': stats['test_metrics']['r2'],
                    'test_mse': stats['test_metrics']['mse'],
                    'test_mae': stats['test_metrics']['mae'],
                    'train_r2': stats['train_metrics']['r2'],
                    'train_mse': stats['train_metrics']['mse']
                }
        
        # 转换numpy类型为Python原生类型以便JSON序列化
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        # 保存报告
        report_file = self.output_dir / f"enhanced_comprehensive_report_{self.split_name}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(convert_to_serializable(report), f, indent=2, ensure_ascii=False)
        
        print(f"\n综合报告已保存到: {report_file}")
        
        # 生成文本摘要
        self._generate_text_summary(report)
    
    def _generate_text_summary(self, report):
        """生成文本摘要"""
        summary_file = self.output_dir / f"enhanced_analysis_summary_{self.split_name}.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"增强Parameter Split分析摘要 - {report['split_name']}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"数据文件: {report['results_file']}\n")
            f.write(f"分析时间: {report['analysis_timestamp']}\n\n")
            
            f.write("数据摘要:\n")
            f.write(f"  总样本数: {report['data_summary']['total_samples']}\n")
            f.write(f"  训练集样本数: {report['data_summary']['train_samples']}\n")
            f.write(f"  测试集样本数: {report['data_summary']['test_samples']}\n")
            f.write(f"  分析模型数: {report['data_summary']['models_analyzed']}\n\n")
            
            f.write("性能摘要:\n")
            for model_key, perf in report['performance_summary'].items():
                f.write(f"  {perf['model_name']}:\n")
                f.write(f"    测试集 R²: {perf['test_r2']:.4f}\n")
                f.write(f"    测试集 MSE: {perf['test_mse']:.6f}\n")
                f.write(f"    测试集 MAE: {perf['test_mae']:.6f}\n")
                f.write(f"    训练集 R²: {perf['train_r2']:.4f}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("分析完成\n")
            f.write("=" * 80 + "\n")
        
        print(f"文本摘要已保存到: {summary_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='增强Parameter Split深度分析')
    parser.add_argument('--split', type=str, default='parameter_amb',
                       choices=['problem', 'parameter_amb', 'parameter_ev_extreme'],
                       help='要分析的划分名称')
    parser.add_argument('--results', type=str, help='训练结果JSON文件路径')
    parser.add_argument('--output', type=str, help='输出目录路径')
    parser.add_argument('--plots', type=str, choices=['minimal', 'full'], default='minimal',
                       help='minimal=仅误差分析图+报告；full=全部图（参数分布、校准曲线等）')
    
    args = parser.parse_args()
    
    print("增强Parameter Split深度分析")
    print("=" * 80)
    
    # 创建分析器
    analyzer = EnhancedParameterSplitAnalyzer(
        split_name=args.split,
        results_file=Path(args.results) if args.results else None,
        output_dir=Path(args.output) if args.output else None
    )
    
    # 运行分析（默认 minimal 仅出 1 张核心图）
    success = analyzer.run_comprehensive_analysis(plots=args.plots)
    
    if success:
        print("\n分析完成!")
    else:
        print("\n分析失败!")
    
    return success


if __name__ == "__main__":
    main()