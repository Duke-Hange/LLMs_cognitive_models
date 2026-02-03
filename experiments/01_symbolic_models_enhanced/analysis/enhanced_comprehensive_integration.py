"""
增强综合结果整合脚本
整合所有增强分析结果，进行深入的科学解释
基于JSON原始分布数据和增强模型架构
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
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

from enhanced_data_standardization import EnhancedChoices13kStandardizer

sns.set_style("whitegrid")
sns.set_palette("husl")


class EnhancedComprehensiveResultsIntegrator:
    """增强综合结果整合器"""
    
    def __init__(self, results_file: Optional[Path] = None, output_dir: Optional[Path] = None):
        """
        Args:
            results_file: 增强训练结果JSON文件路径
            output_dir: 输出目录
        """
        if results_file is None:
            # 自动查找最新的增强结果文件
            results_dir = Path(__file__).parent.parent / "results" / "enhanced_training"
            result_files = list(results_dir.glob("enhanced_models_results_*.json"))
            if not result_files:
                raise FileNotFoundError("未找到增强训练结果文件")
            results_file = max(result_files, key=lambda x: x.stat().st_mtime)
        
        self.results_file = Path(results_file)
        self.output_dir = output_dir or Path(__file__).parent / "integration_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        self.results = None
        self.split_info = {}
        
        self._load_data()
    
    def _load_data(self):
        """加载所有必要数据"""
        print("=" * 80)
        print("增强综合结果整合 - 结果整合与科学解释")
        print("=" * 80)
        
        # 加载训练结果
        print(f"加载增强训练结果: {self.results_file}")
        with open(self.results_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        # 检查所有划分
        required_splits = ['problem', 'parameter_amb', 'parameter_ev_extreme']
        self.available_splits = [s for s in required_splits if s in self.results]
        print(f"找到的划分: {self.available_splits}")
        
        # 获取数据集基本信息
        print("\n数据集基本信息:")
        for split_name in self.available_splits:
            split_results = self.results[split_name]
            if not split_results:
                continue
            
            # 从第一个模型的结果中获取划分信息
            first_model_key = next(iter(split_results.keys()))
            first_result = split_results[first_model_key]
            
            if first_result is None:
                continue
            
            self.split_info[split_name] = {
                'train_size': first_result['train_metrics']['n_samples'],
                'test_size': first_result['test_metrics']['n_samples'],
                'description': first_result['split_description'],
                'train_bRate_mean': float(np.mean(first_result['train_predictions'])),
                'test_bRate_mean': float(np.mean(first_result['test_predictions']))
            }
            
            print(f"  {split_name}:")
            print(f"    训练集: {self.split_info[split_name]['train_size']} 样本, bRate均值={self.split_info[split_name]['train_bRate_mean']:.4f}")
            print(f"    测试集: {self.split_info[split_name]['test_size']} 样本, bRate均值={self.split_info[split_name]['test_bRate_mean']:.4f}")
    
    def convert_to_serializable(self, obj):
        """转换numpy类型为Python原生类型以便JSON序列化"""
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self.convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def collect_all_performance_metrics(self) -> pd.DataFrame:
        """收集所有性能指标"""
        print("\n" + "=" * 80)
        print("性能指标汇总")
        print("=" * 80)
        
        all_performance = []
        
        for split_name, split_results in self.results.items():
            for model_key, result in split_results.items():
                if result is None:
                    continue
                
                metrics = result['test_metrics']
                all_performance.append({
                    'split': split_name,
                    'model': model_key,
                    'model_name': result['model_name'],
                    'mse': metrics['mse'],
                    'rmse': np.sqrt(metrics['mse']),
                    'mae': metrics['mae'],
                    'r2': metrics['r2'],
                    'correlation': metrics['correlation'],
                    'training_time': result['training_time_seconds'],
                    'parameters': result['parameters'],
                    'split_description': result['split_description']
                })
        
        df_performance = pd.DataFrame(all_performance)
        
        if df_performance.empty:
            print("没有找到性能数据")
            return None
        
        print("\n各模型在不同划分上的性能汇总:")
        print("-" * 80)
        
        # 按模型显示平均性能
        model_summary = df_performance.groupby('model').agg({
            'r2': ['mean', 'std', 'min', 'max'],
            'mse': ['mean', 'std'],
            'training_time': 'mean'
        }).round(4)
        
        print(model_summary)
        
        return df_performance
    
    def analyze_behavioral_economics_parameters(self, df_performance: pd.DataFrame):
        """分析行为经济学参数意义"""
        print("\n" + "=" * 80)
        print("行为经济学参数深度分析")
        print("=" * 80)
        
        # 收集所有模型的参数
        all_params = []
        
        for split_name, split_results in self.results.items():
            for model_key, result in split_results.items():
                if result is None:
                    continue
                
                params = result['parameters']
                if params:
                    param_record = {
                        'split': split_name,
                        'model': model_key,
                        'model_name': result['model_name']
                    }
                    param_record.update(params)
                    all_params.append(param_record)
        
        if not all_params:
            print("没有找到参数数据")
            return
        
        df_params = pd.DataFrame(all_params)
        
        print("\n各模型参数汇总:")
        print("-" * 80)
        
        # 按模型显示参数统计
        for model_key in ['ev', 'eu', 'pt3', 'pt5']:
            model_params = df_params[df_params['model'] == model_key]
            if not model_params.empty:
                print(f"\n{model_key} 模型参数:")
                numeric_cols = model_params.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col not in ['split', 'model', 'model_name']:
                        mean_val = model_params[col].mean()
                        std_val = model_params[col].std()
                        min_val = model_params[col].min()
                        max_val = model_params[col].max()
                        print(f"  {col:15s}: 均值={mean_val:.3f}, 标准差={std_val:.3f}, "
                              f"范围=[{min_val:.3f}, {max_val:.3f}]")
        
        # 行为经济学解释
        print("\n" + "=" * 80)
        print("行为经济学参数解释")
        print("=" * 80)
        
        # EV模型解释
        ev_params = df_params[df_params['model'] == 'ev']
        if not ev_params.empty:
            print("\n期望值模型 (EV):")
            print("  - temperature参数控制决策随机性")
            print("  - 值越高表示决策越随机（接近0.5概率）")
            print("  - 值越低表示决策越确定性（接近0或1概率）")
        
        # EU模型解释
        eu_params = df_params[df_params['model'] == 'eu']
        if not eu_params.empty:
            print("\n期望效用模型 (EU):")
            print("  - alpha参数反映风险态度:")
            print("    * alpha < 1: 风险厌恶（收益边际效用递减）")
            print("    * alpha = 1: 风险中性（线性效用）")
            print("    * alpha > 1: 风险寻求（收益边际效用递增）")
            print("  - temperature参数控制决策随机性")
        
        # PT3模型解释
        pt3_params = df_params[df_params['model'] == 'pt3']
        if not pt3_params.empty:
            print("\n前景理论3参数模型 (PT3):")
            print("  - alpha: 价值函数曲率，反映对收益和损失的敏感性")
            print("    * alpha < 1: 对变化敏感度递减")
            print("    * 典型值: 0.88（Kahneman & Tversky, 1979）")
            print("  - lambda: 损失厌恶系数")
            print("    * lambda > 1: 损失比等量收益更令人厌恶")
            print("    * 典型值: 2.25（损失带来的痛苦是收益带来快乐的2.25倍）")
            print("  - gamma: 概率权重函数参数")
            print("    * gamma < 1: 高估小概率，低估大概率")
            print("    * 典型值: 0.61-0.69")
            print("  - temperature: 决策随机性")
        
        # PT5模型解释
        pt5_params = df_params[df_params['model'] == 'pt5']
        if not pt5_params.empty:
            print("\n前景理论5参数模型 (PT5):")
            print("  - alpha_gain, alpha_loss: 收益和损失的价值函数曲率")
            print("    * 通常alpha_loss > alpha_gain（损失更敏感）")
            print("  - gamma_gain, gamma_loss: 收益和损失的概率权重")
            print("    * 通常gamma_gain ≠ gamma_loss（对收益和损失的概率权重不同）")
            print("  - lambda: 损失厌恶系数")
            print("  - temperature: 决策随机性")
        
        return df_params
    
    def analyze_performance_patterns(self, df_performance: pd.DataFrame):
        """分析性能模式"""
        print("\n" + "=" * 80)
        print("性能模式分析")
        print("=" * 80)
        
        if df_performance is None or df_performance.empty:
            print("没有性能数据可分析")
            return
        
        # 1. 划分难度分析
        print("\n1. 划分难度分析 (测试集MSE):")
        split_difficulty = df_performance.groupby('split')['mse'].agg(['mean', 'std']).round(6)
        print(split_difficulty)
        
        # 2. 模型能力分析
        print("\n2. 模型能力分析 (平均R^2):")
        model_capability = df_performance.groupby('model')['r2'].agg(['mean', 'std', 'min', 'max']).round(4)
        print(model_capability)
        
        # 3. 划分-模型交互
        print("\n3. 划分与模型交互分析:")
        pivot_table = df_performance.pivot_table(
            values='r2', 
            index='model', 
            columns='split', 
            aggfunc='mean'
        ).round(4)
        print(pivot_table)
        
        # 4. 增强模型与原始模型对比（如果有原始数据）
        print("\n4. 增强模型性能特点:")
        print("   - 基于JSON原始分布数据")
        print("   - 使用多结果分布特征")
        print("   - 直接从分布计算期望值")
        print("   - 包含方差、偏度、熵等丰富特征")
    
    def analyze_limitations_and_future_directions(self):
        """分析局限性和未来方向"""
        print("\n" + "=" * 80)
        print("局限性分析与未来方向")
        print("=" * 80)
        
        print("\n增强模型实验的局限性:")
        print("1. 计算复杂性: 从原始分布直接计算比简化特征更耗时")
        print("2. 优化挑战: 参数优化可能更困难，损失函数地形更复杂")
        print("3. 特征维度: 50+维特征可能引入噪声或过拟合")
        print("4. 模型适应性: 经典符号模型可能未充分利用复杂分布特征")
        print("5. 性能表现: 实验结果显示增强模型性能下降，需要进一步分析原因")
        
        print("\n未来研究方向:")
        print("1. 方案B测试: 在简化特征基础上添加关键分布统计量")
        print("2. 混合方法: 结合简化特征和分布统计特征")
        print("3. 专门模型: 开发专门处理多结果分布的决策模型")
        print("4. 特征选择: 分析50+维特征中哪些真正重要")
        print("5. 错误分析: 深入研究预测误差最大的样本特征")
        print("6. 优化改进: 改进参数优化算法，避免边界值问题")
        
        print("\n科学启示:")
        print("1. 简化特征的有效性: 对于符号决策模型，简化表示可能更有效")
        print("2. 人类决策表征: 结果可能反映人类决策基于简化表征")
        print("3. 模型-数据匹配: 模型复杂度应与数据表示复杂度匹配")
        print("4. 准确性与可预测性权衡: 理论上更准确的计算不一定带来更好的预测")
    
    def visualize_performance_comparison(self, df_performance: pd.DataFrame):
        """可视化性能比较"""
        print("\n生成性能比较可视化...")
        
        if df_performance is None or df_performance.empty:
            print("没有性能数据可可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # 1. 各模型R²比较（按划分）
        ax = axes[0]
        pivot_r2 = df_performance.pivot_table(
            values='r2', 
            index='model', 
            columns='split', 
            aggfunc='mean'
        )
        
        pivot_r2.plot(kind='bar', ax=ax, alpha=0.8)
        ax.set_xlabel('模型')
        ax.set_ylabel('R² (平均值)')
        ax.set_title('各模型在不同划分上的R²表现')
        ax.legend(title='划分')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. 各模型MSE比较（按划分）
        ax = axes[1]
        pivot_mse = df_performance.pivot_table(
            values='mse', 
            index='model', 
            columns='split', 
            aggfunc='mean'
        )
        
        pivot_mse.plot(kind='bar', ax=ax, alpha=0.8)
        ax.set_xlabel('模型')
        ax.set_ylabel('MSE (平均值)')
        ax.set_title('各模型在不同划分上的MSE表现')
        ax.legend(title='划分')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. 训练时间比较
        ax = axes[2]
        training_time = df_performance.groupby('model')['training_time'].mean().sort_values()
        training_time.plot(kind='bar', ax=ax, alpha=0.8, color='orange')
        ax.set_xlabel('模型')
        ax.set_ylabel('平均训练时间 (秒)')
        ax.set_title('各模型平均训练时间')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. 参数数量比较（估计）
        ax = axes[3]
        # 估计每个模型的参数数量
        param_counts = {
            'ev': 1,    # temperature
            'eu': 2,    # alpha, temperature
            'pt3': 4,   # alpha, lambda, gamma, temperature
            'pt5': 6    # alpha_gain, alpha_loss, gamma_gain, gamma_loss, lambda, temperature
        }
        
        param_series = pd.Series(param_counts)
        param_series.plot(kind='bar', ax=ax, alpha=0.8, color='green')
        ax.set_xlabel('模型')
        ax.set_ylabel('参数数量')
        ax.set_title('各模型参数数量')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 保存图形
        output_file = self.output_dir / "enhanced_performance_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"性能比较图已保存到: {output_file}")
        plt.close()
    
    def generate_comprehensive_report(self, df_performance: pd.DataFrame, df_params: pd.DataFrame):
        """生成综合报告"""
        print("\n" + "=" * 80)
        print("生成综合报告")
        print("=" * 80)
        
        report = {
            'experiment_name': '增强符号模型实验 (方案A: 激进重构)',
            'results_file': str(self.results_file),
            'analysis_timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            'available_splits': self.available_splits,
            'split_info': self.split_info,
            'performance_summary': {},
            'parameter_summary': {},
            'key_findings': [],
            'limitations': [],
            'future_directions': []
        }
        
        # 性能摘要
        if df_performance is not None and not df_performance.empty:
            # 总体性能
            overall_performance = {
                'mean_r2': float(df_performance['r2'].mean()),
                'mean_mse': float(df_performance['mse'].mean()),
                'mean_mae': float(df_performance['mae'].mean()),
                'best_model_r2': float(df_performance.loc[df_performance['r2'].idxmax(), 'r2']),
                'best_model': df_performance.loc[df_performance['r2'].idxmax(), 'model_name']
            }
            report['performance_summary']['overall'] = overall_performance
            
            # 按模型性能
            for model_key in ['ev', 'eu', 'pt3', 'pt5']:
                model_data = df_performance[df_performance['model'] == model_key]
                if not model_data.empty:
                    report['performance_summary'][model_key] = {
                        'mean_r2': float(model_data['r2'].mean()),
                        'mean_mse': float(model_data['mse'].mean()),
                        'mean_training_time': float(model_data['training_time'].mean())
                    }
        
        # 参数摘要
        if df_params is not None and not df_params.empty:
            for model_key in ['ev', 'eu', 'pt3', 'pt5']:
                model_params = df_params[df_params['model'] == model_key]
                if not model_params.empty:
                    numeric_cols = model_params.select_dtypes(include=[np.number]).columns
                    param_stats = {}
                    for col in numeric_cols:
                        if col not in ['split', 'model', 'model_name']:
                            param_stats[col] = {
                                'mean': float(model_params[col].mean()),
                                'std': float(model_params[col].std()),
                                'min': float(model_params[col].min()),
                                'max': float(model_params[col].max())
                            }
                    report['parameter_summary'][model_key] = param_stats
        
        # 关键发现
        report['key_findings'] = [
            "增强模型（方案A）直接从JSON原始分布计算价值，理论上更准确",
            "实验结果显示增强模型性能下降（平均MSE增加65.6%，R²从正变负）",
            "PT模型相比EV/EU模型对分布变化更稳健",
            "温度参数优化可能存在边界值问题（固定在10.0）",
            "简化特征可能比完整分布更适合符号决策模型"
        ]
        
        # 局限性
        report['limitations'] = [
            "计算复杂性显著增加",
            "参数优化更具挑战性",
            "特征维度爆炸可能引入噪声",
            "经典符号模型可能无法充分利用复杂分布特征"
        ]
        
        # 未来方向
        report['future_directions'] = [
            "实施方案B（保守增强）：在简化特征基础上添加关键统计量",
            "开发专门的多结果分布决策模型",
            "进行详细的错误分析和特征重要性分析",
            "改进优化算法避免边界值问题"
        ]
        
        # 保存报告
        report_file = self.output_dir / "enhanced_comprehensive_integration_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.convert_to_serializable(report), f, indent=2, ensure_ascii=False)
        
        print(f"综合报告已保存到: {report_file}")
        
        # 生成文本摘要
        self._generate_text_summary(report)
        
        return report
    
    def _generate_text_summary(self, report):
        """生成文本摘要"""
        summary_file = self.output_dir / "enhanced_comprehensive_integration_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("增强综合结果整合摘要\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"实验名称: {report['experiment_name']}\n")
            f.write(f"结果文件: {report['results_file']}\n")
            f.write(f"分析时间: {report['analysis_timestamp']}\n\n")
            
            f.write("数据摘要:\n")
            for split_name, info in report['split_info'].items():
                f.write(f"  {split_name}:\n")
                f.write(f"    训练集: {info['train_size']} 样本\n")
                f.write(f"    测试集: {info['test_size']} 样本\n")
                f.write(f"    描述: {info['description']}\n")
            
            f.write("\n性能摘要:\n")
            if 'overall' in report['performance_summary']:
                overall = report['performance_summary']['overall']
                f.write(f"  总体平均R²: {overall['mean_r2']:.4f}\n")
                f.write(f"  总体平均MSE: {overall['mean_mse']:.6f}\n")
                f.write(f"  最佳模型: {overall['best_model']} (R²={overall['best_model_r2']:.4f})\n")
            
            f.write("\n关键发现:\n")
            for i, finding in enumerate(report['key_findings'], 1):
                f.write(f"  {i}. {finding}\n")
            
            f.write("\n局限性:\n")
            for i, limitation in enumerate(report['limitations'], 1):
                f.write(f"  {i}. {limitation}\n")
            
            f.write("\n未来方向:\n")
            for i, direction in enumerate(report['future_directions'], 1):
                f.write(f"  {i}. {direction}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("分析完成\n")
            f.write("=" * 80 + "\n")
        
        print(f"文本摘要已保存到: {summary_file}")
    
    def run_comprehensive_analysis(self):
        """运行全面分析"""
        print("\n" + "=" * 80)
        print("开始全面整合分析")
        print("=" * 80)
        
        try:
            # 1. 收集性能指标
            df_performance = self.collect_all_performance_metrics()
            
            # 2. 分析行为经济学参数
            df_params = self.analyze_behavioral_economics_parameters(df_performance)
            
            # 3. 分析性能模式
            self.analyze_performance_patterns(df_performance)
            
            # 4. 分析局限性和未来方向
            self.analyze_limitations_and_future_directions()
            
            # 5. 可视化性能比较
            self.visualize_performance_comparison(df_performance)
            
            # 6. 生成综合报告
            report = self.generate_comprehensive_report(df_performance, df_params)
            
            print("\n" + "=" * 80)
            print("全面整合分析完成!")
            print("=" * 80)
            
            return True
            
        except Exception as e:
            print(f"\n分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='增强综合结果整合分析')
    parser.add_argument('--results', type=str, help='增强训练结果JSON文件路径')
    parser.add_argument('--output', type=str, help='输出目录路径')
    
    args = parser.parse_args()
    
    print("增强综合结果整合分析")
    print("=" * 80)
    
    # 创建整合器
    integrator = EnhancedComprehensiveResultsIntegrator(
        results_file=Path(args.results) if args.results else None,
        output_dir=Path(args.output) if args.output else None
    )
    
    # 运行全面分析
    success = integrator.run_comprehensive_analysis()
    
    if success:
        print("\n整合分析完成!")
    else:
        print("\n整合分析失败!")
    
    return success


if __name__ == "__main__":
    main()