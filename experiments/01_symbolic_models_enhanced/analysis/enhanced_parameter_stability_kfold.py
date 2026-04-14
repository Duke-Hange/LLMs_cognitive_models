"""
增强K折参数稳定性验证脚本
对每个数据划分进行K折交叉验证，评估增强模型参数估计的稳定性
基于JSON原始分布数据和增强模型架构
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 添加增强模型路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from enhanced_data_standardization import (
    EnhancedChoices13kStandardizer,
    create_enhanced_splits
)
from enhanced_symbolic_models import (
    create_enhanced_model,
    EnhancedModelAdapter,
    EnhancedSymbolicModel
)

# 导入K折划分
from sklearn.model_selection import KFold

# 设置随机种子以确保可重复性
np.random.seed(42)


class EnhancedKFoldStabilityAnalyzer:
    """增强K折参数稳定性分析器"""
    
    def __init__(self, n_folds: int = 5, output_dir: Optional[Path] = None):
        """
        Args:
            n_folds: K折数量
            output_dir: 输出目录
        """
        self.n_folds = n_folds
        self.output_dir = output_dir or Path(__file__).parent / "stability_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据标准化器
        self.standardizer = EnhancedChoices13kStandardizer(
            selections_path=str(Path(__file__).parent.parent.parent.parent / "数据集" / "choices13k-main" / "c13k_selections.csv"),
            problems_path=str(Path(__file__).parent.parent.parent.parent / "数据集" / "choices13k-main" / "c13k_problems.json")
        )
        
        # 定义要分析的模型
        self.models = {
            'ev': 'ev',
            'eu': 'eu', 
            'pt3': 'pt3',
            'pt5': 'pt5'
        }
        
        # 定义要分析的划分
        self.splits = ['problem', 'parameter_amb', 'parameter_ev_extreme']
        
        # 存储结果
        self.all_results = {}  # split -> model -> fold -> results
        self.parameter_summary = {}  # 参数稳定性汇总
        
        # 缓存数据
        self.standardized_data = None
        self.y = None
        
    def load_standardized_data(self) -> Tuple[List[Dict], np.ndarray]:
        """
        加载标准化数据
        
        Returns:
            (standardized_data, y): 标准化数据列表和目标向量
        """
        if self.standardized_data is None or self.y is None:
            print("加载和标准化数据...")
            df = self.standardizer.load_and_merge_data()
            standardized_data = self.standardizer.standardize_all()
            
            # 提取目标变量 bRate
            y = np.array([item['action']['bRate'] for item in standardized_data])
            
            self.standardized_data = standardized_data
            self.y = y
            
            print(f"数据加载完成: {len(standardized_data)} 条记录")
        
        return self.standardized_data, self.y
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        评估模型性能
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            性能指标字典
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 计算相关性
        if len(y_true) > 1:
            corr = np.corrcoef(y_true, y_pred)[0, 1]
        else:
            corr = 0.0
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2),
            'correlation': float(corr) if not np.isnan(corr) else 0.0,
            'n_samples': len(y_true)
        }
    
    def get_distributions_from_data(self, data_indices: List[int]) -> Tuple[List[List[List[float]]], List[List[List[float]]]]:
        """
        从标准化数据中提取分布
        
        Args:
            data_indices: 数据索引列表
            
        Returns:
            (distributions_a, distributions_b): Gamble A和B的分布列表
        """
        standardized_data = self.standardized_data
        
        distributions_a = []
        distributions_b = []
        
        for idx in data_indices:
            item = standardized_data[idx]
            distributions_a.append(item['context']['gamble_a']['distribution'])
            distributions_b.append(item['context']['gamble_b']['distribution'])
        
        return distributions_a, distributions_b
    
    def run_kfold_for_split_model(self, split_name: str, model_key: str, 
                                 model_name: str) -> Dict[int, Dict]:
        """
        对特定划分和模型运行K折验证
        
        Args:
            split_name: 划分名称
            model_key: 模型键名
            model_name: 模型名称
            
        Returns:
            字典: fold_index -> 结果字典
        """
        print(f"\n{'='*60}")
        print(f"开始K折验证: {model_key} 模型在 {split_name} 上")
        print(f"{'='*60}")
        
        # 加载数据
        standardized_data, y = self.load_standardized_data()
        
        # 创建数据划分
        train_idx, test_idx, split_info = create_enhanced_splits(
            standardized_data, split_type=split_name
        )
        
        print(f"原始训练集大小: {len(train_idx)}")
        print(f"原始测试集大小: {len(test_idx)}")
        print(f"划分描述: {split_info['description']}")
        
        # 初始化KFold
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        fold_results = {}
        
        for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(kf.split(train_idx)):
            print(f"\n--- 折叠 {fold_idx+1}/{self.n_folds} ---")
            
            # 获取当前折叠的实际索引
            actual_train_idx = [train_idx[i] for i in fold_train_idx]
            actual_val_idx = [train_idx[i] for i in fold_val_idx]
            
            print(f"  折叠训练集: {len(actual_train_idx)} 样本")
            print(f"  折叠验证集: {len(actual_val_idx)} 样本")
            
            # 创建模型和适配器
            model = create_enhanced_model(model_name)
            adapter = EnhancedModelAdapter(model)
            
            try:
                # 提取分布数据
                train_dist_a, train_dist_b = self.get_distributions_from_data(actual_train_idx)
                val_dist_a, val_dist_b = self.get_distributions_from_data(actual_val_idx)
                test_dist_a, test_dist_b = self.get_distributions_from_data(test_idx)
                
                # 训练模型
                start_time = datetime.now()
                adapter.fit_from_standardized(
                    [standardized_data[i] for i in actual_train_idx],
                    y[actual_train_idx]
                )
                training_time = (datetime.now() - start_time).total_seconds()
                
                # 在验证集上评估
                y_val_pred = adapter.predict_from_standardized(
                    [standardized_data[i] for i in actual_val_idx]
                )
                y_val_true = y[actual_val_idx]
                val_metrics = self.evaluate_model(y_val_true, y_val_pred)
                
                # 在整个训练集上评估
                y_train_pred = adapter.predict_from_standardized(
                    [standardized_data[i] for i in train_idx]
                )
                y_train_true = y[train_idx]
                train_metrics = self.evaluate_model(y_train_true, y_train_pred)
                
                # 在测试集上评估
                y_test_pred = adapter.predict_from_standardized(
                    [standardized_data[i] for i in test_idx]
                )
                y_test_true = y[test_idx]
                test_metrics = self.evaluate_model(y_test_true, y_test_pred)
                
                # 收集结果
                fold_results[fold_idx] = {
                    'parameters': model.get_parameters(),
                    'training_time': training_time,
                    'val_metrics': val_metrics,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'train_indices': actual_train_idx,
                    'val_indices': actual_val_idx,
                    'test_indices': test_idx
                }
                
                print(f"  验证集 R^2: {val_metrics['r2']:.4f}, 测试集 R^2: {test_metrics['r2']:.4f}")
                
            except Exception as e:
                print(f"  折叠 {fold_idx+1} 训练失败: {e}")
                import traceback
                traceback.print_exc()
                fold_results[fold_idx] = None
        
        return fold_results
    
    def calculate_parameter_stability(self, fold_results: Dict[int, Dict]) -> Dict:
        """
        计算参数稳定性指标
        
        Args:
            fold_results: K折结果字典
            
        Returns:
            稳定性指标字典
        """
        # 过滤掉失败的结果
        valid_results = [r for r in fold_results.values() if r is not None]
        if not valid_results:
            return {}
        
        # 提取所有参数
        all_params = []
        for result in valid_results:
            params = result['parameters'].copy()
            params['training_time'] = result['training_time']
            all_params.append(params)
        
        # 转换为DataFrame以便分析
        df_params = pd.DataFrame(all_params)
        
        stability_metrics = {}
        
        for column in df_params.columns:
            if df_params[column].dtype in [np.float64, np.int64]:
                values = df_params[column].dropna()
                if len(values) > 1:
                    mean_val = values.mean()
                    std_val = values.std()
                    cv = std_val / mean_val if mean_val != 0 else np.nan
                    min_val = values.min()
                    max_val = values.max()
                    range_val = max_val - min_val
                    
                    stability_metrics[column] = {
                        'mean': float(mean_val),
                        'std': float(std_val),
                        'cv': float(cv),
                        'min': float(min_val),
                        'max': float(max_val),
                        'range': float(range_val),
                        'n_valid': len(values)
                    }
        
        return stability_metrics
    
    def run_all_analyses(self) -> bool:
        """
        运行所有K折分析
        
        Returns:
            成功与否
        """
        print("=" * 80)
        print("增强模型K折参数稳定性验证")
        print("=" * 80)
        print(f"K折数量: {self.n_folds}")
        print(f"模型数量: {len(self.models)}")
        print(f"划分数量: {len(self.splits)}")
        print(f"总训练次数: {self.n_folds * len(self.models) * len(self.splits)}")
        print("=" * 80)
        
        try:
            # 预加载数据
            self.load_standardized_data()
            
            for split_name in self.splits:
                self.all_results[split_name] = {}
                
                for model_key, model_name in self.models.items():
                    print(f"\n处理: {split_name} -> {model_key}")
                    
                    # 运行K折验证
                    fold_results = self.run_kfold_for_split_model(split_name, model_key, model_name)
                    self.all_results[split_name][model_key] = fold_results
                    
                    # 计算参数稳定性
                    stability_metrics = self.calculate_parameter_stability(fold_results)
                    if stability_metrics:
                        if split_name not in self.parameter_summary:
                            self.parameter_summary[split_name] = {}
                        self.parameter_summary[split_name][model_key] = stability_metrics
            
            # 保存结果
            self.save_results()
            
            # 生成报告
            self.generate_report()
            
            return True
            
        except Exception as e:
            print(f"\nK折分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_results(self):
        """保存所有结果到文件"""
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detailed_file = self.output_dir / f"enhanced_kfold_detailed_results_{timestamp}.json"
        
        # 转换为可序列化的格式
        serializable_results = {}
        for split_name, split_results in self.all_results.items():
            serializable_results[split_name] = {}
            for model_key, fold_results in split_results.items():
                serializable_results[split_name][model_key] = {}
                for fold_idx, result in fold_results.items():
                    if result is not None:
                        serializable_results[split_name][model_key][str(fold_idx)] = result
        
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细结果已保存到: {detailed_file}")
        
        # 保存参数稳定性汇总
        if self.parameter_summary:
            summary_file = self.output_dir / f"enhanced_parameter_stability_summary_{timestamp}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(self.parameter_summary, f, indent=2, ensure_ascii=False)
            print(f"参数稳定性汇总已保存到: {summary_file}")
            
            # 保存为CSV格式以便分析
            self.save_summary_csv(timestamp)
    
    def save_summary_csv(self, timestamp: str):
        """保存参数稳定性汇总为CSV格式"""
        if not self.parameter_summary:
            return
        
        rows = []
        for split_name, split_data in self.parameter_summary.items():
            for model_key, model_data in split_data.items():
                for param_name, param_stats in model_data.items():
                    row = {
                        'split': split_name,
                        'model': model_key,
                        'parameter': param_name,
                        'mean': param_stats['mean'],
                        'std': param_stats['std'],
                        'cv': param_stats['cv'],
                        'min': param_stats['min'],
                        'max': param_stats['max'],
                        'range': param_stats['range'],
                        'n_valid': param_stats['n_valid']
                    }
                    rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            csv_file = self.output_dir / f"enhanced_parameter_stability_summary_{timestamp}.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"参数稳定性汇总CSV已保存到: {csv_file}")
    
    def generate_report(self):
        """生成分析报告"""
        print("\n" + "=" * 80)
        print("增强K折参数稳定性验证报告")
        print("=" * 80)
        
        if not self.parameter_summary:
            print("没有有效的参数稳定性数据")
            return
        
        print("\n参数稳定性汇总:")
        print("-" * 80)
        
        for split_name, split_data in self.parameter_summary.items():
            print(f"\n划分: {split_name}")
            print("-" * 40)
            
            for model_key, model_data in split_data.items():
                print(f"\n  模型: {model_key}")
                
                for param_name, param_stats in model_data.items():
                    cv = param_stats['cv']
                    cv_str = f"{cv:.3f}" if not np.isnan(cv) else "NaN"
                    
                    print(f"    {param_name}:")
                    print(f"      均值: {param_stats['mean']:.4f}")
                    print(f"      标准差: {param_stats['std']:.4f}")
                    print(f"      变异系数: {cv_str}")
                    print(f"      范围: [{param_stats['min']:.4f}, {param_stats['max']:.4f}]")
        
        print("\n" + "=" * 80)
        print("稳定性等级评估:")
        print("-" * 80)
        
        # 评估参数稳定性
        stable_params = []
        moderately_stable_params = []
        unstable_params = []
        
        for split_name, split_data in self.parameter_summary.items():
            for model_key, model_data in split_data.items():
                for param_name, param_stats in model_data.items():
                    cv = param_stats['cv']
                    
                    if np.isnan(cv):
                        continue
                    
                    if cv < 0.1:
                        stable_params.append((split_name, model_key, param_name, cv))
                    elif cv < 0.3:
                        moderately_stable_params.append((split_name, model_key, param_name, cv))
                    else:
                        unstable_params.append((split_name, model_key, param_name, cv))
        
        print(f"\n高度稳定参数 (CV < 0.1): {len(stable_params)} 个")
        for split, model, param, cv in stable_params[:5]:  # 显示前5个
            print(f"  {split}/{model}/{param}: CV={cv:.3f}")
        
        print(f"\n中等稳定参数 (0.1 ≤ CV < 0.3): {len(moderately_stable_params)} 个")
        for split, model, param, cv in moderately_stable_params[:5]:
            print(f"  {split}/{model}/{param}: CV={cv:.3f}")
        
        print(f"\n不稳定参数 (CV ≥ 0.3): {len(unstable_params)} 个")
        for split, model, param, cv in unstable_params[:5]:
            print(f"  {split}/{model}/{param}: CV={cv:.3f}")
        
        # 保存评估报告
        self.save_stability_assessment(
            stable_params, moderately_stable_params, unstable_params
        )
    
    def save_stability_assessment(self, stable_params, moderately_stable_params, unstable_params):
        """保存稳定性评估结果"""
        assessment = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'summary': {
                'total_parameters': len(stable_params) + len(moderately_stable_params) + len(unstable_params),
                'stable_count': len(stable_params),
                'moderately_stable_count': len(moderately_stable_params),
                'unstable_count': len(unstable_params)
            },
            'stable_parameters': [
                {'split': s, 'model': m, 'parameter': p, 'cv': cv}
                for s, m, p, cv in stable_params
            ],
            'moderately_stable_parameters': [
                {'split': s, 'model': m, 'parameter': p, 'cv': cv}
                for s, m, p, cv in moderately_stable_params
            ],
            'unstable_parameters': [
                {'split': s, 'model': m, 'parameter': p, 'cv': cv}
                for s, m, p, cv in unstable_params
            ]
        }
        
        assessment_file = self.output_dir / f"enhanced_stability_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(assessment_file, 'w', encoding='utf-8') as f:
            json.dump(assessment, f, indent=2, ensure_ascii=False)
        
        print(f"\n稳定性评估报告已保存到: {assessment_file}")


def main():
    """主函数"""
    print("增强K折参数稳定性验证")
    print("=" * 80)
    
    # 创建分析器
    analyzer = EnhancedKFoldStabilityAnalyzer(n_folds=3)
    
    # 运行分析
    success = analyzer.run_all_analyses()
    
    if success:
        print("\n" + "=" * 80)
        print("增强K折参数稳定性验证完成!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("增强K折参数稳定性验证失败!")
        print("=" * 80)
    
    return success


if __name__ == "__main__":
    main()