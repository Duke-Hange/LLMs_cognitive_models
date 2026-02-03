"""
增强符号模型训练框架
训练和评估增强版符号模型在不同数据划分上的表现
使用多结果分布数据
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

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 项目根路径并设置中文字体（避免中文/负号显示为方框）
_p = Path(__file__).resolve().parent
for _ in range(8):
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
from enhanced_symbolic_models import (
    create_enhanced_model,
    EnhancedModelAdapter,
    EnhancedSymbolicModel
)


class EnhancedModelTrainer:
    """增强模型训练器"""
    
    def __init__(self, standardizer: EnhancedChoices13kStandardizer,
                 results_dir: Optional[Path] = None,
                 curves: str = "mean_only"):
        """
        Args:
            standardizer: 增强数据标准化器
            results_dir: 结果保存目录
            curves: 训练曲线图模式，all=每种子+均值图，mean_only=仅均值图，none=不生成图（仍保存 JSON）
        """
        self.standardizer = standardizer
        self.results_dir = results_dir or Path(__file__).parent / "results" / "enhanced_training"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.curves_dir = self.results_dir / "curves"
        self.curves_dir.mkdir(parents=True, exist_ok=True)
        self.curves_mode = curves

        # 定义要训练的模型
        self.model_names = ['ev', 'eu', 'pt3', 'pt5']

        # 定义要测试的划分
        self.split_types = ['problem', 'parameter_amb', 'parameter_ev_extreme']

        # 存储结果
        self.results = {}
        self.results_aggregated: List[Dict[str, Any]] = []  # 多种子时汇总行
        self.n_seeds = 1
        self.standardized_data = None
    
    def load_and_prepare_data(self) -> Tuple[List[Dict], np.ndarray]:
        """加载和准备数据"""
        print("加载和标准化数据...")
        
        if self.standardized_data is None:
            self.standardized_data = self.standardizer.standardize_all()
        
        y = self.standardizer.get_target_vector()
        
        print(f"  标准化数据记录数: {len(self.standardized_data)}")
        print(f"  目标向量形状: {y.shape}")
        
        return self.standardized_data, y
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        评估模型性能
        
        Args:
            y_true: 真实值
            y_pred: 预测值
        
        Returns:
            评估指标字典
        """
        # 计算各种指标
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)
        
        # R^2 分数
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # 相关系数
        if len(y_true) > 1:
            corr = np.corrcoef(y_true, y_pred)[0, 1]
        else:
            corr = 0
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'correlation': float(corr) if not np.isnan(corr) else 0.0,
            'n_samples': len(y_true)
        }
    
    def _plot_single_curve(self, history: List[Dict], save_path: Path) -> None:
        """绘制单次训练曲线（迭代步 vs train_loss）并保存。"""
        if not history:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        iterations = [h["iteration"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(iterations, train_loss, label="Train Loss (MSE)", alpha=0.8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Train MSE")
        ax.legend()
        ax.set_title("Symbolic model training curve")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    
    def _generate_mean_curves(self, timestamp: str) -> None:
        """遍历 curves_dir 下本 run 的 JSON，按 (split_type, model_name) 分组画均值±标准差曲线。"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        pattern = f"run_{timestamp}_*.json"
        jsons = list(self.curves_dir.glob(pattern))
        if not jsons:
            return
        by_key: Dict[tuple, List[Dict]] = {}
        for p in jsons:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            key = (data["split_type"], data["model_name"])
            by_key.setdefault(key, []).append(data)
        for (split_type, model_name), group in by_key.items():
            max_iter = max(len(d.get("history") or []) for d in group)
            if max_iter == 0:
                continue
            train_arr = np.full((len(group), max_iter), np.nan)
            for i, d in enumerate(group):
                hist = d.get("history") or []
                for j, h in enumerate(hist):
                    if j < max_iter:
                        train_arr[i, j] = h["train_loss"]
            iters = list(range(1, max_iter + 1))
            mean_ = np.nanmean(train_arr, axis=0)
            std_ = np.nanstd(train_arr, axis=0)
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(iters, mean_, label="Train MSE (mean)", color="C0")
            ax.fill_between(iters, mean_ - std_, mean_ + std_, alpha=0.3, color="C0")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Train MSE")
            ax.set_title(f"Mean curve ± std — {split_type} / {model_name} (n={len(group)})")
            fig.tight_layout()
            out_path = self.curves_dir / f"mean_curve_{timestamp}_{split_type}_{model_name}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"  符号均值曲线已保存: {out_path}")
    
    def train_model_on_split(self, model_name: str, split_type: str, 
                            standardized_data: List[Dict], y: np.ndarray,
                            random_state: Optional[int] = None,
                            run_timestamp: Optional[str] = None,
                            seed: Optional[int] = None) -> Optional[Dict]:
        """
        在指定划分上训练模型
        
        Args:
            model_name: 模型名称 ('ev', 'eu', 'pt3', 'pt5')
            split_type: 划分类型
            standardized_data: 标准化数据
            y: 目标向量
            random_state: 划分随机种子（仅 problem 划分使用）
            run_timestamp: 本 run 时间戳，用于保存曲线文件
            seed: 当前种子，用于曲线文件名
        
        Returns:
            训练结果字典
        """
        print(f"\n训练 {model_name.upper()} 模型在 {split_type} 划分上...")
        
        # 创建数据划分
        train_idx, test_idx, split_info = create_enhanced_splits(
            standardized_data, split_type=split_type, random_state=random_state
        )
        
        # 准备训练和测试数据
        train_data = [standardized_data[i] for i in train_idx]
        test_data = [standardized_data[i] for i in test_idx]
        
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        print(f"  训练集: {len(train_data)} 样本")
        print(f"  测试集: {len(test_data)} 样本")
        print(f"  划分描述: {split_info['description']}")
        
        # 创建模型和适配器
        model = create_enhanced_model(model_name)
        adapter = EnhancedModelAdapter(model)
        
        try:
            # 训练模型
            print(f"  开始训练...")
            start_time = datetime.now()
            adapter.fit_from_standardized(train_data, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # 单次训练曲线：保存 JSON 与 PNG
            model = adapter.get_model()
            history = getattr(model, "fit_history", None) or []
            if run_timestamp is not None and seed is not None:
                curve_json = self.curves_dir / f"run_{run_timestamp}_seed{seed}_{split_type}_{model_name}.json"
                curve_png = self.curves_dir / f"run_{run_timestamp}_seed{seed}_{split_type}_{model_name}.png"
                with open(curve_json, "w", encoding="utf-8") as f:
                    json.dump({
                        "seed": seed,
                        "split_type": split_type,
                        "model_name": model_name,
                        "timestamp": run_timestamp,
                        "history": history,
                    }, f, indent=2, ensure_ascii=False)
                if self.curves_mode == "all":
                    self._plot_single_curve(history, curve_png)
            
            # 在训练集上评估
            y_train_pred = adapter.predict_from_standardized(train_data)
            train_metrics = self.evaluate_model(y_train, y_train_pred)
            
            # 在测试集上评估
            y_test_pred = adapter.predict_from_standardized(test_data)
            test_metrics = self.evaluate_model(y_test, y_test_pred)
            
            # 获取模型参数
            params = model.get_parameters()
            
            result = {
                'model_name': model_name,
                'split_type': split_type,
                'split_description': split_info['description'],
                'parameters': params,
                'training_time_seconds': training_time,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'train_predictions': y_train_pred.tolist(),
                'test_predictions': y_test_pred.tolist(),
                'train_indices': train_idx,
                'test_indices': test_idx,
                'split_info': split_info
            }
            
            print(f"  训练完成! 训练时间: {training_time:.2f} 秒")
            print(f"  训练集 MSE: {train_metrics['mse']:.6f}, R^2: {train_metrics['r2']:.4f}")
            print(f"  测试集 MSE: {test_metrics['mse']:.6f}, R^2: {test_metrics['r2']:.4f}")
            
            return result
            
        except Exception as e:
            print(f"  训练失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_all_models(self, seeds: Optional[List[int]] = None):
        """训练所有模型在所有划分上；多种子时对每个 seed 跑一遍并聚合 mean±std。"""
        if seeds is None:
            seeds = [42]
        self.n_seeds = len(seeds)
        print("=" * 80)
        print("开始训练所有增强符号模型" + (f"（多种子: {seeds}）" if self.n_seeds > 1 else ""))
        print("=" * 80)
        
        # 加载数据
        standardized_data, y = self.load_and_prepare_data()
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_per_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
        for seed in seeds:
            if self.n_seeds > 1:
                np.random.seed(seed)
            all_results = {}
            for split_type in self.split_types:
                print(f"\n{'='*60}")
                print(f"[seed={seed}] 处理划分: {split_type}")
                print(f"{'='*60}")
                split_results = {}
                for model_name in self.model_names:
                    result = self.train_model_on_split(
                        model_name, split_type, standardized_data, y,
                        random_state=seed,
                        run_timestamp=run_timestamp,
                        seed=seed,
                    )
                    if result:
                        split_results[model_name] = result
                all_results[split_type] = split_results
            results_per_seed[seed] = all_results
        
        # 保留第一个 seed 的完整结果（JSON 与单 seed 汇总兼容）
        self.results = results_per_seed[seeds[0]]
        
        if self.n_seeds > 1:
            # 聚合：每个 (split_type, model_name) 的 test_mse, test_r2, test_correlation 的 mean 和 std
            self.results_aggregated = []
            for split_type in self.split_types:
                for model_name in self.model_names:
                    mse_list = [
                        results_per_seed[s][split_type][model_name]["test_metrics"]["mse"]
                        for s in seeds
                        if results_per_seed[s][split_type].get(model_name)
                    ]
                    r2_list = [
                        results_per_seed[s][split_type][model_name]["test_metrics"]["r2"]
                        for s in seeds
                        if results_per_seed[s][split_type].get(model_name)
                    ]
                    corr_list = [
                        results_per_seed[s][split_type][model_name]["test_metrics"]["correlation"]
                        for s in seeds
                        if results_per_seed[s][split_type].get(model_name)
                    ]
                    if not mse_list:
                        continue
                    rep = results_per_seed[seeds[0]][split_type][model_name]
                    self.results_aggregated.append({
                        "split_type": split_type,
                        "model": model_name,
                        "split_description": rep["split_description"],
                        "test_mse_mean": float(np.mean(mse_list)),
                        "test_mse_std": float(np.std(mse_list)),
                        "test_r2_mean": float(np.mean(r2_list)),
                        "test_r2_std": float(np.std(r2_list)),
                        "test_correlation_mean": float(np.mean(corr_list)),
                        "test_correlation_std": float(np.std(corr_list)),
                        "n_seeds": self.n_seeds,
                    })
        else:
            self.results_aggregated = []
        # 生成符号模型均值曲线（同 split_type / model_name 多 seed 聚合）；--curves none 时跳过
        if self.curves_mode != "none":
            print("\n生成符号模型均值曲线...")
            self._generate_mean_curves(run_timestamp)
        return self.results
    
    def save_results(self):
        """保存训练结果"""
        if not self.results:
            print("没有结果可保存")
            return
        
        # 创建时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存完整结果
        results_file = self.results_dir / f"enhanced_models_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            # 转换 numpy 类型
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
            
            serializable_results = convert_to_serializable(self.results)
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n完整结果已保存到: {results_file}")
        
        # 保存汇总表格
        self.save_summary_table(timestamp)
        
        return results_file
    
    def save_summary_table(self, timestamp: str):
        """保存结果汇总表格；多种子时输出 test_*_mean/std，单 seed 时输出 test_*。"""
        if not self.results and not getattr(self, "results_aggregated", None):
            return
        
        summary_rows: List[Dict[str, Any]] = []
        multi_seed = self.n_seeds > 1 and getattr(self, "results_aggregated", None) and len(self.results_aggregated) > 0
        
        if multi_seed:
            summary_rows = list(self.results_aggregated)
        else:
            for split_type, split_results in self.results.items():
                for model_name, result in split_results.items():
                    if result is None:
                        continue
                    row = {
                        'split_type': split_type,
                        'model': model_name,
                        'model_name': result['model_name'],
                        'split_description': result['split_description'],
                        'training_time': result['training_time_seconds'],
                        'train_mse': result['train_metrics']['mse'],
                        'train_r2': result['train_metrics']['r2'],
                        'train_mae': result['train_metrics']['mae'],
                        'test_mse': result['test_metrics']['mse'],
                        'test_r2': result['test_metrics']['r2'],
                        'test_mae': result['test_metrics']['mae'],
                        'test_correlation': result['test_metrics']['correlation'],
                        'n_train': result['train_metrics']['n_samples'],
                        'n_test': result['test_metrics']['n_samples']
                    }
                    for param_name, param_value in result['parameters'].items():
                        if param_name != 'model_name':
                            row[f'param_{param_name}'] = param_value
                    summary_rows.append(row)
        
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            csv_file = self.results_dir / f"enhanced_models_summary_{timestamp}.csv"
            df_summary.to_csv(csv_file, index=False, encoding='utf-8')
            try:
                excel_file = self.results_dir / f"enhanced_models_summary_{timestamp}.xlsx"
                df_summary.to_excel(excel_file, index=False)
                print(f"Excel 汇总表格已保存到: {excel_file}")
            except Exception as e:
                print(f"无法保存 Excel 文件: {e}")
            print(f"CSV 汇总表格已保存到: {csv_file}")
            print("\n" + "=" * 80)
            print("增强模型训练结果汇总" + ("（多种子均值 ± 标准差）" if multi_seed else ""))
            print("=" * 80)
            for split_type in self.split_types:
                split_df = df_summary[df_summary['split_type'] == split_type]
                if split_df.empty:
                    continue
                print(f"\n划分: {split_type}")
                print("-" * 40)
                for _, row in split_df.iterrows():
                    if multi_seed:
                        print(f"  {row['model']:4s}: "
                              f"测试 MSE={row['test_mse_mean']:.6f} ± {row['test_mse_std']:.6f}, "
                              f"R^2={row['test_r2_mean']:.4f} ± {row['test_r2_std']:.4f}")
                    else:
                        print(f"  {row['model']:4s}: "
                              f"测试 MSE={row['test_mse']:.6f}, "
                              f"R^2={row['test_r2']:.4f}, "
                              f"相关性={row['test_correlation']:.4f}")
    
    def analyze_performance_comparison(self):
        """分析所有划分的性能比较"""
        if not self.results:
            print("训练结果不存在")
            return
        
        print("\n" + "=" * 80)
        print("增强模型性能比较分析 (三种划分)")
        print("=" * 80)
        
        # 分析每个划分
        for split_type in self.split_types:
            if split_type not in self.results:
                print(f"\n{split_type}: 结果不存在")
                continue
            
            split_results = self.results[split_type]
            if not split_results:
                print(f"\n{split_type}: 无模型结果")
                continue
            
            # 找出最佳模型
            best_model = None
            best_r2 = -float('inf')
            
            for model_name, result in split_results.items():
                if result is None:
                    continue
                
                test_r2 = result['test_metrics']['r2']
                if test_r2 > best_r2:
                    best_r2 = test_r2
                    best_model = model_name
            
            if best_model:
                print(f"\n{split_type}:")
                print(f"  最佳模型: {best_model} (测试集 R^2 = {best_r2:.4f})")
                print(f"  训练集大小: {split_results[best_model]['train_metrics']['n_samples']}")
                print(f"  测试集大小: {split_results[best_model]['test_metrics']['n_samples']}")
                print(f"  划分描述: {split_results[best_model]['split_description']}")
        
        # 总体比较
        print("\n" + "=" * 80)
        print("增强模型总体性能比较")
        print("=" * 80)
        
        comparison_data = []
        for split_type in self.split_types:
            if split_type in self.results and self.results[split_type]:
                for model_name, result in self.results[split_type].items():
                    if result is None:
                        continue
                    
                    comparison_data.append({
                        'split': split_type,
                        'model': model_name,
                        'test_mse': result['test_metrics']['mse'],
                        'test_r2': result['test_metrics']['r2'],
                        'test_corr': result['test_metrics']['correlation']
                    })
        
        # 按模型分组比较
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            
            print("\n按模型分类的性能比较:")
            for model in df_comparison['model'].unique():
                model_data = df_comparison[df_comparison['model'] == model]
                print(f"\n{model}:")
                for _, row in model_data.iterrows():
                    print(f"  {row['split']}: MSE={row['test_mse']:.6f}, R^2={row['test_r2']:.4f}, 相关性={row['test_corr']:.4f}")


def main():
    """主函数；支持 --seeds 多种子运行。"""
    import argparse
    parser = argparse.ArgumentParser(description="增强符号模型训练（支持多种子）")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="随机种子列表，如 --seeds 42 43 44 45 46（默认: 42）",
    )
    parser.add_argument(
        "--curves",
        type=str,
        choices=["all", "mean_only", "none"],
        default="mean_only",
        help="训练曲线图：all=每种子+均值图，mean_only=仅均值图，none=不生成图（仍保存 JSON）",
    )
    args = parser.parse_args()
    seeds: List[int] = args.seeds

    print("增强符号模型训练框架")
    print("=" * 80)
    
    try:
        # 初始化增强标准化器
        print("初始化增强数据标准化器...")
        project_root = Path(__file__).parent.parent.parent
        
        standardizer = EnhancedChoices13kStandardizer(
            selections_path=str(project_root / "数据集" / "choices13k-main" / "c13k_selections.csv"),
            problems_path=str(project_root / "数据集" / "choices13k-main" / "c13k_problems.json")
        )
        
        # 初始化训练器
        trainer = EnhancedModelTrainer(standardizer, curves=args.curves)

        # 训练所有模型（多种子时聚合 mean±std）
        results = trainer.train_all_models(seeds=seeds)
        
        # 保存结果
        trainer.save_results()
        
        # 分析性能比较
        trainer.analyze_performance_comparison()

        print("\n" + "=" * 80)
        print("增强模型训练完成!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    main()