"""
增强实验与原始实验结果比较
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json

# 项目根路径并设置中文字体（避免中文/负号显示为方框）
_p = Path(__file__).resolve().parent
for _ in range(8):
    _p = _p.parent
    if (_p / "shared").is_dir():
        sys.path.insert(0, str(_p))
        break
from shared.visualization import setup_chinese_font
setup_chinese_font()

def load_results():
    """加载两个实验的结果（原始基线已内置在 results/comparison/ 下，增强目录可独立运行）"""
    base = Path(__file__).parent
    enhanced_path = base / "results" / "enhanced_training" / "enhanced_models_summary_20260120_234923.csv"
    # 优先使用本目录内嵌的原始基线，不依赖 01_symbolic_models / 01_symbolic_models_legacy
    original_path = base / "results" / "comparison" / "original_baseline_summary.csv"
    if not original_path.exists():
        original_path = base.parent / "01_symbolic_models" / "results" / "training" / "symbolic_models_summary_20260120_194604.csv"
    if not original_path.exists():
        original_path = base.parent / "01_symbolic_models_legacy" / "results" / "training" / "symbolic_models_summary_20260120_194604.csv"
    if not original_path.exists():
        raise FileNotFoundError(
            "未找到原始基线结果。请确保存在 results/comparison/original_baseline_summary.csv 或原实验目录下的 symbolic_models_summary_20260120_194604.csv"
        )

    enhanced_df = pd.read_csv(enhanced_path)
    original_df = pd.read_csv(original_path)
    
    # 标准化列名以便比较
    enhanced_df = enhanced_df.rename(columns={'split_type': 'split', 'train_mae': 'train_mae_enhanced', 
                                              'test_mae': 'test_mae_enhanced'})
    
    # 标准化原始CSV中的split值（移除"_split"后缀）
    original_df['split'] = original_df['split'].str.replace('_split', '', regex=False)
    
    original_df = original_df.rename(columns={'train_mse': 'train_mse_original', 'test_mse': 'test_mse_original',
                                              'train_r2': 'train_r2_original', 'test_r2': 'test_r2_original'})
    
    # 合并数据
    comparison_df = pd.merge(
        enhanced_df, 
        original_df[['split', 'model', 'train_mse_original', 'test_mse_original', 
                     'train_r2_original', 'test_r2_original']],
        on=['split', 'model'],
        how='inner'
    )
    
    # 计算差异
    comparison_df['mse_diff'] = comparison_df['test_mse'] - comparison_df['test_mse_original']
    comparison_df['r2_diff'] = comparison_df['test_r2'] - comparison_df['test_r2_original']
    comparison_df['mse_ratio'] = comparison_df['test_mse'] / comparison_df['test_mse_original']
    
    # 重命名增强结果列
    comparison_df = comparison_df.rename(columns={
        'train_mse': 'train_mse_enhanced',
        'test_mse': 'test_mse_enhanced',
        'train_r2': 'train_r2_enhanced', 
        'test_r2': 'test_r2_enhanced'
    })
    
    return comparison_df

def analyze_comparison(comparison_df):
    """分析比较结果"""
    print("=" * 80)
    print("增强实验 vs 原始实验 - 性能比较")
    print("=" * 80)
    
    # 整体统计
    print("\n整体性能比较:")
    print(f"总比较条目数: {len(comparison_df)}")
    print(f"平均MSE比率 (增强/原始): {comparison_df['mse_ratio'].mean():.3f}")
    print(f"平均R²差异 (增强-原始): {comparison_df['r2_diff'].mean():.3f}")
    
    # 按划分类型分析
    print("\n按划分类型分析:")
    for split in comparison_df['split'].unique():
        split_df = comparison_df[comparison_df['split'] == split]
        print(f"\n{split}:")
        print(f"  条目数: {len(split_df)}")
        print(f"  平均MSE比率: {split_df['mse_ratio'].mean():.3f}")
        print(f"  平均R²差异: {split_df['r2_diff'].mean():.3f}")
        
    # 按模型类型分析
    print("\n按模型类型分析:")
    for model in comparison_df['model'].unique():
        model_df = comparison_df[comparison_df['model'] == model]
        print(f"\n{model}:")
        print(f"  平均MSE比率: {model_df['mse_ratio'].mean():.3f}")
        print(f"  平均R²差异: {model_df['r2_diff'].mean():.3f}")
    
    # 性能改进/退步统计
    print("\n性能变化统计:")
    improved_mse = (comparison_df['mse_ratio'] < 1.0).sum()
    improved_r2 = (comparison_df['r2_diff'] > 0).sum()
    total = len(comparison_df)
    
    print(f"MSE改进的模型数: {improved_mse}/{total} ({improved_mse/total*100:.1f}%)")
    print(f"R²改进的模型数: {improved_r2}/{total} ({improved_r2/total*100:.1f}%)")
    
    # 最佳和最差表现
    best_mse = comparison_df.loc[comparison_df['mse_ratio'].idxmin()]
    worst_mse = comparison_df.loc[comparison_df['mse_ratio'].idxmax()]
    
    print(f"\n最佳MSE改进: {best_mse['model']} ({best_mse['split']}) - 比率: {best_mse['mse_ratio']:.3f}")
    print(f"最差MSE表现: {worst_mse['model']} ({worst_mse['split']}) - 比率: {worst_mse['mse_ratio']:.3f}")
    
    return comparison_df

def save_comparison_report(comparison_df):
    """保存比较报告"""
    report_dir = Path(__file__).parent / "results" / "comparison"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存CSV
    csv_path = report_dir / "enhanced_vs_original_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n详细比较结果已保存至: {csv_path}")
    
    # 创建摘要报告
    summary = {
        "timestamp": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
        "total_comparisons": len(comparison_df),
        "avg_mse_ratio": float(comparison_df['mse_ratio'].mean()),
        "avg_r2_diff": float(comparison_df['r2_diff'].mean()),
        "mse_improved_count": int((comparison_df['mse_ratio'] < 1.0).sum()),
        "r2_improved_count": int((comparison_df['r2_diff'] > 0).sum()),
        "by_split": {},
        "by_model": {}
    }
    
    for split in comparison_df['split'].unique():
        split_df = comparison_df[comparison_df['split'] == split]
        summary["by_split"][split] = {
            "avg_mse_ratio": float(split_df['mse_ratio'].mean()),
            "avg_r2_diff": float(split_df['r2_diff'].mean())
        }
    
    for model in comparison_df['model'].unique():
        model_df = comparison_df[comparison_df['model'] == model]
        summary["by_model"][model] = {
            "avg_mse_ratio": float(model_df['mse_ratio'].mean()),
            "avg_r2_diff": float(model_df['r2_diff'].mean())
        }
    
    json_path = report_dir / "comparison_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"摘要报告已保存至: {json_path}")
    
    return csv_path, json_path

def main():
    """主函数"""
    print("开始比较增强实验与原始实验结果...")
    
    # 加载结果
    comparison_df = load_results()
    
    # 分析比较
    print(f"\n加载了 {len(comparison_df)} 个比较条目")
    print("列名:", comparison_df.columns.tolist())
    
    # 分析
    comparison_df = analyze_comparison(comparison_df)
    
    # 保存报告
    csv_path, json_path = save_comparison_report(comparison_df)
    
    print("\n比较完成!")
    
    # 显示关键发现
    print("\n" + "=" * 80)
    print("关键发现摘要:")
    print("=" * 80)
    
    avg_mse_ratio = comparison_df['mse_ratio'].mean()
    if avg_mse_ratio < 1.0:
        print(f"✓ 增强实验平均MSE为原始实验的 {avg_mse_ratio:.1%} (改进)")
    else:
        print(f"✗ 增强实验平均MSE为原始实验的 {avg_mse_ratio:.1%} (退步)")
    
    avg_r2_diff = comparison_df['r2_diff'].mean()
    if avg_r2_diff > 0:
        print(f"✓ 增强实验平均R²比原始实验高 {avg_r2_diff:.3f} (改进)")
    else:
        print(f"✗ 增强实验平均R²比原始实验低 {abs(avg_r2_diff):.3f} (退步)")
    
    improved_count = (comparison_df['mse_ratio'] < 1.0).sum()
    total = len(comparison_df)
    print(f"✓ {improved_count}/{total} 个模型在MSE上有所改进")
    
    return comparison_df

if __name__ == "__main__":
    main()