"""
增强版数据准备脚本
加载和标准化 Choices13k 数据集，基于JSON原始分布提取完整特征
为增强符号模型实验准备数据，支持三种划分策略
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 项目根路径并设置中文字体（避免中文/负号显示为方框）
_p = Path(__file__).resolve().parent
for _ in range(8):
    _p = _p.parent
    if (_p / "shared").is_dir():
        sys.path.insert(0, str(_p))
        break
from shared.visualization import setup_chinese_font
setup_chinese_font()

# 使用增强实验目录（项目仅保留 01_symbolic_models_enhanced）
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "01_symbolic_models_enhanced"))
from enhanced_data_standardization import EnhancedChoices13kStandardizer, create_enhanced_splits


def setup_paths(strategy_name="problem_split"):
    """
    设置路径
    
    Args:
        strategy_name: 策略名称 ('problem_split', 'parameter_amb_split', 'parameter_ev_extreme_split')
    
    Returns:
        路径字典
    """
    project_root = Path(__file__).parent.parent.parent.parent
    strategies_dir = project_root / "experiments" / "00_data_preparation" / "strategies"
    
    # 验证策略目录存在
    strategy_dir = strategies_dir / strategy_name
    if not strategy_dir.exists():
        raise ValueError(f"策略目录不存在: {strategy_dir}")
    
    # 策略目录中的文件路径
    selections_path = strategy_dir / "c13k_selections.csv"
    problems_path = strategy_dir / "c13k_problems.json"
    
    # 输出目录
    output_dir = project_root / "experiments" / "00_data_preparation" / "outputs" / strategy_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'strategy_name': strategy_name,
        'strategy_dir': strategy_dir,
        'selections': str(selections_path),
        'problems': str(problems_path),
        'output_dir': output_dir,
        'standardized_json': str(output_dir / "c13k_enhanced_standardized.json"),
        'features_npy': str(output_dir / "enhanced_features.npy"),
        'targets_npy': str(output_dir / "enhanced_targets.npy"),
        'metadata_csv': str(output_dir / "enhanced_metadata.csv"),
        'feature_names_txt': str(output_dir / "enhanced_feature_names.txt"),
        'stats_txt': str(output_dir / "enhanced_data_statistics.txt"),
        'splits_json': str(output_dir / "enhanced_data_splits.json")
    }


def standardize_enhanced_data(paths):
    """
    标准化增强数据
    
    Args:
        paths: 路径字典
        
    Returns:
        (standardizer, standardized_data, X, y, feature_names)
    """
    print("=" * 60)
    print(f"开始增强数据标准化 - {paths['strategy_name']}")
    print("=" * 60)
    
    # 初始化增强标准化器
    standardizer = EnhancedChoices13kStandardizer(
        selections_path=paths['selections'],
        problems_path=paths['problems']
    )
    
    # 加载和标准化所有数据
    standardized_data = standardizer.standardize_all(
        save_path=paths['standardized_json']
    )
    
    # 获取增强特征矩阵和目标向量
    X, feature_names = standardizer.get_enhanced_feature_matrix()
    y = standardizer.get_target_vector()
    
    # 保存为numpy格式以便快速加载
    np.save(paths['features_npy'], X)
    np.save(paths['targets_npy'], y)
    
    # 保存特征名称
    with open(paths['feature_names_txt'], 'w', encoding='utf-8') as f:
        f.write('\n'.join(feature_names))
    
    print(f"\n增强特征矩阵形状: {X.shape}")
    print(f"目标向量形状: {y.shape}")
    print(f"特征数量: {len(feature_names)}")
    
    # 显示特征统计
    print(f"\n前15个特征:")
    for i, name in enumerate(feature_names[:15]):
        print(f"  {i:3d}: {name}")
    
    return standardizer, standardized_data, X, y, feature_names


def create_enhanced_metadata_dataframe(standardized_data):
    """
    创建增强元数据DataFrame用于划分
    
    Args:
        standardized_data: 标准化数据
        
    Returns:
        元数据DataFrame
    """
    metadata_list = []
    for item in standardized_data:
        features = item['context']['features']
        metadata_list.append({
            'index': item['metadata']['index'],
            'problem_id': item['metadata']['problem_id'],
            'json_problem_id': item['metadata']['json_problem_id'],
            'feedback': item['metadata']['feedback_condition'],
            'block': item['metadata']['block'],
            'bRate': item['action']['bRate'],
            'bRate_std': item['action']['bRate_std'],
            'n_subjects': item['action']['n_subjects'],
            'ev_diff': features.get('ev_diff', 0),
            'Amb': features.get('Amb', False),
            'Corr': features.get('Corr', 0),
            'Ha': features.get('Ha', 0),
            'pHa': features.get('pHa', 0),
            'La': features.get('La', 0),
            'Hb': features.get('Hb', 0),
            'pHb': features.get('pHb', 0),
            'Lb': features.get('Lb', 0),
        })
    
    return pd.DataFrame(metadata_list)


def analyze_enhanced_data(standardizer, standardized_data, X, y, feature_names, paths):
    """
    分析增强数据并生成统计信息
    
    Args:
        standardizer: 增强标准化器
        standardized_data: 标准化数据
        X: 特征矩阵
        y: 目标向量
        feature_names: 特征名称列表
        paths: 路径字典
        
    Returns:
        元数据DataFrame
    """
    print("\n" + "=" * 60)
    print("增强数据分析")
    print("=" * 60)
    
    # 创建元数据DataFrame
    df_meta = create_enhanced_metadata_dataframe(standardized_data)
    
    # 保存元数据
    df_meta.to_csv(paths['metadata_csv'], index=False)
    
    # 生成统计信息
    stats_lines = []
    
    # 基本统计
    stats_lines.append("=" * 70)
    stats_lines.append("CHOICES13K 增强数据集统计")
    stats_lines.append("=" * 70)
    stats_lines.append(f"策略名称: {paths['strategy_name']}")
    stats_lines.append(f"总样本数: {len(standardized_data)}")
    stats_lines.append(f"增强特征维度: {X.shape[1]}")
    
    # 数据来源统计
    stats_lines.append(f"\n数据来源:")
    stats_lines.append(f"  CSV文件: {paths['selections']}")
    stats_lines.append(f"  JSON文件: {paths['problems']}")
    
    # Feedback条件统计
    feedback_counts = df_meta['feedback'].value_counts()
    stats_lines.append(f"\nFeedback条件分布:")
    stats_lines.append(f"  有反馈 (Feedback=True): {feedback_counts.get(True, 0)} 条")
    stats_lines.append(f"  无反馈 (Feedback=False): {feedback_counts.get(False, 0)} 条")
    
    # Block分布
    block_counts = df_meta['block'].value_counts().sort_index()
    stats_lines.append(f"\nBlock分布:")
    for block, count in block_counts.items():
        stats_lines.append(f"  Block {block}: {count} 条")
    
    # bRate统计
    stats_lines.append(f"\nbRate统计:")
    stats_lines.append(f"  均值: {y.mean():.4f}")
    stats_lines.append(f"  标准差: {y.std():.4f}")
    stats_lines.append(f"  最小值: {y.min():.4f}")
    stats_lines.append(f"  最大值: {y.max():.4f}")
    stats_lines.append(f"  中位数: {np.median(y):.4f}")
    
    # Amb（模糊性）统计
    amb_counts = df_meta['Amb'].value_counts()
    stats_lines.append(f"\n模糊性(Amb)分布:")
    stats_lines.append(f"  非模糊 (Amb=0): {amb_counts.get(0, 0)} 条")
    stats_lines.append(f"  模糊 (Amb=1): {amb_counts.get(1, 0)} 条")
    
    # EV_diff统计
    stats_lines.append(f"\n期望值差(EV_diff)统计:")
    stats_lines.append(f"  均值: {df_meta['ev_diff'].mean():.4f}")
    stats_lines.append(f"  标准差: {df_meta['ev_diff'].std():.4f}")
    stats_lines.append(f"  最小值: {df_meta['ev_diff'].min():.4f}")
    stats_lines.append(f"  最大值: {df_meta['ev_diff'].max():.4f}")
    stats_lines.append(f"  25th百分位数: {df_meta['ev_diff'].quantile(0.25):.4f}")
    stats_lines.append(f"  75th百分位数: {df_meta['ev_diff'].quantile(0.75):.4f}")
    
    # 增强特征统计
    stats_lines.append(f"\n增强特征统计 (前10个特征):")
    for i in range(min(10, len(feature_names))):
        feat = X[:, i]
        feat_name = feature_names[i]
        stats_lines.append(f"  {feat_name}: 均值={feat.mean():.4f}, 标准差={feat.std():.4f}, 范围=[{feat.min():.4f}, {feat.max():.4f}]")
    
    # 保存统计信息
    with open(paths['stats_txt'], 'w', encoding='utf-8') as f:
        f.write('\n'.join(stats_lines))
    
    # 打印统计信息
    for line in stats_lines[:50]:  # 限制打印行数
        print(line)
    
    if len(stats_lines) > 50:
        print(f"\n... 更多统计信息已保存到: {paths['stats_txt']}")
    
    return df_meta


def create_enhanced_data_splits(standardized_data, df_meta, paths):
    """
    创建增强数据划分（三种策略）
    
    Args:
        standardized_data: 标准化数据
        df_meta: 元数据DataFrame
        paths: 路径字典
        
    Returns:
        划分字典
    """
    print("\n" + "=" * 60)
    print("创建增强数据划分")
    print("=" * 60)
    
    splits = {}
    split_types = ['problem', 'parameter_amb', 'parameter_ev_extreme']
    
    for split_type in split_types:
        print(f"\n创建 {split_type} 划分...")
        
        train_indices, test_indices, split_info = create_enhanced_splits(
            standardized_data,
            split_type=split_type,
            test_size=0.2 if split_type == 'problem' else None,
            random_state=42
        )
        
        # 构建划分信息
        splits[split_type] = {
            'train_indices': train_indices,
            'test_indices': test_indices,
            'split_info': split_info,
            'train_bRate_mean': float(df_meta.iloc[train_indices]['bRate'].mean()) if train_indices else 0,
            'test_bRate_mean': float(df_meta.iloc[test_indices]['bRate'].mean()) if test_indices else 0
        }
        
        print(f"  {split_info['description']}")
        print(f"  训练集大小: {len(train_indices)}")
        print(f"  测试集大小: {len(test_indices)}")
        print(f"  训练集bRate均值: {splits[split_type]['train_bRate_mean']:.4f}")
        print(f"  测试集bRate均值: {splits[split_type]['test_bRate_mean']:.4f}")
    
    # 保存划分
    with open(paths['splits_json'], 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2, default=str)
    
    print(f"\n所有划分已保存到: {paths['splits_json']}")
    
    return splits


def create_enhanced_visualizations(df_meta, X, y, feature_names, paths):
    """
    创建增强可视化图表
    
    Args:
        df_meta: 元数据DataFrame
        X: 特征矩阵
        y: 目标向量
        feature_names: 特征名称列表
        paths: 路径字典
    """
    print("\n" + "=" * 60)
    print("创建增强可视化图表")
    print("=" * 60)
    
    # 图表风格（中文字体已在模块开头通过 setup_chinese_font() 设置）
    sns.set_style("whitegrid")

    fig_dir = paths['output_dir'] / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # 1. bRate分布直方图
    plt.figure(figsize=(12, 8))
    plt.hist(y, bins=50, edgecolor='black', alpha=0.7, density=True)
    plt.xlabel('bRate (选择Gamble B的概率)')
    plt.ylabel('密度')
    plt.title(f'bRate分布直方图 - {paths["strategy_name"]}')
    plt.axvline(x=y.mean(), color='r', linestyle='--', label=f'均值: {y.mean():.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "bRate_distribution.png", dpi=150)
    plt.close()
    
    # 2. Feedback条件对比
    plt.figure(figsize=(10, 6))
    feedback_labels = ['无反馈', '有反馈']
    feedback_data = [
        y[df_meta['feedback'] == False],
        y[df_meta['feedback'] == True]
    ]
    
    plt.boxplot(feedback_data, labels=feedback_labels)
    plt.ylabel('bRate')
    plt.title(f'Feedback条件对bRate的影响 - {paths["strategy_name"]}')
    plt.tight_layout()
    plt.savefig(fig_dir / "feedback_comparison.png", dpi=150)
    plt.close()
    
    # 3. EV_diff与bRate的关系
    plt.figure(figsize=(12, 8))
    plt.scatter(df_meta['ev_diff'], y, alpha=0.3, s=20, c=df_meta['block'], cmap='viridis')
    plt.xlabel('EV_diff (Gamble B - Gamble A的期望值差)')
    plt.ylabel('bRate (选择Gamble B的概率)')
    plt.title(f'期望值差与选择概率的关系 - {paths["strategy_name"]}')
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='EV_diff=0')
    plt.colorbar(label='Block')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "ev_diff_vs_bRate.png", dpi=150)
    plt.close()
    
    # 4. 增强特征相关性热图 (前20个特征)
    if len(feature_names) >= 20:
        # 计算相关性矩阵
        corr_matrix = np.corrcoef(X[:, :20].T)
        
        plt.figure(figsize=(16, 14))
        sns.heatmap(corr_matrix, 
                    annot=False,  # 太多特征时不显示数值
                    cmap='coolwarm', center=0,
                    xticklabels=feature_names[:20],
                    yticklabels=feature_names[:20])
        plt.title(f'增强特征相关性热图 (前20个特征) - {paths["strategy_name"]}')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(fig_dir / "enhanced_feature_correlation.png", dpi=150)
        plt.close()
    
    # 5. 特征重要性预览（基于与bRate的相关性）
    plt.figure(figsize=(14, 8))
    correlations = []
    for i in range(min(25, len(feature_names))):
        correlation = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append((feature_names[i], abs(correlation)))
    
    # 按相关性绝对值排序
    correlations.sort(key=lambda x: x[1], reverse=True)
    top_features = [c[0] for c in correlations[:15]]
    top_corrs = [c[1] for c in correlations[:15]]
    
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_corrs, color='skyblue')
    plt.yticks(y_pos, top_features)
    plt.xlabel('与bRate的绝对相关系数')
    plt.title(f'与bRate最相关的特征 (前15个) - {paths["strategy_name"]}')
    plt.gca().invert_yaxis()  # 最高的在顶部
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_correlation_with_bRate.png", dpi=150)
    plt.close()
    
    print(f"可视化图表已保存到: {fig_dir}")


def process_strategy(strategy_name):
    """
    处理单个策略的数据准备
    
    Args:
        strategy_name: 策略名称
        
    Returns:
        处理成功返回True，否则返回False
    """
    print("\n" + "=" * 70)
    print(f"处理策略: {strategy_name}")
    print("=" * 70)
    
    try:
        # 1. 设置路径
        paths = setup_paths(strategy_name)
        
        # 2. 标准化增强数据
        standardizer, standardized_data, X, y, feature_names = standardize_enhanced_data(paths)
        
        # 3. 分析增强数据
        df_meta = analyze_enhanced_data(standardizer, standardized_data, X, y, feature_names, paths)
        
        # 4. 创建增强数据划分
        splits = create_enhanced_data_splits(standardized_data, df_meta, paths)
        
        # 5. 创建增强可视化图表
        create_enhanced_visualizations(df_meta, X, y, feature_names, paths)
        
        print(f"\n" + "=" * 70)
        print(f"策略 '{strategy_name}' 数据处理完成!")
        print("=" * 70)
        
        # 输出关键文件位置
        print(f"\n生成的文件:")
        print(f"  标准化数据: {paths['standardized_json']}")
        print(f"  特征矩阵: {paths['features_npy']}")
        print(f"  目标向量: {paths['targets_npy']}")
        print(f"  特征名称: {paths['feature_names_txt']}")
        print(f"  元数据: {paths['metadata_csv']}")
        print(f"  统计信息: {paths['stats_txt']}")
        print(f"  数据划分: {paths['splits_json']}")
        print(f"  可视化图表: {paths['output_dir'] / 'figures'}")
        
        # 数据规模总结
        print(f"\n数据规模总结:")
        print(f"  总样本数: {len(standardized_data)}")
        print(f"  增强特征维度: {X.shape[1]}")
        print(f"  特征数量: {len(feature_names)}")
        
        # 划分策略总结
        print(f"\n划分策略总结:")
        for split_type in ['problem', 'parameter_amb', 'parameter_ev_extreme']:
            if split_type in splits:
                split_info = splits[split_type]
                print(f"  {split_type}:")
                print(f"    训练集大小: {len(split_info['train_indices'])}")
                print(f"    测试集大小: {len(split_info['test_indices'])}")
        
        return True
        
    except Exception as e:
        print(f"\n处理策略 '{strategy_name}' 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("Choices13k 增强数据准备脚本")
    print("=" * 70)
    print("基于JSON原始分布数据提取完整特征")
    print("支持三种划分策略: problem_split, parameter_amb_split, parameter_ev_extreme_split")
    print("=" * 70)
    
    # 定义要处理的所有策略
    all_strategies = [
        'problem_split',
        'parameter_amb_split', 
        'parameter_ev_extreme_split'
    ]
    
    # 用户可以选择处理特定策略或所有策略
    import argparse
    parser = argparse.ArgumentParser(description='处理Choices13k增强数据准备')
    parser.add_argument('--strategy', type=str, choices=all_strategies + ['all'], 
                       default='all', help='要处理的策略名称，或使用"all"处理所有策略')
    parser.add_argument('--skip-existing', action='store_true', 
                       help='跳过已存在输出文件的策略')
    
    args = parser.parse_args()
    
    # 确定要处理的策略
    if args.strategy == 'all':
        strategies_to_process = all_strategies
    else:
        strategies_to_process = [args.strategy]
    
    success_count = 0
    total_strategies = len(strategies_to_process)
    
    for i, strategy_name in enumerate(strategies_to_process, 1):
        print(f"\n[{i}/{total_strategies}] 开始处理策略: {strategy_name}")
        
        # 检查是否跳过已存在的输出
        if args.skip_existing:
            # 检查主要输出文件是否存在
            project_root = Path(__file__).parent.parent.parent.parent
            output_dir = project_root / "experiments" / "00_data_preparation" / "outputs" / strategy_name
            standardized_json = output_dir / "c13k_enhanced_standardized.json"
            
            if standardized_json.exists():
                print(f"  跳过策略 '{strategy_name}'，输出文件已存在: {standardized_json}")
                success_count += 1
                continue
        
        # 处理策略
        if process_strategy(strategy_name):
            success_count += 1
    
    print("\n" + "=" * 70)
    print("增强数据准备完成总结")
    print("=" * 70)
    print(f"成功处理的策略: {success_count}/{total_strategies}")
    
    if success_count == total_strategies:
        print("✓ 所有策略处理成功!")
        return 0
    else:
        print(f"⚠  {total_strategies - success_count} 个策略处理失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())