"""
测试增强版数据标准化模块
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from enhanced_data_standardization import (
    EnhancedChoices13kStandardizer,
    create_enhanced_splits
)

def test_data_loading():
    """测试数据加载"""
    print("=" * 60)
    print("测试增强版数据标准化器")
    print("=" * 60)
    
    # 初始化标准化器
    standardizer = EnhancedChoices13kStandardizer(
        selections_path=str(project_root / "数据集" / "choices13k-main" / "c13k_selections.csv"),
        problems_path=str(project_root / "数据集" / "choices13k-main" / "c13k_problems.json")
    )
    
    # 测试数据加载
    print("\n1. 测试数据加载...")
    df = standardizer.load_and_merge_data()
    print(f"   合并后的DataFrame形状: {df.shape}")
    print(f"   列名: {list(df.columns)[:10]}...")
    
    # 检查关键列是否存在
    required_cols = ['gamble_a', 'gamble_b', 'bRate', 'Feedback', 'Amb']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"   警告: 缺少列 {missing_cols}")
    else:
        print(f"   所有必需列都存在")
    
    # 检查数据样本
    print(f"\n2. 检查数据样本...")
    sample_row = df.iloc[0]
    print(f"   样本索引: 0")
    print(f"   Gamble A: {len(sample_row['gamble_a'])} 个结果")
    print(f"   Gamble B: {len(sample_row['gamble_b'])} 个结果")
    print(f"   bRate: {sample_row['bRate']}")
    
    # 测试分布统计计算
    print(f"\n3. 测试分布统计计算...")
    stats_a = standardizer.calculate_distribution_stats(sample_row['gamble_a'])
    stats_b = standardizer.calculate_distribution_stats(sample_row['gamble_b'])
    
    print(f"   Gamble A统计:")
    for key, value in list(stats_a.items())[:5]:
        print(f"     {key}: {value}")
    
    print(f"   Gamble B统计:")
    for key, value in list(stats_b.items())[:5]:
        print(f"     {key}: {value}")
    
    # 测试特征计算
    print(f"\n4. 测试特征计算...")
    features = standardizer.calculate_enhanced_features(sample_row)
    print(f"   计算的特征数量: {len(features)}")
    
    # 显示一些关键特征
    key_features = ['ev_diff', 'a_ev', 'b_ev', 'a_variance', 'b_variance', 
                   'a_skewness', 'b_skewness', 'a_entropy', 'b_entropy']
    print(f"\n   关键特征值:")
    for feat in key_features:
        if feat in features:
            print(f"     {feat}: {features[feat]}")
    
    # 测试完整标准化
    print(f"\n5. 测试完整标准化...")
    standardized_data = standardizer.standardize_all()
    print(f"   标准化记录数: {len(standardized_data)}")
    
    if standardized_data:
        sample_std = standardized_data[0]
        print(f"   标准化样本结构:")
        print(f"     context.keys: {list(sample_std['context'].keys())}")
        print(f"     features数量: {len(sample_std['context']['features'])}")
        
        # 测试特征矩阵
        print(f"\n6. 测试特征矩阵...")
        X, feature_names = standardizer.get_enhanced_feature_matrix()
        print(f"   特征矩阵形状: {X.shape}")
        print(f"   特征名称数量: {len(feature_names)}")
        
        # 显示特征统计
        print(f"\n7. 特征统计:")
        stats = standardizer.get_feature_statistics()
        for i, (feat_name, feat_stats) in enumerate(list(stats.items())[:5]):
            print(f"   {feat_name}:")
            for stat_name, stat_value in feat_stats.items():
                print(f"     {stat_name}: {stat_value:.4f}")
        
        # 测试数据划分
        print(f"\n8. 测试数据划分...")
        split_types = ['problem', 'parameter_amb', 'parameter_ev_extreme']
        
        for split_type in split_types:
            train_idx, test_idx, split_info = create_enhanced_splits(
                standardized_data, split_type=split_type
            )
            print(f"\n   {split_type}:")
            print(f"     描述: {split_info['description']}")
            print(f"     训练集大小: {split_info['train_size']}")
            print(f"     测试集大小: {split_info['test_size']}")
            
        return True
    
    return False

def test_distribution_complexity():
    """测试分布复杂性分析"""
    print("\n" + "=" * 60)
    print("分析分布复杂性")
    print("=" * 60)
    
    # 初始化标准化器
    standardizer = EnhancedChoices13kStandardizer(
        selections_path=str(project_root / "数据集" / "choices13k-main" / "c13k_selections.csv"),
        problems_path=str(project_root / "数据集" / "choices13k-main" / "c13k_problems.json")
    )
    
    standardizer.load_and_merge_data()
    
    # 分析结果数量分布
    print("\n分析Gamble B的结果数量分布:")
    num_outcomes_b = []
    
    for idx in range(min(1000, len(standardizer.df))):
        row = standardizer.df.iloc[idx]
        gamble_b = row['gamble_b']
        num_outcomes_b.append(len(gamble_b))
    
    num_outcomes_b = np.array(num_outcomes_b)
    
    print(f"   样本数: {len(num_outcomes_b)}")
    print(f"   平均结果数: {np.mean(num_outcomes_b):.2f}")
    print(f"   标准差: {np.std(num_outcomes_b):.2f}")
    print(f"   最小值: {np.min(num_outcomes_b)}")
    print(f"   最大值: {np.max(num_outcomes_b)}")
    
    # 结果数量分布
    unique_counts, counts = np.unique(num_outcomes_b, return_counts=True)
    print(f"\n   结果数量分布:")
    for count, freq in zip(unique_counts, counts):
        print(f"     {count}个结果: {freq}个问题 ({freq/len(num_outcomes_b)*100:.1f}%)")
    
    # 测试多结果样本
    print(f"\n查找多结果样本 (>2个结果):")
    multi_outcome_samples = []
    
    for idx in range(min(20, len(standardizer.df))):
        row = standardizer.df.iloc[idx]
        gamble_b = row['gamble_b']
        if len(gamble_b) > 2:
            multi_outcome_samples.append((idx, len(gamble_b), gamble_b))
    
    if multi_outcome_samples:
        for idx, num_outcomes, gamble_b in multi_outcome_samples[:3]:
            print(f"\n   样本 {idx}: {num_outcomes}个结果")
            print(f"     分布: {gamble_b}")
            
            # 计算统计
            stats = standardizer.calculate_distribution_stats(gamble_b)
            print(f"     期望值: {stats.get('ev', 'N/A')}")
            print(f"     方差: {stats.get('variance', 'N/A')}")
            print(f"     偏度: {stats.get('skewness', 'N/A')}")
    
    return True

if __name__ == "__main__":
    import numpy as np
    
    try:
        # 运行测试
        success = test_data_loading()
        
        if success:
            # 运行复杂性分析
            test_distribution_complexity()
            
            print("\n" + "=" * 60)
            print("增强版数据标准化测试完成!")
            print("=" * 60)
        else:
            print("\n测试失败!")
            
    except Exception as e:
        print(f"\n测试过程中出错: {e}")
        import traceback
        traceback.print_exc()