"""
快速集成测试 - 验证增强数据标准化与增强模型的集成
"""

import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from enhanced_data_standardization import EnhancedChoices13kStandardizer
from enhanced_symbolic_models import create_enhanced_model, EnhancedModelAdapter


def quick_test():
    """快速测试"""
    print("快速集成测试")
    print("=" * 50)
    
    # 1. 加载少量数据
    print("\n1. 加载数据...")
    standardizer = EnhancedChoices13kStandardizer(
        selections_path=str(project_root / "数据集" / "choices13k-main" / "c13k_selections.csv"),
        problems_path=str(project_root / "数据集" / "choices13k-main" / "c13k_problems.json")
    )
    
    # 只加载前1000行以加快速度
    df = standardizer.load_and_merge_data()
    df_small = df.head(1000).copy()
    standardizer.df = df_small
    
    print(f"   加载数据: {len(df_small)} 行")
    
    # 2. 标准化数据
    print("\n2. 标准化数据...")
    standardized_data = standardizer.standardize_all()
    print(f"   标准化数据: {len(standardized_data)} 条记录")
    
    if not standardized_data:
        print("   错误: 无标准化数据")
        return
    
    # 3. 测试模型预测（不拟合）
    print("\n3. 测试模型预测（无拟合）...")
    model_names = ['ev', 'eu', 'pt3', 'pt5']
    
    for model_name in model_names:
        print(f"\n   {model_name.upper()} 模型:")
        
        # 创建模型
        model = create_enhanced_model(model_name)
        
        # 使用适配器
        adapter = EnhancedModelAdapter(model)
        
        # 在第一个样本上测试预测
        sample = standardized_data[0]
        gamble_a = sample['context']['gamble_a']['distribution']
        gamble_b = sample['context']['gamble_b']['distribution']
        
        # 直接使用模型预测
        prob = model.predict_from_distributions(gamble_a, gamble_b)
        value_a, value_b = model.compute_gamble_values_from_distributions(gamble_a, gamble_b)
        
        print(f"     价值: A={value_a:.4f}, B={value_b:.4f}")
        print(f"     选择B概率: {prob:.4f}")
        
        # 使用适配器批量预测
        predictions = adapter.predict_from_standardized(standardized_data[:10])
        print(f"     前10个样本预测范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    # 4. 测试简化特征与真实分布的差异
    print("\n4. 测试特征差异...")
    sample = standardized_data[0]
    features = sample['context']['features']
    
    print(f"   特征数量: {len(features)}")
    
    # 检查关键特征
    key_features = ['a_ev', 'b_ev', 'ev_diff', 'a_num_outcomes', 'b_num_outcomes']
    for feat in key_features:
        if feat in features:
            print(f"   {feat}: {features[feat]}")
    
    # 5. 验证多结果分布
    print("\n5. 验证多结果分布...")
    multi_outcome_count = 0
    for i, item in enumerate(standardized_data[:100]):
        gamble_b = item['context']['gamble_b']['distribution']
        if len(gamble_b) > 2:
            multi_outcome_count += 1
            if multi_outcome_count <= 3:
                print(f"   样本 {i}: {len(gamble_b)} 个结果")
                print(f"     分布: {gamble_b}")
    
    print(f"\n   前100个样本中多结果分布数量: {multi_outcome_count}")
    
    print("\n" + "=" * 50)
    print("快速测试完成!")
    print("=" * 50)


if __name__ == "__main__":
    try:
        quick_test()
    except Exception as e:
        print(f"测试出错: {e}")
        import traceback
        traceback.print_exc()