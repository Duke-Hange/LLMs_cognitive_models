"""
快速训练测试 - 验证增强训练脚本
"""

import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from enhanced_data_standardization import EnhancedChoices13kStandardizer, create_enhanced_splits
from enhanced_symbolic_models import create_enhanced_model, EnhancedModelAdapter


def quick_train_test():
    """快速训练测试"""
    print("快速训练测试")
    print("=" * 50)
    
    # 1. 加载少量数据
    print("\n1. 加载数据...")
    standardizer = EnhancedChoices13kStandardizer(
        selections_path=str(project_root / "数据集" / "choices13k-main" / "c13k_selections.csv"),
        problems_path=str(project_root / "数据集" / "choices13k-main" / "c13k_problems.json")
    )
    
    # 加载数据
    df = standardizer.load_and_merge_data()
    
    # 只使用前500个样本以加快速度
    df_small = df.head(500).copy()
    standardizer.df = df_small
    
    print(f"   加载数据: {len(df_small)} 行")
    
    # 2. 标准化数据
    print("\n2. 标准化数据...")
    standardized_data = standardizer.standardize_all()
    print(f"   标准化数据: {len(standardized_data)} 条记录")
    
    if not standardized_data:
        print("   错误: 无标准化数据")
        return
    
    # 获取目标向量
    y = standardizer.get_target_vector()
    print(f"   目标向量形状: {y.shape}")
    
    # 3. 创建数据划分 (problem split)
    print("\n3. 创建数据划分...")
    train_idx, test_idx, split_info = create_enhanced_splits(
        standardized_data, split_type='problem', test_size=0.3, random_state=42
    )
    
    train_data = [standardized_data[i] for i in train_idx]
    test_data = [standardized_data[i] for i in test_idx]
    
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    print(f"   训练集大小: {len(train_data)}")
    print(f"   测试集大小: {len(test_data)}")
    print(f"   划分描述: {split_info['description']}")
    
    # 4. 训练增强EV模型
    print("\n4. 训练增强EV模型...")
    model = create_enhanced_model('ev', temperature=1.0)
    adapter = EnhancedModelAdapter(model)
    
    # 拟合模型
    print("   开始拟合...")
    try:
        adapter.fit_from_standardized(train_data, y_train)
        print("   拟合成功!")
        
        # 获取参数
        params = model.get_parameters()
        print(f"   拟合参数: {params}")
        
        # 预测
        print("   进行预测...")
        y_train_pred = adapter.predict_from_standardized(train_data)
        y_test_pred = adapter.predict_from_standardized(test_data)
        
        # 计算指标
        train_mse = np.mean((y_train - y_train_pred) ** 2)
        test_mse = np.mean((y_test - y_test_pred) ** 2)
        train_r2 = 1 - np.sum((y_train - y_train_pred) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
        test_r2 = 1 - np.sum((y_test - y_test_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
        
        print(f"   训练集 MSE: {train_mse:.6f}, R2: {train_r2:.4f}")
        print(f"   测试集 MSE: {test_mse:.6f}, R2: {test_r2:.4f}")
        
        # 5. 测试其他模型（不拟合，只预测）
        print("\n5. 测试其他模型（无拟合）...")
        other_models = ['eu', 'pt3', 'pt5']
        
        for model_name in other_models:
            print(f"\n   {model_name.upper()} 模型:")
            model_test = create_enhanced_model(model_name)
            adapter_test = EnhancedModelAdapter(model_test)
            
            # 使用默认参数预测
            y_test_pred_default = adapter_test.predict_from_standardized(test_data)
            test_mse_default = np.mean((y_test - y_test_pred_default) ** 2)
            test_r2_default = 1 - np.sum((y_test - y_test_pred_default) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
            
            print(f"     默认参数测试集 MSE: {test_mse_default:.6f}, R2: {test_r2_default:.4f}")
        
        print("\n" + "=" * 50)
        print("快速训练测试完成!")
        print("=" * 50)
        
    except Exception as e:
        print(f"   训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    quick_train_test()