"""
增强模型集成测试
测试增强数据标准化与增强符号模型的集成
"""

import sys
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 设置中文字体（避免中文/负号显示为方框）
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
    EnhancedModelAdapter
)


def test_enhanced_model_integration():
    """测试增强模型与标准化数据的集成"""
    print("=" * 70)
    print("增强模型集成测试")
    print("=" * 70)
    
    # 1. 加载和标准化数据
    print("\n1. 加载和标准化数据...")
    standardizer = EnhancedChoices13kStandardizer(
        selections_path=str(project_root / "数据集" / "choices13k-main" / "c13k_selections.csv"),
        problems_path=str(project_root / "数据集" / "choices13k-main" / "c13k_problems.json")
    )
    
    standardized_data = standardizer.standardize_all()
    print(f"   标准化数据记录数: {len(standardized_data)}")
    
    # 获取目标向量
    y = standardizer.get_target_vector()
    print(f"   目标向量形状: {y.shape}")
    
    # 2. 创建数据划分
    print("\n2. 创建数据划分...")
    split_types = ['problem', 'parameter_amb', 'parameter_ev_extreme']
    
    for split_type in split_types:
        print(f"\n   划分类型: {split_type}")
        train_idx, test_idx, split_info = create_enhanced_splits(
            standardized_data, split_type=split_type
        )
        
        print(f"     描述: {split_info['description']}")
        print(f"     训练集大小: {split_info['train_size']}")
        print(f"     测试集大小: {split_info['test_size']}")
        
        # 准备训练和测试数据
        train_data = [standardized_data[i] for i in train_idx]
        test_data = [standardized_data[i] for i in test_idx]
        
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # 3. 测试所有增强模型
        print(f"\n3. 测试增强模型在 {split_type} 划分上的表现:")
        model_names = ['ev', 'eu', 'pt3', 'pt5']
        
        for model_name in model_names:
            print(f"\n   {model_name.upper()} 模型:")
            
            # 创建模型和适配器
            model = create_enhanced_model(model_name)
            adapter = EnhancedModelAdapter(model)
            
            # 训练模型
            print(f"     训练模型...")
            adapter.fit_from_standardized(train_data, y_train)
            
            # 预测
            print(f"     预测...")
            y_train_pred = adapter.predict_from_standardized(train_data)
            y_test_pred = adapter.predict_from_standardized(test_data)
            
            # 计算评估指标
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            print(f"     训练集 MSE: {train_mse:.6f}, R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
            print(f"     测试集 MSE: {test_mse:.6f}, R²: {test_r2:.4f}, MAE: {test_mae:.4f}")
            
            # 显示模型参数
            params = model.get_parameters()
            print(f"     拟合参数: {params}")
            
            # 保存预测结果用于可视化
            if model_name == 'ev':  # 只保存EV模型的结果用于示例
                save_predictions(
                    y_test, y_test_pred, 
                    f"enhanced_{model_name}_{split_type}_predictions.png",
                    model_name=f"Enhanced {model_name.upper()}", 
                    split_type=split_type
                )
    
    print("\n" + "=" * 70)
    print("增强模型集成测试完成!")
    print("=" * 70)


def save_predictions(y_true, y_pred, filename, model_name, split_type):
    """保存预测结果可视化"""
    try:
        plt.figure(figsize=(10, 6))
        
        # 散点图
        plt.scatter(y_true, y_pred, alpha=0.5, s=10)
        
        # 添加对角线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
        
        plt.xlabel('True bRate')
        plt.ylabel('Predicted bRate')
        plt.title(f'{model_name} - {split_type} Split\nPredictions vs True Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        save_path = Path(__file__).parent / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"     预测可视化已保存到: {save_path}")
        
    except Exception as e:
        print(f"     保存可视化时出错: {e}")


def compare_enhanced_vs_original():
    """比较增强模型与原始模型"""
    print("\n" + "=" * 70)
    print("增强模型与原始模型比较")
    print("=" * 70)
    
    # 注意: 原始模型使用简化的二结果特征
    # 增强模型使用完整的多结果分布
    # 这里我们只进行概念比较
    
    print("\n模型对比:")
    print("1. 原始模型 (Original Models):")
    print("   - 使用简化的二结果特征 (Ha, pHa, La, Hb, pHb, Lb)")
    print("   - 假设每个赌博只有两个可能结果")
    print("   - 期望值计算: EV = p_high * high + (1 - p_high) * low")
    print("   - 无法处理多结果分布")
    
    print("\n2. 增强模型 (Enhanced Models):")
    print("   - 使用完整的多结果分布")
    print("   - 支持任意数量的结果")
    print("   - 期望值计算: EV = Σ(p_i * x_i)")
    print("   - 更准确地表示真实赌博问题")
    
    print("\n3. 预期改进:")
    print("   - 更准确的特征表示")
    print("   - 更好的模型拟合")
    print("   - 更科学的泛化评估")
    print("   - 增强的行为经济学洞察")


def analyze_distribution_impact():
    """分析多结果分布的影响"""
    print("\n" + "=" * 70)
    print("多结果分布影响分析")
    print("=" * 70)
    
    # 加载数据
    standardizer = EnhancedChoices13kStandardizer(
        selections_path=str(project_root / "数据集" / "choices13k-main" / "c13k_selections.csv"),
        problems_path=str(project_root / "数据集" / "choices13k-main" / "c13k_problems.json")
    )
    
    standardizer.load_and_merge_data()
    
    # 分析结果数量分布
    print("\n1. Gamble B 结果数量分布:")
    num_outcomes_b = []
    
    for idx in range(min(2000, len(standardizer.df))):
        row = standardizer.df.iloc[idx]
        gamble_b = row['gamble_b']
        num_outcomes_b.append(len(gamble_b))
    
    num_outcomes_b = np.array(num_outcomes_b)
    
    print(f"   样本数: {len(num_outcomes_b)}")
    print(f"   平均结果数: {np.mean(num_outcomes_b):.2f}")
    print(f"   中位数: {np.median(num_outcomes_b)}")
    print(f"   标准差: {np.std(num_outcomes_b):.2f}")
    print(f"   范围: [{np.min(num_outcomes_b)}, {np.max(num_outcomes_b)}]")
    
    # 统计分布
    unique_counts, counts = np.unique(num_outcomes_b, return_counts=True)
    print(f"\n   详细分布:")
    for count, freq in zip(unique_counts, counts):
        percentage = freq / len(num_outcomes_b) * 100
        print(f"     {count}个结果: {freq}个问题 ({percentage:.1f}%)")
    
    # 分析简化特征与真实分布的差异
    print("\n2. 简化特征 vs 真实分布差异分析:")
    
    # 随机选择一些样本
    sample_indices = np.random.choice(len(standardizer.df), min(10, len(standardizer.df)), replace=False)
    
    for idx in sample_indices:
        row = standardizer.df.iloc[idx]
        
        # 简化的期望值 (基于CSV特征)
        Ha, pHa, La = row['Ha'], row['pHa'], row['La']
        Hb, pHb, Lb = row['Hb'], row['pHb'], row['Lb']
        
        ev_a_simple = pHa * Ha + (1 - pHa) * La
        ev_b_simple = pHb * Hb + (1 - pHb) * Lb
        
        # 真实的期望值 (基于完整分布)
        gamble_a = row['gamble_a']
        gamble_b = row['gamble_b']
        
        ev_a_real = sum(p * x for p, x in gamble_a)
        ev_b_real = sum(p * x for p, x in gamble_b)
        
        # 计算差异
        diff_a = abs(ev_a_real - ev_a_simple)
        diff_b = abs(ev_b_real - ev_b_simple)
        
        if diff_a > 0.01 or diff_b > 0.01:
            print(f"\n   样本 {idx}:")
            print(f"     Gamble A - 简化EV: {ev_a_simple:.4f}, 真实EV: {ev_a_real:.4f}, 差异: {diff_a:.4f}")
            print(f"     Gamble B - 简化EV: {ev_b_simple:.4f}, 真实EV: {ev_b_real:.4f}, 差异: {diff_b:.4f}")
            print(f"     Gamble B 结果数: {len(gamble_b)}")
            
            if len(gamble_b) > 2:
                print(f"     多结果分布示例: {gamble_b[:3]}...")
    
    print("\n3. 结论:")
    print("   - 大部分Gamble B包含多个结果 (平均 {:.1f}个)".format(np.mean(num_outcomes_b)))
    print("   - 简化特征与真实分布存在显著差异")
    print("   - 增强模型能更准确地表示决策问题")


if __name__ == "__main__":
    try:
        # 运行集成测试
        test_enhanced_model_integration()
        
        # 运行比较分析
        compare_enhanced_vs_original()
        
        # 运行分布影响分析
        analyze_distribution_impact()
        
        print("\n所有测试完成!")
        
    except Exception as e:
        print(f"\n测试过程中出错: {e}")
        import traceback
        traceback.print_exc()