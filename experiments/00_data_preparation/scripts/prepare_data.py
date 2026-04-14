"""
数据准备脚本
加载和标准化 Choices13k 数据集，为符号模型实验准备数据
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

# 添加原始数据标准化模块的路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "数据集" / "choices13k-main"))
from data_standardization import Choices13kStandardizer, create_splits

# 创建自定义标准化器类，修复 JSON 序列化问题
class CustomChoices13kStandardizer(Choices13kStandardizer):
    """自定义标准化器，修复 numpy 类型序列化问题"""
    
    def standardize_all(self, save_path: str = None):
        """
        标准化所有数据，处理 numpy 类型的序列化问题
        """
        if self.df is None:
            self.load_data()
        
        print("开始标准化数据...")
        standardized_data = []
        
        for idx in range(len(self.df)):
            try:
                standardized = self.standardize_row(idx)
                standardized_data.append(standardized)
            except Exception as e:
                print(f"处理第 {idx} 行时出错: {e}")
                continue
        
        self.standardized_data = standardized_data
        print(f"标准化完成: {len(standardized_data)} 条记录")
        
        # 保存到文件，处理 numpy 类型
        if save_path:
            # 转换为可序列化的 Python 原生类型
            serializable_data = self._convert_to_serializable(standardized_data)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            print(f"已保存到: {save_path}")
        
        return standardized_data
    
    def _convert_to_serializable(self, data):
        """将数据中的 numpy 类型转换为 Python 原生类型"""
        if isinstance(data, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(data)
        elif isinstance(data, (np.floating, np.float64, np.float32, np.float16)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.bool_):
            return bool(data)
        elif isinstance(data, dict):
            return {key: self._convert_to_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_to_serializable(item) for item in data]
        else:
            return data

def setup_paths():
    """设置路径"""
    project_root = Path(__file__).parent.parent.parent.parent
    data_dir = project_root / "数据集" / "choices13k-main"
    output_dir = project_root / "experiments" / "00_data_preparation" / "outputs"
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'selections': str(data_dir / "c13k_selections.csv"),
        'problems': str(data_dir / "c13k_problems.json"),
        'output_dir': output_dir,
        'standardized_json': str(output_dir / "c13k_standardized.json"),
        'features_npy': str(output_dir / "features.npy"),
        'targets_npy': str(output_dir / "targets.npy"),
        'metadata_csv': str(output_dir / "metadata.csv"),
        'stats_txt': str(output_dir / "data_statistics.txt")
    }

def standardize_data(paths):
    """标准化数据"""
    print("=" * 60)
    print("开始数据标准化")
    print("=" * 60)
    
    # 初始化自定义标准化器
    standardizer = CustomChoices13kStandardizer(
        selections_path=paths['selections'],
        problems_path=paths['problems']
    )
    
    # 加载和标准化所有数据
    standardized_data = standardizer.standardize_all(
        save_path=paths['standardized_json']
    )
    
    # 获取特征矩阵和目标向量
    X = standardizer.get_feature_matrix()
    y = standardizer.get_target_vector()
    
    # 保存为numpy格式以便快速加载
    np.save(paths['features_npy'], X)
    np.save(paths['targets_npy'], y)
    
    print(f"\n特征矩阵形状: {X.shape}")
    print(f"目标向量形状: {y.shape}")
    
    return standardizer, standardized_data, X, y

def create_metadata_dataframe(standardized_data):
    """创建元数据DataFrame用于划分"""
    metadata_list = []
    for item in standardized_data:
        metadata_list.append({
            'index': item['metadata']['index'],
            'problem_id': item['metadata']['problem_id'],
            'feedback': item['metadata']['feedback_condition'],
            'block': item['metadata']['block'],
            'bRate': item['action']['bRate'],
            'bRate_std': item['action']['bRate_std'],
            'n_subjects': item['action']['n_subjects'],
            'EV_diff': item['context']['features']['EV_diff'],
            'Amb': item['context']['features']['Amb']
        })
    
    return pd.DataFrame(metadata_list)

def analyze_data(standardizer, standardized_data, X, y, paths):
    """分析数据并生成统计信息"""
    print("\n" + "=" * 60)
    print("数据分析")
    print("=" * 60)
    
    # 创建元数据DataFrame
    df_meta = create_metadata_dataframe(standardized_data)
    
    # 保存元数据
    df_meta.to_csv(paths['metadata_csv'], index=False)
    
    # 生成统计信息
    stats_lines = []
    
    # 基本统计
    stats_lines.append("=" * 60)
    stats_lines.append("CHOICES13K 数据集统计")
    stats_lines.append("=" * 60)
    stats_lines.append(f"总样本数: {len(standardized_data)}")
    stats_lines.append(f"特征维度: {X.shape[1]}")
    
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
    
    # 特征统计
    feature_names = [
        'Ha', 'pHa', 'La', 'Hb', 'pHb', 'Lb', 'LotShapeB', 'LotNumB',
        'Amb', 'Corr', 'EV_A', 'EV_B', 'EV_diff', 'risk_A', 'risk_B',
        'feedback', 'block'
    ]
    
    stats_lines.append(f"\n特征统计 (前5个特征):")
    for i in range(min(5, len(feature_names))):
        feat = X[:, i]
        stats_lines.append(f"  {feature_names[i]}: 均值={feat.mean():.4f}, 标准差={feat.std():.4f}")
    
    # 保存统计信息
    with open(paths['stats_txt'], 'w', encoding='utf-8') as f:
        f.write('\n'.join(stats_lines))
    
    # 打印统计信息
    for line in stats_lines:
        print(line)
    
    return df_meta

def create_data_splits(df_meta, paths):
    """创建数据划分"""
    print("\n" + "=" * 60)
    print("创建数据划分")
    print("=" * 60)
    
    # 复制DataFrame并重命名列以匹配原始函数的期望
    df_for_splits = df_meta.copy()
    df_for_splits = df_for_splits.rename(columns={
        'feedback': 'Feedback',
        'problem_id': 'Problem',
        'block': 'Block',
        'EV_diff': 'EV_diff'
    })
    
    splits = {}
    
    # 1. Problem-Split (标准划分)
    train_idx_prob, test_idx_prob = create_splits(
        df_for_splits,
        split_type='problem',
        test_size=0.2,
        random_state=42
    )
    splits['problem_split'] = {
        'train_indices': train_idx_prob,
        'test_indices': test_idx_prob,
        'description': '训练集: 80%问题, 测试集: 20%问题'
    }
    
    # 2. Parameter-Amb-Split (模糊性划分)
    train_mask_amb = df_meta['Amb'] == 0  # 非模糊样本
    test_mask_amb = df_meta['Amb'] == 1   # 模糊样本
    train_indices_amb = df_meta[train_mask_amb].index.tolist()
    test_indices_amb = df_meta[test_mask_amb].index.tolist()
    
    splits['parameter_amb_split'] = {
        'train_indices': train_indices_amb,
        'test_indices': test_indices_amb,
        'description': '训练集: 非模糊样本(Amb=0), 测试集: 模糊样本(Amb=1)'
    }
    
    # 3. Parameter-EV-Extreme-Split (期望值极端划分)
    q1 = df_meta['EV_diff'].quantile(0.25)
    q3 = df_meta['EV_diff'].quantile(0.75)
    train_mask_ev = df_meta['EV_diff'] < q1    # EV_diff < 25th百分位数
    test_mask_ev = df_meta['EV_diff'] > q3     # EV_diff > 75th百分位数
    
    train_indices_ev = df_meta[train_mask_ev].index.tolist()
    test_indices_ev = df_meta[test_mask_ev].index.tolist()
    
    splits['parameter_ev_extreme_split'] = {
        'train_indices': train_indices_ev,
        'test_indices': test_indices_ev,
        'description': '训练集: EV_diff < 25th百分位数, 测试集: EV_diff > 75th百分位数'
    }
    
    # 保存划分
    splits_path = paths['output_dir'] / "data_splits.json"
    with open(splits_path, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2, default=str)
    
    print("创建的划分策略:")
    for split_name, split_info in splits.items():
        print(f"\n{split_name}:")
        print(f"  {split_info['description']}")
        print(f"  训练集大小: {len(split_info['train_indices'])}")
        print(f"  测试集大小: {len(split_info['test_indices'])}")
    
    return splits

def create_visualizations(df_meta, X, y, paths):
    """创建可视化图表"""
    print("\n" + "=" * 60)
    print("创建可视化图表")
    print("=" * 60)
    
    # 图表风格（中文字体已在模块开头通过 setup_chinese_font() 设置）
    sns.set_style("whitegrid")

    fig_dir = paths['output_dir'] / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # 1. bRate分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(y, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('bRate (选择Gamble B的概率)')
    plt.ylabel('频数')
    plt.title('bRate分布直方图')
    plt.tight_layout()
    plt.savefig(fig_dir / "bRate_distribution.png", dpi=150)
    plt.close()
    
    # 2. Feedback条件对比
    plt.figure(figsize=(8, 6))
    feedback_labels = ['无反馈', '有反馈']
    feedback_data = [
        y[df_meta['feedback'] == False],
        y[df_meta['feedback'] == True]
    ]
    
    plt.boxplot(feedback_data, labels=feedback_labels)
    plt.ylabel('bRate')
    plt.title('Feedback条件对bRate的影响')
    plt.tight_layout()
    plt.savefig(fig_dir / "feedback_comparison.png", dpi=150)
    plt.close()
    
    # 3. EV_diff与bRate的关系
    plt.figure(figsize=(10, 6))
    plt.scatter(df_meta['EV_diff'], y, alpha=0.5, s=10)
    plt.xlabel('EV_diff (Gamble B - Gamble A的期望值差)')
    plt.ylabel('bRate')
    plt.title('期望值差与选择概率的关系')
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "ev_diff_vs_bRate.png", dpi=150)
    plt.close()
    
    # 4. 特征相关性热图 (前10个特征)
    feature_names = [
        'Ha', 'pHa', 'La', 'Hb', 'pHb', 'Lb', 'LotShapeB', 'LotNumB',
        'Amb', 'Corr', 'EV_A', 'EV_B', 'EV_diff', 'risk_A', 'risk_B',
        'feedback', 'block'
    ]
    
    # 计算相关性矩阵
    corr_matrix = np.corrcoef(X.T)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix[:10, :10], 
                annot=True, fmt='.2f', 
                cmap='coolwarm', center=0,
                xticklabels=feature_names[:10],
                yticklabels=feature_names[:10])
    plt.title('特征相关性热图 (前10个特征)')
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_correlation.png", dpi=150)
    plt.close()
    
    print(f"可视化图表已保存到: {fig_dir}")

def main():
    """主函数"""
    print("Choices13k 数据准备脚本")
    print("=" * 60)
    
    # 设置路径
    paths = setup_paths()
    
    try:
        # 1. 标准化数据
        standardizer, standardized_data, X, y = standardize_data(paths)
        
        # 2. 创建元数据DataFrame
        df_meta = create_metadata_dataframe(standardized_data)
        
        # 3. 分析数据
        analyze_data(standardizer, standardized_data, X, y, paths)
        
        # 4. 创建数据划分
        splits = create_data_splits(df_meta, paths)
        
        # 5. 创建可视化图表
        create_visualizations(df_meta, X, y, paths)
        
        print("\n" + "=" * 60)
        print("数据准备完成!")
        print("=" * 60)
        
        # 输出关键文件位置
        print(f"\n生成的文件:")
        print(f"  标准化数据: {paths['standardized_json']}")
        print(f"  特征矩阵: {paths['features_npy']}")
        print(f"  目标向量: {paths['targets_npy']}")
        print(f"  元数据: {paths['metadata_csv']}")
        print(f"  统计信息: {paths['stats_txt']}")
        print(f"  数据划分: {paths['output_dir'] / 'data_splits.json'}")
        print(f"  可视化图表: {paths['output_dir'] / 'figures'}")
        
        # 数据规模总结
        print(f"\n数据规模总结:")
        print(f"  总样本数: {len(standardized_data)}")
        print(f"  特征维度: {X.shape[1]}")
        
        # Feedback-Split 详情
        train_fb = sum(df_meta['feedback'] == True)
        test_fb = sum(df_meta['feedback'] == False)
        print(f"\nFeedback-Split (核心实验):")
        print(f"  训练集 (有反馈): {train_fb} 条")
        print(f"  测试集 (无反馈): {test_fb} 条")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())