"""
探索性数据分析脚本
对 Choices13k 数据集进行深入分析，生成详细报告
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 项目根路径并设置中文字体（避免中文/负号显示为方框）
_p = Path(__file__).resolve().parent
for _ in range(8):
    _p = _p.parent
    if (_p / "shared").is_dir():
        sys.path.insert(0, str(_p))
        break
from shared.visualization import setup_chinese_font
setup_chinese_font()
sns.set_style("whitegrid")

def setup_paths():
    """设置路径"""
    project_root = Path(__file__).parent.parent.parent.parent
    output_dir = project_root / "experiments" / "00_data_preparation" / "outputs"
    
    return {
        'standardized_json': str(output_dir / "c13k_standardized.json"),
        'features_npy': str(output_dir / "features.npy"),
        'targets_npy': str(output_dir / "targets.npy"),
        'metadata_csv': str(output_dir / "metadata.csv"),
        'splits_json': str(output_dir / "data_splits.json"),
        'analysis_dir': output_dir / "analysis",
        'report_txt': str(output_dir / "exploratory_analysis_report.txt")
    }

def load_data(paths):
    """加载数据"""
    print("加载数据...")
    
    # 加载标准化数据
    with open(paths['standardized_json'], 'r', encoding='utf-8') as f:
        standardized_data = json.load(f)
    
    # 加载特征矩阵和目标向量
    X = np.load(paths['features_npy'])
    y = np.load(paths['targets_npy'])
    
    # 加载元数据
    df_meta = pd.read_csv(paths['metadata_csv'])
    
    # 加载数据划分
    with open(paths['splits_json'], 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    print(f"数据加载完成: {len(standardized_data)} 条记录")
    return standardized_data, X, y, df_meta, splits

def analyze_dataset_structure(standardized_data, df_meta):
    """分析数据集结构"""
    print("\n" + "="*60)
    print("数据集结构分析")
    print("="*60)
    
    report_lines = []
    
    # 基本统计
    report_lines.append("数据集基本信息:")
    report_lines.append(f"  总记录数: {len(standardized_data)}")
    report_lines.append(f"  特征维度: 17")
    report_lines.append(f"  问题ID数量: {df_meta['problem_id'].nunique()}")
    
    # 检查缺失值
    missing_values = df_meta.isnull().sum().sum()
    report_lines.append(f"\n数据质量检查:")
    report_lines.append(f"  缺失值总数: {missing_values}")
    
    # 检查重复记录
    duplicates = df_meta.duplicated().sum()
    report_lines.append(f"  重复记录数: {duplicates}")
    
    # 数据分布检查
    report_lines.append(f"\n数据分布检查:")
    report_lines.append(f"  bRate为0的记录数: {(df_meta['bRate'] == 0).sum()}")
    report_lines.append(f"  bRate为1的记录数: {(df_meta['bRate'] == 1).sum()}")
    report_lines.append(f"  bRate在(0,1)之间的记录数: {((df_meta['bRate'] > 0) & (df_meta['bRate'] < 1)).sum()}")
    
    return report_lines

def analyze_feedback_conditions(df_meta, X, feature_names):
    """分析Feedback条件的影响"""
    print("\n" + "="*60)
    print("Feedback条件分析")
    print("="*60)
    
    report_lines = []
    
    # 分离有反馈和无反馈的数据
    feedback_true = df_meta['feedback'] == True
    feedback_false = df_meta['feedback'] == False
    
    report_lines.append("Feedback条件分布:")
    report_lines.append(f"  有反馈 (Feedback=True): {feedback_true.sum()} 条 ({feedback_true.sum()/len(df_meta)*100:.1f}%)")
    report_lines.append(f"  无反馈 (Feedback=False): {feedback_false.sum()} 条 ({feedback_false.sum()/len(df_meta)*100:.1f}%)")
    
    # 比较bRate统计
    bRate_feedback_true = df_meta.loc[feedback_true, 'bRate']
    bRate_feedback_false = df_meta.loc[feedback_false, 'bRate']
    
    report_lines.append(f"\nbRate统计比较:")
    report_lines.append(f"  有反馈 - 均值: {bRate_feedback_true.mean():.4f}, 标准差: {bRate_feedback_true.std():.4f}")
    report_lines.append(f"  无反馈 - 均值: {bRate_feedback_false.mean():.4f}, 标准差: {bRate_feedback_false.std():.4f}")
    
    # t检验
    t_stat, p_value = stats.ttest_ind(bRate_feedback_true, bRate_feedback_false, equal_var=False)
    report_lines.append(f"  t检验: t={t_stat:.4f}, p={p_value:.4e}")
    report_lines.append(f"  效应大小 (Cohen's d): {abs(t_stat * np.sqrt(1/len(bRate_feedback_true) + 1/len(bRate_feedback_false))):.4f}")
    
    # 比较特征差异
    report_lines.append(f"\n特征差异 (有反馈 vs 无反馈):")
    
    # 定义特征名称
    if feature_names is None:
        feature_names = [
            'Ha', 'pHa', 'La', 'Hb', 'pHb', 'Lb', 'LotShapeB', 'LotNumB',
            'Amb', 'Corr', 'EV_A', 'EV_B', 'EV_diff', 'risk_A', 'risk_B',
            'feedback', 'block'
        ]
    
    # 检查前5个特征的差异
    for i in range(min(5, len(feature_names))):
        feat_true = X[feedback_true, i]
        feat_false = X[feedback_false, i]
        t_stat_feat, p_value_feat = stats.ttest_ind(feat_true, feat_false, equal_var=False)
        
        if p_value_feat < 0.05:
            sig = "**"
        elif p_value_feat < 0.01:
            sig = "***"
        else:
            sig = ""
        
        report_lines.append(f"  {feature_names[i]}: 有反馈={feat_true.mean():.4f}, 无反馈={feat_false.mean():.4f}, p={p_value_feat:.4e}{sig}")
    
    return report_lines

def analyze_correlations(X, y, feature_names):
    """分析相关性"""
    print("\n" + "="*60)
    print("相关性分析")
    print("="*60)
    
    report_lines = []
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # 计算特征与bRate的相关性
    correlations = []
    for i in range(X.shape[1]):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        if not np.isnan(corr):
            correlations.append((feature_names[i], corr, abs(corr)))
    
    # 按绝对值排序
    correlations.sort(key=lambda x: x[2], reverse=True)
    
    report_lines.append("特征与bRate的相关性 (Top 10):")
    for i, (name, corr, abs_corr) in enumerate(correlations[:10]):
        report_lines.append(f"  {i+1:2d}. {name:15s}: r = {corr:7.4f} {'(强相关)' if abs_corr > 0.5 else '(中等相关)' if abs_corr > 0.3 else '(弱相关)'}")
    
    # 检查多重共线性
    report_lines.append(f"\n特征间多重共线性检查:")
    
    # 计算特征间的相关性矩阵
    corr_matrix = np.corrcoef(X.T)
    
    # 找出高度相关的特征对 (|r| > 0.8)
    high_corr_pairs = []
    for i in range(corr_matrix.shape[0]):
        for j in range(i+1, corr_matrix.shape[1]):
            corr_val = corr_matrix[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append((feature_names[i], feature_names[j], corr_val))
    
    if high_corr_pairs:
        report_lines.append(f"  发现 {len(high_corr_pairs)} 对高度相关的特征 (|r| > 0.8):")
        for feat1, feat2, corr_val in high_corr_pairs[:5]:  # 只显示前5对
            report_lines.append(f"    {feat1} 与 {feat2}: r = {corr_val:.4f}")
    else:
        report_lines.append("  没有发现高度相关的特征对 (|r| > 0.8)")
    
    return report_lines

def analyze_problem_blocks(df_meta):
    """分析问题区块"""
    print("\n" + "="*60)
    print("问题区块分析")
    print("="*60)
    
    report_lines = []
    
    # Block分布
    block_counts = df_meta['block'].value_counts().sort_index()
    
    report_lines.append("Block分布:")
    for block, count in block_counts.items():
        percentage = count / len(df_meta) * 100
        report_lines.append(f"  Block {block}: {count:4d} 条 ({percentage:.1f}%)")
    
    # 检查每个Block内的bRate变化
    report_lines.append(f"\n各Block的bRate统计:")
    block_stats = df_meta.groupby('block')['bRate'].agg(['mean', 'std', 'count'])
    
    for block, row in block_stats.iterrows():
        report_lines.append(f"  Block {block}: 均值={row['mean']:.4f}, 标准差={row['std']:.4f}, 样本数={row['count']}")
    
    # ANOVA检验Block间差异
    from scipy import stats
    
    block_groups = [group['bRate'].values for _, group in df_meta.groupby('block')]
    if len(block_groups) >= 2:
        f_stat, p_value = stats.f_oneway(*block_groups)
        report_lines.append(f"\nBlock间bRate差异的ANOVA检验:")
        report_lines.append(f"  F统计量: {f_stat:.4f}")
        report_lines.append(f"  p值: {p_value:.4e}")
        report_lines.append(f"  {'存在显著差异' if p_value < 0.05 else '无显著差异'}")
    
    return report_lines

def analyze_feature_distributions(X, feature_names):
    """分析特征分布"""
    print("\n" + "="*60)
    print("特征分布分析")
    print("="*60)
    
    report_lines = []
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    report_lines.append("特征统计摘要 (前10个特征):")
    
    for i in range(min(10, X.shape[1])):
        feat = X[:, i]
        report_lines.append(f"\n  {feature_names[i]}:")
        report_lines.append(f"    均值: {feat.mean():.4f}")
        report_lines.append(f"    标准差: {feat.std():.4f}")
        report_lines.append(f"    最小值: {feat.min():.4f}")
        report_lines.append(f"    最大值: {feat.max():.4f}")
        report_lines.append(f"    偏度: {stats.skew(feat):.4f}")
        report_lines.append(f"    峰度: {stats.kurtosis(feat):.4f}")
        
        # 检查正态性
        if len(feat) > 3:
            stat, p_value = stats.normaltest(feat)
            report_lines.append(f"    正态性检验: p={p_value:.4e} {'(正态)' if p_value > 0.05 else '(非正态)'}")
    
    return report_lines

def analyze_outliers(X, y, feature_names):
    """分析异常值"""
    print("\n" + "="*60)
    print("异常值分析")
    print("="*60)
    
    report_lines = []
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    report_lines.append("异常值检测 (使用IQR方法):")
    
    outlier_counts = {}
    
    for i in range(min(10, X.shape[1])):
        feat = X[:, i]
        Q1 = np.percentile(feat, 25)
        Q3 = np.percentile(feat, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = np.where((feat < lower_bound) | (feat > upper_bound))[0]
        outlier_counts[feature_names[i]] = len(outliers)
    
    # 汇总异常值最多的特征
    sorted_outliers = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)
    
    report_lines.append(f"\n异常值最多的特征 (Top 5):")
    for i, (feat_name, count) in enumerate(sorted_outliers[:5]):
        percentage = count / X.shape[0] * 100
        report_lines.append(f"  {i+1}. {feat_name}: {count} 个异常值 ({percentage:.1f}%)")
    
    # bRate的异常值
    bRate = y
    Q1 = np.percentile(bRate, 25)
    Q3 = np.percentile(bRate, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    bRate_outliers = np.where((bRate < lower_bound) | (bRate > upper_bound))[0]
    
    report_lines.append(f"\nbRate异常值:")
    report_lines.append(f"  异常值数量: {len(bRate_outliers)} ({len(bRate_outliers)/len(bRate)*100:.1f}%)")
    report_lines.append(f"  异常值范围: {bRate[bRate_outliers].min():.4f} 到 {bRate[bRate_outliers].max():.4f}")
    
    return report_lines

def create_advanced_visualizations(df_meta, X, y, paths, feature_names):
    """创建高级可视化图表"""
    print("\n" + "="*60)
    print("创建高级可视化图表")
    print("="*60)
    
    # 确保分析目录存在
    paths['analysis_dir'].mkdir(exist_ok=True)
    
    if feature_names is None:
        feature_names = [
            'Ha', 'pHa', 'La', 'Hb', 'pHb', 'Lb', 'LotShapeB', 'LotNumB',
            'Amb', 'Corr', 'EV_A', 'EV_B', 'EV_diff', 'risk_A', 'risk_B',
            'feedback', 'block'
        ]
    
    # 1. Feedback条件下的bRate分布对比图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    feedback_true = df_meta['feedback'] == True
    feedback_false = df_meta['feedback'] == False
    
    plt.hist(y[feedback_true], bins=50, alpha=0.5, label='有反馈', density=True)
    plt.hist(y[feedback_false], bins=50, alpha=0.5, label='无反馈', density=True)
    plt.xlabel('bRate')
    plt.ylabel('密度')
    plt.title('Feedback条件下的bRate分布')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    box_data = [y[feedback_true], y[feedback_false]]
    plt.boxplot(box_data, tick_labels=['有反馈', '无反馈'])
    plt.ylabel('bRate')
    plt.title('Feedback条件下的bRate箱线图')
    
    plt.tight_layout()
    plt.savefig(paths['analysis_dir'] / 'feedback_bRate_comparison.png', dpi=150)
    plt.close()
    
    # 2. EV_diff与bRate的关系（按Feedback条件着色）
    plt.figure(figsize=(10, 6))
    
    scatter1 = plt.scatter(df_meta.loc[feedback_true, 'EV_diff'], y[feedback_true], 
                          alpha=0.5, s=10, label='有反馈', c='blue')
    scatter2 = plt.scatter(df_meta.loc[feedback_false, 'EV_diff'], y[feedback_false], 
                          alpha=0.5, s=10, label='无反馈', c='red')
    
    plt.xlabel('EV_diff (Gamble B - Gamble A的期望值差)')
    plt.ylabel('bRate')
    plt.title('EV_diff与bRate的关系（按Feedback条件着色）')
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths['analysis_dir'] / 'ev_diff_bRate_feedback.png', dpi=150)
    plt.close()
    
    # 3. 各Block的bRate分布
    plt.figure(figsize=(12, 6))
    
    unique_blocks = sorted(df_meta['block'].unique())
    box_data = [y[df_meta['block'] == block] for block in unique_blocks]
    
    plt.boxplot(box_data, tick_labels=[f'Block {b}' for b in unique_blocks])
    plt.xlabel('Block')
    plt.ylabel('bRate')
    plt.title('各Block的bRate分布')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(paths['analysis_dir'] / 'block_bRate_distribution.png', dpi=150)
    plt.close()
    
    # 4. 特征重要性排序（基于与bRate的相关性）
    correlations = []
    for i in range(X.shape[1]):
        if i < len(feature_names):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            if not np.isnan(corr):
                correlations.append((feature_names[i], abs(corr), corr))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    plt.figure(figsize=(12, 8))
    top_n = min(15, len(correlations))
    feature_names_top = [c[0] for c in correlations[:top_n]]
    corr_values_top = [c[2] for c in correlations[:top_n]]
    colors = ['red' if v < 0 else 'blue' for v in corr_values_top]
    
    y_pos = np.arange(top_n)
    plt.barh(y_pos, corr_values_top, color=colors)
    plt.yticks(y_pos, feature_names_top)
    plt.xlabel('与bRate的相关系数')
    plt.title('特征与bRate的相关性（Top 15）')
    plt.tight_layout()
    plt.savefig(paths['analysis_dir'] / 'feature_correlation_ranking.png', dpi=150)
    plt.close()
    
    print(f"高级可视化图表已保存到: {paths['analysis_dir']}")

def generate_report(report_sections, paths):
    """生成分析报告"""
    print("\n" + "="*60)
    print("生成分析报告")
    print("="*60)
    
    with open(paths['report_txt'], 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CHOICES13K 探索性数据分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        for section_name, section_lines in report_sections:
            f.write(section_name + "\n")
            f.write("-" * len(section_name) + "\n")
            f.write("\n".join(section_lines))
            f.write("\n\n")
    
    print(f"分析报告已保存到: {paths['report_txt']}")
    
    # 打印报告摘要
    print("\n分析报告摘要:")
    for section_name, section_lines in report_sections:
        print(f"\n{section_name}:")
        for line in section_lines[:3]:  # 只打印前3行作为摘要
            print(f"  {line}")

def main():
    """主函数"""
    print("Choices13k 探索性数据分析")
    print("=" * 80)
    
    # 设置路径
    paths = setup_paths()
    
    # 确保分析目录存在
    paths['analysis_dir'].mkdir(exist_ok=True)
    
    # 加载数据
    standardized_data, X, y, df_meta, splits = load_data(paths)
    
    # 定义特征名称
    feature_names = [
        'Ha', 'pHa', 'La', 'Hb', 'pHb', 'Lb', 'LotShapeB', 'LotNumB',
        'Amb', 'Corr', 'EV_A', 'EV_B', 'EV_diff', 'risk_A', 'risk_B',
        'feedback', 'block'
    ]
    
    # 执行各项分析
    report_sections = []
    
    # 1. 数据集结构分析
    print("\n执行数据集结构分析...")
    structure_report = analyze_dataset_structure(standardized_data, df_meta)
    report_sections.append(("数据集结构分析", structure_report))
    
    # 2. Feedback条件分析
    print("执行Feedback条件分析...")
    feedback_report = analyze_feedback_conditions(df_meta, X, feature_names)
    report_sections.append(("Feedback条件分析", feedback_report))
    
    # 3. 相关性分析
    print("执行相关性分析...")
    correlation_report = analyze_correlations(X, y, feature_names)
    report_sections.append(("相关性分析", correlation_report))
    
    # 4. 问题区块分析
    print("执行问题区块分析...")
    block_report = analyze_problem_blocks(df_meta)
    report_sections.append(("问题区块分析", block_report))
    
    # 5. 特征分布分析
    print("执行特征分布分析...")
    feature_report = analyze_feature_distributions(X, feature_names)
    report_sections.append(("特征分布分析", feature_report))
    
    # 6. 异常值分析
    print("执行异常值分析...")
    outlier_report = analyze_outliers(X, y, feature_names)
    report_sections.append(("异常值分析", outlier_report))
    
    # 7. 创建高级可视化图表
    print("创建高级可视化图表...")
    create_advanced_visualizations(df_meta, X, y, paths, feature_names)
    
    # 8. 生成完整报告
    generate_report(report_sections, paths)
    
    print("\n" + "=" * 80)
    print("探索性数据分析完成!")
    print("=" * 80)
    
    # 输出关键发现
    print("\n关键发现摘要:")
    print(f"1. 数据集包含 {len(standardized_data)} 条记录，{X.shape[1]} 个特征")
    
    # Feedback条件差异
    feedback_true = df_meta['feedback'] == True
    feedback_false = df_meta['feedback'] == False
    bRate_diff = df_meta.loc[feedback_true, 'bRate'].mean() - df_meta.loc[feedback_false, 'bRate'].mean()
    print(f"2. Feedback条件: 有反馈={feedback_true.sum()}条, 无反馈={feedback_false.sum()}条")
    print(f"   bRate差异: 有反馈-无反馈 = {bRate_diff:.4f}")
    
    # 最重要的特征
    correlations = []
    for i in range(X.shape[1]):
        if i < len(feature_names):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            if not np.isnan(corr):
                correlations.append((feature_names[i], abs(corr)))
    
    if correlations:
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_feature = correlations[0][0]
        top_corr = correlations[0][1]
        print(f"3. 与bRate最相关的特征: {top_feature} (|r| = {top_corr:.4f})")
    
    # 数据质量问题
    missing_values = df_meta.isnull().sum().sum()
    duplicates = df_meta.duplicated().sum()
    print(f"4. 数据质量: 缺失值={missing_values}, 重复记录={duplicates}")
    
    print(f"\n详细报告: {paths['report_txt']}")
    print(f"可视化图表: {paths['analysis_dir']}")

if __name__ == "__main__":
    main()