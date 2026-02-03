"""
数据划分和清洗脚本
创建3个文件夹，分别按照策略1、2、3划分数据，并进行数据清洗
"""

import os
import shutil
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from data_standardization import Choices13kStandardizer


def create_directories():
    """创建3个策略文件夹"""
    strategies = {
        'strategy1_problem_split': '策略1: Problem-Split（问题划分）',
        'strategy2_feedback_split': '策略2: Feedback-Split（反馈条件划分）',
        'strategy3_parameter_split': '策略3: Parameter-Split（参数空间划分）'
    }
    
    for folder_name, description in strategies.items():
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"[OK] 创建文件夹: {folder_name} - {description}")
        else:
            print(f"[WARN] 文件夹已存在: {folder_name}")
    
    return list(strategies.keys())


def copy_data_files(folder_path):
    """复制原始数据文件到指定文件夹"""
    files_to_copy = ['c13k_problems.json', 'c13k_selections.csv']
    
    for file_name in files_to_copy:
        if os.path.exists(file_name):
            dest_path = os.path.join(folder_path, file_name)
            shutil.copy2(file_name, dest_path)
            print(f"  [OK] 复制文件: {file_name} -> {dest_path}")
        else:
            print(f"  [WARN] 文件不存在: {file_name}")


def clean_data(df):
    """
    数据清洗函数
    
    Args:
        df: 原始DataFrame
        
    Returns:
        清洗后的DataFrame和清洗报告
    """
    original_count = len(df)
    report = {
        'original_count': original_count,
        'missing_values': {},
        'invalid_values': {},
        'cleaned_count': 0,
        'removed_count': 0
    }
    
    # 1. 检查缺失值
    missing = df.isnull().sum()
    report['missing_values'] = missing[missing > 0].to_dict()
    
    # 2. 检查关键字段的有效性
    # bRate 应该在 [0, 1] 范围内
    invalid_bRate = ((df['bRate'] < 0) | (df['bRate'] > 1)).sum()
    if invalid_bRate > 0:
        report['invalid_values']['bRate'] = int(invalid_bRate)
        df = df[(df['bRate'] >= 0) & (df['bRate'] <= 1)]
    
    # 概率值应该在 [0, 1] 范围内
    for col in ['pHa', 'pHb']:
        invalid_prob = ((df[col] < 0) | (df[col] > 1)).sum()
        if invalid_prob > 0:
            report['invalid_values'][col] = int(invalid_prob)
            df = df[(df[col] >= 0) & (df[col] <= 1)]
    
    # n (被试数量) 应该 > 0
    invalid_n = (df['n'] <= 0).sum()
    if invalid_n > 0:
        report['invalid_values']['n'] = int(invalid_n)
        df = df[df['n'] > 0]
    
    # 3. 检查异常值（使用IQR方法）
    numeric_cols = ['Ha', 'La', 'Hb', 'Lb', 'bRate', 'bRate_std']
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # 使用3倍IQR，更宽松
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                # 记录异常值数量，但不删除（可能是有效数据）
                if 'outliers' not in report:
                    report['outliers'] = {}
                report['outliers'][col] = int(outliers)
    
    # 4. 删除重复行（如果有）
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        report['duplicates_removed'] = int(duplicates)
    
    report['cleaned_count'] = len(df)
    report['removed_count'] = original_count - len(df)
    
    return df, report


def strategy1_problem_split(folder_path):
    """策略1: Problem-Split（问题划分）"""
    print(f"\n{'='*60}")
    print(f"策略1: Problem-Split（问题划分）")
    print(f"{'='*60}")
    
    # 加载数据
    selections_path = os.path.join(folder_path, 'c13k_selections.csv')
    problems_path = os.path.join(folder_path, 'c13k_problems.json')
    
    df = pd.read_csv(selections_path)
    print(f"原始数据量: {len(df)} 条")
    
    # 数据清洗
    print("\n进行数据清洗...")
    df_cleaned, clean_report = clean_data(df)
    print_clean_report(clean_report)
    
    # 按问题ID划分
    problem_ids = df_cleaned['Problem'].unique()
    print(f"\n唯一问题数: {len(problem_ids)}")
    
    train_problems, test_problems = train_test_split(
        problem_ids,
        test_size=0.2,
        random_state=42
    )
    
    train_mask = df_cleaned['Problem'].isin(train_problems)
    test_mask = df_cleaned['Problem'].isin(test_problems)
    
    train_df = df_cleaned[train_mask].copy()
    test_df = df_cleaned[test_mask].copy()
    
    print(f"\n划分结果:")
    print(f"  训练集: {len(train_df)} 条 ({len(train_problems)} 个问题)")
    print(f"  测试集: {len(test_df)} 条 ({len(test_problems)} 个问题)")
    
    # 保存划分后的数据
    train_path = os.path.join(folder_path, 'train.csv')
    test_path = os.path.join(folder_path, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"  [OK] 训练集已保存: {train_path}")
    print(f"  [OK] 测试集已保存: {test_path}")
    
    # 保存划分信息
    split_info = {
        'strategy': 'Problem-Split',
        'train_problems': sorted(train_problems.tolist()),
        'test_problems': sorted(test_problems.tolist()),
        'train_size': len(train_df),
        'test_size': len(test_df),
        'clean_report': clean_report
    }
    
    info_path = os.path.join(folder_path, 'split_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    
    print(f"  [OK] 划分信息已保存: {info_path}")


def strategy2_feedback_split(folder_path):
    """策略2: Feedback-Split（反馈条件划分）"""
    print(f"\n{'='*60}")
    print(f"策略2: Feedback-Split（反馈条件划分）")
    print(f"{'='*60}")
    
    # 加载数据
    selections_path = os.path.join(folder_path, 'c13k_selections.csv')
    problems_path = os.path.join(folder_path, 'c13k_problems.json')
    
    df = pd.read_csv(selections_path)
    print(f"原始数据量: {len(df)} 条")
    
    # 数据清洗
    print("\n进行数据清洗...")
    df_cleaned, clean_report = clean_data(df)
    print_clean_report(clean_report)
    
    # 按反馈条件划分
    train_mask = df_cleaned['Feedback'] == True
    test_mask = df_cleaned['Feedback'] == False
    
    train_df = df_cleaned[train_mask].copy()
    test_df = df_cleaned[test_mask].copy()
    
    print(f"\n划分结果:")
    print(f"  训练集 (有反馈): {len(train_df)} 条")
    print(f"  测试集 (无反馈): {len(test_df)} 条")
    
    # 保存划分后的数据
    train_path = os.path.join(folder_path, 'train.csv')
    test_path = os.path.join(folder_path, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"  [OK] 训练集已保存: {train_path}")
    print(f"  [OK] 测试集已保存: {test_path}")
    
    # 保存划分信息
    split_info = {
        'strategy': 'Feedback-Split',
        'train_condition': 'Feedback=True',
        'test_condition': 'Feedback=False',
        'train_size': len(train_df),
        'test_size': len(test_df),
        'train_problems': sorted(train_df['Problem'].unique().tolist()),
        'test_problems': sorted(test_df['Problem'].unique().tolist()),
        'clean_report': clean_report
    }
    
    info_path = os.path.join(folder_path, 'split_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    
    print(f"  [OK] 划分信息已保存: {info_path}")


def strategy3_parameter_split(folder_path):
    """策略3: Parameter-Split（参数空间划分）"""
    print(f"\n{'='*60}")
    print(f"策略3: Parameter-Split（参数空间划分）")
    print(f"{'='*60}")
    
    # 加载数据
    selections_path = os.path.join(folder_path, 'c13k_selections.csv')
    problems_path = os.path.join(folder_path, 'c13k_problems.json')
    
    df = pd.read_csv(selections_path)
    print(f"原始数据量: {len(df)} 条")
    
    # 数据清洗
    print("\n进行数据清洗...")
    df_cleaned, clean_report = clean_data(df)
    print_clean_report(clean_report)
    
    # 计算 EV_diff（期望值差异）
    # EV_A = pHa * Ha + (1-pHa) * La
    # EV_B = pHb * Hb + (1-pHb) * Lb
    df_cleaned['EV_A'] = df_cleaned['pHa'] * df_cleaned['Ha'] + (1 - df_cleaned['pHa']) * df_cleaned['La']
    df_cleaned['EV_B'] = df_cleaned['pHb'] * df_cleaned['Hb'] + (1 - df_cleaned['pHb']) * df_cleaned['Lb']
    df_cleaned['EV_diff'] = df_cleaned['EV_B'] - df_cleaned['EV_A']
    
    # 按 EV_diff 划分（使用中位数作为阈值）
    threshold = df_cleaned['EV_diff'].abs().median()
    print(f"\nEV_diff 阈值 (中位数): {threshold:.4f}")
    
    train_mask = df_cleaned['EV_diff'].abs() < threshold
    test_mask = df_cleaned['EV_diff'].abs() >= threshold
    
    train_df = df_cleaned[train_mask].copy()
    test_df = df_cleaned[test_mask].copy()
    
    print(f"\n划分结果:")
    print(f"  训练集 (|EV_diff| < {threshold:.4f}): {len(train_df)} 条")
    print(f"  测试集 (|EV_diff| >= {threshold:.4f}): {len(test_df)} 条")
    
    # 保存划分后的数据
    train_path = os.path.join(folder_path, 'train.csv')
    test_path = os.path.join(folder_path, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"  [OK] 训练集已保存: {train_path}")
    print(f"  [OK] 测试集已保存: {test_path}")
    
    # 保存划分信息
    split_info = {
        'strategy': 'Parameter-Split',
        'split_parameter': 'EV_diff',
        'threshold': float(threshold),
        'train_condition': f'|EV_diff| < {threshold:.4f}',
        'test_condition': f'|EV_diff| >= {threshold:.4f}',
        'train_size': len(train_df),
        'test_size': len(test_df),
        'train_problems': sorted(train_df['Problem'].unique().tolist()),
        'test_problems': sorted(test_df['Problem'].unique().tolist()),
        'clean_report': clean_report
    }
    
    info_path = os.path.join(folder_path, 'split_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    
    print(f"  [OK] 划分信息已保存: {info_path}")


def print_clean_report(report):
    """打印清洗报告"""
    print(f"  原始数据量: {report['original_count']}")
    
    if report['missing_values']:
        print(f"  缺失值:")
        for col, count in report['missing_values'].items():
            print(f"    - {col}: {count} 个")
    
    if report['invalid_values']:
        print(f"  无效值:")
        for col, count in report['invalid_values'].items():
            print(f"    - {col}: {count} 个")
    
    if 'outliers' in report:
        print(f"  异常值 (已记录但未删除):")
        for col, count in report['outliers'].items():
            print(f"    - {col}: {count} 个")
    
    if 'duplicates_removed' in report:
        print(f"  重复行: {report['duplicates_removed']} 个")
    
    print(f"  清洗后数据量: {report['cleaned_count']}")
    print(f"  删除数据量: {report['removed_count']}")


def main():
    """主函数"""
    print("="*60)
    print("Choices13k 数据划分和清洗")
    print("="*60)
    
    # 1. 创建文件夹
    print("\n步骤 1: 创建文件夹")
    print("-"*60)
    folders = create_directories()
    
    # 2. 复制数据文件
    print("\n步骤 2: 复制原始数据文件")
    print("-"*60)
    for folder in folders:
        print(f"\n处理文件夹: {folder}")
        copy_data_files(folder)
    
    # 3. 按策略划分数据
    print("\n步骤 3: 按策略划分数据")
    print("-"*60)
    
    strategy1_problem_split(folders[0])
    strategy2_feedback_split(folders[1])
    strategy3_parameter_split(folders[2])
    
    print("\n" + "="*60)
    print("完成！所有数据已划分并清洗")
    print("="*60)
    print("\n文件夹结构:")
    for folder in folders:
        print(f"  {folder}/")
        print(f"    ├── c13k_problems.json")
        print(f"    ├── c13k_selections.csv")
        print(f"    ├── train.csv")
        print(f"    ├── test.csv")
        print(f"    └── split_info.json")


if __name__ == "__main__":
    main()
