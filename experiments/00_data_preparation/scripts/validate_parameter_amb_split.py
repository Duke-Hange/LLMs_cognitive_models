#!/usr/bin/env python3
"""
Parameter-Amb-Split完整性验证脚本
验证当前Parameter-Amb-Split划分中问题级别的重叠情况
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# 脚本位于 00_data_preparation/scripts/，outputs 位于 00_data_preparation/outputs/
_00_ROOT = Path(__file__).resolve().parent.parent


def load_split_indices() -> Tuple[List[int], List[int]]:
    """加载Parameter-Amb-Split的划分索引"""
    splits_path = _00_ROOT / "outputs" / "parameter_amb_split" / "enhanced_data_splits.json"

    if not splits_path.exists():
        print(f"错误: 划分文件不存在: {splits_path}")
        sys.exit(1)

    with open(splits_path, 'r', encoding='utf-8') as f:
        splits_data = json.load(f)

    if "parameter_amb" not in splits_data:
        print("错误: 划分文件中未找到parameter_amb部分")
        sys.exit(1)

    param_amb = splits_data["parameter_amb"]
    train_indices = param_amb["train_indices"]
    test_indices = param_amb["test_indices"]

    print(f"加载划分索引成功:")
    print(f"  训练集样本数: {len(train_indices)}")
    print(f"  测试集样本数: {len(test_indices)}")
    print(f"  总样本数: {len(train_indices) + len(test_indices)}")

    return train_indices, test_indices


def load_standardized_data() -> List[Dict]:
    """加载标准化数据"""
    data_path = _00_ROOT / "outputs" / "parameter_amb_split" / "c13k_enhanced_standardized.json"

    if not data_path.exists():
        print(f"错误: 标准化数据文件不存在: {data_path}")
        sys.exit(1)

    print(f"加载标准化数据: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  总记录数: {len(data)}")
    return data


def extract_problem_ids(data: List[Dict], indices: List[int]) -> Set[int]:
    """从数据中提取指定索引的problem_id集合"""
    problem_ids = set()
    for idx in indices:
        if idx < 0 or idx >= len(data):
            print(f"警告: 索引 {idx} 超出数据范围 (0-{len(data)-1})")
            continue
        problem_id = data[idx]["metadata"]["problem_id"]
        problem_ids.add(problem_id)
    return problem_ids


def analyze_overlap(train_ids: Set[int], test_ids: Set[int]) -> Dict:
    """分析问题ID重叠情况"""
    overlap = train_ids.intersection(test_ids)

    return {
        "total_problems": len(train_ids.union(test_ids)),
        "train_problems": len(train_ids),
        "test_problems": len(test_ids),
        "overlap_problems": len(overlap),
        "overlap_ratio": len(overlap) / len(train_ids.union(test_ids)) if train_ids.union(test_ids) else 0,
        "overlap_problem_ids": list(overlap),
        "train_only": len(train_ids - test_ids),
        "test_only": len(test_ids - train_ids),
        "is_problem_level_split": len(overlap) == 0
    }


def main():
    print("=" * 70)
    print("Parameter-Amb-Split完整性验证")
    print("=" * 70)

    # 1. 加载划分索引
    print("\n[1/3] 加载划分索引...")
    train_indices, test_indices = load_split_indices()

    # 2. 加载标准化数据
    print("\n[2/3] 加载标准化数据...")
    standardized_data = load_standardized_data()

    # 3. 提取问题ID
    print("\n[3/3] 提取问题ID并分析重叠...")
    train_problem_ids = extract_problem_ids(standardized_data, train_indices)
    test_problem_ids = extract_problem_ids(standardized_data, test_indices)

    print(f"  训练集问题数: {len(train_problem_ids)}")
    print(f"  测试集问题数: {len(test_problem_ids)}")

    # 4. 分析重叠
    analysis = analyze_overlap(train_problem_ids, test_problem_ids)

    # 5. 输出结果
    print("\n" + "=" * 70)
    print("分析结果:")
    print("=" * 70)
    print(f"总问题数: {analysis['total_problems']}")
    print(f"训练集问题数: {analysis['train_problems']}")
    print(f"测试集问题数: {analysis['test_problems']}")
    print(f"重叠问题数: {analysis['overlap_problems']}")
    print(f"重叠比例: {analysis['overlap_ratio']:.2%}")
    print(f"仅在训练集的问题数: {analysis['train_only']}")
    print(f"仅在测试集的问题数: {analysis['test_only']}")
    print(f"是否实现问题级别划分: {'是' if analysis['is_problem_level_split'] else '否'}")

    if analysis['overlap_problems'] > 0:
        print(f"\n警告: 发现 {analysis['overlap_problems']} 个问题同时出现在训练集和测试集")
        print(f"重叠问题ID: {analysis['overlap_problem_ids'][:10]}")  # 只显示前10个
        if len(analysis['overlap_problem_ids']) > 10:
            print(f"  ... 和另外 {len(analysis['overlap_problem_ids']) - 10} 个问题")
    else:
        print(f"\n✓ 验证通过: 实现了严格的问题级别划分")

    # 保存结果到 00_data_preparation/outputs/
    _00_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir = _00_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "parameter_amb_split_validation_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    print(f"\n详细报告已保存到: {output_path}")

    return analysis['is_problem_level_split']


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
