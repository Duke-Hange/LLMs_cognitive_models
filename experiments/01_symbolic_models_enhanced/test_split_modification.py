#!/usr/bin/env python3
"""
测试修改后的Parameter-Amb-Split划分逻辑（使用增强实验模块）
在 01_symbolic_models_enhanced 目录下运行: python test_split_modification.py
"""

import sys
from pathlib import Path

# 保证当前目录在路径中，便于从本目录直接运行
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from enhanced_data_standardization import create_enhanced_splits
import json

# 创建模拟数据来测试划分逻辑（格式与 create_enhanced_splits 一致）
def create_test_data():
    """创建测试数据，包含一些有多个样本的问题"""
    test_data = []
    # 增强划分需要: metadata.index, problem_id, json_problem_id, feedback_condition, block; context.features.ev_diff, Amb
    def row(problem_id, index, Amb, bRate=0.5):
        return {
            'context': {'features': {'Amb': Amb, 'ev_diff': 0.0}},
            'metadata': {'problem_id': problem_id, 'index': index, 'json_problem_id': problem_id, 'feedback_condition': 0, 'block': 0},
            'action': {'bRate': bRate}
        }
    # 问题1: Amb=0, 2个样本
    test_data.append(row(1, 0, 0, 0.5))
    test_data.append(row(1, 1, 0, 0.6))
    # 问题2: Amb=1, 2个样本
    test_data.append(row(2, 2, 1, 0.7))
    test_data.append(row(2, 3, 1, 0.8))
    # 问题3: Amb=0, 1个样本
    test_data.append(row(3, 4, 0, 0.9))
    # 问题4: Amb=1, 1个样本
    test_data.append(row(4, 5, 1, 0.3))
    return test_data

def test_parameter_amb_split():
    """测试parameter_amb划分"""
    print("测试修改后的Parameter-Amb-Split划分逻辑（增强模块）")
    print("=" * 60)

    test_data = create_test_data()

    # 运行划分（增强版 API）
    train_idx, test_idx, split_info = create_enhanced_splits(
        test_data, split_type='parameter_amb'
    )

    print(f"划分描述: {split_info['description']}")
    print(f"训练集样本数: {len(train_idx)}")
    print(f"测试集样本数: {len(test_idx)}")

    # 提取问题ID
    train_problems = set()
    test_problems = set()

    for idx in train_idx:
        train_problems.add(test_data[idx]['metadata']['problem_id'])

    for idx in test_idx:
        test_problems.add(test_data[idx]['metadata']['problem_id'])

    print(f"\n训练集问题ID: {sorted(train_problems)}")
    print(f"测试集问题ID: {sorted(test_problems)}")

    # 检查重叠
    overlap = train_problems.intersection(test_problems)
    print(f"\n重叠问题数: {len(overlap)}")
    if overlap:
        print(f"重叠问题ID: {sorted(overlap)}")
    else:
        print("✓ 无重叠问题 - 划分正确")

    # 验证每个问题的所有样本都在同一划分中
    problem_samples = {}
    for item in test_data:
        pid = item['metadata']['problem_id']
        if pid not in problem_samples:
            problem_samples[pid] = {'train': 0, 'test': 0}

        idx = item['metadata']['index']
        if idx in train_idx:
            problem_samples[pid]['train'] += 1
        elif idx in test_idx:
            problem_samples[pid]['test'] += 1

    print("\n问题样本分布:")
    all_correct = True
    for pid, counts in sorted(problem_samples.items()):
        if counts['train'] > 0 and counts['test'] > 0:
            print(f"  问题{pid}: ❌ 分割在训练集({counts['train']})和测试集({counts['test']})")
            all_correct = False
        elif counts['train'] > 0:
            print(f"  问题{pid}: ✓ 全部在训练集({counts['train']}样本)")
        elif counts['test'] > 0:
            print(f"  问题{pid}: ✓ 全部在测试集({counts['test']}样本)")

    if all_correct:
        print("\n✓ 所有测试通过: 修改后的划分实现了问题级别分离")
    else:
        print("\n❌ 测试失败: 有些问题被分割了")

    return all_correct

if __name__ == "__main__":
    success = test_parameter_amb_split()
    sys.exit(0 if success else 1)
