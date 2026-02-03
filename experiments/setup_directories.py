"""
目录结构设置脚本 - 符号模型层实验第1天
创建标准化的实验目录结构
"""

import os
import sys

def create_directory_structure():
    """创建完整的目录结构"""
    
    # 基础目录列表
    directories = [
        # 数据准备目录
        'experiments/00_data_preparation',
        'experiments/00_data_preparation/notebooks',
        'experiments/00_data_preparation/scripts',
        'experiments/00_data_preparation/outputs',
        
        # 符号模型增强实验目录（唯一保留的符号模型实验）
        'experiments/01_symbolic_models_enhanced',
        'experiments/01_symbolic_models_enhanced/analysis',
        'experiments/01_symbolic_models_enhanced/results',
        'experiments/01_symbolic_models_enhanced/results/enhanced_training',
        'experiments/01_symbolic_models_enhanced/results/comparison',
        
        # 跨模型比较
        'experiments/04_comparison',
        'experiments/04_comparison/output',
        
        # 共享工具目录
        'shared',
        'shared/utils',
        'shared/visualization',
    ]
    
    print("=" * 60)
    print("创建目录结构")
    print("=" * 60)
    
    created_dirs = []
    existing_dirs = []
    
    for dir_path in directories:
        # 转换为绝对路径
        abs_path = os.path.abspath(dir_path)
        
        if os.path.exists(abs_path):
            print(f"[OK] {dir_path:50} [已存在]")
            existing_dirs.append(dir_path)
        else:
            try:
                os.makedirs(abs_path, exist_ok=True)
                print(f"[OK] {dir_path:50} [创建成功]")
                created_dirs.append(dir_path)
            except Exception as e:
                print(f"[FAIL] {dir_path:50} [创建失败: {e}]")
    
    return created_dirs, existing_dirs

def create_readme_files():
    """在各个目录创建README占位文件"""
    
    readme_files = [
        ('experiments/00_data_preparation', '数据准备阶段：加载、标准化和探索Choices13k数据'),
        ('experiments/01_symbolic_models_enhanced', '符号模型增强实验：多结果分布、期望效用与前景理论'),
        ('experiments/01_symbolic_models_enhanced/analysis', '分析与稳定性验证'),
        ('experiments/01_symbolic_models_enhanced/results', '实验结果与对比'),
        ('experiments/04_comparison', '跨模型比较：符号、神经及后续模型统一比较'),
        ('shared', '共享工具和函数'),
        ('shared/utils', '通用工具函数'),
        ('shared/visualization', '可视化工具'),
    ]
    
    print("\n" + "=" * 60)
    print("创建README文件")
    print("=" * 60)
    
    created_files = []
    
    for dir_path, description in readme_files:
        readme_path = os.path.join(dir_path, 'README.md')
        
        if os.path.exists(readme_path):
            print(f"[OK] {readme_path:50} [已存在]")
        else:
            try:
                content = f"""# {os.path.basename(dir_path)}

{description}

**创建时间**: {os.path.basename(dir_path)}目录
**所属项目**: Choices13k - 符号模型层实验

## 目录说明

{description}

## 文件说明

暂无文件。此目录将用于存放相关实验文件。

---
*此文件自动生成*"""
                
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"[OK] {readme_path:50} [创建成功]")
                created_files.append(readme_path)
                
            except Exception as e:
                print(f"[FAIL] {readme_path:50} [创建失败: {e}]")
    
    return created_files

def create_project_readme():
    """创建项目总README"""
    
    readme_path = 'README.md'
    
    if os.path.exists(readme_path):
        print(f"[OK] {readme_path:50} [已存在]")
        return False
    
    content = """# Choices13k 风险决策建模项目

## 项目概述

本项目旨在通过对比三种模型来解释人类风险决策行为：
1. **符号模型** (Symbolic Models) - 传统认知心理学模型
2. **神经网络** (Neural Networks) - 无语言先验的统计学习
3. **大语言模型** (LLMs) - 拥有语言先验的通用推理能力

## 当前阶段：符号模型层实验

### 实验目标
- 实现并评估4个符号模型：期望值模型、期望效用理论、前景理论（3参数和5参数）
- 使用Choices13k数据集（13,006个风险选择问题）
- 重点测试模型是否理解反馈机制（Feedback-Split）

### 目录结构

```
D:\\桌面文件夹\\合作项目\\
├── 代码\\                          # 理论文档
├── 数据集\\choices13k-main\\       # 原始数据集
├── experiments\\                   # 实验目录
│   ├── 00_data_preparation\\      # 数据准备阶段
│   ├── 01_symbolic_models_enhanced\\  # 符号模型增强实验（当前阶段）
│   ├── 02_neural_models\\         # 神经网络实验（未来）
│   └── 03_llm_models\\            # LLM实验（未来）
├── shared\\                        # 共享工具
└── README.md                      # 本文件
```

### 快速开始

1. **环境设置**：
   ```bash
   conda activate yh311_G
   ```

2. **验证环境**：
   ```bash
   python experiments/env_verification.py
   ```

3. **设置目录**：
   ```bash
   python experiments/setup_directories.py
   ```

### 实验进度

| 阶段 | 状态 | 开始日期 | 完成日期 |
|------|------|----------|----------|
| 第1天：环境验证与目录创建 | 🟢 进行中 | 2025-01-20 | - |
| 第2-3天：数据准备与探索 | ⚪ 未开始 | - | - |
| 第4天：符号模型实现 | ⚪ 未开始 | - | - |
| 第5天：模型验证与拟合框架 | ⚪ 未开始 | - | - |
| 第6天：Feedback-Split深度实验 | ⚪ 未开始 | - | - |
| 第7天：快速扩展实验 | ⚪ 未开始 | - | - |
| 第8天：结果分析与解释 | ⚪ 未开始 | - | - |
| 第9天：报告撰写与代码整理 | ⚪ 未开始 | - | - |

### 联系信息

项目负责人：用户
开始日期：2025年1月20日

---
*此文件自动生成*"""
    
    try:
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[OK] {readme_path:50} [创建成功]")
        return True
    except Exception as e:
        print(f"[FAIL] {readme_path:50} [创建失败: {e}]")
        return False

def main():
    """主函数"""
    print("符号模型层实验 - 目录结构设置")
    print("=" * 60)
    
    # 创建目录
    created_dirs, existing_dirs = create_directory_structure()
    
    # 创建README文件
    created_files = create_readme_files()
    
    # 创建项目README
    project_readme_created = create_project_readme()
    
    # 总结
    print("\n" + "=" * 60)
    print("设置完成总结")
    print("=" * 60)
    
    print(f"创建的目录: {len(created_dirs)} 个")
    print(f"已存在的目录: {len(existing_dirs)} 个")
    print(f"创建的README文件: {len(created_files)} 个")
    
    if project_readme_created:
        print("[OK] 项目README创建成功")
    
    print("\n[OK] 目录结构设置完成！")
    print("\n下一步建议：")
    print("1. 运行环境验证: python experiments/env_verification.py")
    print("2. 开始数据准备: 创建 experiments/00_data_preparation/scripts/prepare_data.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)