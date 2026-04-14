"""
experiments：仅创建缺失的实验目录结构，不创建、不覆盖任何 README。
用于新克隆或缺少子目录时的初始化。请在仓库根目录或 experiments 目录下运行。
"""

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = _SCRIPT_DIR


def create_directory_structure():
    """创建 experiments 下的标准目录结构"""
    directories = [
        "00_data_preparation",
        "00_data_preparation/notebooks",
        "00_data_preparation/scripts",
        "00_data_preparation/outputs",
        "01_symbolic_models_enhanced",
        "01_symbolic_models_enhanced/analysis",
        "01_symbolic_models_enhanced/results",
        "01_symbolic_models_enhanced/results/enhanced_training",
        "01_symbolic_models_enhanced/results/enhanced_training/curves",
        "01_symbolic_models_enhanced/results/comparison",
        "02_neural_models",
        "02_neural_models/results",
        "02_neural_models/results/curves",
        "04_comparison",
        "04_comparison/output",
        "results",
        "results/learning_curves_all",
    ]

    print("=" * 60)
    print("创建 experiments 目录结构")
    print("=" * 60)

    created_dirs = []
    existing_dirs = []

    for dir_path in directories:
        abs_path = BASE_DIR / dir_path
        if abs_path.exists():
            print(f"[OK] {dir_path:55} [已存在]")
            existing_dirs.append(dir_path)
        else:
            try:
                abs_path.mkdir(parents=True, exist_ok=True)
                print(f"[OK] {dir_path:55} [创建成功]")
                created_dirs.append(dir_path)
            except Exception as e:
                print(f"[FAIL] {dir_path:55} [创建失败: {e}]")

    return created_dirs, existing_dirs


def main():
    print("experiments 目录结构初始化（仅创建缺失目录）")
    print("=" * 60)
    created_dirs, existing_dirs = create_directory_structure()
    print("\n" + "=" * 60)
    print(f"本次创建: {len(created_dirs)} 个目录，已存在: {len(existing_dirs)} 个")
    print("[OK] 完成。")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
