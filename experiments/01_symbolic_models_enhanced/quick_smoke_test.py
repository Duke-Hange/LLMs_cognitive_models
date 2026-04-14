"""
快速冒烟测试：用少量样本验证增强数据标准化与 4 个符号模型的 fit/predict 流程无报错。
开发/CI 用；主流程请运行 train_enhanced_models.py。
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from enhanced_data_standardization import EnhancedChoices13kStandardizer, create_enhanced_splits
from enhanced_symbolic_models import create_enhanced_model, EnhancedModelAdapter

N_SMOKE = 500  # 冒烟测试样本数


def main():
    print("快速冒烟测试（增强符号模型）")
    print("=" * 50)

    # 1. 加载少量数据并标准化
    print("\n1. 加载与标准化...")
    standardizer = EnhancedChoices13kStandardizer(
        selections_path=str(project_root / "数据集" / "choices13k-main" / "c13k_selections.csv"),
        problems_path=str(project_root / "数据集" / "choices13k-main" / "c13k_problems.json"),
    )
    df = standardizer.load_and_merge_data()
    df_small = df.head(N_SMOKE).copy()
    standardizer.df = df_small
    standardized_data = standardizer.standardize_all()
    if not standardized_data:
        print("错误: 无标准化数据")
        return 1
    y = standardizer.get_target_vector()
    print(f"   样本数: {len(standardized_data)}, y.shape: {y.shape}")

    # 2. 一次划分
    print("\n2. 划分 (problem, random_state=42)...")
    train_idx, test_idx, split_info = create_enhanced_splits(
        standardized_data, split_type="problem", random_state=42
    )
    train_data = [standardized_data[i] for i in train_idx]
    test_data = [standardized_data[i] for i in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    print(f"   训练: {len(train_data)}, 测试: {len(test_data)}")

    # 3. 对 4 个模型各做 fit + predict，打印简要指标
    print("\n3. 各模型 fit + predict...")
    for model_name in ["ev", "eu", "pt3", "pt5"]:
        model = create_enhanced_model(model_name)
        adapter = EnhancedModelAdapter(model)
        adapter.fit_from_standardized(train_data, y_train)
        y_test_pred = adapter.predict_from_standardized(test_data)
        mse = float(np.mean((y_test - y_test_pred) ** 2))
        ss_res = np.sum((y_test - y_test_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
        print(f"   {model_name:4s}: test_mse={mse:.6f}, test_r2={r2:.4f}")

    print("\n" + "=" * 50)
    print("冒烟测试通过。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
