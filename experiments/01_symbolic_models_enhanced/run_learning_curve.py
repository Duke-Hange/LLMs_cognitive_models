"""
符号模型数据量–性能曲线：在不同训练集比例下训练，固定测试集评估，记录 test MSE 并绘图。
对三种划分 × (ev, eu, pt3, pt5) × 多种训练比例 做实验；支持多种子聚合。
"""

import sys
import csv
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

# 项目根路径并设置中文字体（避免中文/负号显示为方框）
_p = Path(__file__).resolve().parent
for _ in range(8):
    _p = _p.parent
    if (_p / "shared").is_dir():
        sys.path.insert(0, str(_p))
        break
from shared.visualization import setup_chinese_font
setup_chinese_font()

from enhanced_data_standardization import create_enhanced_splits
from enhanced_symbolic_models import create_enhanced_model, EnhancedModelAdapter

# 训练集比例列表（与 02 数据量曲线一致）
TRAIN_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SPLIT_TYPES = ["problem", "parameter_amb", "parameter_ev_extreme"]
MODEL_NAMES = ["ev", "eu", "pt3", "pt5"]


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
    return {"mse": mse, "r2": r2}


def run_one_fraction(
    train_data: List[Dict],
    y_train: np.ndarray,
    test_data: List[Dict],
    y_test: np.ndarray,
    model_name: str,
) -> float:
    """在给定训练集上训练符号模型，返回测试集 MSE。"""
    model = create_enhanced_model(model_name)
    adapter = EnhancedModelAdapter(model)
    adapter.fit_from_standardized(train_data, y_train)
    y_pred = adapter.predict_from_standardized(test_data)
    metrics = evaluate_model(y_test, y_pred)
    return metrics["mse"]


def main() -> None:
    parser = argparse.ArgumentParser(description="符号模型数据量–性能曲线")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="随机种子列表")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="标准化数据目录（默认使用项目 01 的 c13k_enhanced_standardized.json 路径）",
    )
    args = parser.parse_args()
    seeds: List[int] = args.seeds

    # 加载标准化数据
    project_root = Path(__file__).parent.parent.parent
    standardized_data = None
    if args.data_dir:
        data_path = Path(args.data_dir)
        if data_path.exists():
            with open(data_path, "r", encoding="utf-8") as f:
                standardized_data = json.load(f)
    if standardized_data is None:
        data_path = Path(__file__).parent / "c13k_enhanced_standardized.json"
        if data_path.exists():
            with open(data_path, "r", encoding="utf-8") as f:
                standardized_data = json.load(f)
    if standardized_data is None:
        data_path = Path(__file__).parent.parent / "00_data_preparation" / "outputs" / "problem_split" / "c13k_enhanced_standardized.json"
        if data_path.exists():
            with open(data_path, "r", encoding="utf-8") as f:
                standardized_data = json.load(f)
    if standardized_data is None:
        # 现场标准化（与 train_enhanced_models 一致）
        from enhanced_data_standardization import EnhancedChoices13kStandardizer
        standardizer = EnhancedChoices13kStandardizer(
            selections_path=str(project_root / "数据集" / "choices13k-main" / "c13k_selections.csv"),
            problems_path=str(project_root / "数据集" / "choices13k-main" / "c13k_problems.json"),
        )
        standardized_data = standardizer.standardize_all()
    y = np.array([item["action"]["bRate"] for item in standardized_data], dtype=np.float64)
    print(f"加载数据: {len(standardized_data)} 条")

    results_dir = Path(__file__).parent / "results" / "enhanced_training"
    curves_dir = results_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        np.random.seed(seed)
        for split_type in SPLIT_TYPES:
            train_idx, test_idx, _ = create_enhanced_splits(
                standardized_data, split_type=split_type, random_state=seed
            )
            train_idx = np.array(train_idx)
            test_idx = np.array(test_idx)
            n_train = len(train_idx)
            train_data_full = [standardized_data[i] for i in train_idx]
            test_data = [standardized_data[i] for i in test_idx]
            y_test = y[test_idx]

            for fraction in TRAIN_FRACTIONS:
                n_use = max(1, int(n_train * fraction))
                train_data_f = train_data_full[:n_use]
                y_train_f = y[train_idx[:n_use]]
                for model_name in MODEL_NAMES:
                    print(f"  [seed={seed}] {split_type} frac={fraction:.2f} {model_name}...")
                    try:
                        test_mse = run_one_fraction(
                            train_data_f, y_train_f, test_data, y_test, model_name
                        )
                    except Exception as e:
                        print(f"    失败: {e}")
                        test_mse = float("nan")
                    rows.append({
                        "split_type": split_type,
                        "model_name": model_name,
                        "fraction": fraction,
                        "seed": seed,
                        "test_mse": test_mse,
                    })

    csv_path = curves_dir / f"learning_curve_symbolic_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split_type", "model_name", "fraction", "seed", "test_mse"])
        w.writeheader()
        w.writerows(rows)
    print(f"学习曲线数据已保存: {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图。")
        return

    for split_type in SPLIT_TYPES:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        for model_name in MODEL_NAMES:
            sub = [r for r in rows if r["split_type"] == split_type and r["model_name"] == model_name]
            if not sub:
                continue
            by_frac: Dict[float, List[float]] = {}
            for r in sub:
                f = r["fraction"]
                m = r["test_mse"]
                if not np.isnan(m):
                    by_frac.setdefault(f, []).append(m)
            if not by_frac:
                continue
            fractions = sorted(by_frac.keys())
            means = [np.mean(by_frac[f]) for f in fractions]
            stds = [np.std(by_frac[f]) for f in fractions]
            ax.plot(fractions, means, "o-", label=model_name)
            ax.fill_between(fractions, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.3)
        ax.set_xlabel("Training set fraction")
        ax.set_ylabel("Test MSE")
        ax.set_title(f"Symbolic — Data quantity vs test MSE ({split_type})")
        ax.legend()
        fig.tight_layout()
        out_path = curves_dir / f"data_quantity_curve_symbolic_{timestamp}_{split_type}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"数据量曲线已保存: {out_path}")

    json_path = curves_dir / f"learning_curve_symbolic_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "rows": rows, "seeds": seeds}, f, indent=2, ensure_ascii=False)
    print(f"学习曲线 JSON 已保存: {json_path}")


if __name__ == "__main__":
    main()
