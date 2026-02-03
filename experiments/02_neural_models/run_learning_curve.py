"""
数据量–性能曲线：在不同训练集比例下训练模型，固定测试集评估，记录 test MSE 并绘图。
50 个比例点，ValueBased + Context-Dependent，多种子，平滑平均线（window=5）。
"""

import sys
import csv
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

# 项目根路径并设置中文字体（避免中文/负号显示为方框）
_p = Path(__file__).resolve().parent
for _ in range(8):
    _p = _p.parent
    if (_p / "shared").is_dir():
        sys.path.insert(0, str(_p))
        break
from shared.visualization import setup_chinese_font
setup_chinese_font()

from config import (
    SEEDS,
    SPLIT_TYPES,
    DATA_QUANTITY_N_FRACTIONS,
    RESULTS_DIR,
    CURVES_DIR,
)
from data_loader import (
    load_standardized_data,
    build_distribution_encodings,
    get_target_vector,
    get_split_data,
)
from train import set_seed, run_one_fraction_split

SMOOTH_WINDOW = 5
MODEL_DISPLAY_NAMES = {"value_based": "ValueBased", "context_dependent": "Context-Dependent"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="数据量–性能曲线（训练集比例 0~100% vs 测试 MSE，50 点，双模型，多种子，平滑）"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="随机种子列表",
    )
    parser.add_argument(
        "--n-fractions",
        type=int,
        default=DATA_QUANTITY_N_FRACTIONS,
        help=f"比例点个数（默认 {DATA_QUANTITY_N_FRACTIONS}）",
    )
    args = parser.parse_args()
    seeds: List[int] = args.seeds
    n_fractions = max(1, args.n_fractions)
    fractions = [i / n_fractions for i in range(1, n_fractions + 1)]

    set_seed(seeds[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Seeds: {seeds}, Fractions: {n_fractions}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)

    print("加载数据...")
    standardized_data = load_standardized_data()
    enc_A, enc_B, enc_full = build_distribution_encodings(standardized_data)
    y = get_target_vector(standardized_data)

    model_types_lc = ["value_based", "context_dependent"]
    rows: List[Dict[str, Any]] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for seed in seeds:
        set_seed(seed)
        for split_type in SPLIT_TYPES:
            split_data = get_split_data(
                standardized_data, enc_A, enc_B, enc_full, y, split_type, random_state=seed
            )
            for fraction in fractions:
                for model_type in model_types_lc:
                    print(f"  [seed={seed}] {split_type} frac={fraction:.2f} {model_type}...")
                    test_mse = run_one_fraction_split(
                        split_data, fraction, model_type, device, seed
                    )
                    rows.append({
                        "split_type": split_type,
                        "model_type": model_type,
                        "fraction": fraction,
                        "seed": seed,
                        "test_mse": test_mse,
                    })

    csv_path = CURVES_DIR / f"learning_curve_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split_type", "model_type", "fraction", "seed", "test_mse"])
        w.writeheader()
        w.writerows(rows)
    print(f"学习曲线数据已保存: {csv_path}")

    # 绘图：按 split_type 分子图，每图内双模型，平滑平均线 + 可选每种子线 + 可选 std 带
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图。")
        json_path = CURVES_DIR / f"learning_curve_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"timestamp": timestamp, "rows": rows, "seeds": seeds}, f, indent=2, ensure_ascii=False)
        print(f"学习曲线 JSON 已保存: {json_path}")
        return

    for split_type in SPLIT_TYPES:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for model_type in model_types_lc:
            sub = [r for r in rows if r["split_type"] == split_type and r["model_type"] == model_type]
            if not sub:
                continue
            # 按 fraction 聚合：每个 fraction 对应多种子的 test_mse 列表
            by_frac: Dict[float, List[float]] = {}
            for r in sub:
                by_frac.setdefault(r["fraction"], []).append(r["test_mse"])
            fracs_sorted = sorted(by_frac.keys())
            mean_arr = np.array([np.mean(by_frac[f]) for f in fracs_sorted])
            std_arr = np.array([np.std(by_frac[f]) for f in fracs_sorted])
            # 每种子一条线（细线半透明）
            seeds_here = sorted({r["seed"] for r in sub})
            for s in seeds_here:
                points = [(r["fraction"], r["test_mse"]) for r in sub if r["seed"] == s]
                points.sort(key=lambda x: x[0])
                xs = [p[0] * 100 for p in points]
                ys = [p[1] for p in points]
                ax.plot(xs, ys, alpha=0.4, linewidth=1)
            # 平滑平均线（window=5）作为主曲线
            if len(mean_arr) >= SMOOTH_WINDOW:
                mean_smooth = np.convolve(mean_arr, np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW, mode="valid")
                std_smooth = np.convolve(std_arr, np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW, mode="valid")
                n_smooth = len(mean_smooth)
                x_smooth = np.linspace(0, 100, n_smooth)
                ax.plot(x_smooth, mean_smooth, linewidth=3, alpha=0.9, label=MODEL_DISPLAY_NAMES[model_type])
                ax.fill_between(
                    x_smooth,
                    mean_smooth - std_smooth,
                    mean_smooth + std_smooth,
                    alpha=0.25,
                )
            else:
                ax.plot(
                    [f * 100 for f in fracs_sorted],
                    mean_arr,
                    "o-",
                    linewidth=2,
                    label=MODEL_DISPLAY_NAMES[model_type],
                )
                ax.fill_between(
                    [f * 100 for f in fracs_sorted],
                    mean_arr - std_arr,
                    mean_arr + std_arr,
                    alpha=0.25,
                )
        ax.set_xlabel("Percent training data used (%)", fontsize=12)
        ax.set_ylabel("Test set MSE", fontsize=12)
        ax.set_title(
            f"Data quantity vs test MSE — {split_type}\n"
            f"(Mean ± std across seeds; mean smoothed with window={SMOOTH_WINDOW})"
        )
        ax.legend()
        ax.set_xlim(0, 100)
        fig.tight_layout()
        out_path = CURVES_DIR / f"data_quantity_curve_{timestamp}_{split_type}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"数据量曲线已保存: {out_path}")

    json_path = CURVES_DIR / f"learning_curve_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "rows": rows, "seeds": seeds}, f, indent=2, ensure_ascii=False)
    print(f"学习曲线 JSON 已保存: {json_path}")


if __name__ == "__main__":
    main()
