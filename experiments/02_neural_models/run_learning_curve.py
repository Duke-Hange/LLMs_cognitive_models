"""
数据量–性能曲线：在不同训练集比例下训练模型，固定测试集评估，记录 test MSE 并绘图。
与可参考代码协议一致：50 个比例点、固定测试集、横轴训练数据比例(%)、纵轴 Test MSE。
支持 --plot-style reference（无平滑，默认）与 smoothed（平滑 + 每种子细线）。
experiments 使用单一 train_test_split（不包含 split_type 维度）。
"""

import sys
import csv
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

_exp = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_exp))
from shared.path_bootstrap import ensure_experiments_on_path

ensure_experiments_on_path(setup_font=True)

from config import (
    SEEDS,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    DATA_QUANTITY_N_FRACTIONS,
    RESULTS_DIR,
    CURVES_DIR,
)
from data_loader import (
    load_csv_data,
    build_distribution_encodings,
    get_target_vector,
)
from train import set_seed, collect_learning_curve_data
from plotting import plot_data_quantity_curves, NEURAL_MODEL_TYPES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="数据量–性能曲线（与可参考代码协议一致：50 点、固定测试集、参考式绘图）"
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
    parser.add_argument(
        "--plot-style",
        type=str,
        choices=["reference", "smoothed"],
        default="reference",
        help="reference=无平滑、参考式绘图；smoothed=平滑平均线+每种子细线",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="按 model_type 保存 test_mse 数组为 .npy（多种子时取 mean）",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="跑满 EPOCHS、不早停，与 train.py --no-early-stop 行为一致",
    )
    args = parser.parse_args()
    seeds: List[int] = args.seeds
    n_fractions = max(1, args.n_fractions)
    plot_style = args.plot_style
    if args.no_early_stop:
        run_epochs = EPOCHS
        run_patience = EPOCHS
        print(f"长跑模式：{EPOCHS} epoch，无早停")
    else:
        run_epochs = EPOCHS
        run_patience = EARLY_STOPPING_PATIENCE

    set_seed(seeds[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Seeds: {seeds}, Fractions: {n_fractions}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)

    print("加载 CSV 数据...")
    data_df = load_csv_data()
    enc_A, enc_B, enc_full = build_distribution_encodings(data_df)
    y = get_target_vector(data_df)
    print(f"样本数: {len(y)}, enc_A: {enc_A.shape}, enc_full: {enc_full.shape}")
    b_rate_head = data_df["bRate"].head(5).to_numpy()
    y_head = y[:5]
    print(f"标签检查（前5条） y + bRate: {np.round(y_head + b_rate_head, 6)}")

    rows, timestamp = collect_learning_curve_data(
        enc_A, enc_B, enc_full, y, seeds, device,
        n_fractions=n_fractions,
        epochs=run_epochs,
        patience=run_patience,
        catch_errors=True,
    )

    csv_path = CURVES_DIR / f"learning_curve_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "split_type",
                "model_type",
                "fraction",
                "seed",
                "test_mse",
                "test_cross_entropy",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"学习曲线数据已保存: {csv_path}")

    json_path = CURVES_DIR / f"learning_curve_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "rows": rows, "seeds": seeds, "split": "single_train_test_split"}, f, indent=2, ensure_ascii=False)
    print(f"学习曲线 JSON 已保存: {json_path}")

    try:
        plot_data_quantity_curves(rows, timestamp, CURVES_DIR, plot_style=plot_style)
    except ImportError:
        print("未安装 matplotlib，跳过绘图。")

    if args.save_npy:
        for model_type in NEURAL_MODEL_TYPES:
            sub = [r for r in rows if r["model_type"] == model_type]
            if not sub:
                continue
            by_frac: Dict[float, List[float]] = {}
            for r in sub:
                by_frac.setdefault(r["fraction"], []).append(r["test_mse"])
            fracs_sorted = sorted(by_frac.keys())
            mean_arr = np.array([np.mean(by_frac[f]) for f in fracs_sorted])
            npy_path = CURVES_DIR / f"learning_curve_{timestamp}_{model_type}.npy"
            np.save(npy_path, mean_arr)
        print(f".npy 已保存至 {CURVES_DIR}（按 model_type）")


if __name__ == "__main__":
    main()
