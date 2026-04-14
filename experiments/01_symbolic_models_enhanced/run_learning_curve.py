"""
符号模型数据量–性能曲线：在不同训练集比例下训练，固定测试集评估，记录 test MSE 并绘图。
与可参考代码协议一致：50 个比例点（2%–100%）、固定测试集、横轴训练数据比例(%)、纵轴 Test MSE、无平滑。
对三种划分（对比条件）× (ev, eu, pt3, pt5) × 多种训练比例 做实验；支持多种子聚合。
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

_exp = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_exp))
from shared.path_bootstrap import ensure_experiments_on_path

ensure_experiments_on_path(setup_font=True)

from enhanced_data_standardization import create_enhanced_splits, SPLIT_TYPES
from enhanced_symbolic_models import create_enhanced_model, EnhancedModelAdapter

# 与可参考代码一致：50 个比例点（2%, 4%, …, 100%）；与 02 的 DATA_QUANTITY_N_FRACTIONS 保持一致
N_FRACTIONS = 50
MODEL_NAMES = ["ev", "eu", "pt3", "pt5"]


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """同时返回 MSE、R² 和二元交叉熵（与 02 神经侧字段一致）。"""
    # MSE 与 R²
    mse = float(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
    # 交叉熵（将 bRate 视作 [0,1] 上的软标签）
    eps = 1e-8
    y_clip = np.clip(y_pred, eps, 1.0 - eps)
    ce = float(-np.mean(y_true * np.log(y_clip) + (1.0 - y_true) * np.log(1.0 - y_clip)))
    return {"mse": mse, "r2": r2, "cross_entropy": ce}


def run_one_fraction(
    train_data: List[Dict],
    y_train: np.ndarray,
    test_data: List[Dict],
    y_test: np.ndarray,
    model_name: str,
) -> float:
    """在给定训练集上训练符号模型，返回测试集 MSE 与 CE。"""
    model = create_enhanced_model(model_name)
    adapter = EnhancedModelAdapter(model)
    adapter.fit_from_standardized(train_data, y_train)
    y_pred = adapter.predict_from_standardized(test_data)
    metrics = evaluate_model(y_test, y_pred)
    return metrics["mse"], metrics["cross_entropy"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="符号模型数据量–性能曲线（与可参考代码协议一致：50 点、固定测试集、参考式绘图）"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="随机种子列表")
    parser.add_argument(
        "--n-fractions",
        type=int,
        default=N_FRACTIONS,
        help=f"训练数据比例点个数（默认 {N_FRACTIONS}，与可参考代码一致）",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="标准化数据 JSON 的显式路径（默认使用本目录 c13k_enhanced_standardized.json，缺失时自动生成并写入）",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="按 split_type 与 model_name 保存 test_mse 数组为 .npy（多种子时取 mean）",
    )
    parser.add_argument(
        "--fractions-mode",
        type=str,
        choices=["full", "simple"],
        default="full",
        help="full=高分辨率比例列表（接近原文 Fig. S1）; simple=简化版比例列表",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["json", "csv"],
        default="json",
        help="json=c13k JSON + 三种划分；csv=data/data.csv + train_test（与 02 联合图对齐）",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="data-source=csv 时的 CSV 路径（默认 experiments/data/data.csv）",
    )
    args = parser.parse_args()
    seeds: List[int] = args.seeds
    n_fractions = max(1, args.n_fractions)
    if args.fractions_mode == "full":
        # 接近原文：从很小比例到 100%，默认仍采用均匀 1/n..1 的 50 个点
        fractions = [i / n_fractions for i in range(1, n_fractions + 1)]
    else:
        # 简化版：少量代表性点，便于快速实验
        fractions = [0.05, 0.1, 0.2, 0.5, 1.0]

    project_root = _project_root
    if args.data_source == "csv":
        from csv_data import CSV_DATA_PATH, load_csv_standardized_and_targets

        csv_p = Path(args.csv_path) if args.csv_path else CSV_DATA_PATH
        standardized_data, y = load_csv_standardized_and_targets(csv_p)
        split_types_list = ["train_test"]
        print(f"CSV 数据源: {csv_p}（仅 train_test 划分，与 02 一致）")
    else:
        canonical_path = Path(__file__).parent / "c13k_enhanced_standardized.json"
        data_path = Path(args.data_path) if args.data_path else canonical_path
        standardized_data = None
        if data_path.exists():
            with open(data_path, "r", encoding="utf-8") as f:
                standardized_data = json.load(f)
        if standardized_data is None and data_path.resolve() == canonical_path.resolve():
            from enhanced_data_standardization import EnhancedChoices13kStandardizer

            standardizer = EnhancedChoices13kStandardizer(
                selections_path=str(project_root / "数据集" / "choices13k-main" / "c13k_selections.csv"),
                problems_path=str(project_root / "数据集" / "choices13k-main" / "c13k_problems.json"),
            )
            standardized_data = standardizer.standardize_all(save_path=str(canonical_path))
            print(f"已生成并保存标准化数据至: {canonical_path}")
        elif standardized_data is None:
            raise FileNotFoundError(
                f"未找到标准化数据: {data_path}。"
                "请先在 01 目录运行标准化或训练以生成 c13k_enhanced_standardized.json，或指定存在的 --data-path。"
            )
        y = np.array([item["action"]["bRate"] for item in standardized_data], dtype=np.float64)
        split_types_list = list(SPLIT_TYPES)

    print(f"加载数据: {len(standardized_data)} 条")
    print(f"比例点数: {n_fractions}（与可参考代码一致）")

    results_dir = Path(__file__).parent / "results" / "enhanced_training"
    curves_dir = results_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        np.random.seed(seed)
        for split_type in split_types_list:
            if split_type == "train_test":
                from csv_data import SPLIT_RANDOM_STATE as _CSV_RS, TEST_SIZE as _CSV_TS

                train_idx, test_idx, _ = create_enhanced_splits(
                    standardized_data,
                    split_type="train_test",
                    test_size=_CSV_TS,
                    random_state=_CSV_RS + seed,
                )
            else:
                train_idx, test_idx, _ = create_enhanced_splits(
                    standardized_data, split_type=split_type, random_state=seed
                )
            train_idx = np.array(train_idx)
            test_idx = np.array(test_idx)
            n_train = len(train_idx)
            train_data_full = [standardized_data[i] for i in train_idx]
            test_data = [standardized_data[i] for i in test_idx]
            y_test = y[test_idx]

            for fraction in fractions:
                n_use = max(1, int(n_train * fraction))
                train_data_f = train_data_full[:n_use]
                y_train_f = y[train_idx[:n_use]]
                for model_name in MODEL_NAMES:
                    print(f"  [seed={seed}] {split_type} frac={fraction:.2f} {model_name}...")
                    try:
                        test_mse, test_ce = run_one_fraction(
                            train_data_f, y_train_f, test_data, y_test, model_name
                        )
                    except Exception as e:
                        print(f"    失败: {e}")
                        test_mse = float("nan")
                        test_ce = float("nan")
                    rows.append(
                        {
                            "split_type": split_type,
                            "model_name": model_name,
                            "fraction": fraction,
                            "seed": seed,
                            "test_mse": test_mse,
                            "test_cross_entropy": test_ce,
                        }
                    )

    csv_path = curves_dir / f"learning_curve_symbolic_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "split_type",
                "model_name",
                "fraction",
                "seed",
                "test_mse",
                "test_cross_entropy",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"学习曲线数据已保存: {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图。")
    else:
        for split_type in split_types_list:
            # MSE 曲线
            fig_mse, ax_mse = plt.subplots(1, 1, figsize=(8, 5))
            # Cross-Entropy 曲线（Fig. S1 风格）
            fig_ce, ax_ce = plt.subplots(1, 1, figsize=(8, 5))
            for model_name in MODEL_NAMES:
                sub = [r for r in rows if r["split_type"] == split_type and r["model_name"] == model_name]
                if not sub:
                    continue
                by_frac_mse: Dict[float, List[float]] = {}
                by_frac_ce: Dict[float, List[float]] = {}
                for r in sub:
                    f = r["fraction"]
                    m = r["test_mse"]
                    ce = r.get("test_cross_entropy", float("nan"))
                    if not np.isnan(m):
                        by_frac_mse.setdefault(f, []).append(m)
                    if not np.isnan(ce):
                        by_frac_ce.setdefault(f, []).append(ce)
                # MSE
                if by_frac_mse:
                    fracs_sorted = sorted(by_frac_mse.keys())
                    means = np.array([np.mean(by_frac_mse[f]) for f in fracs_sorted])
                    stds = np.array([np.std(by_frac_mse[f]) for f in fracs_sorted])
                    xs = [f * 100 for f in fracs_sorted]
                    ax_mse.plot(xs, means, "o-", label=model_name, linewidth=6, alpha=0.8)
                    ax_mse.fill_between(xs, means - stds, means + stds, alpha=0.25)
                # CE
                if by_frac_ce:
                    fracs_sorted_ce = sorted(by_frac_ce.keys())
                    means_ce = np.array([np.mean(by_frac_ce[f]) for f in fracs_sorted_ce])
                    stds_ce = np.array([np.std(by_frac_ce[f]) for f in fracs_sorted_ce])
                    xs_ce = [f * 100 for f in fracs_sorted_ce]
                    ax_ce.plot(xs_ce, means_ce, "o-", label=model_name, linewidth=6, alpha=0.8)
                    ax_ce.fill_between(xs_ce, means_ce - stds_ce, means_ce + stds_ce, alpha=0.25)

            # 美化 MSE 图
            ax_mse.set_xlabel("Percent training data used (%)", fontsize=17, fontweight="bold")
            ax_mse.set_ylabel("Test Set MSE", fontsize=17, fontweight="bold")
            title_mse = f"Data quantity vs test MSE — {split_type}"
            if len(seeds) > 1:
                title_mse += "\n(Mean ± std across seeds)"
            ax_mse.set_title(title_mse)
            ax_mse.legend()
            ax_mse.set_xlim(0, 100)
            for spine_name in ["top", "right"]:
                ax_mse.spines[spine_name].set_visible(False)
            for spine_name in ["bottom", "left"]:
                ax_mse.spines[spine_name].set_linewidth(2.5)
            ax_mse.tick_params(axis="both", which="major", labelsize=12)
            fig_mse.tight_layout()
            out_path_mse = curves_dir / f"data_quantity_curve_symbolic_mse_{timestamp}_{split_type}.png"
            fig_mse.savefig(out_path_mse, dpi=150)
            plt.close(fig_mse)
            print(f"数据量曲线(MSE)已保存: {out_path_mse}")

            # 美化 CE 图（Fig. S1 风格）
            ax_ce.set_xlabel("Percent training data used (%)", fontsize=17, fontweight="bold")
            ax_ce.set_ylabel("Test Set Cross Entropy", fontsize=17, fontweight="bold")
            title_ce = f"Data quantity vs test Cross Entropy — {split_type}"
            if len(seeds) > 1:
                title_ce += "\n(Mean ± std across seeds)"
            ax_ce.set_title(title_ce)
            ax_ce.legend()
            ax_ce.set_xlim(0, 100)
            for spine_name in ["top", "right"]:
                ax_ce.spines[spine_name].set_visible(False)
            for spine_name in ["bottom", "left"]:
                ax_ce.spines[spine_name].set_linewidth(2.5)
            ax_ce.tick_params(axis="both", which="major", labelsize=12)
            fig_ce.tight_layout()
            out_path_ce = curves_dir / f"data_quantity_curve_symbolic_ce_{timestamp}_{split_type}.png"
            fig_ce.savefig(out_path_ce, dpi=150)
            plt.close(fig_ce)
            print(f"数据量曲线(CE)已保存: {out_path_ce}")

    json_path = curves_dir / f"learning_curve_symbolic_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "rows": rows, "seeds": seeds}, f, indent=2, ensure_ascii=False)
    print(f"学习曲线 JSON 已保存: {json_path}")

    if args.save_npy:
        for split_type in split_types_list:
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
                fracs_sorted = sorted(by_frac.keys())
                mean_arr = np.array([np.mean(by_frac[f]) for f in fracs_sorted])
                npy_path = curves_dir / f"learning_curve_symbolic_{timestamp}_{split_type}_{model_name}.npy"
                np.save(npy_path, mean_arr)
        print(f".npy 已保存至 {curves_dir}（按 split_type 与 model_name）")


if __name__ == "__main__":
    main()
