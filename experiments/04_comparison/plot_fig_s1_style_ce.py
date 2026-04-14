"""
合并符号与神经的学习曲线，绘制联合图：--metric ce 为 Fig. S1 风格（Test Cross-Entropy），--metric mse 为训练比例 vs Test MSE。
支持 --run-dir（run_learning_curves_all 产出目录）或 --symbolic-csv / --neural-csv。
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
DIR_01_CURVES = EXPERIMENTS_DIR / "01_symbolic_models_enhanced" / "results" / "enhanced_training" / "curves"
DIR_02_CURVES = EXPERIMENTS_DIR / "02_neural_models" / "results" / "curves"
OUTPUT_DIR_DEFAULT = Path(__file__).resolve().parent / "output"

SYMBOLIC_LABELS = {"ev": "EV", "eu": "EU", "pt3": "PT3", "pt5": "PT5"}
NEURAL_LABELS = {
    "value_based": "Value-Based",
    "context_dependent": "Context-Dependent (L)",
    "context_dependent_sigmoid": "Context-Dependent (S)",
}


def _load_rows(csv_path: Path, need_ce: bool = True) -> List[Dict]:
    """加载学习曲线 CSV；need_ce 时解析 test_cross_entropy，否则解析 test_mse。统一 model / family。"""
    rows: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if "model_name" in r and "model_type" not in r:
                r["model"] = r["model_name"]
                r["family"] = "symbolic"
            elif "model_type" in r:
                r["model"] = r["model_type"]
                r["family"] = "neural"
            # 旧版神经 CSV 无 split_type 列时，与 experiments/02 约定为 train_test
            r["split_type"] = r.get("split_type") or r.get("split", "") or "train_test"
            r["fraction"] = float(r["fraction"])
            r["seed"] = int(r.get("seed", 0))
            r["test_mse"] = float(r.get("test_mse", "nan"))
            r["test_cross_entropy"] = float(r.get("test_cross_entropy", "nan"))
            rows.append(r)
    return rows


def aggregate_ce(rows: List[Dict]) -> Dict:
    """按 (family, model, split_type) 聚合 CE，返回 fractions/means/stds 排序后。"""
    agg: Dict = {}
    for r in rows:
        key = (r["family"], r["model"], r["split_type"], r["fraction"])
        agg.setdefault(key, []).append(r["test_cross_entropy"])
    out: Dict = {}
    for key, vals in agg.items():
        arr = np.array([v for v in vals if not np.isnan(v)], dtype=float)
        if arr.size == 0:
            continue
        family, model, split_type, fraction = key
        out.setdefault((family, model, split_type), {"fractions": [], "means": [], "stds": []})
        out[(family, model, split_type)]["fractions"].append(fraction)
        out[(family, model, split_type)]["means"].append(float(arr.mean()))
        out[(family, model, split_type)]["stds"].append(float(arr.std()))
    for key, d in out.items():
        fracs = np.array(d["fractions"])
        order = np.argsort(fracs)
        d["fractions"] = fracs[order].tolist()
        d["means"] = np.array(d["means"])[order].tolist()
        d["stds"] = np.array(d["stds"])[order].tolist()
    return out


def aggregate_mse(
    rows: List[Dict], split_key: str = "split_type", model_key: str = "model"
) -> Dict[Tuple[str, str], Tuple[List[float], List[float]]]:
    """按 (split_type, model) 聚合 test_mse，返回 (split, model) -> (percent_list, mean_mse_list)。"""
    by_key: Dict[Tuple[str, str], Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        st = r.get(split_key) or r.get("split")
        model = r.get(model_key)
        if st is None or model is None:
            continue
        f = r["fraction"]
        m = r["test_mse"]
        if not np.isnan(m):
            by_key[(st, model)][f].append(m)
    out = {}
    for (st, model), by_frac in by_key.items():
        fracs_sorted = sorted(by_frac.keys())
        percent = [f * 100 for f in fracs_sorted]
        means = [float(np.mean(by_frac[f])) for f in fracs_sorted]
        out[(st, model)] = (percent, means)
    return out


def plot_ce(symbolic_csv: Path, neural_csv: Path, out_dir: Path, mode: str) -> None:
    """Fig. S1 风格：Test Set Cross-Entropy vs 训练比例。"""
    import matplotlib.pyplot as plt
    rows_sym = _load_rows(symbolic_csv)
    rows_neu = _load_rows(neural_csv)
    rows_all = rows_sym + rows_neu
    agg = aggregate_ce(rows_all)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_types = sorted({r["split_type"] for r in rows_all})
    for split_type in split_types:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for (family, model, split), d in agg.items():
            if split != split_type:
                continue
            xs = [f * 100 for f in d["fractions"]]
            ys = np.array(d["means"])
            stds = np.array(d["stds"])
            label = (SYMBOLIC_LABELS if family == "symbolic" else NEURAL_LABELS).get(model, model)
            ax.plot(xs, ys, "o-", label=label, linewidth=4, alpha=0.85)
            ax.fill_between(xs, ys - stds, ys + stds, alpha=0.2)
        ax.set_xlabel("Percent training data used (%)", fontsize=15, fontweight="bold")
        ax.set_ylabel("Test Set Cross Entropy", fontsize=15, fontweight="bold")
        ax.set_title(f"Fig. S1-style CE curves — {split_type} ({mode})")
        ax.legend()
        ax.set_xlim(0, 100)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_linewidth(2.0)
        ax.tick_params(axis="both", which="major", labelsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_s1_style_ce_{mode}_{split_type}.png", dpi=150)
        plt.close(fig)


def plot_mse(symbolic_csv: Path, neural_csv: Path, out_dir: Path) -> None:
    """符号 vs 神经：Test Set MSE vs 训练比例，一图多线。"""
    import matplotlib.pyplot as plt
    rows_sym = _load_rows(symbolic_csv)
    rows_neu = _load_rows(neural_csv)
    agg_sym = aggregate_mse(rows_sym, "split_type", "model")
    agg_neu = aggregate_mse(rows_neu, "split_type", "model")
    out_dir.mkdir(parents=True, exist_ok=True)
    split_types = sorted({r["split_type"] for r in rows_sym + rows_neu})
    _repo = Path(__file__).resolve().parent.parent.parent
    if (_repo / "shared").is_dir():
        sys.path.insert(0, str(_repo))
        try:
            from shared.visualization import setup_chinese_font
            setup_chinese_font()
        except Exception:
            pass
    for split_type in split_types:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for (st, model), (percent, means) in agg_sym.items():
            if st != split_type:
                continue
            label = SYMBOLIC_LABELS.get(model, model)
            ax.plot(percent, means, "o-", color="C0", linewidth=2, label=label)
        for (st, model), (percent, means) in agg_neu.items():
            if st != split_type:
                continue
            label = NEURAL_LABELS.get(model, model)
            ax.plot(percent, means, "s-", color="C1", linewidth=2, label=label)
        ax.set_xlabel("Percent training data used (%)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Test Set MSE", fontsize=12, fontweight="bold")
        ax.set_title(f"Symbolic vs Neural learning curves — {split_type}")
        ax.legend(loc="best", fontsize=8)
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"joint_learning_curve_{split_type}.png", dpi=150)
        plt.close(fig)


def resolve_csvs(
    run_dir: Path = None,
    symbolic_csv: Path = None,
    neural_csv: Path = None,
) -> Tuple[Path, Path]:
    """解析符号/神经 CSV 路径：优先 run_dir/symbolic 与 run_dir/neural，否则用显式路径或默认最新。"""
    if run_dir and run_dir.exists():
        run_dir = run_dir.resolve()
        sym_files = list((run_dir / "symbolic").glob("*.csv")) if (run_dir / "symbolic").exists() else []
        neu_files = list((run_dir / "neural").glob("*.csv")) if (run_dir / "neural").exists() else []
        sym = max(sym_files, key=lambda p: p.stat().st_mtime) if sym_files else None
        neu = max(neu_files, key=lambda p: p.stat().st_mtime) if neu_files else None
        if sym and neu:
            return sym, neu
    if symbolic_csv and neural_csv:
        return Path(symbolic_csv), Path(neural_csv)
    sym = Path(symbolic_csv) if symbolic_csv else None
    neu = Path(neural_csv) if neural_csv else None
    if not sym and DIR_01_CURVES.exists():
        files = list(DIR_01_CURVES.glob("learning_curve_symbolic_*.csv"))
        sym = max(files, key=lambda p: p.stat().st_mtime) if files else None
    if not neu and DIR_02_CURVES.exists():
        files = [f for f in DIR_02_CURVES.glob("learning_curve_*.csv") if f.suffix == ".csv"]
        neu = max(files, key=lambda p: p.stat().st_mtime) if files else None
    if not sym or not neu:
        raise FileNotFoundError("请提供 --symbolic-csv 与 --neural-csv，或 --run-dir 下 symbolic/*.csv 与 neural/*.csv，或先运行 01/02 学习曲线。")
    return sym, neu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并符号与神经学习曲线：--metric ce 为 Fig.S1 风格 CE 图，--metric mse 为 MSE 联合图")
    parser.add_argument("--metric", type=str, choices=["mse", "ce"], default="ce", help="mse=训练比例 vs Test MSE；ce=Fig.S1 风格 Cross-Entropy")
    parser.add_argument("--symbolic-csv", type=Path, default=None, help="符号学习曲线 CSV")
    parser.add_argument("--neural-csv", type=Path, default=None, help="神经学习曲线 CSV")
    parser.add_argument("--run-dir", type=Path, default=None, help="run_learning_curves_all 产出目录，内有 symbolic/ 与 neural/ 子目录")
    parser.add_argument("--mode", type=str, choices=["full", "simple"], default="full", help="仅 --metric ce 时有效")
    parser.add_argument("--out-dir", type=Path, default=None, help=f"输出目录（默认 {OUTPUT_DIR_DEFAULT}）")
    args = parser.parse_args()
    out_dir = args.out_dir or OUTPUT_DIR_DEFAULT
    try:
        symbolic_csv, neural_csv = resolve_csvs(args.run_dir, args.symbolic_csv, args.neural_csv)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    if args.metric == "ce":
        plot_ce(symbolic_csv, neural_csv, out_dir, args.mode)
        print("Fig. S1 风格 CE 图已保存。")
    else:
        plot_mse(symbolic_csv, neural_csv, out_dir)
        print("MSE 联合学习曲线图已保存。")
