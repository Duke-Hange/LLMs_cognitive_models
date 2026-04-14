"""
04_comparison 跨模型比较
加载 01 符号、02 神经、03 LLM summary CSV，按 split_type 对齐，输出比较表、按划分小结与带可比性说明的 Markdown 报告。
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Any

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent

# 添加项目根目录到路径
sys.path.insert(0, str(_THIS_DIR.parent.parent))

# 从共享模块导入中文字体设置 
from shared import setup_chinese_font
setup_chinese_font()


def _to_json_serializable(obj: Any) -> Any:
    """将 numpy/pandas 类型转为 Python 原生类型，便于 json.dump。"""
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if isinstance(obj, (dict,)):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list,)):
        return [_to_json_serializable(x) for x in obj]
    return obj

try:
    from config import (
        get_latest_symbolic_path,
        get_latest_neural_path,
        get_latest_llm_path,
        OUTPUT_DIR,
    )
except ImportError:
    _EXPERIMENTS_DIR = _THIS_DIR.parent
    OUTPUT_DIR = _THIS_DIR / "output"

    def get_latest_symbolic_path() -> Path:
        d = _EXPERIMENTS_DIR / "01_symbolic_models_enhanced" / "results" / "enhanced_training"
        if not d.exists():
            raise FileNotFoundError(f"未找到目录: {d}")
        files = list(d.glob("enhanced_models_summary_*.csv"))
        if not files:
            raise FileNotFoundError(f"未找到 enhanced_models_summary_*.csv 于 {d}")
        return max(files, key=lambda p: p.stat().st_mtime)

    def get_latest_neural_path() -> Path:
        d = _EXPERIMENTS_DIR / "02_neural_models" / "results"
        if not d.exists():
            raise FileNotFoundError(f"未找到目录: {d}")
        files = list(d.glob("neural_models_summary_*.csv"))
        if not files:
            raise FileNotFoundError(f"未找到 neural_models_summary_*.csv 于 {d}")
        return max(files, key=lambda p: p.stat().st_mtime)

    def get_latest_llm_path() -> Path:
        d = _EXPERIMENTS_DIR / "03_LLM" / "results"
        if not d.exists():
            raise FileNotFoundError(f"未找到目录: {d}")
        files = list(d.glob("llm_models_summary_*.csv"))
        if not files:
            raise FileNotFoundError(f"未找到 llm_models_summary_*.csv 于 {d}")
        return max(files, key=lambda p: p.stat().st_mtime)


def _is_multi_seed_summary(df: pd.DataFrame, mean_col: str = "test_mse_mean") -> bool:
    """判断 summary 是否为多种子汇总（含 *_mean / *_std）。"""
    return mean_col in df.columns


def load_symbolic(csv_path: Path) -> Tuple[pd.DataFrame, bool]:
    """加载符号 summary CSV。返回 (df, multi_seed)；多种子时含 test_*_mean/std, n_seeds。"""
    df = pd.read_csv(csv_path)
    if "split_type" not in df.columns and "split" in df.columns:
        df = df.rename(columns={"split": "split_type"})
    multi_seed = _is_multi_seed_summary(df)
    if multi_seed:
        for col in ["split_type", "model", "test_mse_mean", "test_mse_std", "test_r2_mean", "test_r2_std", "test_correlation_mean", "test_correlation_std"]:
            if col not in df.columns:
                raise ValueError(f"符号多种子 summary 缺少列: {col}")
    else:
        for col in ["split_type", "model", "test_mse", "test_r2", "test_correlation"]:
            if col not in df.columns:
                raise ValueError(f"符号 summary 缺少列: {col}")
    return df, multi_seed


def load_neural(csv_path: Path) -> Tuple[pd.DataFrame, bool]:
    """加载神经 summary CSV，统一为 model 列。返回 (df, multi_seed)。"""
    df = pd.read_csv(csv_path)
    multi_seed = _is_multi_seed_summary(df)
    if multi_seed:
        for col in ["model_type", "split_type", "test_mse_mean", "test_mse_std", "test_r2_mean", "test_r2_std", "test_correlation_mean", "test_correlation_std"]:
            if col not in df.columns:
                raise ValueError(f"神经多种子 summary 缺少列: {col}")
    else:
        for col in ["model_type", "split_type", "test_mse", "test_r2", "test_correlation"]:
            if col not in df.columns:
                raise ValueError(f"神经 summary 缺少列: {col}")
    df = df.rename(columns={"model_type": "model"})
    return df, multi_seed


def load_llm(csv_path: Path) -> Tuple[pd.DataFrame, bool]:
    """加载 LLM summary CSV，统一列名并返回 (df, multi_seed)。"""
    df = pd.read_csv(csv_path)
    if "split_type" not in df.columns and "split" in df.columns:
        df = df.rename(columns={"split": "split_type"})
    if "model_type" in df.columns and "model" not in df.columns:
        df = df.rename(columns={"model_type": "model"})
    multi_seed = _is_multi_seed_summary(df)
    if multi_seed:
        for col in ["model", "split_type", "test_mse_mean", "test_mse_std", "test_r2_mean", "test_r2_std", "test_correlation_mean", "test_correlation_std"]:
            if col not in df.columns:
                raise ValueError(f"LLM 多种子 summary 缺少列: {col}")
    else:
        for col in ["model", "split_type", "test_mse", "test_r2", "test_correlation"]:
            if col not in df.columns:
                raise ValueError(f"LLM summary 缺少列: {col}")
    return df, multi_seed


def build_long_table(
    symbolic_df: pd.DataFrame,
    neural_df: pd.DataFrame,
    llm_df: Optional[pd.DataFrame] = None,
    symbolic_multi: bool = False,
    neural_multi: bool = False,
    llm_multi: bool = False,
) -> pd.DataFrame:
    """统一为长表：split_type, family, model, test_mse, test_r2, test_correlation；多种子时含 *_std, n_seeds。"""
    rows = []
    for _, row in symbolic_df.iterrows():
        if symbolic_multi:
            r = {
                "split_type": row["split_type"],
                "family": "symbolic",
                "model": row["model"],
                "test_mse": float(row["test_mse_mean"]),
                "test_r2": float(row["test_r2_mean"]),
                "test_correlation": float(row["test_correlation_mean"]),
                "test_mse_std": float(row["test_mse_std"]),
                "test_r2_std": float(row["test_r2_std"]),
                "test_correlation_std": float(row["test_correlation_std"]),
                "n_seeds": int(row.get("n_seeds", 1)),
            }
        else:
            r = {
                "split_type": row["split_type"],
                "family": "symbolic",
                "model": row["model"],
                "test_mse": float(row["test_mse"]),
                "test_r2": float(row["test_r2"]),
                "test_correlation": float(row["test_correlation"]),
                "test_mse_std": 0.0,
                "test_r2_std": 0.0,
                "test_correlation_std": 0.0,
                "n_seeds": 1,
            }
        rows.append(r)
    for _, row in neural_df.iterrows():
        if neural_multi:
            r = {
                "split_type": row["split_type"],
                "family": "neural",
                "model": row["model"],
                "test_mse": float(row["test_mse_mean"]),
                "test_r2": float(row["test_r2_mean"]),
                "test_correlation": float(row["test_correlation_mean"]),
                "test_mse_std": float(row["test_mse_std"]),
                "test_r2_std": float(row["test_r2_std"]),
                "test_correlation_std": float(row["test_correlation_std"]),
                "n_seeds": int(row.get("n_seeds", 1)),
            }
        else:
            r = {
                "split_type": row["split_type"],
                "family": "neural",
                "model": row["model"],
                "test_mse": float(row["test_mse"]),
                "test_r2": float(row["test_r2"]),
                "test_correlation": float(row["test_correlation"]),
                "test_mse_std": 0.0,
                "test_r2_std": 0.0,
                "test_correlation_std": 0.0,
                "n_seeds": 1,
            }
        rows.append(r)
    if llm_df is not None:
        for _, row in llm_df.iterrows():
            if llm_multi:
                r = {
                    "split_type": row["split_type"],
                    "family": "llm",
                    "model": row["model"],
                    "test_mse": float(row["test_mse_mean"]),
                    "test_r2": float(row["test_r2_mean"]),
                    "test_correlation": float(row["test_correlation_mean"]),
                    "test_mse_std": float(row["test_mse_std"]),
                    "test_r2_std": float(row["test_r2_std"]),
                    "test_correlation_std": float(row["test_correlation_std"]),
                    "n_seeds": int(row.get("n_seeds", 1)),
                }
            else:
                r = {
                    "split_type": row["split_type"],
                    "family": "llm",
                    "model": row["model"],
                    "test_mse": float(row["test_mse"]),
                    "test_r2": float(row["test_r2"]),
                    "test_correlation": float(row["test_correlation"]),
                    "test_mse_std": 0.0,
                    "test_r2_std": 0.0,
                    "test_correlation_std": 0.0,
                    "n_seeds": 1,
                }
            rows.append(r)
    return pd.DataFrame(rows)


def build_summary_by_split(long_df: pd.DataFrame) -> dict:
    """按 split_type 汇总：符号/神经最佳 R²、平均 R²（及最佳模型名）；多种子时含 *_std。"""
    has_std = "test_r2_std" in long_df.columns
    by_split = {}
    for st in long_df["split_type"].unique():
        sub = long_df[long_df["split_type"] == st]
        sym = sub[sub["family"] == "symbolic"]
        neu = sub[sub["family"] == "neural"]
        llm = sub[sub["family"] == "llm"]
        entry = {}
        if len(sym):
            best_idx = sym["test_r2"].idxmax()
            entry["symbolic_best_r2"] = float(sym.loc[best_idx, "test_r2"])
            entry["symbolic_best_model"] = sym.loc[best_idx, "model"]
            entry["symbolic_mean_r2"] = float(sym["test_r2"].mean())
            entry["symbolic_mean_mse"] = float(sym["test_mse"].mean())
            if has_std:
                entry["symbolic_best_r2_std"] = float(sym.loc[best_idx, "test_r2_std"])
        if len(neu):
            best_idx = neu["test_r2"].idxmax()
            entry["neural_best_r2"] = float(neu.loc[best_idx, "test_r2"])
            entry["neural_best_model"] = neu.loc[best_idx, "model"]
            entry["neural_mean_r2"] = float(neu["test_r2"].mean())
            entry["neural_mean_mse"] = float(neu["test_mse"].mean())
            if has_std:
                entry["neural_best_r2_std"] = float(neu.loc[best_idx, "test_r2_std"])
        if len(llm):
            best_idx = llm["test_r2"].idxmax()
            entry["llm_best_r2"] = float(llm.loc[best_idx, "test_r2"])
            entry["llm_best_model"] = llm.loc[best_idx, "model"]
            entry["llm_mean_r2"] = float(llm["test_r2"].mean())
            entry["llm_mean_mse"] = float(llm["test_mse"].mean())
            if has_std:
                entry["llm_best_r2_std"] = float(llm.loc[best_idx, "test_r2_std"])
        by_split[st] = entry
    return by_split


def plot_comparison_figure(
    long_df: pd.DataFrame,
    out_path: Path,
    has_std: bool,
) -> None:
    """根据 long_df 绘制跨模型比较图：一张图两个子图（Test R²、Test MSE），按划分分组柱状图，颜色区分族。"""
    import matplotlib.pyplot as plt
    split_types = long_df["split_type"].unique().tolist()
    if not split_types:
        return
    n_splits = len(split_types)
    fig, axes = plt.subplots(1, 2, figsize=(4 * n_splits, 5))
    family_colors = {"symbolic": "C0", "neural": "C1", "llm": "C2"}
    for ax, ycol, yerr_col, ylabel in [
        (axes[0], "test_r2", "test_r2_std", "Test R²"),
        (axes[1], "test_mse", "test_mse_std", "Test MSE"),
    ]:
        x_offset = 0
        width = 0.35
        for st in split_types:
            sub = long_df[long_df["split_type"] == st]
            n_models = len(sub)
            x_pos = list(range(x_offset, x_offset + n_models))
            heights = sub[ycol].tolist()
            colors = [family_colors.get(f, "gray") for f in sub["family"]]
            yerr = sub[yerr_col].tolist() if has_std and yerr_col in sub.columns else None
            ax.bar(x_pos, heights, width=width, color=colors, yerr=yerr, capsize=2)
            ax.set_xticks([x_offset + (n_models - 1) / 2])
            ax.set_xticklabels([st], fontsize=9)
            x_offset += n_models + 1
        ax.set_ylabel(ylabel)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")
    axes[0].set_title("Test R²")
    axes[1].set_title("Test MSE")
    axes[1].set_ylim(0.04, 0.07)
    axes[0].legend(
        [
            plt.Rectangle((0, 0), 1, 1, fc=family_colors["symbolic"]),
            plt.Rectangle((0, 0), 1, 1, fc=family_colors["neural"]),
            plt.Rectangle((0, 0), 1, 1, fc=family_colors["llm"]),
        ],
        ["symbolic", "neural", "llm"],
        loc="upper right",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_mse_r2(
    long_df: pd.DataFrame,
    out_path: Path,
    has_std: bool,
) -> None:
    """绘制 MSE 与 R² 的模型比较图：两子图（MSE、R²），按划分分组，每模型一柱并带图例。"""
    import matplotlib.pyplot as plt
    split_types = long_df["split_type"].unique().tolist()
    if not split_types:
        return
    preferred = ["ev", "eu", "pt3", "pt5", "value_based", "context_dependent", "context_dependent_sigmoid"]
    model_display = {
        "ev": "EV", "eu": "EU", "pt3": "PT3", "pt5": "PT5",
        "value_based": "Value-Based",
        "context_dependent": "Context-Dependent (L)",
        "context_dependent_sigmoid": "Context-Dependent (S)",
    }
    seen = long_df["model"].unique().tolist()
    models_order = [m for m in preferred if m in seen] + [m for m in seen if m not in preferred]
    n_models = len(models_order)
    width = 0.8 / n_models if n_models else 0.2
    gap = 0.8
    model_colors = {
        "ev": "#1f77b4",
        "eu": "#3787c0",
        "pt3": "#4f97cc",
        "pt5": "#69a7d8",
        "value_based": "#ff7f0e",
        "context_dependent": "#ff9f3e",
        "context_dependent_sigmoid": "#2e6db4",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, ycol, yerr_col, title in [
        (axes[0], "test_mse", "test_mse_std", "Test MSE"),
        (axes[1], "test_r2", "test_r2_std", "Test R²"),
    ]:
        x_ticks = []
        x_tick_labels = []
        used = 0
        for st in split_types:
            sub = long_df[long_df["split_type"] == st]
            for i, model in enumerate(models_order):
                row = sub[sub["model"] == model]
                if row.empty:
                    continue
                row = row.iloc[0]
                pos = used + i * width
                h = float(row[ycol])
                c = model_colors.get(model, "gray")
                yerr = float(row[yerr_col]) if has_std and yerr_col in row and (row.get(yerr_col) or 0) else None
                ax.bar(pos, h, width=width * 0.9, color=c, yerr=yerr, capsize=1.5)
            group_center = used + (n_models - 1) * width / 2
            x_ticks.append(group_center)
            x_tick_labels.append(st)
            used += n_models * width + gap
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels, fontsize=10)
        ax.set_ylabel(title)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_title(title)
        if title == "Test MSE":
            ax.set_ylim(0.04, 0.07)
    legend_labels = [model_display.get(m, m) for m in models_order]
    handles = [plt.Rectangle((0, 0), 1, 1, fc=model_colors.get(m, "gray")) for m in models_order]
    fig.legend(handles, legend_labels, loc="upper center", ncol=min(7, len(models_order)), bbox_to_anchor=(0.5, 0.02), fontsize=8)
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(
    long_df: pd.DataFrame,
    summary_by_split: dict,
    report_path: Path,
    symbolic_path: Optional[Path] = None,
    neural_path: Optional[Path] = None,
    llm_path: Optional[Path] = None,
    has_std: bool = False,
    n_seeds_max: int = 1,
    comparison_figure_basename: Optional[str] = None,
    comparison_mse_r2_basename: Optional[str] = None,
) -> None:
    """生成带比较设计、可比性说明、结果表与按划分小结的 Markdown 报告；多种子时显示 mean±std。"""
    lines = [
        "# 跨模型比较报告",
        "",
        "## 比较目的",
        "",
        "在相同数据、相同划分与相同评估指标下，比较**符号模型（增强）**、**神经网络（Value-Based、Context-Dependent）**与**LLM 模型**在预测 Choices13k **群体选择倾向**上的表现（语义上均为 Option B 侧，但各轨**标量定义**见下表）。",
        "",
        "## 比较设计",
        "",
        "| 类型 | 内容 |",
        "|------|------|",
        "| **固定** | 数据源（01 标准化数据或与本仓库 `data/data.csv` 对齐的 CSV 轨）、划分（`problem` / `parameter_amb` / `parameter_ev_extreme` 或 `train_test`）、指标（test MSE, R², correlation）、评估方式（仅测试集） |",
        "| **变化** | 模型族（符号 vs 神经 vs LLM）、神经两类（Value-Based / Context-Dependent）、输入表示（符号：50+ 维增强特征；神经：完整分布编码；LLM：自然语言提示） |",
        "",
        "## 标签空间与数值可比性",
        "",
        "| 模型族 | Summary 中指标所对应的监督目标 | 说明 |",
        "|--------|----------------------------------|------|",
        "| **01 符号** | **bRate**（选 B 的比例） | 拟合对象即 bRate。 |",
        "| **02 神经**（本仓库 CSV 轨默认） | **y = 1 − bRate** | 网络输出与该标量在同一空间比较；与 bRate 互为仿射，**MSE 等价**，R² 在该设定下与 bRate 空间一致；若将神经输出解释为 bRate，应使用 **1 − ŷ**。 |",
        "| **03 LLM** | 通常为 **bRate**（Prompt 要求 `P_B`） | 与 02 并列时需确认 summary 是否已变换到与神经同一空间；`05_program` 在 `target_mode: one_minus_bRate` 下会对 **1 − pred_bRate** 计分。 |",
        "",
        "跨族比较表格中的 MSE/R²/correlation **仅在各列已按同一 `split`、同一目标空间解释时**可直接对比；撰写结论时建议显式写出神经轨为 **1−bRate** 或同时报告 bRate 空间的等价变换。",
        "",
        "## 可比性说明",
        "",
        "同一划分、同一指标公式；**输入表示与标签命名可能不同**（符号用摘要特征拟合 bRate；神经用分布编码拟合 **1−bRate**；LLM 用语言预测 bRate），因此差异为**模型族、表示与目标参数化的联合效应**。神经侧包含两类模型：Value-Based、Context-Dependent。",
        "",
        "## 复现与可比性前提",
        "",
        "本报告表内比较为「同一测试集」下的结果，当且仅当满足以下约定时成立：",
        "",
        "1. **数据源**：符号与神经使用**同一**标准化数据文件，即 `experiments/01_symbolic_models_enhanced/c13k_enhanced_standardized.json`（02 从该路径读取，01 学习曲线与训练亦应以该文件为权威）。",
        "2. **划分**：**problem_split** 依赖随机种子；01 与 02 须使用**相同 random_state/seed**，否则训练/测试集不一致。**parameter_amb** 与 **parameter_ev_extreme** 为按特征确定性划分，同数据即同测试集。",
        "3. **本报告对应 run**：下表与图基于本次指定的 01 符号 summary 与 02 神经 summary；若未显式指定 `--symbolic` / `--neural`，则使用各目录「最新」文件，**严格可比**建议先跑 01→02（同 seed），再跑 04 并传入本次产出的 summary 路径。",
        "",
        "## 结果表",
        "",
    ]
    if has_std:
        lines.append("| 划分 | 族 | 模型 | 测试 MSE | 测试 R² | 测试相关性 |")
        lines.append("|------|-----|------|----------|---------|------------|")
        for _, row in long_df.iterrows():
            mse_str = f"{row['test_mse']:.6f} ± {row['test_mse_std']:.6f}" if row.get("test_mse_std", 0) != 0 else f"{row['test_mse']:.6f}"
            r2_str = f"{row['test_r2']:.4f} ± {row['test_r2_std']:.4f}" if row.get("test_r2_std", 0) != 0 else f"{row['test_r2']:.4f}"
            corr_str = f"{row['test_correlation']:.4f} ± {row['test_correlation_std']:.4f}" if row.get("test_correlation_std", 0) != 0 else f"{row['test_correlation']:.4f}"
            lines.append(f"| {row['split_type']} | {row['family']} | {row['model']} | {mse_str} | {r2_str} | {corr_str} |")
    else:
        lines.append("| 划分 | 族 | 模型 | 测试 MSE | 测试 R² | 测试相关性 |")
        lines.append("|------|-----|------|----------|---------|------------|")
        for _, row in long_df.iterrows():
            lines.append(
                f"| {row['split_type']} | {row['family']} | {row['model']} | {row['test_mse']:.6f} | {row['test_r2']:.4f} | {row['test_correlation']:.4f} |"
            )
    if comparison_figure_basename:
        lines.extend(["", "## 跨模型比较图", "", f"![跨模型比较]({comparison_figure_basename})", ""])
    if comparison_mse_r2_basename:
        lines.extend(["", "## MSE 与 R² 模型比较图", "", f"![MSE与R²比较]({comparison_mse_r2_basename})", ""])
    lines.extend(["", "## 按划分小结", ""])
    for st, entry in summary_by_split.items():
        parts = []
        if "symbolic_best_r2" in entry:
            if has_std and "symbolic_best_r2_std" in entry:
                parts.append(f"符号最佳 R²={entry['symbolic_best_r2']:.4f} ± {entry['symbolic_best_r2_std']:.4f}（{entry['symbolic_best_model']}），平均 R²={entry['symbolic_mean_r2']:.4f}")
            else:
                parts.append(f"符号最佳 R²={entry['symbolic_best_r2']:.4f}（{entry['symbolic_best_model']}），平均 R²={entry['symbolic_mean_r2']:.4f}")
        if "neural_best_r2" in entry:
            if has_std and "neural_best_r2_std" in entry:
                parts.append(f"神经最佳 R²={entry['neural_best_r2']:.4f} ± {entry['neural_best_r2_std']:.4f}（{entry['neural_best_model']}），平均 R²={entry['neural_mean_r2']:.4f}")
            else:
                parts.append(f"神经最佳 R²={entry['neural_best_r2']:.4f}（{entry['neural_best_model']}），平均 R²={entry['neural_mean_r2']:.4f}")
        if "llm_best_r2" in entry:
            if has_std and "llm_best_r2_std" in entry:
                parts.append(f"LLM 最佳 R²={entry['llm_best_r2']:.4f} ± {entry['llm_best_r2_std']:.4f}（{entry['llm_best_model']}），平均 R²={entry['llm_mean_r2']:.4f}")
            else:
                parts.append(f"LLM 最佳 R²={entry['llm_best_r2']:.4f}（{entry['llm_best_model']}），平均 R²={entry['llm_mean_r2']:.4f}")
        lines.append(f"- **{st}**: " + "；".join(parts))
    lines.extend(["", "## 局限与复现信息", ""])
    if has_std and n_seeds_max > 1:
        lines.append(f"- 比较基于多种子运行结果（种子数: 最多 {n_seeds_max}），表中为均值 ± 标准差。")
        # 若存在标准差为 0 的行，说明同一划分/模型在多种子下结果完全一致（如符号模型在 parameter_amb/parameter_ev_extreme 上常因确定性优化而无波动）
        if (long_df.get("test_r2_std", pd.Series(dtype=float)).fillna(0).abs() == 0).any():
            lines.append("- 表中未显示「±」的单元格表示该划分/模型在多种子下指标方差为 0（多次运行结果一致），并非数据缺失。")
    else:
        lines.append("- 比较基于单次运行结果；未评估随机性（多种子）。")
    lines.append("- 三类模型使用不同输入表示，结论限于「当前表示设定 + 符号族/神经族/LLM 族」下的相对表现。")
    lines.append("")
    if symbolic_path:
        lines.append(f"- 符号 summary: `{symbolic_path}`")
    if neural_path:
        lines.append(f"- 神经 summary: `{neural_path}`")
    if llm_path:
        lines.append(f"- LLM summary: `{llm_path}`")
    lines.extend([
        "",
        f"- 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已保存: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="04_comparison 跨模型比较")
    parser.add_argument("--symbolic", type=Path, default=None, help="01 符号 summary CSV 路径（默认: 最新）")
    parser.add_argument("--neural", type=Path, default=None, help="02 神经 summary CSV 路径（默认: 最新）")
    parser.add_argument("--llm", type=Path, default=None, help="03 LLM summary CSV 路径（默认: 自动检测最新）")
    parser.add_argument("--out-dir", type=Path, default=None, help="输出目录（默认: 04_comparison/output）")
    args = parser.parse_args()

    symbolic_path = args.symbolic or get_latest_symbolic_path()
    neural_path = args.neural or get_latest_neural_path()
    llm_path: Optional[Path] = args.llm
    if llm_path is None:
        try:
            llm_path = get_latest_llm_path()
        except FileNotFoundError:
            llm_path = None
    out_dir = args.out_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载符号: {symbolic_path}")
    symbolic_df, symbolic_multi = load_symbolic(symbolic_path)
    print(f"加载神经: {neural_path}")
    neural_df, neural_multi = load_neural(neural_path)
    llm_df: Optional[pd.DataFrame] = None
    llm_multi = False
    if llm_path is not None:
        print(f"加载LLM: {llm_path}")
        llm_df, llm_multi = load_llm(llm_path)
    else:
        print("未检测到 LLM summary，将仅比较 symbolic 与 neural。")

    long_df = build_long_table(
        symbolic_df, neural_df, llm_df=llm_df,
        symbolic_multi=symbolic_multi,
        neural_multi=neural_multi,
        llm_multi=llm_multi,
    )
    summary_by_split = build_summary_by_split(long_df)

    has_std = "test_r2_std" in long_df.columns and (long_df["test_r2_std"].fillna(0).abs().sum() > 0 or long_df["n_seeds"].max() > 1)
    n_seeds_max = int(long_df["n_seeds"].max()) if "n_seeds" in long_df.columns else 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_path = out_dir / f"comparison_table_{timestamp}.csv"
    summary_path = out_dir / f"comparison_summary_{timestamp}.json"
    report_path = out_dir / f"comparison_report_{timestamp}.md"

    long_df.to_csv(table_path, index=False, encoding="utf-8")
    print(f"比较表已保存: {table_path}")

    sym_r2_col = "test_r2_mean" if symbolic_multi else "test_r2"
    neu_r2_col = "test_r2_mean" if neural_multi else "test_r2"
    summary_json = {
        "timestamp": timestamp,
        "symbolic_source": str(symbolic_path),
        "neural_source": str(neural_path),
        "llm_source": str(llm_path) if llm_path is not None else None,
        "by_split": summary_by_split,
        "symbolic_mean_r2_overall": float(symbolic_df[sym_r2_col].mean()),
        "neural_mean_r2_overall": float(neural_df[neu_r2_col].mean()),
        "has_std": bool(has_std),
        "n_seeds_max": int(n_seeds_max),
    }
    if llm_df is not None:
        llm_r2_col = "test_r2_mean" if llm_multi else "test_r2"
        summary_json["llm_mean_r2_overall"] = float(llm_df[llm_r2_col].mean())
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_serializable(summary_json), f, indent=2, ensure_ascii=False)
    print(f"汇总 JSON 已保存: {summary_path}")

    comparison_figure_basename = f"comparison_r2_{timestamp}.png"
    comparison_figure_path = out_dir / comparison_figure_basename
    plot_comparison_figure(long_df, comparison_figure_path, has_std=has_std)
    print(f"比较图已保存: {comparison_figure_path}")

    comparison_mse_r2_basename = f"comparison_mse_r2_{timestamp}.png"
    comparison_mse_r2_path = out_dir / comparison_mse_r2_basename
    plot_comparison_mse_r2(long_df, comparison_mse_r2_path, has_std=has_std)
    print(f"MSE/R² 比较图已保存: {comparison_mse_r2_path}")

    write_report(
        long_df, summary_by_split, report_path,
        symbolic_path=symbolic_path,
        neural_path=neural_path,
        llm_path=llm_path,
        has_std=has_std,
        n_seeds_max=n_seeds_max,
        comparison_figure_basename=comparison_figure_basename,
        comparison_mse_r2_basename=comparison_mse_r2_basename,
    )
    print("跨模型比较完成。")


if __name__ == "__main__":
    main()
