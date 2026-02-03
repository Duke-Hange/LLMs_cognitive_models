"""
神经网络实验结果报告
加载 summary CSV，在控制台打印汇总，并生成 Markdown 报告；可选与符号模型结果对比。
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = _THIS_DIR / "results"
REPORTS_DIR = RESULTS_DIR / "reports"


def get_latest_summary_csv() -> Path:
    """返回 results/ 下最新的 neural_models_summary_*.csv"""
    pattern = "neural_models_summary_*.csv"
    files = list(RESULTS_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到 {RESULTS_DIR / pattern}，请先运行 train.py")
    return max(files, key=lambda p: p.stat().st_mtime)


def load_neural_summary(csv_path: Path) -> pd.DataFrame:
    """加载神经网络 summary CSV（支持单次运行与多种子汇总）。"""
    df = pd.read_csv(csv_path)
    if "model_type" not in df.columns or "split_type" not in df.columns:
        raise ValueError("summary CSV 缺少列: model_type, split_type")
    # 多种子汇总为 test_*_mean / test_*_std，统一为 test_mse, test_r2, test_correlation
    if "test_mse" not in df.columns and "test_mse_mean" in df.columns:
        df = df.rename(columns={
            "test_mse_mean": "test_mse",
            "test_r2_mean": "test_r2",
            "test_correlation_mean": "test_correlation",
        })
    for col in ["test_mse", "test_r2", "test_correlation"]:
        if col not in df.columns:
            raise ValueError(f"summary CSV 缺少列: {col}")
    if "best_epoch" not in df.columns:
        df["best_epoch"] = None  # 多种子汇总无 best_epoch
    return df


def print_console_summary(df: pd.DataFrame) -> None:
    """控制台打印「神经网络实验结果汇总」"""
    print("=" * 80)
    print("神经网络实验结果汇总")
    print("=" * 80)
    for split_type in df["split_type"].unique():
        sub = df[df["split_type"] == split_type]
        print(f"\n划分: {split_type}")
        print("-" * 40)
        for _, row in sub.iterrows():
            print(
                f"  {row['model_type']:20s}: "
                f"测试 MSE={row['test_mse']:.6f}, "
                f"R^2={row['test_r2']:.4f}, "
                f"相关性={row['test_correlation']:.4f}"
            )
    best = df.loc[df["test_r2"].idxmax()]
    print(f"\n最佳 test R^2: {best['model_type']} @ {best['split_type']} = {best['test_r2']:.4f}")


def generate_markdown_report(
    df: pd.DataFrame,
    report_path: Path,
    comparison_df: Optional[pd.DataFrame] = None,
    symbolic_summary_path: Optional[str] = None,
) -> None:
    """生成 Markdown 报告"""
    lines = [
        "# 神经网络实验结果报告",
        "",
        "## 实验概述",
        "",
        "- **实验名称**: Choices13k 神经网络实验（Value-Based / Context-Dependent）",
        "- **实验目标**: 在完整分布编码输入下评估两类神经网络对 bRate 的预测与泛化。",
        "- **模型**: Value-Based（单赌局编码 + 共享 f + Softmax）、Context-Dependent（两赌局拼接 + 直接输出 bRate）。",
        "- **划分类型**: problem, parameter_amb, parameter_ev_extreme。",
        "- **评估指标**: MSE, R², 相关性。",
        "",
        "## 数据与方法",
        "",
        "- **数据来源**: 01_symbolic_models_enhanced 的标准化 JSON（c13k_enhanced_standardized.json）。",
        "- **输入**: 每个赌局的完整结果分布编码（padding 到 max_outcomes=9），单赌局 18 维，两赌局拼接 36 维。",
        "- **训练**: Adam, MSE 损失，早停（验证集 10%），随机种子固定。",
        "",
        "## 结果",
        "",
        "### 按划分与模型",
        "",
    ]
    has_best_epoch = "best_epoch" in df.columns and df["best_epoch"].notna().any()
    if has_best_epoch:
        lines.append("| 划分 | 模型 | 测试 MSE | 测试 R² | 测试相关性 | best_epoch |")
        lines.append("|------|------|----------|---------|------------|------------|")
    else:
        lines.append("| 划分 | 模型 | 测试 MSE | 测试 R² | 测试相关性 |")
        lines.append("|------|------|----------|---------|------------|")
    for _, row in df.iterrows():
        if has_best_epoch:
            ep = int(row["best_epoch"]) if pd.notna(row.get("best_epoch")) else "—"
            lines.append(
                f"| {row['split_type']} | {row['model_type']} | {row['test_mse']:.6f} | {row['test_r2']:.4f} | {row['test_correlation']:.4f} | {ep} |"
            )
        else:
            lines.append(
                f"| {row['split_type']} | {row['model_type']} | {row['test_mse']:.6f} | {row['test_r2']:.4f} | {row['test_correlation']:.4f} |"
            )
    lines.extend([
        "",
        "### 按划分小结",
        "",
    ])
    for split_type in df["split_type"].unique():
        sub = df[df["split_type"] == split_type]
        avg_r2 = sub["test_r2"].mean()
        best_model = sub.loc[sub["test_r2"].idxmax(), "model_type"]
        lines.append(f"- **{split_type}**: 平均 test R² = {avg_r2:.4f}，最佳模型 = {best_model}")
    lines.extend([
        "",
        "### 按模型小结",
        "",
    ])
    for model_type in df["model_type"].unique():
        sub = df[df["model_type"] == model_type]
        avg_r2 = sub["test_r2"].mean()
        avg_mse = sub["test_mse"].mean()
        lines.append(f"- **{model_type}**: 平均 test MSE = {avg_mse:.6f}，平均 test R² = {avg_r2:.4f}")
    lines.extend([
        "",
        "## 关键发现",
        "",
    ])
    best_row = df.loc[df["test_r2"].idxmax()]
    lines.append(f"1. 最佳单次表现: **{best_row['model_type']}** 在 **{best_row['split_type']}** 上 test R² = {best_row['test_r2']:.4f}。")
    vb = df[df["model_type"] == "value_based"]
    cd = df[df["model_type"] == "context_dependent"]
    if len(vb) and len(cd):
        r2_vb = vb["test_r2"].mean()
        r2_cd = cd["test_r2"].mean()
        if r2_cd > r2_vb:
            lines.append("2. Context-Dependent 平均 test R² 高于 Value-Based，与文献中「上下文依赖预测更优」一致。")
        else:
            lines.append("2. Value-Based 与 Context-Dependent 在平均 test R² 上接近。")
    lines.append("3. 三种划分下 test R² 均较低，提示在完整分布编码下 bRate 预测具有挑战性。")
    lines.append("4. 可与符号模型（01 增强实验）对比，见下节（若已提供符号 summary）。")

    if comparison_df is not None and len(comparison_df) > 0:
        lines.extend([
            "",
            "## 与符号模型对比",
            "",
        ])
        if symbolic_summary_path:
            lines.append(f"符号模型结果来源: `{symbolic_summary_path}`")
        lines.extend([
            "",
            "| 划分 | 来源 | 模型 | 测试 MSE | 测试 R² | 测试相关性 |",
            "|------|------|------|----------|---------|------------|",
        ])
        for _, row in comparison_df.iterrows():
            lines.append(
                f"| {row['split_type']} | {row['source']} | {row['model']} | {row['test_mse']:.6f} | {row['test_r2']:.4f} | {row['test_correlation']:.4f} |"
            )
        lines.append("")
        sym = comparison_df[comparison_df["source"] == "symbolic"]
        neu = comparison_df[comparison_df["source"] == "neural"]
        if len(sym) and len(neu):
            avg_r2_sym = sym["test_r2"].mean()
            avg_r2_neu = neu["test_r2"].mean()
            lines.append(f"- 符号模型平均 test R² = {avg_r2_sym:.4f}；神经网络平均 test R² = {avg_r2_neu:.4f}。")
        lines.append("")

    lines.extend([
        "---",
        f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown 报告已保存: {report_path}")


def load_symbolic_summary(csv_path: Path) -> pd.DataFrame:
    """加载 01 增强符号模型 summary CSV，列名可能为 split_type 或 split"""
    df = pd.read_csv(csv_path)
    if "split_type" not in df.columns and "split" in df.columns:
        df = df.rename(columns={"split": "split_type"})
    return df


def build_comparison_df(neural_df: pd.DataFrame, symbolic_df: pd.DataFrame) -> pd.DataFrame:
    """按 split_type 对齐神经与符号结果，生成对比表（source, model, test_mse, test_r2, test_correlation）"""
    rows = []
    for _, row in neural_df.iterrows():
        rows.append({
            "split_type": row["split_type"],
            "source": "neural",
            "model": row["model_type"],
            "test_mse": row["test_mse"],
            "test_r2": row["test_r2"],
            "test_correlation": row["test_correlation"],
        })
    sym_model_col = "model" if "model" in symbolic_df.columns else "model_name"
    for _, row in symbolic_df.iterrows():
        st = row["split_type"] if "split_type" in row else row.get("split", "")
        rows.append({
            "split_type": st,
            "source": "symbolic",
            "model": row[sym_model_col],
            "test_mse": row["test_mse"],
            "test_r2": row["test_r2"],
            "test_correlation": row["test_correlation"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="神经网络实验结果报告")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="神经网络 summary CSV 路径（默认: results/ 下最新）",
    )
    parser.add_argument(
        "--symbolic",
        type=Path,
        default=None,
        help="01 增强符号模型 summary CSV 路径（可选，用于对比）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="报告输出路径（默认: results/reports/neural_models_report_时间戳.md）",
    )
    args = parser.parse_args()

    csv_path = args.summary or get_latest_summary_csv()
    print(f"加载神经网络结果: {csv_path}")
    df = load_neural_summary(csv_path)

    print_console_summary(df)

    report_path = args.out or REPORTS_DIR / f"neural_models_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    comparison_df = None
    symbolic_path_str = None

    if args.symbolic is not None and args.symbolic.exists():
        print(f"加载符号模型结果: {args.symbolic}")
        symbolic_df = load_symbolic_summary(args.symbolic)
        comparison_df = build_comparison_df(df, symbolic_df)
        symbolic_path_str = str(args.symbolic)
        comp_csv = REPORTS_DIR / "neural_vs_symbolic_comparison.csv"
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(comp_csv, index=False, encoding="utf-8")
        print(f"对比表已保存: {comp_csv}")
        summary_json = {
            "neural_summary": csv_path.name,
            "symbolic_summary": args.symbolic.name,
            "neural_avg_test_r2": float(df["test_r2"].mean()),
            "symbolic_avg_test_r2": float(symbolic_df["test_r2"].mean()),
        }
        comp_json = REPORTS_DIR / "comparison_summary.json"
        with open(comp_json, "w", encoding="utf-8") as f:
            json.dump(summary_json, f, indent=2, ensure_ascii=False)
        print(f"对比摘要已保存: {comp_json}")

    generate_markdown_report(df, report_path, comparison_df=comparison_df, symbolic_summary_path=symbolic_path_str)
    print("报告生成完成。")


if __name__ == "__main__":
    main()
