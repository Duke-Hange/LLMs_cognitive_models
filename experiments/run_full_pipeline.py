"""
一键流程（experiments）：同 seed 下依次 01 标准化（若需）→ 01 训练 → 02 训练（原始编码，无 StandardScaler）→ 04 比较，并写入 manifest。

EXPERIMENTS_DIR 为本文件所在目录（即 `experiments/`）；02 使用原始编码，不对分布编码做 StandardScaler。
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

EXPERIMENTS_DIR = Path(__file__).resolve().parent
DIR_01 = EXPERIMENTS_DIR / "01_symbolic_models_enhanced"
DIR_02 = EXPERIMENTS_DIR / "02_neural_models"
DIR_04 = EXPERIMENTS_DIR / "04_comparison"
CANONICAL_JSON = DIR_01 / "c13k_enhanced_standardized.json"
RESULTS_01 = DIR_01 / "results" / "enhanced_training"
RESULTS_02 = DIR_02 / "results"
OUTPUT_04 = DIR_04 / "output"


def run_cmd(cmd: List[str], cwd: Path, description: str) -> bool:
    print(f"\n{'='*60}\n{description}\n{'='*60}")
    ret = subprocess.run(cmd, cwd=str(cwd))
    if ret.returncode != 0:
        print(f"失败: {description} 退出码 {ret.returncode}", file=sys.stderr)
        return False
    return True


def latest_file(directory: Path, pattern: str):
    """返回目录下匹配 pattern 的最新文件（按 mtime）；无则 None。"""
    files = list(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="experiments：01→02（原始编码）→04 一键流程，并写 manifest"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="随机种子列表（01 与 02 共用）")
    parser.add_argument("--skip-01", action="store_true", help="跳过 01 符号训练")
    parser.add_argument("--skip-02", action="store_true", help="跳过 02 神经训练")
    parser.add_argument("--skip-04", action="store_true", help="跳过 04 比较")
    parser.add_argument(
        "--symbolic-data",
        type=str,
        choices=["json", "csv"],
        default="json",
        help="01 符号模型数据源：json=c13k；csv=data/data.csv（与 02 同源、train_test 划分）",
    )
    parser.add_argument("--curves", type=str, choices=["all", "mean_only", "full", "none"], default="mean_only", help="01/02 训练曲线模式（默认 mean_only）；02 的 full=对比图+单模型细图")
    args = parser.parse_args()

    seeds: List[int] = args.seeds
    manifest = {
        "run_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "experiments2": True,
        "standardize_features": False,
        "standardized_data_path": str(CANONICAL_JSON.resolve()),
        "seeds": seeds,
        "symbolic_summary_path": None,
        "neural_summary_path": None,
        "comparison_report_path": None,
        "comparison_table_path": None,
        "comparison_figure_path": None,
        "comparison_mse_r2_path": None,
    }

    # 0. 仅 json 模式需要权威标准化 JSON；csv 模式与 02 共用 data.csv
    if args.symbolic_data == "json":
        if not CANONICAL_JSON.exists():
            print("\n标准化数据不存在，正在生成并保存至 01 目录...")
            if not run_cmd(
                [sys.executable, "enhanced_data_standardization.py"],
                DIR_01,
                "01 生成 c13k_enhanced_standardized.json",
            ):
                return 1
            if not CANONICAL_JSON.exists():
                print("生成后仍未找到 JSON，退出。", file=sys.stderr)
                return 1
        else:
            print(f"\n使用已有标准化数据: {CANONICAL_JSON}")
    else:
        print("\n01 使用 CSV 模式（与 02 同源 data.csv），跳过 c13k JSON 检查。")

    # 1. 01 训练
    if not args.skip_01:
        cmd_01 = [
            sys.executable, "train_enhanced_models.py",
            "--seeds", *map(str, seeds),
            "--curves", args.curves,
            "--data-source", args.symbolic_data,
        ]
        if not run_cmd(cmd_01, DIR_01, "01 符号模型训练"):
            return 1
        latest = latest_file(RESULTS_01, "enhanced_models_summary_*.csv")
        if latest:
            manifest["symbolic_summary_path"] = str(latest.resolve())
            print(f"  01 summary: {latest.name}")
        else:
            print("  未找到 01 summary CSV，请检查 results/enhanced_training/")

    # 2. 02 训练（原始编码，无 StandardScaler）
    if not args.skip_02:
        cmd_02 = [sys.executable, "train.py", "--seeds", *map(str, seeds), "--curves", args.curves]
        if not run_cmd(cmd_02, DIR_02, "02 神经网络训练（原始编码）"):
            return 1
        latest = latest_file(RESULTS_02, "neural_models_summary_*.csv")
        if latest:
            manifest["neural_summary_path"] = str(latest.resolve())
            print(f"  02 summary: {latest.name}")
        else:
            print("  未找到 02 summary CSV，请检查 results/")

    # 3. 04 比较（使用本 run 产出的 summary）
    if not args.skip_04 and (manifest["symbolic_summary_path"] and manifest["neural_summary_path"]):
        OUTPUT_04.mkdir(parents=True, exist_ok=True)
        cmd_04 = [
            sys.executable, "run_comparison.py",
            "--symbolic", manifest["symbolic_summary_path"],
            "--neural", manifest["neural_summary_path"],
            "--out-dir", str(OUTPUT_04),
        ]
        if not run_cmd(cmd_04, DIR_04, "04 跨模型比较"):
            return 1
        report = latest_file(OUTPUT_04, "comparison_report_*.md")
        table = latest_file(OUTPUT_04, "comparison_table_*.csv")
        fig = latest_file(OUTPUT_04, "comparison_r2_*.png")
        fig_mse = latest_file(OUTPUT_04, "comparison_mse_r2_*.png")
        if report:
            manifest["comparison_report_path"] = str(report.resolve())
        if table:
            manifest["comparison_table_path"] = str(table.resolve())
        if fig:
            manifest["comparison_figure_path"] = str(fig.resolve())
        if fig_mse:
            manifest["comparison_mse_r2_path"] = str(fig_mse.resolve())
    elif args.skip_04 or not manifest["symbolic_summary_path"] or not manifest["neural_summary_path"]:
        if args.skip_04:
            print("\n已跳过 04，不写比较相关路径。")
        else:
            print("\n缺少 01 或 02 summary，跳过 04。")

    # 4. 写 manifest
    manifest_path = OUTPUT_04 / "manifest.json"
    OUTPUT_04.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nmanifest 已写入: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
