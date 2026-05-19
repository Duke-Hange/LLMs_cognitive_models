"""
统一入口（experiments）：依次运行 01 符号模型与 02 神经网络的数据量–性能曲线，并汇总结果与联合可视化。

- 默认 `--symbolic-data csv`：01 与 02 均基于 `data/data.csv`，划分均为 `train_test`，便于 `plot_fig_s1_style_ce` 在同一张图对比符号与神经。
- `--symbolic-data json`：01 使用 c13k JSON 多划分（对照轨），与主协议默认口径不同。

运行后将本次生成的 CSV/JSON/PNG 复制到 `results/learning_curves_all/{run_timestamp}/`，并可选生成 04 联合 CE/MSE 图。

注意：本脚本是 01+02 对照轨学习曲线入口，不会生成 05 主协议的 `split_manifest.json` 与 `configs_snapshot/`。
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

EXPERIMENTS_DIR = Path(__file__).resolve().parent
DIR_01 = EXPERIMENTS_DIR / "01_symbolic_models_enhanced"
DIR_02 = EXPERIMENTS_DIR / "02_neural_models"
DIR_04 = EXPERIMENTS_DIR / "04_comparison"
CURVES_01 = DIR_01 / "results" / "enhanced_training" / "curves"
CURVES_02 = DIR_02 / "results" / "curves"
OUTPUT_ROOT = EXPERIMENTS_DIR / "results" / "learning_curves_all"


def run_cmd(cmd: list, cwd: Path, description: str) -> bool:
    print(f"\n{'='*60}\n{description}\n{'='*60}")
    ret = subprocess.run(cmd, cwd=str(cwd))
    if ret.returncode != 0:
        print(f"失败: {description} 退出码 {ret.returncode}", file=sys.stderr)
        return False
    return True


def latest_timestamp_from_dir(directory: Path, prefix: str) -> Optional[str]:
    """从目录中匹配 prefix 的文件里取最新文件的时间戳（文件名中的 YYYYMMDD_HHMMSS）。"""
    files = [f for f in directory.glob(f"{prefix}*") if f.is_file()]
    if not files:
        return None
    newest = max(files, key=lambda p: p.stat().st_mtime)
    m = re.search(r"(\d{8}_\d{6})", newest.name)
    return m.group(1) if m else None


def copy_files_with_timestamp(curves_dir: Path, out_subdir: Path, ts: str) -> int:
    """复制目录中文件名包含 ts 的所有文件到 out_subdir。"""
    out_subdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in curves_dir.glob(f"*{ts}*"):
        if f.is_file():
            shutil.copy2(f, out_subdir / f.name)
            print(f"  已复制: {f.name}")
            n += 1
    return n


def write_run_audit(
    out_run: Path,
    symbolic_enabled: bool,
    neural_enabled: bool,
    fig_s1_enabled: bool,
) -> None:
    """
    写出本次 run 的完整性自检报告，帮助判断结果是否满足最小可读标准。
    """
    symbolic_dir = out_run / "symbolic"
    neural_dir = out_run / "neural"
    output_04 = EXPERIMENTS_DIR / "04_comparison" / "output"

    symbolic_csv = len(list(symbolic_dir.glob("*.csv"))) if symbolic_dir.exists() else 0
    symbolic_png = len(list(symbolic_dir.glob("*.png"))) if symbolic_dir.exists() else 0
    neural_csv = len(list(neural_dir.glob("*.csv"))) if neural_dir.exists() else 0
    neural_png = len(list(neural_dir.glob("*.png"))) if neural_dir.exists() else 0

    latest_fig_s1_ce = latest_timestamp_from_dir(output_04, "fig_s1_style_ce_")
    fig_s1_ce_present = latest_fig_s1_ce is not None

    checks = {
        "symbolic_csv_present": (not symbolic_enabled) or (symbolic_csv > 0),
        "neural_csv_present": (not neural_enabled) or (neural_csv > 0),
        "symbolic_png_present": (not symbolic_enabled) or (symbolic_png > 0),
        "neural_png_present": (not neural_enabled) or (neural_png > 0),
        "fig_s1_ce_present_if_enabled": (not fig_s1_enabled) or fig_s1_ce_present,
    }
    all_passed = all(checks.values())
    audit = {
        "script": "run_learning_curves_all.py",
        "note": (
            "This is a 01/02 comparative track. "
            "For frozen main protocol artifacts, run experiments/05_program/src/runners/run_all.py."
        ),
        "counts": {
            "symbolic_csv": symbolic_csv,
            "symbolic_png": symbolic_png,
            "neural_csv": neural_csv,
            "neural_png": neural_png,
        },
        "checks": checks,
        "all_passed": all_passed,
    }
    path = out_run / "run_audit.json"
    path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    if all_passed:
        print(f"[自检] 通过，详情见: {path}")
    else:
        print(f"[自检] 未全部通过，请检查: {path}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="experiments：运行 01 符号 + 02 神经学习曲线，汇总到 results/learning_curves_all/"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="随机种子列表")
    parser.add_argument("--n-fractions", type=int, default=50, help="比例点个数（默认 50）")
    parser.add_argument("--save-npy", action="store_true", help="同时保存 .npy 数组")
    parser.add_argument("--skip-symbolic", action="store_true", help="跳过 01 符号模型")
    parser.add_argument("--skip-neural", action="store_true", help="跳过 02 神经网络")
    parser.add_argument("--plot-style", type=str, choices=["reference", "smoothed"], default="reference", help="02 绘图风格（默认 reference）")
    parser.add_argument("--no-plot-fig-s1", action="store_true", help="不自动绘制 Fig. S1 风格 CE 联合图")
    parser.add_argument(
        "--symbolic-data",
        type=str,
        choices=["json", "csv"],
        default="csv",
        help="01 数据源：csv=data.csv+train_test（与 02 对齐联合图，默认）；json=c13k+三种划分",
    )
    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_run = OUTPUT_ROOT / run_timestamp
    out_run.mkdir(parents=True, exist_ok=True)
    print(f"本次结果将汇总到: {out_run}")
    print(
        "[提示] 当前脚本属于 01/02 学习曲线对照轨；"
        "若用于主结论复现，请运行 05_program/src/runners/run_all.py 生成 split_manifest 与配置快照。",
        flush=True,
    )

    seeds_str = " ".join(str(s) for s in args.seeds)
    n_frac = args.n_fractions

    if not args.skip_symbolic:
        cmd_01 = [
            sys.executable,
            "run_learning_curve.py",
            "--seeds",
            *map(str, args.seeds),
            "--n-fractions",
            str(n_frac),
            "--data-source",
            args.symbolic_data,
        ]
        if args.save_npy:
            cmd_01.append("--save-npy")
        if not run_cmd(cmd_01, DIR_01, "01 符号模型学习曲线"):
            return 1
        ts_01 = latest_timestamp_from_dir(CURVES_01, "learning_curve_symbolic_")
        if ts_01:
            copy_files_with_timestamp(CURVES_01, out_run / "symbolic", ts_01)
        else:
            print("未找到 01 本次生成的学习曲线文件，跳过复制。")

    if not args.skip_neural:
        cmd_02 = [
            sys.executable, "run_learning_curve.py",
            "--seeds", *map(str, args.seeds),
            "--n-fractions", str(n_frac),
            "--plot-style", args.plot_style,
        ]
        if args.save_npy:
            cmd_02.append("--save-npy")
        if not run_cmd(cmd_02, DIR_02, "02 神经网络学习曲线（原始编码）"):
            return 1
        ts_02 = latest_timestamp_from_dir(CURVES_02, "learning_curve_")
        if ts_02:
            copy_files_with_timestamp(CURVES_02, out_run / "neural", ts_02)
        else:
            print("未找到 02 本次生成的学习曲线文件，跳过复制。")

    fig_s1_enabled = not args.no_plot_fig_s1
    if fig_s1_enabled:
        sym_csvs = list((out_run / "symbolic").glob("*.csv")) if (out_run / "symbolic").exists() else []
        neu_csvs = list((out_run / "neural").glob("*.csv")) if (out_run / "neural").exists() else []
        if sym_csvs and neu_csvs:
            run_cmd(
                [
                    sys.executable, "plot_fig_s1_style_ce.py",
                    "--metric", "ce",
                    "--run-dir", str(out_run.resolve()),
                ],
                DIR_04,
                "04 Fig. S1 风格 CE 联合图（训练比例 vs Test Cross-Entropy）",
            )
        else:
            print("（未同时存在 symbolic 与 neural 的 CSV，跳过 Fig. S1 风格图）")

    write_run_audit(
        out_run=out_run,
        symbolic_enabled=not args.skip_symbolic,
        neural_enabled=not args.skip_neural,
        fig_s1_enabled=fig_s1_enabled,
    )
    print(f"\n汇总完成: {out_run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
