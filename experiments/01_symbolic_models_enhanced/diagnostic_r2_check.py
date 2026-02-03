"""
R² 快速诊断：检查 (1) 训练 vs 测试 R² (2) 预测 vs 真实散点图
"""
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 项目根路径并设置中文字体（避免中文/负号显示为方框）
_p = Path(__file__).resolve().parent
for _ in range(8):
    _p = _p.parent
    if (_p / "shared").is_dir():
        sys.path.insert(0, str(_p))
        break
from shared.visualization import setup_chinese_font
setup_chinese_font()

def main():
    base = Path(__file__).resolve().parent
    project_root = base.parent.parent
    results_dir = base / "results" / "enhanced_training"

    # 1) 找最新结果文件
    result_files = sorted(results_dir.glob("enhanced_models_results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not result_files:
        print("未找到 enhanced_models_results_*.json")
        return
    latest = result_files[0]
    print("使用结果文件:", latest.name)

    with open(latest, "r", encoding="utf-8") as f:
        results = json.load(f)

    # 2) 打印 训练 R² vs 测试 R² 表
    print("\n" + "=" * 70)
    print("1. 训练集 R² vs 测试集 R²（确认 R² 在测试集上计算且与训练对比）")
    print("=" * 70)
    for split_type in ["problem", "parameter_amb", "parameter_ev_extreme"]:
        if split_type not in results:
            continue
        print(f"\n划分: {split_type}")
        for model_name in ["ev", "eu", "pt3", "pt5"]:
            if model_name not in results[split_type]:
                continue
            r = results[split_type][model_name]
            tr2 = r["train_metrics"]["r2"]
            te2 = r["test_metrics"]["r2"]
            print(f"  {model_name:4s}: train_r2 = {tr2:+.4f},  test_r2 = {te2:+.4f}  (测试样本数={r['test_metrics']['n_samples']})")

    # 3) 预测 vs 真实 散点图（取 problem_split + pt5）
    split_type = "problem"
    model_name = "pt5"
    if split_type not in results or model_name not in results[split_type]:
        print("\n无法绘制: 缺少 problem / pt5 结果")
        return

    # y 与标准化数据同序 = CSV 行序（merge 时左表顺序）
    csv_path = project_root / "数据集" / "choices13k-main" / "c13k_selections.csv"
    if not csv_path.exists():
        print("\n未找到 c13k_selections.csv，跳过散点图")
        return
    df = pd.read_csv(csv_path)
    y_full = df["bRate"].values

    r = results[split_type][model_name]
    test_idx = r["test_indices"]
    y_test = y_full[test_idx]
    y_pred = np.array(r["test_predictions"])

    # 复算 R² 做一次交叉验证
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_recomputed = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    corr = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else 0
    print("\n" + "=" * 70)
    print(f"2. 复算检查: {split_type} / {model_name} 测试集 R² = {r2_recomputed:.6f}, 相关系数 = {corr:.6f}")
    print("=" * 70)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.3, s=8)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="y=x")
    ax.set_xlabel("真实 bRate (y_test)")
    ax.set_ylabel("预测 bRate (y_pred)")
    ax.set_title(f"测试集: {split_type} / {model_name}\nR²={r2_recomputed:.4f}, corr={corr:.4f}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    out_path = base / "results" / "diagnostic_r2_pred_vs_actual.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n散点图已保存: {out_path}")

if __name__ == "__main__":
    main()
