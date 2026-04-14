from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_curve(curve_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    sub = curve_df[curve_df["metric"] == metric].copy()
    if sub.empty:
        raise ValueError(f"No rows for metric={metric}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for model_id, g in sub.groupby("model_id"):
        g2 = g.sort_values("N")
        ax.plot(g2["N"], g2["mean"], marker="o", linewidth=2, label=str(model_id))
        ax.fill_between(g2["N"], g2["ci_low"], g2["ci_high"], alpha=0.2)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Training sample size (N)")
    ax.set_ylabel(metric)
    ax.set_title(f"Learning Curve - {metric}")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
