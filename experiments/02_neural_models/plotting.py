"""
experiments 绘图：单一 train_test_split 的数据量曲线。
"""

from pathlib import Path
from typing import Dict, List, Any

import numpy as np

from config import DATA_QUANTITY_SMOOTH_WINDOW

FIGURE_DPI = 150
NEURAL_MODEL_TYPES = ["value_based", "context_dependent", "context_dependent_sigmoid"]
DATA_QUANTITY_DISPLAY_NAMES = {
    "value_based": "Value-Based",
    "context_dependent": "Context-Dependent (L)",
    "context_dependent_sigmoid": "Context-Dependent (S)",
}


def plot_data_quantity_curves(
    rows: List[Dict[str, Any]],
    timestamp: str,
    curves_dir: Path,
    smooth_window: int = DATA_QUANTITY_SMOOTH_WINDOW,
    plot_style: str = "reference",
) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return

    seeds = sorted({r["seed"] for r in rows})
    n_seeds = len(seeds)

    def _plot_one_metric(metric_key: str, y_label: str, title_base: str, file_name: str) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for model_type in NEURAL_MODEL_TYPES:
            sub = [r for r in rows if r["model_type"] == model_type]
            if not sub:
                continue
            by_frac = {}
            for r in sub:
                by_frac.setdefault(r["fraction"], []).append(r[metric_key])
            fracs_sorted = sorted(by_frac.keys())
            mean_arr = np.array([np.mean(by_frac[f]) for f in fracs_sorted])
            std_arr = np.array([np.std(by_frac[f]) for f in fracs_sorted])
            xs = [f * 100 for f in fracs_sorted]

            if plot_style == "smoothed" and len(mean_arr) >= smooth_window:
                for s in seeds:
                    points = [(r["fraction"], r[metric_key]) for r in sub if r["seed"] == s]
                    points.sort(key=lambda x: x[0])
                    ax.plot([p[0] * 100 for p in points], [p[1] for p in points], alpha=0.35, linewidth=1)
                mean_smooth = np.convolve(mean_arr, np.ones(smooth_window) / smooth_window, mode="valid")
                std_smooth = np.convolve(std_arr, np.ones(smooth_window) / smooth_window, mode="valid")
                x_smooth = np.linspace(0, 100, len(mean_smooth))
                ax.plot(
                    x_smooth,
                    mean_smooth,
                    linewidth=3,
                    alpha=0.9,
                    label=DATA_QUANTITY_DISPLAY_NAMES.get(model_type, model_type),
                )
                ax.fill_between(x_smooth, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.25)
            else:
                ax.plot(
                    xs,
                    mean_arr,
                    "o-",
                    linewidth=2.5,
                    alpha=0.9,
                    label=DATA_QUANTITY_DISPLAY_NAMES.get(model_type, model_type),
                )
                ax.fill_between(xs, mean_arr - std_arr, mean_arr + std_arr, alpha=0.25)

        ax.set_xlabel("Percent training data used (%)")
        ax.set_ylabel(y_label)
        title = title_base
        if n_seeds > 1:
            title += "\n(Mean ± std across seeds)"
        ax.set_title(title)
        ax.set_xlim(0, 100)
        ax.legend()
        fig.tight_layout()
        out_path = curves_dir / file_name
        fig.savefig(out_path, dpi=FIGURE_DPI)
        plt.close(fig)
        print(f"数据量曲线已保存: {out_path}")

    _plot_one_metric(
        metric_key="test_mse",
        y_label="Test Set MSE",
        title_base="Data quantity vs test MSE (single train/test split)",
        file_name=f"data_quantity_curve_{timestamp}.png",
    )
    _plot_one_metric(
        metric_key="test_cross_entropy",
        y_label="Test Set Cross-Entropy",
        title_base="Data quantity vs test cross-entropy (single train/test split)",
        file_name=f"data_quantity_curve_ce_{timestamp}.png",
    )
