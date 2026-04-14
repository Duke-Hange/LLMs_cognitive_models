from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from ..common.metrics import EvalPolicy, evaluate_with_policy


def evaluate_predictions_long(pred_df: pd.DataFrame, policy: EvalPolicy) -> pd.DataFrame:
    rows: List[Dict] = []
    keys = ["family", "model_id", "N", "seed"]
    for (family, model_id, n, seed), g in pred_df.groupby(keys, dropna=False):
        y_true = g["y_true"].to_numpy(dtype=np.float64)
        y_pred = g["y_pred"].to_numpy(dtype=np.float64)
        parse_ok = g["parse_ok"].fillna(False).to_numpy(dtype=bool)
        if policy.llm_parse_fail_fill is not None:
            fill = float(policy.llm_parse_fail_fill)
            y_pred = np.where(np.isnan(y_pred), fill, y_pred)
        m = evaluate_with_policy(y_true, y_pred, parse_ok=parse_ok, policy=policy)
        rows.append(
            {
                "family": family,
                "model_id": model_id,
                "N": int(n),
                "seed": int(seed),
                **m,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_ci(vals: np.ndarray, n_boot: int, ci_level: float, seed: int = 1234) -> tuple[float, float]:
    if vals.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(int(n_boot)):
        sample = rng.choice(vals, size=vals.size, replace=True)
        boot.append(float(np.mean(sample)))
    alpha = 1.0 - ci_level
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return lo, hi


def aggregate_curve(metric_df: pd.DataFrame, metric: str, ci_level: float, bootstrap_samples: int) -> pd.DataFrame:
    rows: List[Dict] = []
    for (family, model_id, n), g in metric_df.groupby(["family", "model_id", "N"], dropna=False):
        vals = g[metric].to_numpy(dtype=np.float64)
        mean = float(np.mean(vals)) if vals.size else float("nan")
        std = float(np.std(vals)) if vals.size else float("nan")
        lo, hi = _bootstrap_ci(vals, n_boot=bootstrap_samples, ci_level=ci_level)
        rows.append(
            {
                "family": family,
                "model_id": model_id,
                "N": int(n),
                "metric": metric,
                "mean": mean,
                "std": std,
                "ci_low": lo,
                "ci_high": hi,
                "n_seeds": int(vals.size),
            }
        )
    out = pd.DataFrame(rows).sort_values(["family", "model_id", "N"]).reset_index(drop=True)
    return out


def build_summary_table(metric_df: pd.DataFrame, n_target_metric: str, n_target_value: float) -> pd.DataFrame:
    rows: List[Dict] = []
    for (family, model_id), g in metric_df.groupby(["family", "model_id"], dropna=False):
        g2 = g.sort_values("N")
        x = g2["N"].to_numpy(dtype=np.float64)
        y_mae = g2.groupby("N")[n_target_metric].mean().reset_index()
        # AULC over cross_entropy and mae
        ce_curve = g2.groupby("N")["cross_entropy"].mean().reset_index()
        mae_curve = g2.groupby("N")["mae"].mean().reset_index()
        aulc_ce = float(np.trapz(ce_curve["cross_entropy"].to_numpy(), ce_curve["N"].to_numpy())) if len(ce_curve) > 1 else float("nan")
        aulc_mae = float(np.trapz(mae_curve["mae"].to_numpy(), mae_curve["N"].to_numpy())) if len(mae_curve) > 1 else float("nan")
        n_target = float("nan")
        for _, r in y_mae.iterrows():
            if float(r[n_target_metric]) <= float(n_target_value):
                n_target = float(r["N"])
                break
        rows.append(
            {
                "family": family,
                "model_id": model_id,
                "AULC_cross_entropy": aulc_ce,
                "AULC_mae": aulc_mae,
                f"N_at_{n_target_metric}_{n_target_value}": n_target,
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "model_id"]).reset_index(drop=True)
