from __future__ import annotations

from typing import Dict, List

import pandas as pd


def run_repro_checks(pred_df: pd.DataFrame, metric_df: pd.DataFrame, expected_n_test: int) -> Dict:
    issues: List[str] = []
    checks: Dict[str, bool] = {}

    # 1) Coverage check: each model/N/seed should have full test rows.
    grp = pred_df.groupby(["family", "model_id", "N", "seed"])["row_index"].nunique().reset_index(name="n_rows")
    bad = grp[grp["n_rows"] != int(expected_n_test)]
    checks["full_test_coverage"] = bad.empty
    if not bad.empty:
        issues.append(f"Groups with missing/extra rows: {len(bad)}")

    # 2) Same row_index set across models for same N/seed.
    key_groups = pred_df.groupby(["N", "seed"])
    same_index = True
    for (_, _), g in key_groups:
        sets = g.groupby(["family", "model_id"])["row_index"].apply(lambda s: tuple(sorted(set(s.tolist()))))
        if len(set(sets.tolist())) > 1:
            same_index = False
            break
    checks["same_test_indices_across_models"] = same_index
    if not same_index:
        issues.append("Row index sets differ across models for some N/seed.")

    # 3) Metric table has one row per family/model/N/seed group.
    n_pred_groups = pred_df.groupby(["family", "model_id", "N", "seed"]).ngroups
    checks["metric_group_count_match"] = int(n_pred_groups) == int(len(metric_df))
    if not checks["metric_group_count_match"]:
        issues.append(f"metric rows={len(metric_df)} != prediction groups={n_pred_groups}")

    return {
        "checks": checks,
        "issues": issues,
        "ok": all(bool(v) for v in checks.values()),
    }
