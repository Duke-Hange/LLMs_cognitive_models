from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import sys

_EXPERIMENTS_ROOT = Path(__file__).resolve().parents[3]
if str(_EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_ROOT))

from evaluation_metrics import evaluate_predictions  # type: ignore


@dataclass
class EvalPolicy:
    ce_eps: float = 1e-15
    # None：主指标仅在 parse_ok 子集上计算，失败样本不插补
    llm_parse_fail_fill: Optional[float] = None


def _clip_prob(x: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), eps, 1.0 - eps)


def apply_parse_fail_fill(pred: np.ndarray, parse_ok: np.ndarray, fill_value: float) -> np.ndarray:
    out = np.asarray(pred, dtype=np.float64).copy()
    mask = ~np.asarray(parse_ok, dtype=bool)
    out[mask] = float(fill_value)
    return out


def evaluate_with_policy(y_true: np.ndarray, y_pred: np.ndarray, parse_ok: np.ndarray, policy: EvalPolicy) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = _clip_prob(np.asarray(y_pred, dtype=np.float64), eps=policy.ce_eps)
    parse_ok = np.asarray(parse_ok, dtype=bool)

    nan_pred = np.isnan(y_pred)
    if policy.llm_parse_fail_fill is None:
        # 主指标：仅解析成功且预测非 NaN 的样本（与 03_LLM「禁止静默 0.5」一致）
        valid = parse_ok & ~nan_pred
        if np.any(valid):
            main = evaluate_predictions(y_true[valid], y_pred[valid])
        else:
            main = {k: float("nan") for k in ("mse", "r2", "correlation", "cross_entropy", "rmse", "mae")}
        sens = main
    else:
        y_pred_main = apply_parse_fail_fill(y_pred, parse_ok=parse_ok, fill_value=float(policy.llm_parse_fail_fill))
        main = evaluate_predictions(y_true, y_pred_main)
        if np.any(parse_ok & ~nan_pred):
            sel = parse_ok & ~nan_pred
            sens = evaluate_predictions(y_true[sel], y_pred[sel])
        elif np.any(parse_ok):
            sens = evaluate_predictions(y_true[parse_ok], y_pred[parse_ok])
        else:
            sens = {k: float("nan") for k in ("mse", "r2", "correlation", "cross_entropy", "rmse", "mae")}

    return {
        "mse": float(main["mse"]),
        "r2": float(main["r2"]),
        "correlation": float(main["correlation"]),
        "cross_entropy": float(main["cross_entropy"]),
        "rmse": float(main["rmse"]),
        "mae": float(main["mae"]),
        "parse_success_rate": float(np.mean(parse_ok)) if parse_ok.size else float("nan"),
        "sens_mse_parse_ok": float(sens["mse"]),
        "sens_cross_entropy_parse_ok": float(sens["cross_entropy"]),
        "sens_mae_parse_ok": float(sens["mae"]),
    }
