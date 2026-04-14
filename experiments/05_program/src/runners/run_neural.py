from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
_NEU_DIR = _ROOT / "02_neural_models"
if str(_NEU_DIR) not in sys.path:
    sys.path.insert(0, str(_NEU_DIR))

from train import _split_train_val, train_context_dependent, train_value_based  # type: ignore


def run_neural_family(
    model_defs: List[Dict],
    enc_a: np.ndarray,
    enc_b: np.ndarray,
    enc_full: np.ndarray,
    y_target: np.ndarray,
    test_idx: np.ndarray,
    sample_map: Dict[tuple, np.ndarray],
    epochs: int = 1000,
    patience: int = 100,
    verbose: bool = True,
) -> List[Dict]:
    out: List[Dict] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_jobs = max(1, len(model_defs) * len(sample_map))
    done_jobs = 0
    t0 = time.time()

    x_a_te = enc_a[test_idx]
    x_b_te = enc_b[test_idx]
    x_full_te = enc_full[test_idx]
    y_true_test = y_target[test_idx]

    for model_def in model_defs:
        model_id = str(model_def["model_id"])
        source_name = str(model_def["source_name"])

        for (n, seed), train_idx in sample_map.items():
            if verbose:
                print(
                    f"[05_program][neural] start model={model_id} N={n} seed={seed} "
                    f"epochs={epochs}",
                    flush=True,
                )
            sl = _split_train_val(
                enc_a[train_idx],
                enc_b[train_idx],
                enc_full[train_idx],
                y_target[train_idx],
                seed=int(seed),
            )
            if source_name == "value_based":
                model, scaler_a, scaler_b = train_value_based(
                    sl["enc_A_tr"],
                    sl["enc_B_tr"],
                    sl["y_tr"],
                    sl["enc_A_val"],
                    sl["enc_B_val"],
                    sl["y_val"],
                    device=device,
                    epochs=epochs,
                    patience=patience,
                )
                x_a = scaler_a.transform(x_a_te) if scaler_a is not None else x_a_te
                x_b = scaler_b.transform(x_b_te) if scaler_b is not None else x_b_te
                with torch.no_grad():
                    p = model(
                        torch.FloatTensor(x_a).to(device),
                        torch.FloatTensor(x_b).to(device),
                    ).cpu().numpy()
            elif source_name == "context_dependent_sigmoid":
                model, scaler = train_context_dependent(
                    sl["enc_full_tr"],
                    sl["y_tr"],
                    sl["enc_full_val"],
                    sl["y_val"],
                    device=device,
                    epochs=epochs,
                    patience=patience,
                    use_sigmoid=True,
                )
                x = scaler.transform(x_full_te) if scaler is not None else x_full_te
                with torch.no_grad():
                    p = model(torch.FloatTensor(x).to(device)).cpu().numpy()
            else:
                raise ValueError(f"Unknown neural source_name: {source_name}")

            p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
            for j, row_idx in enumerate(test_idx):
                out.append(
                    {
                        "family": "neural",
                        "model_id": model_id,
                        "source_name": source_name,
                        "N": int(n),
                        "seed": int(seed),
                        "row_index": int(row_idx),
                        "y_true": float(y_true_test[j]),
                        "y_pred": float(p[j]),
                        "parse_ok": True,
                    }
                )
            done_jobs += 1
            if verbose:
                frac = done_jobs / total_jobs
                elapsed = time.time() - t0
                eta = (elapsed / frac - elapsed) if frac > 1e-12 else float("nan")
                print(
                    f"[05_program][neural] progress {done_jobs}/{total_jobs} "
                    f"({frac*100:.1f}%) elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
    return out
