from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
_SYM_DIR = _ROOT / "01_symbolic_models_enhanced"
if str(_SYM_DIR) not in sys.path:
    sys.path.insert(0, str(_SYM_DIR))

from enhanced_symbolic_models import EnhancedModelAdapter, create_enhanced_model  # type: ignore


def run_symbolic_family(
    model_defs: List[Dict],
    standardized_records: List[Dict],
    y_target: np.ndarray,
    y_brate: np.ndarray,
    test_idx: np.ndarray,
    sample_map: Dict[tuple, np.ndarray],
    verbose: bool = True,
) -> List[Dict]:
    out: List[Dict] = []
    test_records = [standardized_records[int(i)] for i in test_idx]
    y_true_test = y_target[test_idx]
    total_jobs = max(1, len(model_defs) * len(sample_map))
    done_jobs = 0
    t0 = time.time()

    for model_def in model_defs:
        model_id = str(model_def["model_id"])
        source_name = str(model_def["source_name"])
        for (n, seed), train_idx in sample_map.items():
            if verbose:
                print(
                    f"[05_program][symbolic] start model={model_id} N={n} seed={seed}",
                    flush=True,
                )
            train_records = [standardized_records[int(i)] for i in train_idx]
            y_train_b = y_brate[train_idx]

            base_model = create_enhanced_model(source_name)
            adapter = EnhancedModelAdapter(base_model)
            adapter.fit_from_standardized(train_records, y_train_b)
            pred_b = np.asarray(adapter.predict_from_standardized(test_records), dtype=np.float64)
            pred_y = 1.0 - pred_b

            for j, row_idx in enumerate(test_idx):
                out.append(
                    {
                        "family": "symbolic",
                        "model_id": model_id,
                        "source_name": source_name,
                        "N": int(n),
                        "seed": int(seed),
                        "row_index": int(row_idx),
                        "y_true": float(y_true_test[j]),
                        "y_pred": float(np.clip(pred_y[j], 1e-15, 1 - 1e-15)),
                        "parse_ok": True,
                    }
                )
            done_jobs += 1
            if verbose:
                frac = done_jobs / total_jobs
                elapsed = time.time() - t0
                eta = (elapsed / frac - elapsed) if frac > 1e-12 else float("nan")
                print(
                    f"[05_program][symbolic] progress {done_jobs}/{total_jobs} "
                    f"({frac*100:.1f}%) elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
    return out
