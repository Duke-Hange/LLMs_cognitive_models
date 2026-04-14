from __future__ import annotations

import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
_LLM_DIR = _ROOT / "03_LLM" / "src"
if str(_LLM_DIR) not in sys.path:
    sys.path.insert(0, str(_LLM_DIR))

from kshot import build_kshot_prompt, select_fixed_pool  # type: ignore
from llm_client import OllamaClient  # type: ignore
from parse_output import parse_with_retry  # type: ignore


def _safe_float(v: float) -> float:
    if isinstance(v, (int, float)) and not math.isnan(float(v)):
        return float(v)
    return float("nan")


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def run_llm_family(
    model_defs: List[Dict],
    df: pd.DataFrame,
    y_target: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    sample_map: Dict[tuple, np.ndarray],
    llm_cfg: Dict,
    verbose: bool = True,
) -> List[Dict]:
    out: List[Dict] = []
    test_df = df.iloc[test_idx].copy().sort_values("row_index")
    dry_run = bool(llm_cfg.get("dry_run", False))
    log_every = int(llm_cfg.get("log_every", 20))
    total_items = max(1, len(model_defs) * len(sample_map) * len(test_df))
    done_items = 0
    t0 = time.time()

    for model_def in model_defs:
        model_id = str(model_def["model_id"])
        source_name = str(model_def["source_name"])
        if verbose:
            print(
                f"[05_program][llm] model started: {model_id} ({source_name}) "
                f"dry_run={dry_run}",
                flush=True,
            )
        client = OllamaClient(
            model=source_name,
            base_url=str(llm_cfg["base_url"]),
            timeout_seconds=int(llm_cfg["timeout_seconds"]),
            temperature=float(llm_cfg["temperature"]),
            num_predict=int(llm_cfg["num_predict"]),
            cache_dir=Path(str(llm_cfg["cache_dir"])),
        )

        for (n, seed), sampled_train_idx in sample_map.items():
            if verbose:
                print(
                    f"[05_program][llm] start model={model_id} N={n} seed={seed} "
                    f"test_rows={len(test_df)}",
                    flush=True,
                )
            train_records = df.iloc[sampled_train_idx].copy().sort_values("row_index").to_dict("records")
            pool = select_fixed_pool(
                train_records=train_records,
                k=int(n),
                pool_seed=int(seed) + 10000,
                decimals=int(llm_cfg["kshot_decimals"]),
            )

            for _, row in test_df.iterrows():
                base_prompt = str(row["prompt"])
                prompt = build_kshot_prompt(
                    base_prompt=base_prompt,
                    pool=pool,
                    dedup_icl=bool(llm_cfg.get("dedup_icl", True)),
                )
                prompt_hash = _hash_text(prompt)
                cache_key = f"{source_name}::{int(row['row_index'])}::N={n}::seed={seed}::{prompt_hash}"

                if dry_run:
                    parsed = {
                        "pred_brate": float("nan"),
                        "parse_ok": False,
                        "parser_method": "dry_run",
                        "retry_count": 0,
                        "raw_completion": "",
                        "latency_seconds": 0.0,
                        "from_cache": False,
                    }
                else:
                    def _infer_once(prompt_text: str):
                        r = client.generate(prompt_text, cache_key=cache_key if prompt_text == prompt else None)
                        return r.text, r.latency_seconds, r.from_cache

                    parsed = parse_with_retry(prompt, _infer_once, max_retries=int(llm_cfg["max_retries"]))

                pred_b = _safe_float(parsed["pred_brate"])
                pred_y = float(1.0 - pred_b) if not math.isnan(pred_b) else float("nan")
                row_idx = int(row["row_index"])
                out.append(
                    {
                        "family": "llm",
                        "model_id": model_id,
                        "source_name": source_name,
                        "N": int(n),
                        "seed": int(seed),
                        "row_index": row_idx,
                        "y_true": float(y_target[row_idx]),
                        "y_pred": float(np.clip(pred_y, 1e-15, 1 - 1e-15)) if not math.isnan(pred_y) else float("nan"),
                        "parse_ok": bool(parsed["parse_ok"]),
                    }
                )
                done_items += 1
                if verbose and (done_items == 1 or done_items % max(1, log_every) == 0 or done_items == total_items):
                    frac = done_items / total_items
                    elapsed = time.time() - t0
                    eta = (elapsed / frac - elapsed) if frac > 1e-12 else float("nan")
                    print(
                        f"[05_program][llm] progress {done_items}/{total_items} "
                        f"({frac*100:.1f}%) model={model_id} N={n} seed={seed} "
                        f"row={int(row['row_index'])} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )
    return out
