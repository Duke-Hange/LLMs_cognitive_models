from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_LLM_DIR = _THIS_DIR.parent
_EXPERIMENTS_DIR = _LLM_DIR.parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from evaluation_metrics import evaluate_predictions  # noqa: E402
from kshot import build_kshot_prompt, select_fixed_pool, swap_option_labels  # noqa: E402
from llm_client import OllamaClient  # noqa: E402
from parse_output import parse_with_retry  # noqa: E402
from splits import build_train_test_split  # noqa: E402


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _parse_k_values(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        v = int(part.strip())
        if v < 0:
            raise ValueError("k must be >= 0")
        out.append(v)
    if not out:
        raise ValueError("k list cannot be empty")
    return sorted(list(dict.fromkeys(out)))


def _safe_float(v: float) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="03_LLM evaluation with fixed-pool k-shot (Ollama)")
    parser.add_argument("--prompts", type=Path, default=_LLM_DIR / "results" / "prompts_en_v2.jsonl")
    parser.add_argument("--output-dir", type=Path, default=_LLM_DIR / "results")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-random-state-base", type=int, default=1017)
    parser.add_argument("--split-type", type=str, default="train_test")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--num-predict", type=int, default=32)
    parser.add_argument("--max-retries", type=int, default=1, help="Retries after parse failure")
    parser.add_argument("--k", type=str, default="0", help="Comma-separated k values, e.g. 0,1,2,5,10")
    parser.add_argument("--pool-seed", type=int, default=42010)
    parser.add_argument("--kshot-decimals", type=int, default=4)
    parser.add_argument(
        "--no-dedup-icl",
        action="store_true",
        help="Repeat full per-row prompt in each k-shot example (legacy; more tokens)",
    )
    parser.add_argument("--run-swap-robustness", action="store_true")
    parser.add_argument("--swap-tolerance", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true", help="Build prompts and splits only, no model calls")
    parser.add_argument("--max-test-rows", type=int, default=0, help="0 means full test set")
    parser.add_argument("--cache-dir", type=Path, default=_LLM_DIR / "results" / "cache")
    parser.add_argument("--log-every", type=int, default=20, help="Print progress every N items")
    parser.add_argument("--quiet", action="store_true", help="Reduce progress logs")
    args = parser.parse_args()

    prompts = _read_jsonl(args.prompts)
    if not prompts:
        raise SystemExit(f"No records found in {args.prompts}")

    df = pd.DataFrame(prompts)
    needed = {"row_index", "prompt_version", "prompt", "prompt_hash", "bRate"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in prompts jsonl: {missing}")
    df = df.sort_values("row_index").reset_index(drop=True)

    split = build_train_test_split(
        n_rows=len(df),
        seed=args.seed,
        test_size=args.test_size,
        base_random_state=args.split_random_state_base,
    )
    k_values = _parse_k_values(args.k)

    train_df = df.iloc[split.train_idx].copy().sort_values("row_index")
    test_df = df.iloc[split.test_idx].copy().sort_values("row_index")
    if args.max_test_rows > 0:
        test_df = test_df.iloc[: args.max_test_rows].copy()

    if not args.quiet:
        print(
            "[03_LLM] start | "
            f"model={args.model} "
            f"seed={args.seed} split_random_state={split.random_state} "
            f"test_rows={len(test_df)} "
            f"k_values={k_values} "
            f"swap={args.run_swap_robustness} dry_run={args.dry_run}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = OllamaClient(
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        temperature=args.temperature,
        num_predict=args.num_predict,
        cache_dir=args.cache_dir,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.replace("/", "_").replace(":", "_")
    pred_path = args.output_dir / f"llm_predictions_{args.split_type}_{model_tag}_{timestamp}.csv"
    summary_path = args.output_dir / f"llm_models_summary_{timestamp}.csv"

    all_rows_out: List[Dict] = []
    summary_rows: List[Dict] = []
    train_records = train_df.to_dict("records")
    total_factor = 2 if args.run_swap_robustness else 1
    total_items = len(test_df) * len(k_values) * total_factor
    completed_items = 0
    t0 = time.time()

    def _print_progress(k_now: int, row_index_now: int, is_swap_now: bool) -> None:
        if args.quiet:
            return
        elapsed = time.time() - t0
        frac = (completed_items / total_items) if total_items else 1.0
        eta = (elapsed / frac - elapsed) if frac > 1e-12 else float("nan")
        tag = "swap" if is_swap_now else "main"
        print(
            "[03_LLM] progress | "
            f"{completed_items}/{total_items} ({frac*100:.1f}%) "
            f"k={k_now} row={row_index_now} mode={tag} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
        )

    for k in k_values:
        if not args.quiet:
            print(f"[03_LLM] k={k} started ...")
        pool = select_fixed_pool(
            train_records=train_records,
            k=k,
            pool_seed=args.pool_seed + k,
            decimals=args.kshot_decimals,
        )
        ex_rows_text = ",".join(str(x) for x in pool.example_row_indices)
        k_rows: List[Dict] = []
        swap_checks: List[float] = []
        for _, row in test_df.iterrows():
            base_prompt = str(row["prompt"])
            prompt = build_kshot_prompt(
                base_prompt=base_prompt,
                pool=pool,
                dedup_icl=not args.no_dedup_icl,
            )
            prompt_full_hash = _hash_text(prompt)
            cache_key = f"{args.model}::{int(row['row_index'])}::k={k}::{prompt_full_hash}"

            if args.dry_run:
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

                parsed = parse_with_retry(prompt, _infer_once, max_retries=args.max_retries)
            completed_items += 1
            if completed_items == 1 or completed_items % max(1, args.log_every) == 0 or completed_items == total_items:
                _print_progress(k_now=k, row_index_now=int(row["row_index"]), is_swap_now=False)

            pred = _safe_float(parsed["pred_brate"])
            row_out = {
                "row_index": int(row["row_index"]),
                "split_type": args.split_type,
                "model": args.model,
                "prompt_version": str(row["prompt_version"]),
                "k": int(k),
                "pool_seed": int(pool.pool_seed),
                "example_row_indices": ex_rows_text,
                "prompt_hash": str(row["prompt_hash"]),
                "prompt_full_hash": prompt_full_hash,
                "bRate": float(row["bRate"]),
                "pred_bRate": pred,
                "pred_y_neural_space": float(1.0 - pred) if not math.isnan(pred) else float("nan"),
                "raw_completion": str(parsed["raw_completion"]),
                "parse_ok": bool(parsed["parse_ok"]),
                "parser_method": str(parsed["parser_method"]),
                "retry_count": int(parsed["retry_count"]),
                "latency_seconds": float(parsed["latency_seconds"]),
                "from_cache": bool(parsed["from_cache"]),
                "is_swap": False,
            }
            all_rows_out.append(row_out)
            k_rows.append(row_out)

            if args.run_swap_robustness:
                swap_prompt = swap_option_labels(prompt)
                swap_hash = _hash_text(swap_prompt)
                swap_cache_key = f"{args.model}::{int(row['row_index'])}::k={k}::swap::{swap_hash}"
                if args.dry_run:
                    swap_parsed = {
                        "pred_brate": float("nan"),
                        "parse_ok": False,
                        "parser_method": "dry_run",
                        "retry_count": 0,
                        "raw_completion": "",
                        "latency_seconds": 0.0,
                        "from_cache": False,
                    }
                else:
                    def _infer_swap_once(prompt_text: str):
                        r = client.generate(prompt_text, cache_key=swap_cache_key if prompt_text == swap_prompt else None)
                        return r.text, r.latency_seconds, r.from_cache

                    swap_parsed = parse_with_retry(swap_prompt, _infer_swap_once, max_retries=args.max_retries)
                completed_items += 1
                if completed_items == 1 or completed_items % max(1, args.log_every) == 0 or completed_items == total_items:
                    _print_progress(k_now=k, row_index_now=int(row["row_index"]), is_swap_now=True)

                swap_pred = _safe_float(swap_parsed["pred_brate"])
                swap_row = {
                    "row_index": int(row["row_index"]),
                    "split_type": args.split_type,
                    "model": args.model,
                    "prompt_version": str(row["prompt_version"]),
                    "k": int(k),
                    "pool_seed": int(pool.pool_seed),
                    "example_row_indices": ex_rows_text,
                    "prompt_hash": str(row["prompt_hash"]),
                    "prompt_full_hash": swap_hash,
                    "bRate": float(row["bRate"]),
                    "pred_bRate": swap_pred,
                    "pred_y_neural_space": float(1.0 - swap_pred) if not math.isnan(swap_pred) else float("nan"),
                    "raw_completion": str(swap_parsed["raw_completion"]),
                    "parse_ok": bool(swap_parsed["parse_ok"]),
                    "parser_method": str(swap_parsed["parser_method"]),
                    "retry_count": int(swap_parsed["retry_count"]),
                    "latency_seconds": float(swap_parsed["latency_seconds"]),
                    "from_cache": bool(swap_parsed["from_cache"]),
                    "is_swap": True,
                }
                all_rows_out.append(swap_row)
                k_rows.append(swap_row)

                if row_out["parse_ok"] and swap_row["parse_ok"]:
                    swap_delta = abs(float(swap_row["pred_bRate"]) - (1.0 - float(row_out["pred_bRate"])))
                    swap_checks.append(swap_delta)

        k_df = pd.DataFrame(k_rows)
        main_df = k_df[~k_df["is_swap"]].copy()
        valid_df = main_df[main_df["parse_ok"] & main_df["pred_bRate"].notna()].copy()

        y_true = valid_df["bRate"].to_numpy(dtype=np.float64)
        y_pred = valid_df["pred_bRate"].to_numpy(dtype=np.float64)
        if len(valid_df) > 0:
            metrics = evaluate_predictions(y_true, y_pred)
        else:
            metrics = {
                "mse": float("nan"),
                "r2": float("nan"),
                "correlation": float("nan"),
                "cross_entropy": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
            }

        if swap_checks:
            swap_pass_rate = float(np.mean(np.asarray(swap_checks, dtype=np.float64) <= args.swap_tolerance))
        else:
            swap_pass_rate = float("nan")

        summary_rows.append(
            {
                "model": args.model,
                "split_type": args.split_type,
                "prompt_version": str(test_df["prompt_version"].iloc[0]),
                "k": int(k),
                "seed": int(args.seed),
                "split_random_state": int(split.random_state),
                "test_mse": float(metrics["mse"]),
                "test_r2": float(metrics["r2"]),
                "test_correlation": float(metrics["correlation"]),
                "test_cross_entropy": float(metrics["cross_entropy"]),
                "test_rmse": float(metrics["rmse"]),
                "test_mae": float(metrics["mae"]),
                "n_test": int(len(main_df)),
                "n_valid": int(len(valid_df)),
                "parse_success_rate": float(len(valid_df) / len(main_df)) if len(main_df) else float("nan"),
                "swap_pass_rate": swap_pass_rate,
                "pool_seed": int(pool.pool_seed),
                "example_row_indices": ex_rows_text,
                "swap_tolerance": float(args.swap_tolerance),
                "predictions_path": str(pred_path),
            }
        )
        if not args.quiet:
            print(
                "[03_LLM] k finished | "
                f"k={k} n_test={len(main_df)} n_valid={len(valid_df)} "
                f"parse_success_rate={summary_rows[-1]['parse_success_rate']:.3f} "
                f"test_mse={summary_rows[-1]['test_mse']:.6f}"
            )

    pred_df = pd.DataFrame(all_rows_out)
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8")

    if not args.quiet:
        print(f"[03_LLM] done | elapsed={(time.time() - t0):.1f}s")
    print(f"[03_LLM] predictions -> {pred_path}")
    print(f"[03_LLM] summary -> {summary_path}")


if __name__ == "__main__":
    main()
