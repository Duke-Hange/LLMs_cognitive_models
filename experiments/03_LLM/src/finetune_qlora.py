from __future__ import annotations

import argparse
import json
import math
import sys
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
from parse_output import parse_probability_from_text  # noqa: E402
from splits import build_train_test_split  # noqa: E402


def _require_training_deps() -> None:
    missing = []
    try:
        import torch  # noqa: F401
    except Exception:
        missing.append("torch")
    try:
        import transformers  # noqa: F401
    except Exception:
        missing.append("transformers")
    try:
        import datasets  # noqa: F401
    except Exception:
        missing.append("datasets")
    try:
        import peft  # noqa: F401
    except Exception:
        missing.append("peft")
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        missing.append("bitsandbytes")
    if missing:
        raise SystemExit(
            "Missing finetune dependencies: "
            + ", ".join(missing)
            + ".\nInstall them before running finetune_qlora.py."
        )


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _fmt_target(b_rate: float, decimals: int = 4) -> str:
    return f"P_B={float(b_rate):.{decimals}f}"


def _split_train_val_indices(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_val = max(1, int(round(n * val_fraction)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return perm[n_val:], perm[:n_val]


def main() -> None:
    parser = argparse.ArgumentParser(description="03_LLM QLoRA finetune and evaluate")
    parser.add_argument("--prompts", type=Path, default=_LLM_DIR / "results" / "prompts_en_v2.jsonl")
    parser.add_argument("--output-dir", type=Path, default=_LLM_DIR / "results")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-type", type=str, default="train_test")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-random-state-base", type=int, default=1017)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--decimals", type=int, default=4)
    parser.add_argument("--report-to", type=str, default="none")
    args = parser.parse_args()

    _require_training_deps()
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    rows = _read_jsonl(args.prompts)
    if not rows:
        raise SystemExit(f"No rows found in {args.prompts}")
    df = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
    needed = {"row_index", "prompt", "prompt_version", "bRate"}
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise SystemExit(f"Missing columns in prompts jsonl: {miss}")

    split = build_train_test_split(
        n_rows=len(df),
        seed=args.seed,
        test_size=args.test_size,
        base_random_state=args.split_random_state_base,
    )
    train_df = df.iloc[split.train_idx].copy().sort_values("row_index").reset_index(drop=True)
    test_df = df.iloc[split.test_idx].copy().sort_values("row_index").reset_index(drop=True)

    tr_idx, va_idx = _split_train_val_indices(len(train_df), args.val_fraction, args.seed)
    fit_df = train_df.iloc[tr_idx].copy().reset_index(drop=True)
    val_df = train_df.iloc[va_idx].copy().reset_index(drop=True)

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    def _make_features(data_df: pd.DataFrame) -> Dataset:
        records = []
        for _, row in data_df.iterrows():
            prompt = str(row["prompt"]).strip()
            target = _fmt_target(float(row["bRate"]), decimals=args.decimals)
            full_text = f"{prompt}\n{target}"
            prompt_text = f"{prompt}\n"

            full = tokenizer(full_text, truncation=True, max_length=args.max_length)
            prompt_only = tokenizer(prompt_text, truncation=True, max_length=args.max_length)

            labels = list(full["input_ids"])
            prompt_len = min(len(prompt_only["input_ids"]), len(labels))
            for i in range(prompt_len):
                labels[i] = -100

            records.append(
                {
                    "input_ids": full["input_ids"],
                    "attention_mask": full["attention_mask"],
                    "labels": labels,
                }
            )
        return Dataset.from_list(records)

    train_ds = _make_features(fit_df)
    eval_ds = _make_features(val_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.base_model.replace("/", "_")
    run_dir = args.output_dir / f"finetune_{model_tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tr_args = TrainingArguments(
        output_dir=str(run_dir),
        overwrite_output_dir=False,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to=[] if args.report_to == "none" else [args.report_to],
        fp16=False,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True),
    )
    trainer.train()
    trainer.save_model(str(run_dir / "adapter"))
    tokenizer.save_pretrained(str(run_dir / "adapter"))

    pred_rows: List[Dict] = []
    model.eval()
    for _, row in test_df.iterrows():
        prompt = str(row["prompt"]).strip()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        parsed = parse_probability_from_text(text)
        pred = float(parsed.pred_brate) if parsed.parse_ok else float("nan")

        pred_rows.append(
            {
                "row_index": int(row["row_index"]),
                "split_type": args.split_type,
                "model": f"qlora::{args.base_model}",
                "prompt_version": str(row["prompt_version"]),
                "k": 0,
                "pool_seed": np.nan,
                "example_row_indices": "",
                "prompt_hash": row.get("prompt_hash", ""),
                "prompt_full_hash": "",
                "bRate": float(row["bRate"]),
                "pred_bRate": pred,
                "pred_y_neural_space": float(1.0 - pred) if not math.isnan(pred) else float("nan"),
                "raw_completion": text,
                "parse_ok": bool(parsed.parse_ok),
                "parser_method": parsed.parser_method,
                "retry_count": 0,
                "latency_seconds": np.nan,
                "from_cache": False,
                "is_swap": False,
            }
        )

    pred_df = pd.DataFrame(pred_rows)
    valid_df = pred_df[pred_df["parse_ok"] & pred_df["pred_bRate"].notna()].copy()
    if len(valid_df) > 0:
        metrics = evaluate_predictions(valid_df["bRate"].to_numpy(), valid_df["pred_bRate"].to_numpy())
    else:
        metrics = {
            "mse": float("nan"),
            "r2": float("nan"),
            "correlation": float("nan"),
            "cross_entropy": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.output_dir / f"llm_predictions_{args.split_type}_qlora_{timestamp}.csv"
    summary_path = args.output_dir / f"llm_models_summary_{timestamp}.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")

    summary = {
        "model": f"qlora::{args.base_model}",
        "split_type": args.split_type,
        "prompt_version": str(test_df["prompt_version"].iloc[0]),
        "k": 0,
        "seed": int(args.seed),
        "split_random_state": int(split.random_state),
        "test_mse": float(metrics["mse"]),
        "test_r2": float(metrics["r2"]),
        "test_correlation": float(metrics["correlation"]),
        "test_cross_entropy": float(metrics["cross_entropy"]),
        "test_rmse": float(metrics["rmse"]),
        "test_mae": float(metrics["mae"]),
        "n_test": int(len(pred_df)),
        "n_valid": int(len(valid_df)),
        "parse_success_rate": float(len(valid_df) / len(pred_df)) if len(pred_df) else float("nan"),
        "swap_pass_rate": float("nan"),
        "pool_seed": np.nan,
        "example_row_indices": "",
        "swap_tolerance": np.nan,
        "predictions_path": str(pred_path),
    }
    pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8")

    meta = {
        "timestamp": timestamp,
        "base_model": args.base_model,
        "seed": args.seed,
        "split_random_state": split.random_state,
        "n_train_fit": len(fit_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "adapter_dir": str(run_dir / "adapter"),
        "predictions_path": str(pred_path),
        "summary_path": str(summary_path),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[03_LLM][QLoRA] adapter -> {run_dir / 'adapter'}")
    print(f"[03_LLM][QLoRA] predictions -> {pred_path}")
    print(f"[03_LLM][QLoRA] summary -> {summary_path}")


if __name__ == "__main__":
    main()
