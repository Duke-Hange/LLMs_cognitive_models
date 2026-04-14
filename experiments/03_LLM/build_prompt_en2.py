"""
Build English prompts that only ask for personal preference (no aggregate bRate).

Reuses gamble prose from build_prompt_en.py (same Option A/B wording as en_v2).

Usage:
  cd experiments/03_LLM
  python build_prompt_en2.py --out results/prompts_en2_preference.jsonl
  python build_prompt_en2.py --out PRINT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd

# Same directory import
from build_prompt_en import (
    format_option_prose,
    parse_gamble_literal,
    prompt_hash,
    validate_probability_mass,
)

PROMPT_VERSION = "en2_preference"

TEMPLATE_EN2_PREFERENCE = """Imagine you are taking part in a decision-making experiment.

Q: Which option do you prefer?

Option A: {option_a_prose}

Option B: {option_b_prose}

Answer with exactly one line: either "Option A" or "Option B" and nothing else.
"""


def build_prompt_preference_only(
    gamble_a: List[Tuple[float, float]],
    gamble_b: List[Tuple[float, float]],
    strict_validate: bool = True,
) -> str:
    if strict_validate:
        validate_probability_mass(gamble_a, "Option A")
        validate_probability_mass(gamble_b, "Option B")
    prose_a = format_option_prose(gamble_a)
    prose_b = format_option_prose(gamble_b)
    return TEMPLATE_EN2_PREFERENCE.format(
        option_a_prose=prose_a,
        option_b_prose=prose_b,
    )


def row_to_record(
    row_index: int,
    row: pd.Series,
    strict_validate: bool = True,
) -> dict:
    a = parse_gamble_literal(row["A"])
    b = parse_gamble_literal(row["B"])
    text = build_prompt_preference_only(a, b, strict_validate=strict_validate)
    return {
        "row_index": int(row_index),
        "prompt_version": PROMPT_VERSION,
        "prompt": text,
        "prompt_hash": prompt_hash(text),
        "bRate": float(row["bRate"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build preference-only English prompts (Which option do you prefer?)"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "data.csv",
        help="Path to data.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output .jsonl path, '-' for stdout, or PRINT for first prompt only",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max rows (0 = all)")
    parser.add_argument(
        "--no-strict-sum",
        action="store_true",
        help="Do not require sum(prob)=1 per option",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    for col in ("bRate", "A", "B"):
        if col not in df.columns:
            sys.exit(f"Missing column {col!r} in {args.csv}")

    n = len(df) if args.limit <= 0 else min(args.limit, len(df))
    records = []
    for i in range(n):
        row = df.iloc[i]
        try:
            rec = row_to_record(i, row, strict_validate=not args.no_strict_sum)
        except ValueError as e:
            sys.exit(f"Row {i}: {e}")
        records.append(rec)

    if args.out == "PRINT" and records:
        print(records[0]["prompt"])
        return

    if args.out == "-":
        for rec in records:
            print(json.dumps(rec, ensure_ascii=False))
        return

    if not args.out:
        sys.exit("Specify --out <path.jsonl> or --out - or --out PRINT")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
