"""
Build rigorous English prompts from experiments/data/data.csv.

Conventions:
- Preserve CSV order of (probability, outcome) pairs (ast.literal_eval).
- Deterministic formatting for stable prompt_hash; validate sum(p)=1 per option (unless --no-strict-sum).
- bRate is never included in the prompt text (evaluation label only).

Usage:
  python build_prompt_en.py --out results/prompts_en_v1.jsonl
  python build_prompt_en.py --version v2 --out results/prompts_en_v2.jsonl
  python build_prompt_en.py --version v2 --out PRINT   # print first v2 prompt only
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import pandas as pd

PROB_SUM_TOL = 1e-5

# Canonical template (keep in sync with prompts/instruction_en_v1.txt)
TEMPLATE_EN_V1 = """You are assisting with a cognitive-science risk-choice task.

Context:
- Many human participants each faced exactly the same binary choice between two risky options, labeled "Gamble A" and "Gamble B".
- Each gamble is a lottery: the listed outcomes are mutually exclusive and exhaustive; exactly one outcome occurs per gamble according to the stated probabilities (which sum to 1 for each gamble).

Your task:
- Estimate the proportion of participants who chose Gamble B (rather than Gamble A).
- Respond with a single real number in the closed interval [0, 1], where 1 means "everyone chose B" and 0 means "no one chose B".

Rules:
- Use only the probability-outcome information below. Do not invent extra payoffs, framing, or context.
- Outcomes are amounts in a single experimental currency (points) comparable across gambles.

Gamble A:
{gamble_a_block}

Gamble B:
{gamble_b_block}

Output format (strict):
- Output exactly one line of the form: P_B=0.6273
- Use a decimal point. No percent sign, no units, no explanation, no extra lines or whitespace before or after.
"""

# Literature-style Q&A (PNAS / decisions-from-description style); Option A/B kept for Choices13k.
# Sync with prompts/instruction_en_v2.txt
TEMPLATE_EN_V2 = """Imagine you are taking part in a decision-making experiment. Many different participants were each presented with the following question. Every participant chose either Option A or Option B.

Q: Which option do you prefer?

Option A: {option_a_prose}

Option B: {option_b_prose}

Your task is to estimate what fraction of participants chose Option B (as opposed to Option A). Respond with a real number between 0 and 1, where 1 means everyone chose Option B and 0 means no one did.

Do not state your own preference. Base your estimate only on the information above.

Output format (strict):
Output exactly one line of the form: P_B=0.6273
Use a decimal point. No percent sign for the final answer, no extra words, no blank lines.
"""

# Compact v3; sync with prompts/instruction_en_v3.txt. Keeps "Option A/B" + "\n\nYour task" for ICL split.
TEMPLATE_EN_V3 = """Many participants each faced the same choice: Option A vs Option B.

Q: Which option do you prefer?

Option A: {option_a_prose}

Option B: {option_b_prose}

Your task: estimate the fraction who chose Option B (vs A). One number in [0, 1].

Use only the lottery text. Do not report your own preference.

Output (strict): exactly one line P_B=0.6273 (decimal point; no %, no extra words or lines).
"""


def _fmt_probability(p: float) -> str:
    """Stable, readable probability string (avoids 0.9500000000000001 in prompts)."""
    x = float(p)
    s = f"{x:.12g}"
    if "e" in s.lower():
        return s
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def _fmt_outcome(v: float) -> str:
    x = float(v)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    s = f"{x:.12g}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _fmt_percent_human(p: float) -> str:
    """Probability as a human-readable percentage (no % sign in return)."""
    q = float(p) * 100.0
    if abs(q - round(q)) < 1e-6:
        return str(int(round(q)))
    s = f"{q:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _dollar_word(magnitude: float) -> str:
    """Singular 'dollar' when |magnitude| is 1, else 'dollars'."""
    return "dollar" if abs(abs(magnitude) - 1.0) < 1e-9 else "dollars"


def format_option_prose(pairs: Sequence[Tuple[float, float]]) -> str:
    """
    Single-line prose for Option A/B (image-style: X% chance you receive/lose ...).
    Preserves CSV order. Negative payoffs phrased as 'lose |x| dollar(s)'.
    """
    parts: List[str] = []
    for prob, outcome in pairs:
        pct = _fmt_percent_human(prob)
        x = float(outcome)
        if x > 0:
            u = _dollar_word(x)
            parts.append(f"{pct}% chance you receive {_fmt_outcome(x)} {u}")
        elif x < 0:
            mag = abs(x)
            u = _dollar_word(mag)
            parts.append(f"{pct}% chance you lose {_fmt_outcome(mag)} {u}")
        else:
            parts.append(f"{pct}% chance you receive 0 dollars")
    return "; ".join(parts) + "."


def parse_gamble_literal(cell: Any) -> List[Tuple[float, float]]:
    """Parse CSV cell into list of (probability, outcome)."""
    if isinstance(cell, str):
        parsed = ast.literal_eval(cell)
    else:
        parsed = cell
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"Gamble must be non-empty list, got {type(parsed)}")
    out: List[Tuple[float, float]] = []
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Bad outcome entry: {item!r}")
        out.append((float(item[0]), float(item[1])))
    return out


def validate_probability_mass(pairs: Sequence[Tuple[float, float]], label: str) -> None:
    s = sum(p for p, _ in pairs)
    if abs(s - 1.0) > PROB_SUM_TOL:
        raise ValueError(
            f"{label}: outcome probabilities sum to {s:.10f}, expected 1.0 (tol={PROB_SUM_TOL})"
        )


def format_gamble_block(pairs: Sequence[Tuple[float, float]]) -> str:
    """One line per outcome, bullet list."""
    lines = []
    for prob, outcome in pairs:
        lines.append(
            f"- With probability {_fmt_probability(prob)}, the outcome is {_fmt_outcome(outcome)}."
        )
    return "\n".join(lines)


def build_prompt_en_v1(gamble_a: List[Tuple[float, float]], gamble_b: List[Tuple[float, float]]) -> str:
    validate_probability_mass(gamble_a, "Gamble A")
    validate_probability_mass(gamble_b, "Gamble B")
    a_block = format_gamble_block(gamble_a)
    b_block = format_gamble_block(gamble_b)
    return TEMPLATE_EN_V1.format(gamble_a_block=a_block, gamble_b_block=b_block)


def build_prompt_en_v2(gamble_a: List[Tuple[float, float]], gamble_b: List[Tuple[float, float]]) -> str:
    validate_probability_mass(gamble_a, "Option A")
    validate_probability_mass(gamble_b, "Option B")
    prose_a = format_option_prose(gamble_a)
    prose_b = format_option_prose(gamble_b)
    return TEMPLATE_EN_V2.format(option_a_prose=prose_a, option_b_prose=prose_b)


def build_prompt_en_v3(gamble_a: List[Tuple[float, float]], gamble_b: List[Tuple[float, float]]) -> str:
    validate_probability_mass(gamble_a, "Option A")
    validate_probability_mass(gamble_b, "Option B")
    prose_a = format_option_prose(gamble_a)
    prose_b = format_option_prose(gamble_b)
    return TEMPLATE_EN_V3.format(option_a_prose=prose_a, option_b_prose=prose_b)


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def row_to_record(
    row_index: int,
    row: pd.Series,
    strict_validate: bool = True,
    version: str = "v1",
) -> dict:
    a = parse_gamble_literal(row["A"])
    b = parse_gamble_literal(row["B"])
    if version == "v1":
        if strict_validate:
            validate_probability_mass(a, "Gamble A")
            validate_probability_mass(b, "Gamble B")
        text = build_prompt_en_v1(a, b)
        pv = "en_v1"
    elif version == "v2":
        if strict_validate:
            validate_probability_mass(a, "Option A")
            validate_probability_mass(b, "Option B")
        text = build_prompt_en_v2(a, b)
        pv = "en_v2"
    elif version == "v3":
        if strict_validate:
            validate_probability_mass(a, "Option A")
            validate_probability_mass(b, "Option B")
        text = build_prompt_en_v3(a, b)
        pv = "en_v3"
    else:
        raise ValueError(f"Unknown version {version!r}, expected 'v1', 'v2', or 'v3'")
    return {
        "row_index": int(row_index),
        "prompt_version": pv,
        "prompt": text,
        "prompt_hash": prompt_hash(text),
        "bRate": float(row["bRate"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build English prompts from data.csv")
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
        help="Output .jsonl path, or '-' for stdout (records), or 'PRINT' for first prompt only",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max rows (0 = all)")
    parser.add_argument(
        "--no-strict-sum",
        action="store_true",
        help="Do not require sum(prob)=1 per gamble (not recommended)",
    )
    parser.add_argument(
        "--version",
        choices=("v1", "v2", "v3"),
        default="v1",
        help="Prompt template: v1=cognitive bullet list, v2=literature Q&A, v3=compact v2 (ICL-friendly)",
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
            rec = row_to_record(
                i,
                row,
                strict_validate=not args.no_strict_sum,
                version=args.version,
            )
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
