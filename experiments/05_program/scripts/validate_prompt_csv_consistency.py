"""
校验：从 LLM prompt 文本解析出的 Option A/B 分布是否与 data.csv 中结构化 A/B 一致。

用法（仓库根目录）：
  python experiments/05_program/scripts/validate_prompt_csv_consistency.py \\
    --csv experiments/data/data.csv \\
    --jsonl experiments/03_LLM/results/prompts_en_v2.jsonl

比较规则：将 (p, x) 按 (x, p) 排序后在容差内逐对相等（忽略 prompt 与 CSV 中结果顺序差异）。
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd

_PROG_ROOT = Path(__file__).resolve().parents[1]
if str(_PROG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROG_ROOT))

from src.data.prompt_parser import parse_prompt_to_distributions  # noqa: E402


def _norm_dist(dist: list, tol: float = 1e-5) -> list[tuple[float, float]]:
    pairs = [(float(p), float(x)) for p, x in dist]
    pairs.sort(key=lambda t: (t[1], t[0]))
    return pairs


def dists_match(a: list, b: list, tol: float) -> bool:
    na, nb = _norm_dist(a, tol), _norm_dist(b, tol)
    if len(na) != len(nb):
        return False
    for (p1, x1), (p2, x2) in zip(na, nb):
        if abs(p1 - p2) > tol or abs(x1 - x2) > tol:
            return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Compare prompt-parsed gambles to CSV A/B.")
    p.add_argument("--csv", type=Path, required=True, help="Path to data.csv")
    p.add_argument("--jsonl", type=Path, required=True, help="Path to prompts jsonl")
    p.add_argument("--max-rows", type=int, default=0, help="0 = all rows present in both")
    p.add_argument("--tol", type=float, default=1e-5)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    for col in ("A", "B"):
        df[col] = df[col].map(lambda s: ast.literal_eval(str(s)))

    rows: list[dict] = []
    with args.jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    jdf = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)

    n = len(jdf) if args.max_rows <= 0 else min(args.max_rows, len(jdf))
    mismatches = 0
    missing_csv = 0
    parse_errors = 0

    for i in range(n):
        rec = jdf.iloc[i]
        ri = int(rec["row_index"])
        if ri >= len(df) or ri < 0:
            missing_csv += 1
            continue
        csv_a = df.iloc[ri]["A"]
        csv_b = df.iloc[ri]["B"]
        try:
            pa, pb = parse_prompt_to_distributions(str(rec["prompt"]))
        except Exception:
            parse_errors += 1
            continue
        ok_a = dists_match(pa, list(csv_a), args.tol)
        ok_b = dists_match(pb, list(csv_b), args.tol)
        if not (ok_a and ok_b):
            mismatches += 1
            if mismatches <= 5:
                print(f"mismatch row_index={ri} ok_a={ok_a} ok_b={ok_b}")

    checked = n - missing_csv
    print(
        f"checked={checked} mismatches={mismatches} parse_errors={parse_errors} "
        f"missing_csv_index={missing_csv} tol={args.tol}"
    )
    if mismatches or parse_errors or missing_csv:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
