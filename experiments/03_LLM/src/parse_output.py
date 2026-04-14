from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


P_B_PATTERN = re.compile(r"\bP_B\s*=\s*([01](?:\.\d+)?|\.\d+)\b", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"(?<![\d.])([01](?:\.\d+)?|\.\d+)(?![\d.])")


@dataclass
class ParseResult:
    pred_brate: float
    parse_ok: bool
    parser_method: str


def _coerce_probability(value: float) -> Optional[float]:
    if math.isnan(value) or math.isinf(value):
        return None
    if 0.0 <= value <= 1.0:
        return float(value)
    return None


def parse_probability_from_text(text: str) -> ParseResult:
    raw = (text or "").strip()
    m = P_B_PATTERN.search(raw)
    if m:
        v = _coerce_probability(float(m.group(1)))
        if v is not None:
            return ParseResult(pred_brate=v, parse_ok=True, parser_method="p_b_pattern")

    nums = NUMBER_PATTERN.findall(raw)
    if len(nums) == 1:
        v = _coerce_probability(float(nums[0]))
        if v is not None:
            return ParseResult(pred_brate=v, parse_ok=True, parser_method="single_number")

    return ParseResult(pred_brate=float("nan"), parse_ok=False, parser_method="failed")


def build_retry_prompt(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "IMPORTANT: Output only one line in this exact format: P_B=0.6273\n"
        "No extra words."
    )


def parse_with_retry(
    prompt: str,
    infer_once: Callable[[str], Tuple[str, float, bool]],
    max_retries: int = 1,
) -> dict:
    attempts = 0
    current_prompt = prompt
    final_text = ""
    latency = 0.0
    from_cache = False
    parse = ParseResult(pred_brate=float("nan"), parse_ok=False, parser_method="failed")

    while attempts <= max_retries:
        text, dt, cache_flag = infer_once(current_prompt)
        final_text = text
        latency += dt
        from_cache = from_cache or cache_flag
        parse = parse_probability_from_text(text)
        if parse.parse_ok:
            break
        attempts += 1
        if attempts <= max_retries:
            current_prompt = build_retry_prompt(prompt)

    return {
        "pred_brate": parse.pred_brate,
        "parse_ok": bool(parse.parse_ok),
        "parser_method": parse.parser_method,
        "retry_count": min(attempts, max_retries),
        "raw_completion": final_text,
        "latency_seconds": latency,
        "from_cache": from_cache,
    }
