from __future__ import annotations

import re
from typing import List, Tuple


OPTION_A_RE = re.compile(r"Option A:\s*(.+?)(?:\n\n|\nOption B:)", re.IGNORECASE | re.DOTALL)
OPTION_B_RE = re.compile(r"Option B:\s*(.+?)(?:\n\nAnswer|\nAnswer|$)", re.IGNORECASE | re.DOTALL)
CLAUSE_RE = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)%\s*chance\s*you\s*(receive|lose)\s*([0-9]+(?:\.[0-9]+)?)\s*dollars?",
    re.IGNORECASE,
)


def _parse_option_text(option_text: str) -> List[List[float]]:
    out: List[List[float]] = []
    for prob_s, verb, amount_s in CLAUSE_RE.findall(option_text):
        p = float(prob_s) / 100.0
        amount = float(amount_s)
        x = amount if verb.lower() == "receive" else -amount
        out.append([p, x])
    if not out:
        raise ValueError(f"Failed to parse option distribution: {option_text!r}")
    return out


def parse_prompt_to_distributions(prompt: str) -> Tuple[List[List[float]], List[List[float]]]:
    m_a = OPTION_A_RE.search(prompt)
    m_b = OPTION_B_RE.search(prompt)
    if not m_a or not m_b:
        raise ValueError("Missing Option A/B block in prompt")
    dist_a = _parse_option_text(m_a.group(1))
    dist_b = _parse_option_text(m_b.group(1))
    return dist_a, dist_b
