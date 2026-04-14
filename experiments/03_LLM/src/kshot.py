from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Option A/B prose followed by task tail: v2/v3 use "\n\nYour task"; en2_preference uses "\n\nAnswer with exactly...".
_OPTION_AB_RE = re.compile(
    r"Option A:\s*(?P<a>.+?)\s*\n\nOption B:\s*(?P<b>.+?)(?=\s*\n\n(?:Your task\b|Answer with exactly one line\b))",
    flags=re.DOTALL | re.IGNORECASE,
)

# Short intro + one-shot task wording (template body is not repeated per example).
_ICL_DEDUP_PREAMBLE = (
    "Few-shot: each block is Option A/B, then an observed P_B= line. "
    "Predict the population fraction who chose B (vs A) in [0,1]; do not state your own preference. "
    "After the examples, answer the last pair with exactly one line P_B=0.#### (decimal; no %; no extra text)."
)


@dataclass
class KShotPool:
    k: int
    pool_seed: int
    example_row_indices: List[int]
    example_blocks: List[str]


def _fmt_pb(v: float, decimals: int = 4) -> str:
    return f"P_B={float(v):.{decimals}f}"


def extract_option_ab_prose(full_prompt: str) -> Optional[Tuple[str, str]]:
    """If prompt matches v2/v3-style Option A/B + Your task, return (prose_a, prose_b)."""
    m = _OPTION_AB_RE.search(full_prompt)
    if not m:
        return None
    a = str(m.group("a")).strip()
    b = str(m.group("b")).strip()
    if not a or not b:
        return None
    return a, b


def _split_legacy_example_block(block: str) -> Optional[Tuple[str, str]]:
    """Split `full_prompt\\nP_B=...` into (full_prompt, pb_line)."""
    if "\n" not in block:
        return None
    body, last = block.rsplit("\n", 1)
    last_stripped = last.strip()
    if not last_stripped.startswith("P_B="):
        return None
    return body, last_stripped


def _format_pair_block(prose_a: str, prose_b: str) -> str:
    return f"Option A: {prose_a}\n\nOption B: {prose_b}"


def build_kshot_example_block(prompt_text: str, b_rate: float, decimals: int = 4) -> str:
    return f"{prompt_text}\n{_fmt_pb(b_rate, decimals=decimals)}"


def select_fixed_pool(
    train_records: Sequence[dict],
    k: int,
    pool_seed: int,
    decimals: int = 4,
) -> KShotPool:
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return KShotPool(k=0, pool_seed=pool_seed, example_row_indices=[], example_blocks=[])
    if k > len(train_records):
        raise ValueError(f"k={k} exceeds train size={len(train_records)}")

    rng = np.random.default_rng(pool_seed)
    chosen = rng.choice(len(train_records), size=k, replace=False)
    chosen = np.asarray(chosen, dtype=np.int64)

    row_indices: List[int] = []
    blocks: List[str] = []
    for i in chosen:
        rec = train_records[int(i)]
        row_indices.append(int(rec["row_index"]))
        blocks.append(build_kshot_example_block(str(rec["prompt"]), float(rec["bRate"]), decimals=decimals))

    order = np.argsort(np.asarray(row_indices, dtype=np.int64))
    row_indices = [row_indices[int(i)] for i in order]
    blocks = [blocks[int(i)] for i in order]
    return KShotPool(k=k, pool_seed=pool_seed, example_row_indices=row_indices, example_blocks=blocks)


def _build_kshot_prompt_dedup(base_prompt: str, pool: KShotPool) -> Optional[str]:
    base_parts = extract_option_ab_prose(base_prompt)
    if base_parts is None:
        return None
    a_t, b_t = base_parts

    compact_blocks: List[str] = []
    for block in pool.example_blocks:
        sp = _split_legacy_example_block(block)
        if sp is None:
            return None
        body, pb_line = sp
        ab = extract_option_ab_prose(body)
        if ab is None:
            return None
        a_i, b_i = ab
        compact_blocks.append(f"{_format_pair_block(a_i, b_i)}\n{pb_line}")

    examples = "\n\n".join(compact_blocks)
    final_pair = _format_pair_block(a_t, b_t)
    return f"{_ICL_DEDUP_PREAMBLE}\n\n{examples}\n\nLast problem:\n\n{final_pair}"


def _build_kshot_prompt_legacy(base_prompt: str, pool: KShotPool) -> str:
    intro = (
        f"{pool.k} solved examples; each ends with P_B= (fraction who chose B). "
        "Output one line P_B= for the last problem only."
    )
    sep = "\n\n"
    examples = sep.join(pool.example_blocks)
    return f"{intro}\n\n{examples}\n\nLast problem:\n\n{base_prompt}"


def build_kshot_prompt(base_prompt: str, pool: KShotPool, *, dedup_icl: bool = True) -> str:
    if pool.k == 0:
        return base_prompt
    if dedup_icl:
        deduped = _build_kshot_prompt_dedup(base_prompt, pool)
        if deduped is not None:
            return deduped
    return _build_kshot_prompt_legacy(base_prompt, pool)


def swap_option_labels(prompt: str) -> str:
    """
    Swap A/B labels in prompt text for robustness checks.
    Uses temporary placeholders to avoid overwrite collisions.
    """
    swapped = prompt
    swapped = swapped.replace("Option A", "__TMP_OPTION_A__")
    swapped = swapped.replace("Option B", "__TMP_OPTION_B__")
    swapped = swapped.replace("Gamble A", "__TMP_GAMBLE_A__")
    swapped = swapped.replace("Gamble B", "__TMP_GAMBLE_B__")

    swapped = swapped.replace("__TMP_OPTION_A__", "Option B")
    swapped = swapped.replace("__TMP_OPTION_B__", "Option A")
    swapped = swapped.replace("__TMP_GAMBLE_A__", "Gamble B")
    swapped = swapped.replace("__TMP_GAMBLE_B__", "Gamble A")
    return swapped
