from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .prompt_parser import parse_prompt_to_distributions


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    standardized_records: List[Dict]
    enc_A: np.ndarray
    enc_B: np.ndarray
    enc_full: np.ndarray
    y_target: np.ndarray
    y_brate: np.ndarray


def _encode_distribution(distribution: List[List[float]], max_outcomes: int) -> np.ndarray:
    out = np.zeros(2 * max_outcomes, dtype=np.float64)
    n = min(len(distribution), max_outcomes)
    for i in range(n):
        p, x = distribution[i]
        out[2 * i] = float(p)
        out[2 * i + 1] = float(x)
    return out


def _build_encodings(dists_a: List[List[List[float]]], dists_b: List[List[List[float]]], max_outcomes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(dists_a)
    enc_a = np.zeros((n, 2 * max_outcomes), dtype=np.float64)
    enc_b = np.zeros((n, 2 * max_outcomes), dtype=np.float64)
    for i in range(n):
        enc_a[i] = _encode_distribution(dists_a[i], max_outcomes=max_outcomes)
        enc_b[i] = _encode_distribution(dists_b[i], max_outcomes=max_outcomes)
    enc_full = np.hstack([enc_a, enc_b])
    return enc_a, enc_b, enc_full


def load_prompts_dataset(path: Path, target_mode: str = "one_minus_bRate", max_outcomes: int = 9) -> DatasetBundle:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    df = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
    needed = {"row_index", "prompt", "bRate", "prompt_hash", "prompt_version"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    dists_a: List[List[List[float]]] = []
    dists_b: List[List[List[float]]] = []
    standardized_records: List[Dict] = []
    for _, row in df.iterrows():
        prompt = str(row["prompt"])
        dist_a, dist_b = parse_prompt_to_distributions(prompt)
        dists_a.append(dist_a)
        dists_b.append(dist_b)
        br = float(row["bRate"])
        pv = str(row.get("prompt_version", "unknown"))
        standardized_records.append(
            {
                "metadata": {
                    "index": int(row["row_index"]),
                    "source": f"jsonl:{pv}",
                    "prompt_version": pv,
                },
                "context": {
                    "gamble_a": {"distribution": dist_a, "description": ""},
                    "gamble_b": {"distribution": dist_b, "description": ""},
                },
                "action": {"bRate": br},
            }
        )

    y_b = df["bRate"].to_numpy(dtype=np.float64)
    if target_mode == "one_minus_bRate":
        y_target = 1.0 - y_b
    elif target_mode == "bRate":
        y_target = y_b.copy()
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    enc_a, enc_b, enc_full = _build_encodings(dists_a, dists_b, max_outcomes=max_outcomes)
    return DatasetBundle(
        df=df,
        standardized_records=standardized_records,
        enc_A=enc_a,
        enc_B=enc_b,
        enc_full=enc_full,
        y_target=y_target,
        y_brate=y_b,
    )
