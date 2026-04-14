from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from shared.splits_core import sklearn_train_test_indices


@dataclass
class MasterSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    random_state: int


def build_master_split(n_rows: int, master_seed: int, test_size: float, split_random_state_base: int) -> MasterSplit:
    random_state = int(split_random_state_base + master_seed)
    train_idx, test_idx = sklearn_train_test_indices(
        n_rows, test_size=test_size, random_state=random_state
    )
    return MasterSplit(
        train_idx=train_idx,
        test_idx=test_idx,
        random_state=random_state,
    )


def build_sample_map(train_idx: np.ndarray, sample_sizes: Iterable[int], seeds: Iterable[int]) -> Dict[Tuple[int, int], np.ndarray]:
    sample_map: Dict[Tuple[int, int], np.ndarray] = {}
    n_train = len(train_idx)
    for n in sample_sizes:
        if n <= 0:
            raise ValueError(f"Sample size must be > 0, got {n}")
        if n > n_train:
            raise ValueError(f"Sample size {n} exceeds train pool size {n_train}")
        for seed in seeds:
            rng = np.random.default_rng(int(seed))
            choose_local = rng.choice(n_train, size=int(n), replace=False)
            sample_map[(int(n), int(seed))] = np.asarray(train_idx[choose_local], dtype=np.int64)
    return sample_map


def sample_map_as_json_ready(sample_map: Dict[Tuple[int, int], np.ndarray]) -> List[dict]:
    rows: List[dict] = []
    for (n, seed), idx in sorted(sample_map.items(), key=lambda x: (x[0][0], x[0][1])):
        rows.append(
            {
                "N": int(n),
                "seed": int(seed),
                "train_indices": [int(x) for x in idx.tolist()],
            }
        )
    return rows
