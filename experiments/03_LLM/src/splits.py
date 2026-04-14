from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.splits_core import sklearn_train_test_indices


@dataclass
class TrainTestSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    random_state: int
    test_size: float


def build_train_test_split(
    n_rows: int,
    seed: int,
    test_size: float = 0.2,
    base_random_state: int = 1017,
) -> TrainTestSplit:
    random_state = int(base_random_state + seed)
    train_idx, test_idx = sklearn_train_test_indices(
        n_rows, test_size=test_size, random_state=random_state
    )
    return TrainTestSplit(
        train_idx=train_idx,
        test_idx=test_idx,
        random_state=random_state,
        test_size=float(test_size),
    )
