"""
共用的 train/test 索引划分（sklearn），供 03_LLM 与 05_program 薄封装复用。
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def sklearn_train_test_indices(
    n_rows: int,
    *,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_rows <= 1:
        raise ValueError("n_rows must be > 1")
    idx = np.arange(n_rows)
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=int(random_state),
        shuffle=True,
    )
    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )
