"""
experiments：从与 02 神经模块相同的 data.csv 构造符号模型所需的最小 standardized 结构。
符号模型拟合目标为 bRate（选择 Gamble B 的概率）；02 使用 y=1-bRate，跨模型比较时需注意标签空间。
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

# 与 experiments/02_neural_models/config.py 保持一致
CSV_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "data.csv"
TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 1017

_NEURAL_DIR = Path(__file__).resolve().parent.parent / "02_neural_models"
if str(_NEURAL_DIR) not in sys.path:
    sys.path.insert(0, str(_NEURAL_DIR))


def load_csv_standardized_and_targets(
    csv_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    读取 data.csv，解析 A/B 分布，构造与 EnhancedModelAdapter 兼容的 standardized_data；
    返回 (records, y) 其中 y 为 bRate 向量。
    """
    from data_loader import load_csv_data

    path = csv_path or CSV_DATA_PATH
    df = load_csv_data(path)
    n = len(df)
    standardized_data: List[Dict[str, Any]] = []
    for i in range(n):
        row = df.iloc[i]
        dist_a = row["A"]
        dist_b = row["B"]
        br = float(row["bRate"])
        standardized_data.append(
            {
                "metadata": {
                    "index": i,
                    "problem_id": -1,
                    "json_problem_id": -1,
                    "feedback_condition": "csv",
                    "block": 0,
                    "source": "data.csv",
                },
                "context": {
                    "features": {
                        "ev_diff": 0.0,
                        "Amb": 0,
                    },
                    "gamble_a": {"distribution": dist_a, "description": ""},
                    "gamble_b": {"distribution": dist_b, "description": ""},
                },
                "action": {"bRate": br},
            }
        )
    y = np.array([item["action"]["bRate"] for item in standardized_data], dtype=np.float64)
    return standardized_data, y
