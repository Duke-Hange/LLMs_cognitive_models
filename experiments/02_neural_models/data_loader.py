"""
数据加载与分布编码
从 01 的标准化数据加载，生成完整分布编码（每个选项内所有结果-概率对），并与 create_enhanced_splits 对齐。
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# 引入 01 的划分函数
_THIS_DIR = Path(__file__).resolve().parent
_DIR_01 = _THIS_DIR.parent / "01_symbolic_models_enhanced"
if str(_DIR_01) not in sys.path:
    sys.path.insert(0, str(_DIR_01))
from enhanced_data_standardization import create_enhanced_splits

try:
    from config import STANDARDIZED_JSON_PATH, MAX_OUTCOMES, SPLIT_TYPES
except ImportError:
    STANDARDIZED_JSON_PATH = _THIS_DIR.parent / "01_symbolic_models_enhanced" / "c13k_enhanced_standardized.json"
    MAX_OUTCOMES = 9
    SPLIT_TYPES = ["problem", "parameter_amb", "parameter_ev_extreme"]


def load_standardized_data(json_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """加载标准化数据（与 01 格式一致）。"""
    path = json_path or STANDARDIZED_JSON_PATH
    if not path.exists():
        raise FileNotFoundError(f"标准化数据未找到: {path}，请先在 01_symbolic_models_enhanced 中生成 c13k_enhanced_standardized.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def encode_distribution(distribution: List[List[float]], max_outcomes: int) -> np.ndarray:
    """
    将单赌局完整分布 [[p1,x1],[p2,x2],...] 编码为固定长度向量。
    顺序: p1, x1, p2, x2, ..., 不足用 0 填充，长度 2*max_outcomes。
    """
    out = np.zeros(2 * max_outcomes, dtype=np.float64)
    if not distribution:
        return out
    n = min(len(distribution), max_outcomes)
    for i in range(n):
        p, x = distribution[i][0], distribution[i][1]
        out[2 * i] = float(p)
        out[2 * i + 1] = float(x)
    return out


def build_distribution_encodings(
    standardized_data: List[Dict],
    max_outcomes: int = MAX_OUTCOMES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    为所有样本构建分布编码。
    Returns:
        enc_A: (n, 2*max_outcomes)
        enc_B: (n, 2*max_outcomes)
        enc_full: (n, 4*max_outcomes) = [enc_A | enc_B]
    """
    n = len(standardized_data)
    enc_A = np.zeros((n, 2 * max_outcomes), dtype=np.float64)
    enc_B = np.zeros((n, 2 * max_outcomes), dtype=np.float64)
    for i, item in enumerate(standardized_data):
        ctx = item["context"]
        dist_a = ctx["gamble_a"]["distribution"]
        dist_b = ctx["gamble_b"]["distribution"]
        enc_A[i] = encode_distribution(dist_a, max_outcomes)
        enc_B[i] = encode_distribution(dist_b, max_outcomes)
    enc_full = np.hstack([enc_A, enc_B])
    return enc_A, enc_B, enc_full


def get_target_vector(standardized_data: List[Dict]) -> np.ndarray:
    """bRate 目标向量 (n,)"""
    return np.array([item["action"]["bRate"] for item in standardized_data], dtype=np.float64)


def get_split_data(
    standardized_data: List[Dict],
    enc_A: np.ndarray,
    enc_B: np.ndarray,
    enc_full: np.ndarray,
    y: np.ndarray,
    split_type: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    按 split_type 划分，返回训练/测试的编码与标签。
    """
    train_idx, test_idx, split_info = create_enhanced_splits(
        standardized_data, split_type=split_type, **kwargs
    )
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "split_info": split_info,
        "enc_A_train": enc_A[train_idx],
        "enc_B_train": enc_B[train_idx],
        "enc_full_train": enc_full[train_idx],
        "y_train": y[train_idx],
        "enc_A_test": enc_A[test_idx],
        "enc_B_test": enc_B[test_idx],
        "enc_full_test": enc_full[test_idx],
        "y_test": y[test_idx],
    }


if __name__ == "__main__":
    data = load_standardized_data()
    print(f"加载样本数: {len(data)}")
    enc_A, enc_B, enc_full = build_distribution_encodings(data)
    y = get_target_vector(data)
    print(f"enc_A shape: {enc_A.shape}, enc_full shape: {enc_full.shape}, y shape: {y.shape}")
    for st in SPLIT_TYPES:
        out = get_split_data(data, enc_A, enc_B, enc_full, y, st)
        print(f"{st}: train {out['y_train'].shape[0]}, test {out['y_test'].shape[0]}, {out['split_info']['description']}")
