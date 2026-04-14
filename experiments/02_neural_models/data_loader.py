"""
experiments 数据加载与分布编码（CSV 版本）
读取 data.csv，解析 A/B 分布；标签由 config.TARGET_MODE 决定（默认 y = 1 - bRate）。
"""

import ast
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from config import CSV_DATA_PATH, MAX_OUTCOMES, TARGET_MODE

REQUIRED_COLUMNS = ("bRate", "A", "B")


def load_csv_data(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """加载并解析 data.csv。"""
    path = csv_path or CSV_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"CSV 数据未找到: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")

    def parse_distribution(value: str) -> list:
        try:
            dist = ast.literal_eval(value)
        except Exception as e:
            raise ValueError(f"A/B 解析失败: {value}") from e
        if not isinstance(dist, list):
            raise ValueError(f"A/B 格式错误，应为 list: {dist}")
        return dist

    df["A"] = df["A"].apply(parse_distribution)
    df["B"] = df["B"].apply(parse_distribution)
    return df


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
    df: pd.DataFrame,
    max_outcomes: int = MAX_OUTCOMES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    为所有样本构建分布编码。
    Returns:
        enc_A: (n, 2*max_outcomes)
        enc_B: (n, 2*max_outcomes)
        enc_full: (n, 4*max_outcomes) = [enc_A | enc_B]
    """
    n = len(df)
    enc_A = np.zeros((n, 2 * max_outcomes), dtype=np.float64)
    enc_B = np.zeros((n, 2 * max_outcomes), dtype=np.float64)
    truncated = 0
    for i, row in df.iterrows():
        dist_a = row["A"]
        dist_b = row["B"]
        if len(dist_a) > max_outcomes or len(dist_b) > max_outcomes:
            truncated += 1
        enc_A[i] = encode_distribution(dist_a, max_outcomes)
        enc_B[i] = encode_distribution(dist_b, max_outcomes)
    if truncated > 0:
        print(f"警告: {truncated} 条样本超出 MAX_OUTCOMES={max_outcomes}，已截断。")
    enc_full = np.hstack([enc_A, enc_B])
    return enc_A, enc_B, enc_full


def get_target_vector(df: pd.DataFrame, target_mode: str = TARGET_MODE) -> np.ndarray:
    """目标向量。默认 one_minus_bRate。"""
    b_rate = df["bRate"].to_numpy(dtype=np.float64)
    if target_mode == "one_minus_bRate":
        return 1.0 - b_rate
    if target_mode == "bRate":
        return b_rate
    raise ValueError(f"未知 target_mode: {target_mode}")


if __name__ == "__main__":
    data = load_csv_data()
    print(f"加载样本数: {len(data)}")
    enc_A, enc_B, enc_full = build_distribution_encodings(data)
    y = get_target_vector(data)
    print(f"enc_A shape: {enc_A.shape}, enc_full shape: {enc_full.shape}, y shape: {y.shape}")
