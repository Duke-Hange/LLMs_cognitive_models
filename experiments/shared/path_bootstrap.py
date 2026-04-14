"""
将 `experiments/` 加入 sys.path，并可选设置中文字体。
供各子目录脚本在 import 本包内 `shared.*` 之前调用。
"""

from __future__ import annotations

import sys
from pathlib import Path


def experiments_root() -> Path:
    """本文件位于 experiments/shared/，返回 experiments 目录。"""
    return Path(__file__).resolve().parent.parent


def ensure_experiments_on_path(*, setup_font: bool = False) -> Path:
    """
    确保 `experiments` 在 sys.path 首位（若尚未包含）。
    setup_font=True 时调用 `shared.visualization.setup_chinese_font()`（需 path 已就绪）。
    """
    exp = experiments_root()
    root_s = str(exp)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    if setup_font:
        from shared.visualization import setup_chinese_font

        setup_chinese_font()
    return exp
