"""
04_comparison 配置
默认路径：01 符号模型增强 summary、02 神经网络 summary、输出目录。
"""

from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _THIS_DIR.parent

# 01 符号模型增强：results/enhanced_training/ 下最新 enhanced_models_summary_*.csv
DIR_01_ENHANCED = _EXPERIMENTS_DIR / "01_symbolic_models_enhanced" / "results" / "enhanced_training"
SYMBOLIC_PATTERN = "enhanced_models_summary_*.csv"

# 02 神经网络：results/ 下最新 neural_models_summary_*.csv
DIR_02_NEURAL = _EXPERIMENTS_DIR / "02_neural_models" / "results"
NEURAL_PATTERN = "neural_models_summary_*.csv"

# 03 LLM：results/ 下最新 llm_models_summary_*.csv
DIR_03_LLM = _EXPERIMENTS_DIR / "03_LLM" / "results"
LLM_PATTERN = "llm_models_summary_*.csv"

# 输出目录
OUTPUT_DIR = _THIS_DIR / "output"


def get_latest_symbolic_path() -> Path:
    """返回 01 目录下最新的符号 summary CSV"""
    if not DIR_01_ENHANCED.exists():
        raise FileNotFoundError(f"未找到目录: {DIR_01_ENHANCED}，请先运行 01 增强实验")
    files = list(DIR_01_ENHANCED.glob(SYMBOLIC_PATTERN))
    if not files:
        raise FileNotFoundError(f"未找到 {SYMBOLIC_PATTERN} 于 {DIR_01_ENHANCED}")
    return max(files, key=lambda p: p.stat().st_mtime)


def get_latest_neural_path() -> Path:
    """返回 02 目录下最新的神经 summary CSV"""
    if not DIR_02_NEURAL.exists():
        raise FileNotFoundError(f"未找到目录: {DIR_02_NEURAL}，请先运行 02 神经网络实验")
    files = list(DIR_02_NEURAL.glob(NEURAL_PATTERN))
    if not files:
        raise FileNotFoundError(f"未找到 {NEURAL_PATTERN} 于 {DIR_02_NEURAL}")
    return max(files, key=lambda p: p.stat().st_mtime)


def get_latest_llm_path() -> Path:
    """返回 03 目录下最新的 LLM summary CSV"""
    if not DIR_03_LLM.exists():
        raise FileNotFoundError(f"未找到目录: {DIR_03_LLM}，请先运行 03 LLM 实验")
    files = list(DIR_03_LLM.glob(LLM_PATTERN))
    if not files:
        raise FileNotFoundError(f"未找到 {LLM_PATTERN} 于 {DIR_03_LLM}")
    return max(files, key=lambda p: p.stat().st_mtime)
