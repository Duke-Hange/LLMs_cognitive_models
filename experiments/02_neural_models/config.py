"""
神经网络实验配置（experiments）
数据源为 CSV（`experiments/data/data.csv`），标签默认 y = 1 - bRate。
划分方式为单一 train_test_split + learning curve（不使用 split_types）。
"""

from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _THIS_DIR.parent  # experiments
_PROJECT_ROOT = _EXPERIMENTS_DIR.parent

CSV_DATA_PATH = _EXPERIMENTS_DIR / "data" / "data.csv"

# 标签：固定与参考代码一致
TARGET_MODE = "one_minus_bRate"  # y = 1 - bRate

# 与早期参考实现一致：不对分布编码做 StandardScaler，直接使用原始编码训练与评估
USE_RAW_ENCODING = True

# 分布编码
MAX_OUTCOMES = 9  # Choices13k 中 B 最多约 9 个结果，单赌局编码长度 2*MAX_OUTCOMES

# 单一划分参数
TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 1017

# 随机种子（单次运行用；多种子时由 CLI --seeds 指定）
SEED = 42
SEEDS = [42]  # 默认多种子列表，供无 CLI 时使用

# 训练（与可参考代码 anwser/results.ipynb 一致：1000 epoch, 1e-3）
EPOCHS = 1000
BATCH_SIZE = 64
LR = 1e-3
# 与可参考代码对齐，学习曲线需更长耐心
EARLY_STOPPING_PATIENCE = 100
VAL_FRACTION = 0.1  # 从训练集中留 10% 作验证

# 数据量曲线专用：与参考文献一致的 50 个比例点（比例序列由 DATA_QUANTITY_N_FRACTIONS 生成 1/n..1.0）
DATA_QUANTITY_N_FRACTIONS = 50
# 数据量曲线绘图平滑窗口（与 train.run_data_quantity_curve、run_learning_curve 共用）
DATA_QUANTITY_SMOOTH_WINDOW = 5

# 输出
RESULTS_DIR = _THIS_DIR / "results"
CURVES_DIR = RESULTS_DIR / "curves"
