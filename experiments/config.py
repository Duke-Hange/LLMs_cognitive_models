"""
统一配置管理模块
统一管理项目的各项配置参数
"""
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent

# 数据目录
DATA_DIR = ROOT_DIR / 'data'
RESULTS_DIR = ROOT_DIR / 'results'
CURVES_DIR = RESULTS_DIR / 'learning_curves'  # 神经网络曲线目录

# 数据源配置
CSV_DATA_PATH = DATA_DIR / 'data.csv'

# 目标模式配置
TARGET_MODE = "one_minus_bRate"  # 选择 "one_minus_bRate" 或 "bRate"

# 划分配置
TEST_SPLIT_SIZE = 0.2
SPLIT_RANDOM_STATE = 1017
TEST_SIZE = 0.2  # 匹配原有的 TRAIN_TEST_SPLIT
VAL_FRACTION = 0.2  # 验证集比例

# 符号模型配置
SYMBOLIC_MAXITER = 5000

# 神经网络配置
EPOCHS = 200  # 训练轮数
NEURAL_EPOCHS = 100
LEARNING_RATE = 0.001  # 匹配LR
LR = 0.001  # 匹配训练脚本
BATCH_SIZE = 32
HIDDEN_DIM = 64
EARLY_STOPPING_PATIENCE = 10

# 绘图平滑配置
DATA_QUANTITY_SMOOTH_WINDOW = 5  # 用于数据量曲线平滑的窗口大小

# 随机种子配置
SEED = 42
SEEDS = [42, 123, 456]  # 默认随机种子列表

# 学习曲线配置
DEFAULT_TRAIN_SIZES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_FRACTIONS = 10
DATA_QUANTITY_N_FRACTIONS = 10

# MAX_OUTCOMES配置（与02_neural_models/config.py对齐）
MAX_OUTCOMES = 9  # 根据实际项目设置调整

# 编码配置
USE_RAW_ENCODING = False  # 根据需要调整