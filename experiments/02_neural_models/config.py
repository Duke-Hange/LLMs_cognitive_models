"""
神经网络实验配置
与 01_symbolic_models_enhanced 数据与划分对齐，主输入为完整分布编码。
"""

from pathlib import Path

# 路径：本目录与 01 目录
_THIS_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _THIS_DIR.parent
_DIR_01 = _EXPERIMENTS_DIR / "01_symbolic_models_enhanced"

# 数据：使用 01 的标准化 JSON（或通过 standardizer 生成）
STANDARDIZED_JSON_PATH = _DIR_01 / "c13k_enhanced_standardized.json"

# 分布编码
MAX_OUTCOMES = 9  # Choices13k 中 B 最多约 9 个结果，单赌局编码长度 2*MAX_OUTCOMES

# 划分类型（与 01 一致）
SPLIT_TYPES = ["problem", "parameter_amb", "parameter_ev_extreme"]

# 随机种子（单次运行用；多种子时由 CLI --seeds 指定）
SEED = 42
SEEDS = [42]  # 默认多种子列表，供无 CLI 时使用

# 训练（与 Peterson et al. 参考一致：2000 epoch, 1e-3）
EPOCHS = 2000
BATCH_SIZE = 64
LR = 1e-3
EARLY_STOPPING_PATIENCE = 15
VAL_FRACTION = 0.1  # 从训练集中留 10% 作验证

# 数据量–性能曲线：训练集比例列表
TRAIN_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# 数据量曲线专用：与参考文献一致的 50 个比例点
DATA_QUANTITY_N_FRACTIONS = 50

# 输出
RESULTS_DIR = _THIS_DIR / "results"
CURVES_DIR = RESULTS_DIR / "curves"
