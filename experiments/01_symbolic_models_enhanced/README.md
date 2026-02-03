# Choices13k 增强符号模型实验

## 🎯 实验目标
**基于多结果分布重构符号模型数据管道**，解决CSV特征过度简化的问题，提高模型准确性和泛化能力。

### 核心科学问题
1. **特征准确性问题**：CSV中的`Hb`、`pHb`、`Lb`等特征是否过度简化真实的多结果分布？
2. **模型改进潜力**：使用完整分布特征是否能提高符号模型的预测性能？
3. **泛化能力验证**：增强模型在参数空间变化时是否具有更好的泛化能力？

## 📊 数据特征分析

### 多结果分布发现
通过分析JSON原始分布数据，我们发现：

```
Gamble B结果分布统计（前1000样本）：
- 平均结果数：3.78个
- 标准差：2.31
- 范围：[1, 9]个结果
- 分布：47.9%为2结果，52.1%为多结果（>2个结果）
```

### CSV特征简化问题
CSV中的特征严重简化了真实分布：
- **简化特征**：`EV_simple = pHb*Hb + (1-pHb)*Lb`
- **真实分布**：`EV_real = Σ(p_i * x_i)`
- **差异**：期望值计算存在显著差异，影响模型准确性

## 🏗️ 增强数据管道

### 1. 增强数据标准化 (`enhanced_data_standardization.py`)
**直接从JSON原始分布提取丰富特征**

#### 特征工程（50+维）
- **基本统计**：期望值、方差、标准差、最小值、最大值、范围
- **高阶矩**：偏度、峰度、超额峰度
- **分位数特征**：q10、q25、q50（中位数）、q75、q90
- **不确定性度量**：熵、结果数量
- **风险度量**：下行方差、下行标准差、损失概率、收益概率
- **对比特征**：EV_diff、方差比、偏度差异、熵差异

#### 数据划分（三种科学策略）
1. **Problem-Split**：80/20问题划分（基准）
2. **Parameter-Amb-Split**：训练集=非模糊样本(Amb=0)，测试集=模糊样本(Amb=1)
3. **Parameter-EV-Extreme-Split**：训练集=EV_diff < 25th百分位数，测试集=EV_diff > 75th百分位数

### 2. 增强符号模型 (`enhanced_symbolic_models.py`)
**完全支持多结果分布的决策理论模型**

#### 模型架构
```
EnhancedSymbolicModel (基类)
├── EnhancedExpectedValueModel (EV)
├── EnhancedExpectedUtilityModel (EU)
├── EnhancedProspectTheory3PModel (PT3P)
└── EnhancedProspectTheory5PModel (PT5P)
```

#### 关键创新
- **分布感知计算**：直接从分布列表计算价值 `EV = Σ(p_i * x_i)`
- **适配器模式**：`EnhancedModelAdapter`连接标准化数据与增强模型
- **批量预测**：支持从分布列表直接批量预测

### 3. 增强训练框架 (`train_enhanced_models.py`)
**完整的增强实验管道**

#### 功能特性
- **数据加载**：自动加载和标准化增强数据
- **模型训练**：4种增强模型在3种划分上训练
- **评估指标**：MSE、MAE、RMSE、R²、相关性
- **结果管理**：自动保存JSON、CSV、Excel格式结果
- **性能可视化**：预测vs真实值散点图生成

## 🚀 快速开始

### 环境要求
```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

### 运行完整增强实验
```bash
# 1. 运行完整增强实验
python train_enhanced_models.py

# 2. 运行快速测试（500样本）
python quick_train_test.py

# 3. 测试增强数据标准化
python test_enhanced_standardization.py

# 4. 测试集成流程
python quick_integration_test.py
```

### 代码示例
```python
# 初始化增强标准化器
from enhanced_data_standardization import EnhancedChoices13kStandardizer
standardizer = EnhancedChoices13kStandardizer(
    selections_path="数据集/choices13k-main/c13k_selections.csv",
    problems_path="数据集/choices13k-main/c13k_problems.json"
)

# 标准化数据
standardized_data = standardizer.standardize_all()

# 创建增强模型
from enhanced_symbolic_models import create_enhanced_model, EnhancedModelAdapter
model = create_enhanced_model('ev')  # 增强EV模型
adapter = EnhancedModelAdapter(model)

# 训练和预测
adapter.fit_from_standardized(train_data, y_train)
predictions = adapter.predict_from_standardized(test_data)
```

## 📈 预期改进

### 1. 准确性提升
- **更准确的特征表示**：使用真实分布而非简化特征
- **消除计算偏差**：正确的期望值计算 `EV = Σ(p_i * x_i)`
- **丰富的行为洞察**：熵、偏度、峰度等新特征提供更多决策心理学洞察

### 2. 泛化能力验证
- **科学的划分策略**：基于准确特征的数据划分更科学
- **可比的结果体系**：与原始实验相同的三种划分，确保可比性
- **增强的行为经济学检验**：更准确地验证期望值、期望效用、前景理论

### 3. 方法学贡献
- **激进重构方案**：直接从JSON原始分布重建数据管道
- **分布感知模型**：符号模型首次支持多结果分布计算
- **特征工程丰富化**：50+维统计特征提供更全面的决策问题表示

## 🔬 科学意义

### 理论贡献
1. **准确决策理论检验**：使用真实分布验证期望值、期望效用、前景理论
2. **消除简化偏差**：避免CSV特征对Gamble B的过度简化
3. **增强行为洞察**：熵、偏度、峰度等特征提供新的行为经济学视角

### 方法学创新
1. **激进重构方案**：直接从JSON原始分布重建数据管道
2. **分布感知模型**：符号模型首次支持多结果分布计算
3. **特征工程丰富化**：50+维统计特征提供更全面的决策问题表示

### 实验严谨性
1. **科学的数据划分**：基于准确`ev_diff`和`Amb`特征的划分策略
2. **完整的评估框架**：与原始实验相同的三种划分，确保可比性
3. **系统的结果管理**：标准化结果保存和比较机制

## 📁 文件结构

```
experiments/01_symbolic_models_enhanced/
├── README.md                          # 本文件
├── enhanced_data_standardization.py   # 增强数据标准化器（核心）
├── enhanced_symbolic_models.py        # 增强符号模型（核心）
├── train_enhanced_models.py           # 增强训练框架（核心）
├── test_enhanced_standardization.py   # 标准化测试
├── quick_integration_test.py          # 集成测试
├── quick_train_test.py                # 训练测试
└── results/enhanced_training/         # 增强实验结果目录
```

## 📊 与原始实验对比

| 维度 | 原始符号模型实验 | 增强符号模型实验 |
|------|------------------|------------------|
| **数据源** | CSV简化特征 | JSON原始分布 |
| **特征维度** | 17维简化特征 | 50+维增强特征 |
| **分布支持** | 仅2结果分布 | 多结果分布（平均3.78个） |
| **期望值计算** | `EV = p_high*high + (1-p_high)*low` | `EV = Σ(p_i * x_i)` |
| **模型架构** | 原始符号模型 | 增强符号模型（分布感知） |
| **训练管道** | 原始训练脚本 | 增强训练框架 |
| **科学问题** | 参数空间泛化 | 特征准确性+参数空间泛化 |

## 🔍 深入分析方向

### 立即执行
```bash
# 运行完整增强实验并生成结果
python train_enhanced_models.py
```

### 分析建议
1. **与原始模型系统比较**：加载原始实验结果，进行MSE/R²对比分析
2. **特征重要性分析**：分析方差、偏度、熵等新特征对预测的贡献
3. **参数稳定性验证**：实现K折交叉验证，检查增强模型参数估计稳定性
4. **可视化分析**：多结果分布的可视化展示与解释

### 扩展研究
1. **混合模型策略**：增强特征 + 原始特征的组合模型
2. **分布复杂度分析**：结果数量与模型性能的关系研究
3. **认知负荷假设**：多结果分布是否增加决策难度的实证检验

## 📝 引用

如果使用本增强实验代码或结果，请引用：

```
@software{choices13k_enhanced_2026,
  title = {Enhanced Symbolic Model Experiment with Multi-Outcome Distribution Support},
  author = {Choices13k Research Team},
  year = {2026},
  url = {https://github.com/your-repo/choices13k},
  note = {Based on original symbolic model experiment with enhanced data pipeline}
}
```

## 📄 许可证

MIT License

## 📧 联系方式

如有问题或建议，请联系实验负责人。

---

**实验完成时间**: 2026-01-20  
**实验版本**: v2.0（增强版）  
**项目状态**: 代码就绪，可运行完整实验  
**独立性**: 本目录为唯一符号模型实验目录，可独立运行；原始基线结果已内嵌于 `results/comparison/original_baseline_summary.csv`