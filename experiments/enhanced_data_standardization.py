"""
增强版数据标准化模块
直接从JSON原始分布数据提取丰富的特征，支持多结果分布
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import scipy.stats as stats
from scipy import integrate
import warnings
warnings.filterwarnings('ignore')


class EnhancedChoices13kStandardizer:
    """增强版Choices13k标准化器 - 直接从JSON提取完整分布特征"""
    
    def __init__(self, selections_path: str, problems_path: str):
        """
        初始化标准化器
        
        Args:
            selections_path: c13k_selections.csv 路径
            problems_path: c13k_problems.json 路径
        """
        self.selections_path = selections_path
        self.problems_path = problems_path
        self.df = None
        self.problems = None
        self.standardized_data = None
        self.enhanced_features = None
        
    def load_and_merge_data(self) -> pd.DataFrame:
        """
        加载并合并JSON和CSV数据
        
        Returns:
            合并后的DataFrame
        """
        print("加载Choices13k数据...")
        
        # 加载CSV行为数据
        df_selections = pd.read_csv(self.selections_path)
        print(f"加载CSV数据: {len(df_selections)} 条记录")
        
        # 加载JSON问题定义
        with open(self.problems_path, 'r') as f:
            self.problems = json.load(f)
        print(f"加载JSON数据: {len(self.problems)} 个问题")
        
        # 将problems转换为DataFrame
        problems_list = []
        for prob_id, gamble_data in self.problems.items():
            problems_list.append({
                'problem_id': int(prob_id),
                'gamble_a': gamble_data['A'],
                'gamble_b': gamble_data['B']
            })
        
        df_problems = pd.DataFrame(problems_list)
        
        # 合并数据 (注意: CSV中的Problem列是1-based, JSON键是0-based)
        df_selections['json_problem_id'] = df_selections['Problem'] - 1
        
        # 合并数据
        df_merged = df_selections.merge(
            df_problems,
            left_on='json_problem_id',
            right_on='problem_id',
            how='left'
        )
        
        print(f"合并后数据: {len(df_merged)} 条记录")
        self.df = df_merged
        return df_merged
    
    def calculate_distribution_stats(self, distribution: List[List[float]]) -> Dict[str, float]:
        """
        计算分布的统计特征
        
        Args:
            distribution: [[概率, 金额], ...] 列表
            
        Returns:
            统计特征字典
        """
        if not distribution:
            return {}
        
        # 分离概率和结果
        probabilities = np.array([p for p, _ in distribution])
        outcomes = np.array([x for _, x in distribution])
        
        # 基本统计
        expected_value = np.sum(probabilities * outcomes)
        variance = np.sum(probabilities * (outcomes - expected_value) ** 2)
        std_dev = np.sqrt(variance) if variance > 0 else 0
        
        # 更高阶矩（需要足够的结果）
        stats_dict = {
            'ev': float(expected_value),
            'variance': float(variance),
            'std_dev': float(std_dev),
            'num_outcomes': len(distribution),
            'min_outcome': float(np.min(outcomes)),
            'max_outcome': float(np.max(outcomes)),
            'range': float(np.max(outcomes) - np.min(outcomes)),
            'mean_abs_dev': float(np.sum(probabilities * np.abs(outcomes - expected_value))),
        }
        
        # 偏度和峰度（需要至少3个不同结果）
        if len(outcomes) >= 3 and variance > 1e-10:
            # 中心矩
            centered = outcomes - expected_value
            skewness = np.sum(probabilities * centered ** 3) / (std_dev ** 3) if std_dev > 0 else 0
            kurtosis = np.sum(probabilities * centered ** 4) / (variance ** 2) if variance > 0 else 0
            
            stats_dict.update({
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'excess_kurtosis': float(kurtosis - 3),  # 超额峰度
            })
        
        # 分位数特征
        sorted_indices = np.argsort(outcomes)
        sorted_outcomes = outcomes[sorted_indices]
        sorted_probs = probabilities[sorted_indices]
        
        # 累积分布函数
        cdf = np.cumsum(sorted_probs)
        
        # 分位数
        quantiles = {}
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            idx = np.searchsorted(cdf, q)
            if idx < len(sorted_outcomes):
                quantiles[f'q{q*100:.0f}'] = float(sorted_outcomes[idx])
            else:
                quantiles[f'q{q*100:.0f}'] = float(sorted_outcomes[-1])
        
        stats_dict.update(quantiles)
        
        # 熵（分布不确定性）
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        stats_dict['entropy'] = float(entropy)
        
        # 风险度量
        # 下行风险（半方差）
        downside_mask = outcomes < expected_value
        if np.any(downside_mask):
            downside_variance = np.sum(probabilities[downside_mask] * (outcomes[downside_mask] - expected_value) ** 2)
            stats_dict['downside_variance'] = float(downside_variance)
            stats_dict['downside_std'] = float(np.sqrt(downside_variance))
        
        # 损失概率
        loss_prob = np.sum(probabilities[outcomes < 0])
        stats_dict['loss_probability'] = float(loss_prob)
        
        # 收益概率
        gain_prob = np.sum(probabilities[outcomes > 0])
        stats_dict['gain_probability'] = float(gain_prob)
        
        # 期望损失/收益
        losses = outcomes[outcomes < 0]
        if len(losses) > 0:
            loss_probs = probabilities[outcomes < 0]
            expected_loss = np.sum(loss_probs * losses) / loss_prob if loss_prob > 0 else 0
            stats_dict['expected_loss'] = float(expected_loss)
        
        gains = outcomes[outcomes > 0]
        if len(gains) > 0:
            gain_probs = probabilities[outcomes > 0]
            expected_gain = np.sum(gain_probs * gains) / gain_prob if gain_prob > 0 else 0
            stats_dict['expected_gain'] = float(expected_gain)
        
        # 损失厌恶指标
        if 'expected_loss' in stats_dict and 'expected_gain' in stats_dict:
            if stats_dict['expected_loss'] != 0:
                loss_aversion_ratio = abs(stats_dict['expected_gain'] / stats_dict['expected_loss'])
                stats_dict['loss_aversion_ratio'] = float(loss_aversion_ratio)
        
        return stats_dict
    
    def calculate_gamble_features(self, gamble_a: List[List[float]], gamble_b: List[List[float]]) -> Dict[str, Any]:
        """
        计算赌博问题的完整特征
        
        Args:
            gamble_a: Gamble A的分布
            gamble_b: Gamble B的分布
            
        Returns:
            特征字典
        """
        # 计算每个分布的统计特征
        stats_a = self.calculate_distribution_stats(gamble_a)
        stats_b = self.calculate_distribution_stats(gamble_b)
        
        # 组合特征
        features = {}
        
        # 原始分布（用于符号模型）
        features['gamble_a_distribution'] = gamble_a
        features['gamble_b_distribution'] = gamble_b
        
        # 基本统计特征
        for stat_name, stat_value in stats_a.items():
            features[f'a_{stat_name}'] = stat_value
        
        for stat_name, stat_value in stats_b.items():
            features[f'b_{stat_name}'] = stat_value
        
        # 对比特征
        if 'ev' in stats_a and 'ev' in stats_b:
            ev_diff = stats_b['ev'] - stats_a['ev']
            features['ev_diff'] = ev_diff
            features['ev_diff_abs'] = abs(ev_diff)
            features['ev_ratio'] = stats_b['ev'] / stats_a['ev'] if stats_a['ev'] != 0 else 0
        
        # 风险对比
        if 'variance' in stats_a and 'variance' in stats_b:
            features['variance_diff'] = stats_b['variance'] - stats_a['variance']
            features['variance_ratio'] = stats_b['variance'] / stats_a['variance'] if stats_a['variance'] > 0 else 0
        
        # 偏度对比
        if 'skewness' in stats_a and 'skewness' in stats_b:
            features['skewness_diff'] = stats_b['skewness'] - stats_a['skewness']
        
        # 复杂程度对比
        features['num_outcomes_diff'] = stats_b.get('num_outcomes', 0) - stats_a.get('num_outcomes', 0)
        features['entropy_diff'] = stats_b.get('entropy', 0) - stats_a.get('entropy', 0)
        
        # 范围对比
        if 'range' in stats_a and 'range' in stats_b:
            features['range_diff'] = stats_b['range'] - stats_a['range']
            features['range_ratio'] = stats_b['range'] / stats_a['range'] if stats_a['range'] > 0 else 0
        
        # 损失厌恶特征
        if 'loss_probability' in stats_a and 'loss_probability' in stats_b:
            features['loss_prob_diff'] = stats_b['loss_probability'] - stats_a['loss_probability']
        
        # 保留原始CSV中的关键特征（用于兼容性）
        # 这些将在后续从CSV列中提取
        
        return features
    
    def calculate_enhanced_features(self, row: pd.Series) -> Dict[str, Any]:
        """
        计算增强特征（整合CSV中的元数据）
        
        Args:
            row: 数据行
            
        Returns:
            增强特征字典
        """
        # 获取赌博分布
        gamble_a = row['gamble_a']
        gamble_b = row['gamble_b']
        
        # 计算分布特征
        distribution_features = self.calculate_gamble_features(gamble_a, gamble_b)
        
        # 添加CSV中的元数据特征
        metadata_features = {
            # 原始CSV特征
            'Ha': float(row['Ha']),
            'pHa': float(row['pHa']),
            'La': float(row['La']),
            'Hb': float(row['Hb']),
            'pHb': float(row['pHb']),
            'Lb': float(row['Lb']),
            'LotShapeB': int(row['LotShapeB']),
            'LotNumB': int(row['LotNumB']),
            'Amb': bool(row['Amb']),
            'Corr': int(row['Corr']),
            'feedback': bool(row['Feedback']),
            'block': int(row['Block']),
            'n_subjects': int(row['n']),
            
            # 行为数据
            'bRate': float(row['bRate']),
            'bRate_std': float(row['bRate_std']),
            
            # 问题ID
            'problem_id': int(row['Problem']),
            'json_problem_id': int(row['json_problem_id']),
        }
        
        # 合并特征
        all_features = {**distribution_features, **metadata_features}
        
        return all_features
    
    def generate_gamble_description(self, distribution: List[List[float]], gamble_name: str = "Gamble") -> str:
        """
        生成赌博选项的自然语言描述
        
        Args:
            distribution: 赌博分布
            gamble_name: 选项名称
            
        Returns:
            自然语言描述
        """
        # 按概率排序（从高到低）
        sorted_dist = sorted(distribution, key=lambda x: x[0], reverse=True)
        
        outcomes = []
        for p, outcome in sorted_dist:
            if p > 1e-10:  # 忽略概率极小的结果
                outcomes.append(f"以概率 {p:.3f} 获得 {outcome} 点")
        
        description = f"{gamble_name}:\n" + "\n".join(f"  - {outcome}" for outcome in outcomes)
        return description
    
    def standardize_row(self, idx: int) -> Dict[str, Any]:
        """
        标准化单行数据
        
        Args:
            idx: 行索引
            
        Returns:
            标准化后的数据字典
        """
        row = self.df.iloc[idx]
        
        # 计算增强特征
        features = self.calculate_enhanced_features(row)
        
        # 生成自然语言描述
        gamble_a_desc = self.generate_gamble_description(row['gamble_a'], "Gamble A")
        gamble_b_desc = self.generate_gamble_description(row['gamble_b'], "Gamble B")
        
        # 构建标准化格式
        standardized = {
            'context': {
                'description': f"选择 {gamble_a_desc} 或 {gamble_b_desc}",
                'gamble_a': {
                    'distribution': row['gamble_a'],
                    'description': gamble_a_desc,
                    'stats': {k:v for k,v in features.items() if k.startswith('a_')}
                },
                'gamble_b': {
                    'distribution': row['gamble_b'],
                    'description': gamble_b_desc,
                    'stats': {k:v for k,v in features.items() if k.startswith('b_')}
                },
                'features': features,
                'feedback': bool(row['Feedback']),
                'block': int(row['Block'])
            },
            'action': {
                'bRate': float(row['bRate']),
                'bRate_std': float(row['bRate_std']),
                'n_subjects': int(row['n'])
            },
            'metadata': {
                'problem_id': int(row['Problem']),
                'json_problem_id': int(row['json_problem_id']),
                'feedback_condition': bool(row['Feedback']),
                'block': int(row['Block']),
                'dataset': 'choices13k_enhanced',
                'index': idx
            },
            'raw_features': {
                'Ha': float(row['Ha']),
                'pHa': float(row['pHa']),
                'La': float(row['La']),
                'Hb': float(row['Hb']),
                'pHb': float(row['pHb']),
                'Lb': float(row['Lb']),
                'LotShapeB': int(row['LotShapeB']),
                'LotNumB': int(row['LotNumB']),
                'Amb': bool(row['Amb']),
                'Corr': int(row['Corr']),
            }
        }
        
        return standardized
    
    def standardize_all(self, save_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        标准化所有数据
        
        Args:
            save_path: 保存路径（可选）
            
        Returns:
            标准化后的数据列表
        """
        if self.df is None:
            self.load_and_merge_data()
        
        print("开始标准化数据...")
        standardized_data = []
        
        for idx in range(len(self.df)):
            try:
                standardized = self.standardize_row(idx)
                standardized_data.append(standardized)
                
                # 进度显示
                if (idx + 1) % 1000 == 0:
                    print(f"  已处理 {idx + 1}/{len(self.df)} 条记录")
                    
            except Exception as e:
                print(f"处理第 {idx} 行时出错: {e}")
                continue
        
        self.standardized_data = standardized_data
        print(f"标准化完成: {len(standardized_data)} 条记录")
        
        # 保存到文件
        if save_path:
            # 转换为可序列化的Python原生类型
            serializable_data = self._convert_to_serializable(standardized_data)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            print(f"已保存到: {save_path}")
        
        return standardized_data
    
    def _convert_to_serializable(self, data):
        """将数据中的numpy类型转换为Python原生类型"""
        if isinstance(data, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(data)
        elif isinstance(data, (np.floating, np.float64, np.float32, np.float16)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.bool_):
            return bool(data)
        elif isinstance(data, dict):
            return {key: self._convert_to_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_to_serializable(item) for item in data]
        else:
            return data
    
    def get_enhanced_feature_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        获取增强特征矩阵
        
        Returns:
            (特征矩阵, 特征名称列表)
        """
        if self.standardized_data is None:
            self.standardize_all()
        
        # 收集所有特征键（从第一行获取）
        if not self.standardized_data:
            return np.array([]), []
        
        first_features = self.standardized_data[0]['context']['features']
        
        # 选择数值特征（排除分布等复杂类型）
        numeric_features = []
        for key, value in first_features.items():
            if isinstance(value, (int, float, bool, np.number)):
                numeric_features.append(key)
        
        # 构建特征矩阵
        features_list = []
        for item in self.standardized_data:
            feat = item['context']['features']
            row = [feat.get(key, 0) for key in numeric_features]
            features_list.append(row)
        
        self.enhanced_features = numeric_features
        return np.array(features_list), numeric_features
    
    def get_target_vector(self) -> np.ndarray:
        """
        获取目标向量（bRate）
        
        Returns:
            目标向量 [n_samples]
        """
        if self.standardized_data is None:
            self.standardize_all()
        
        return np.array([item['action']['bRate'] for item in self.standardized_data])
    
    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        获取特征统计信息
        
        Returns:
            特征统计字典
        """
        if self.enhanced_features is None:
            self.get_enhanced_feature_matrix()
        
        X, feature_names = self.get_enhanced_feature_matrix()
        
        stats = {}
        for i, feat_name in enumerate(feature_names):
            feat_values = X[:, i]
            stats[feat_name] = {
                'mean': float(np.mean(feat_values)),
                'std': float(np.std(feat_values)),
                'min': float(np.min(feat_values)),
                'max': float(np.max(feat_values)),
                'median': float(np.median(feat_values)),
                'q25': float(np.percentile(feat_values, 25)),
                'q75': float(np.percentile(feat_values, 75)),
            }
        
        return stats


# 与 create_enhanced_splits 支持的划分类型一致；02 config、01 run_learning_curve 等须与此保持一致
SPLIT_TYPES = ["problem", "parameter_amb", "parameter_ev_extreme"]


def create_enhanced_splits(standardized_data: List[Dict], 
                          split_type: str = 'problem',
                          **kwargs) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """
    创建增强数据划分
    
    Args:
        standardized_data: 标准化数据
        split_type: 划分类型 ('problem', 'parameter_amb', 'parameter_ev_extreme', 'train_test')
        **kwargs: 划分参数；train_test 时传 test_size、random_state（与 experiments/02 神经模块一致）
        
    Returns:
        (训练索引, 测试索引, 划分信息)
    """
    if split_type == 'train_test':
        from sklearn.model_selection import train_test_split
        n = len(standardized_data)
        idx = np.arange(n)
        tr, te = train_test_split(
            idx,
            test_size=kwargs.get('test_size', 0.2),
            random_state=kwargs.get('random_state', 1017),
            shuffle=True,
        )
        train_indices = tr.tolist()
        test_indices = te.tolist()
        train_b = np.array([standardized_data[i]['action']['bRate'] for i in train_indices])
        test_b = np.array([standardized_data[i]['action']['bRate'] for i in test_indices])
        split_info = {
            'split_type': split_type,
            'description': (
                f"单一 train_test_split（与 experiments/02 一致），"
                f"test_size={kwargs.get('test_size', 0.2)}, random_state={kwargs.get('random_state', 1017)}"
            ),
            'train_size': len(train_indices),
            'test_size': len(test_indices),
            'train_bRate_mean': float(np.mean(train_b)),
            'test_bRate_mean': float(np.mean(test_b)),
            'parameters': kwargs,
        }
        return train_indices, test_indices, split_info

    # 创建DataFrame用于划分
    data_list = []
    for item in standardized_data:
        features = item['context']['features']
        data_list.append({
            'index': item['metadata']['index'],
            'problem_id': item['metadata']['problem_id'],
            'json_problem_id': item['metadata']['json_problem_id'],
            'feedback': item['metadata']['feedback_condition'],
            'block': item['metadata']['block'],
            'bRate': item['action']['bRate'],
            'ev_diff': features.get('ev_diff', 0),
            'Amb': features.get('Amb', False),
        })
    
    df = pd.DataFrame(data_list)
    
    if split_type == 'problem':
        # Problem-Split: 按问题ID划分
        from sklearn.model_selection import train_test_split
        
        problem_ids = df['problem_id'].unique()
        train_problems, test_problems = train_test_split(
            problem_ids,
            test_size=kwargs.get('test_size', 0.2),
            random_state=kwargs.get('random_state', 42)
        )
        
        train_mask = df['problem_id'].isin(train_problems)
        test_mask = df['problem_id'].isin(test_problems)
        
        description = f"训练集: {len(train_problems)}个问题, 测试集: {len(test_problems)}个问题"
        
    elif split_type == 'parameter_amb':
        # Parameter-Amb-Split: 按模糊性划分
        train_mask = df['Amb'] == 0  # 非模糊样本
        test_mask = df['Amb'] == 1   # 模糊样本
        
        description = f"训练集: 非模糊样本(Amb=0), 测试集: 模糊样本(Amb=1)"
        
    elif split_type == 'parameter_ev_extreme':
        # Parameter-EV-Extreme-Split: 按期望值极端划分
        ev_diff_values = df['ev_diff']
        q1 = ev_diff_values.quantile(0.25)
        q3 = ev_diff_values.quantile(0.75)
        
        train_mask = ev_diff_values < q1    # EV_diff < 25th百分位数
        test_mask = ev_diff_values > q3     # EV_diff > 75th百分位数
        
        description = f"训练集: EV_diff < 25th百分位数({q1:.2f}), 测试集: EV_diff > 75th百分位数({q3:.2f})"
        
    else:
        raise ValueError(f"未知的划分类型: {split_type}")
    
    train_indices = df[train_mask]['index'].tolist()
    test_indices = df[test_mask]['index'].tolist()
    
    split_info = {
        'split_type': split_type,
        'description': description,
        'train_size': len(train_indices),
        'test_size': len(test_indices),
        'train_bRate_mean': float(df[train_mask]['bRate'].mean()),
        'test_bRate_mean': float(df[test_mask]['bRate'].mean()),
        'parameters': kwargs
    }
    
    return train_indices, test_indices, split_info


# 使用示例
if __name__ == "__main__":
    # 基于脚本位置解析项目根目录，与运行时的当前工作目录无关
    _script_dir = Path(__file__).resolve().parent
    _project_root = _script_dir.parent.parent
    _data_dir = _project_root / "数据集" / "choices13k-main"

    # 初始化增强标准化器
    standardizer = EnhancedChoices13kStandardizer(
        selections_path=str(_data_dir / "c13k_selections.csv"),
        problems_path=str(_data_dir / "c13k_problems.json")
    )

    # 加载和标准化数据（保存到本脚本所在目录）
    standardized_data = standardizer.standardize_all(
        save_path=str(_script_dir / "c13k_enhanced_standardized.json")
    )
    
    # 获取特征矩阵
    X, feature_names = standardizer.get_enhanced_feature_matrix()
    y = standardizer.get_target_vector()
    
    print(f"\n增强特征矩阵形状: {X.shape}")
    print(f"目标向量形状: {y.shape}")
    print(f"特征数量: {len(feature_names)}")
    
    # 显示前10个特征
    print(f"\n前10个特征:")
    for i, name in enumerate(feature_names[:10]):
        print(f"  {i}: {name}")
    
    # 创建划分
    print(f"\n创建数据划分...")
    for split_type in SPLIT_TYPES:
        train_idx, test_idx, split_info = create_enhanced_splits(
            standardized_data, split_type=split_type
        )
        print(f"\n{split_type}:")
        print(f"  {split_info['description']}")
        print(f"  训练集: {split_info['train_size']} 样本")
        print(f"  测试集: {split_info['test_size']} 样本")
        print(f"  训练集bRate均值: {split_info['train_bRate_mean']:.4f}")
        print(f"  测试集bRate均值: {split_info['test_bRate_mean']:.4f}")