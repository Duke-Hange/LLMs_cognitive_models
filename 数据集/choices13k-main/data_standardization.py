"""
Choices13k 数据标准化模块
将 Choices13k 数据集转换为统一的分析格式
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class Choices13kStandardizer:
    """Choices13k 数据标准化器"""
    
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
        
    def load_data(self):
        """加载原始数据"""
        print("加载 Choices13k 数据...")
        self.df = pd.read_csv(self.selections_path)
        
        with open(self.problems_path, 'r') as f:
            self.problems = json.load(f)
        
        # 将 problems 转换为 DataFrame 并合并
        problems_df = pd.DataFrame(self.problems).T
        problems_df.index = problems_df.index.astype(int)
        self.df = self.df.join(problems_df, how='left')
        
        print(f"加载完成: {len(self.df)} 条记录")
        return self
    
    def calculate_features(self, row: pd.Series) -> Dict:
        """
        计算问题特征
        
        Args:
            row: 数据行
            
        Returns:
            特征字典
        """
        # 提取 Gamble A 和 B 的 outcomes
        gamble_a = row['A']
        gamble_b = row['B']
        
        # 计算期望值
        ev_a = sum(p * outcome for p, outcome in gamble_a)
        ev_b = sum(p * outcome for p, outcome in gamble_b)
        ev_diff = ev_b - ev_a
        
        # 计算风险（标准差）
        risk_a = np.sqrt(sum(p * (outcome - ev_a)**2 for p, outcome in gamble_a))
        risk_b = np.sqrt(sum(p * (outcome - ev_b)**2 for p, outcome in gamble_b))
        
        # 提取其他特征
        features = {
            'Ha': row['Ha'],
            'pHa': row['pHa'],
            'La': row['La'],
            'Hb': row['Hb'],
            'pHb': row['pHb'],
            'Lb': row['Lb'],
            'LotShapeB': row['LotShapeB'],
            'LotNumB': row['LotNumB'],
            'Amb': bool(row['Amb']),
            'Corr': int(row['Corr']),
            'EV_A': ev_a,
            'EV_B': ev_b,
            'EV_diff': ev_diff,
            'risk_A': risk_a,
            'risk_B': risk_b,
        }
        
        return features
    
    def generate_gamble_description(self, gamble: List, gamble_name: str = "Gamble") -> str:
        """
        生成赌博选项的自然语言描述
        
        Args:
            gamble: 赌博选项 [[p, outcome], ...]
            gamble_name: 选项名称
            
        Returns:
            自然语言描述
        """
        # 按概率排序（从高到低）
        sorted_gamble = sorted(gamble, key=lambda x: x[0], reverse=True)
        
        outcomes = []
        for p, outcome in sorted_gamble:
            outcomes.append(f"以概率 {p:.2f} 获得 {outcome} 点")
        
        description = f"{gamble_name}:\n" + "\n".join(f"  - {outcome}" for outcome in outcomes)
        return description
    
    def generate_prompt(self, row: pd.Series, include_feedback: bool = True) -> str:
        """
        生成 LLM 可理解的自然语言 Prompt
        
        Args:
            row: 数据行
            include_feedback: 是否包含反馈信息
            
        Returns:
            Prompt 字符串
        """
        gamble_a = row['A']
        gamble_b = row['B']
        
        # 计算期望值
        ev_a = sum(p * outcome for p, outcome in gamble_a)
        ev_b = sum(p * outcome for p, outcome in gamble_b)
        
        # 生成描述
        desc_a = self.generate_gamble_description(gamble_a, "Gamble A")
        desc_b = self.generate_gamble_description(gamble_b, "Gamble B")
        
        prompt = f"""你面临一个风险选择问题：

{desc_a}
期望值: {ev_a:.2f} 点

{desc_b}
期望值: {ev_b:.2f} 点

"""
        
        # 添加模糊性信息
        if row['Amb']:
            prompt += "注意：Gamble B 的概率信息不完全明确（存在模糊性）。\n"
        
        # 添加相关性信息
        if row['Corr'] == 1:
            prompt += "注意：两个选项的收益存在正相关。\n"
        elif row['Corr'] == -1:
            prompt += "注意：两个选项的收益存在负相关。\n"
        
        # 添加反馈信息
        if include_feedback:
            if row['Feedback']:
                prompt += "\n你将获得反馈：选择后会看到实际获得的奖励和错过的奖励。\n"
            else:
                prompt += "\n你不会获得反馈：选择后不会看到实际结果。\n"
        
        prompt += "\n请预测人类选择 Gamble B 的概率（0-1之间的数值）。"
        
        return prompt
    
    def calculate_reward_distribution(self, gamble: List) -> Dict:
        """
        计算奖励分布
        
        Args:
            gamble: 赌博选项
            
        Returns:
            奖励分布字典
        """
        outcomes = [outcome for _, outcome in gamble]
        probabilities = [p for p, _ in gamble]
        
        return {
            'outcomes': outcomes,
            'probabilities': probabilities,
            'expected_value': sum(p * o for p, o in zip(probabilities, outcomes))
        }
    
    def standardize_row(self, idx: int) -> Dict:
        """
        标准化单行数据
        
        Args:
            idx: 行索引
            
        Returns:
            标准化后的数据字典
        """
        row = self.df.iloc[idx]
        
        # 计算特征
        features = self.calculate_features(row)
        
        # 生成自然语言描述
        gamble_a_desc = self.generate_gamble_description(row['A'], "Gamble A")
        gamble_b_desc = self.generate_gamble_description(row['B'], "Gamble B")
        
        # 计算奖励分布
        reward_a = self.calculate_reward_distribution(row['A'])
        reward_b = self.calculate_reward_distribution(row['B'])
        
        # 构建标准化格式
        standardized = {
            'context': {
                'description': f"选择 {gamble_a_desc} 或 {gamble_b_desc}",
                'gamble_a': {
                    'outcomes': row['A'],
                    'expected_value': features['EV_A'],
                    'description': gamble_a_desc
                },
                'gamble_b': {
                    'outcomes': row['B'],
                    'expected_value': features['EV_B'],
                    'lottery_shape': int(row['LotShapeB']),
                    'lottery_num': int(row['LotNumB']),
                    'description': gamble_b_desc
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
            'reward': {
                'expected_reward_A': features['EV_A'],
                'expected_reward_B': features['EV_B'],
                'feedback_received': bool(row['Feedback']),
                'reward_distribution_A': reward_a,
                'reward_distribution_B': reward_b
            },
            'metadata': {
                'problem_id': int(row['Problem']),
                'feedback_condition': bool(row['Feedback']),
                'block': int(row['Block']),
                'dataset': 'choices13k',
                'index': idx
            },
            'prompt': self.generate_prompt(row)
        }
        
        return standardized
    
    def standardize_all(self, save_path: Optional[str] = None) -> List[Dict]:
        """
        标准化所有数据
        
        Args:
            save_path: 保存路径（可选）
            
        Returns:
            标准化后的数据列表
        """
        if self.df is None:
            self.load_data()
        
        print("开始标准化数据...")
        standardized_data = []
        
        for idx in range(len(self.df)):
            try:
                standardized = self.standardize_row(idx)
                standardized_data.append(standardized)
            except Exception as e:
                print(f"处理第 {idx} 行时出错: {e}")
                continue
        
        self.standardized_data = standardized_data
        print(f"标准化完成: {len(standardized_data)} 条记录")
        
        # 保存到文件
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(standardized_data, f, ensure_ascii=False, indent=2)
            print(f"已保存到: {save_path}")
        
        return standardized_data
    
    def get_feature_matrix(self) -> np.ndarray:
        """
        获取特征矩阵（用于神经网络）
        
        Returns:
            特征矩阵 [n_samples, n_features]
        """
        if self.standardized_data is None:
            self.standardize_all()
        
        features_list = []
        for item in self.standardized_data:
            feat = item['context']['features']
            features_list.append([
                feat['Ha'], feat['pHa'], feat['La'],
                feat['Hb'], feat['pHb'], feat['Lb'],
                feat['LotShapeB'], feat['LotNumB'],
                float(feat['Amb']), feat['Corr'],
                feat['EV_A'], feat['EV_B'], feat['EV_diff'],
                feat['risk_A'], feat['risk_B'],
                float(item['context']['feedback']),
                item['context']['block']
            ])
        
        return np.array(features_list)
    
    def get_target_vector(self) -> np.ndarray:
        """
        获取目标向量（bRate）
        
        Returns:
            目标向量 [n_samples]
        """
        if self.standardized_data is None:
            self.standardize_all()
        
        return np.array([item['action']['bRate'] for item in self.standardized_data])


def create_splits(df: pd.DataFrame, split_type: str = 'problem', **kwargs) -> Tuple:
    """
    创建数据划分
    
    Args:
        df: 标准化后的 DataFrame
        split_type: 划分类型 ('problem', 'feedback', 'parameter', 'block')
        **kwargs: 划分参数
        
    Returns:
        (train_indices, test_indices)
    """
    from sklearn.model_selection import train_test_split
    
    if split_type == 'problem':
        # Problem-Split: 按问题ID划分
        problem_ids = df['Problem'].unique()
        train_problems, test_problems = train_test_split(
            problem_ids, 
            test_size=kwargs.get('test_size', 0.2),
            random_state=kwargs.get('random_state', 42)
        )
        train_mask = df['Problem'].isin(train_problems)
        test_mask = df['Problem'].isin(test_problems)
        
    elif split_type == 'feedback':
        # Feedback-Split: 按反馈条件划分
        train_mask = df['Feedback'] == True
        test_mask = df['Feedback'] == False
        
    elif split_type == 'parameter':
        # Parameter-Split: 按参数空间划分
        # 需要先计算 EV_diff
        if 'EV_diff' not in df.columns:
            # 需要从标准化数据计算
            raise ValueError("需要先计算 EV_diff 特征")
        
        threshold = kwargs.get('threshold', 0.0)
        train_mask = df['EV_diff'].abs() < threshold
        test_mask = df['EV_diff'].abs() >= threshold
        
    elif split_type == 'block':
        # Block-Split: 按区块划分
        train_blocks = kwargs.get('train_blocks', [1, 2, 3])
        test_blocks = kwargs.get('test_blocks', [4, 5])
        train_mask = df['Block'].isin(train_blocks)
        test_mask = df['Block'].isin(test_blocks)
        
    else:
        raise ValueError(f"未知的划分类型: {split_type}")
    
    train_indices = df[train_mask].index.tolist()
    test_indices = df[test_mask].index.tolist()
    
    return train_indices, test_indices


# 使用示例
if __name__ == "__main__":
    # 初始化标准化器
    standardizer = Choices13kStandardizer(
        selections_path='c13k_selections.csv',
        problems_path='c13k_problems.json'
    )
    
    # 加载和标准化数据
    standardized_data = standardizer.standardize_all(save_path='c13k_standardized.json')
    
    # 获取特征矩阵和目标向量
    X = standardizer.get_feature_matrix()
    y = standardizer.get_target_vector()
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"目标向量形状: {y.shape}")
    
    # 转换为 DataFrame 以便划分
    df_standardized = pd.DataFrame([
        {
            'Problem': item['metadata']['problem_id'],
            'Feedback': item['metadata']['feedback_condition'],
            'Block': item['metadata']['block'],
            'bRate': item['action']['bRate'],
            'EV_diff': item['context']['features']['EV_diff']
        }
        for item in standardized_data
    ])
    
    # 创建划分
    train_idx, test_idx = create_splits(df_standardized, split_type='feedback')
    print(f"训练集大小: {len(train_idx)}, 测试集大小: {len(test_idx)}")
