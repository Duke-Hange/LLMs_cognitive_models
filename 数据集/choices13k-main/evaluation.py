"""
评估指标模块
针对 Choices13k 聚合数据的评估方法
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import beta, wasserstein_distance, ks_2samp, spearmanr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

# 项目根路径并设置中文字体（避免中文/负号显示为方框）
_p = Path(__file__).resolve().parent
for _ in range(8):
    _p = _p.parent
    if (_p / "shared").is_dir():
        sys.path.insert(0, str(_p))
        break
from shared.visualization import setup_chinese_font
setup_chinese_font()


# ==================== 1. 拟合优度评估 ====================

def goodness_of_fit(y_true: np.ndarray, y_pred: np.ndarray, 
                    n_subjects: Optional[np.ndarray] = None) -> Dict:
    """
    计算拟合优度指标
    
    Args:
        y_true: 真实 bRate [n_samples]
        y_pred: 预测 bRate [n_samples]
        n_subjects: 每个问题的被试数量（可选，用于 Beta 分布 NLL）
        
    Returns:
        指标字典
    """
    metrics = {}
    
    # 基本回归指标
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
    
    # Beta 分布负对数似然（如果提供了 n_subjects）
    if n_subjects is not None:
        metrics['nll_beta'] = nll_beta_distribution(y_true, y_pred, n_subjects)
    
    return metrics


def nll_beta_distribution(y_true: np.ndarray, y_pred: np.ndarray, 
                         n_subjects: np.ndarray) -> float:
    """
    计算 Beta 分布的负对数似然
    
    假设 bRate ~ Beta(alpha, beta)，其中 alpha + beta = n_subjects
    """
    nll_sum = 0.0
    valid_count = 0
    
    for i in range(len(y_true)):
        bRate_true = y_true[i]
        bRate_pred = y_pred[i]
        n = n_subjects[i]
        
        # 跳过边界值
        if bRate_true <= 0 or bRate_true >= 1:
            continue
        if bRate_pred <= 0 or bRate_pred >= 1:
            continue
        
        # 从 bRate 和 n 估计 alpha, beta
        alpha_true = bRate_true * n
        beta_true = (1 - bRate_true) * n
        
        alpha_pred = bRate_pred * n
        beta_pred = (1 - bRate_pred) * n
        
        # 确保参数有效
        if alpha_pred <= 0 or beta_pred <= 0:
            continue
        
        try:
            # 计算负对数似然
            nll = -beta.logpdf(bRate_true, alpha_pred, beta_pred)
            nll_sum += nll
            valid_count += 1
        except:
            continue
    
    return nll_sum / valid_count if valid_count > 0 else np.inf


# ==================== 2. 生成式验证 ====================

def distribution_match(true_dist: np.ndarray, pred_dist: np.ndarray) -> Dict:
    """
    比较两个分布的匹配程度
    
    Args:
        true_dist: 真实分布
        pred_dist: 预测分布
        
    Returns:
        匹配指标字典
    """
    metrics = {}
    
    # Wasserstein 距离
    metrics['wasserstein'] = wasserstein_distance(true_dist, pred_dist)
    
    # Kolmogorov-Smirnov 检验
    ks_stat, p_value = ks_2samp(true_dist, pred_dist)
    metrics['ks_statistic'] = ks_stat
    metrics['ks_p_value'] = p_value
    
    # 均值差异
    metrics['mean_diff'] = np.abs(np.mean(true_dist) - np.mean(pred_dist))
    
    # 标准差差异
    metrics['std_diff'] = np.abs(np.std(true_dist) - np.std(pred_dist))
    
    return metrics


def conditional_distribution_check(df: pd.DataFrame, predictions: np.ndarray,
                                  condition_cols: List[str]) -> Dict:
    """
    按条件检查分布匹配
    
    Args:
        df: 包含真实值和条件列的 DataFrame
        predictions: 模型预测值
        condition_cols: 条件列名列表（如 ['Feedback', 'Amb']）
        
    Returns:
        按条件分组的评估结果
    """
    results = {}
    
    for col in condition_cols:
        if col not in df.columns:
            continue
        
        results[col] = {}
        for value in df[col].unique():
            mask = df[col] == value
            true_vals = df[mask]['bRate'].values
            pred_vals = predictions[mask]
            
            results[col][value] = {
                'mse': mean_squared_error(true_vals, pred_vals),
                'rmse': np.sqrt(mean_squared_error(true_vals, pred_vals)),
                'correlation': np.corrcoef(true_vals, pred_vals)[0, 1],
                'mean_true': np.mean(true_vals),
                'mean_pred': np.mean(pred_vals),
                'n_samples': len(true_vals)
            }
    
    return results


def parameter_space_analysis(df: pd.DataFrame, predictions: np.ndarray,
                             param_col: str = 'EV_diff', bins: int = 10) -> Dict:
    """
    分析模型在不同参数空间的行为
    
    Args:
        df: 包含参数列的 DataFrame
        predictions: 模型预测值
        param_col: 参数列名
        bins: 分箱数量
        
    Returns:
        参数空间分析结果
    """
    if param_col not in df.columns:
        raise ValueError(f"参数列 '{param_col}' 不存在")
    
    # 分箱
    df['param_bin'] = pd.cut(df[param_col], bins=bins)
    
    # 按箱聚合
    true_by_bin = df.groupby('param_bin')['bRate'].agg(['mean', 'std', 'count'])
    pred_by_bin = pd.DataFrame({
        'mean': predictions,
        'param_bin': df['param_bin']
    }).groupby('param_bin')['mean'].agg(['mean', 'std', 'count'])
    
    # 计算相关性
    correlation = np.corrcoef(
        true_by_bin['mean'].values,
        pred_by_bin['mean'].values
    )[0, 1]
    
    return {
        'param_col': param_col,
        'bins': bins,
        'true_by_bin': true_by_bin,
        'pred_by_bin': pred_by_bin,
        'correlation': correlation,
        'bin_centers': [interval.mid for interval in true_by_bin.index]
    }


# ==================== 3. 模型恢复 ====================

def model_recovery(X_features: np.ndarray, model_labels: np.ndarray,
                  test_size: float = 0.2) -> Dict:
    """
    模型恢复实验：给定问题特征，判断数据是由哪个模型生成的
    
    Args:
        X_features: 特征矩阵 [n_samples, n_features]
        model_labels: 模型标签 [n_samples] (0, 1, 2 分别代表不同模型)
        test_size: 测试集比例
        
    Returns:
        恢复准确率等指标
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, model_labels, test_size=test_size, random_state=42
    )
    
    # 训练分类器
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 预测
    y_pred = clf.predict(X_test)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'feature_importances': clf.feature_importances_
    }


# ==================== 4. 内部表征对齐 (RSA) ====================

def compute_rsa(symbolic_vars: np.ndarray, neural_embeddings: np.ndarray,
                llm_embeddings: Optional[np.ndarray] = None) -> Dict:
    """
    计算表征相似性分析 (Representational Similarity Analysis)
    
    Args:
        symbolic_vars: 符号模型的潜变量 [n_samples, n_vars]
                      例如：[EU_diff, PT_diff, Risk_diff]
        neural_embeddings: 神经网络的内部表征 [n_samples, embedding_dim]
        llm_embeddings: LLM 的内部表征 [n_samples, embedding_dim]（可选）
        
    Returns:
        RSA 结果字典
    """
    results = {}
    
    # 计算距离矩阵
    symbolic_dist = squareform(pdist(symbolic_vars, metric='euclidean'))
    neural_dist = squareform(pdist(neural_embeddings, metric='euclidean'))
    
    # 符号模型 vs 神经网络
    rsa_neural, p_value_neural = spearmanr(
        squareform(symbolic_dist),
        squareform(neural_dist)
    )
    results['symbolic_vs_neural'] = {
        'rsa': rsa_neural,
        'p_value': p_value_neural
    }
    
    # 符号模型 vs LLM（如果提供）
    if llm_embeddings is not None:
        llm_dist = squareform(pdist(llm_embeddings, metric='euclidean'))
        rsa_llm, p_value_llm = spearmanr(
            squareform(symbolic_dist),
            squareform(llm_dist)
        )
        results['symbolic_vs_llm'] = {
            'rsa': rsa_llm,
            'p_value': p_value_llm
        }
        
        # 神经网络 vs LLM
        rsa_neural_llm, p_value_neural_llm = spearmanr(
            squareform(neural_dist),
            squareform(llm_dist)
        )
        results['neural_vs_llm'] = {
            'rsa': rsa_neural_llm,
            'p_value': p_value_neural_llm
        }
    
    return results


def extract_symbolic_variables(contexts: List[Dict], 
                               models: Dict[str, object]) -> np.ndarray:
    """
    从符号模型中提取潜变量
    
    Args:
        contexts: 上下文列表
        models: 符号模型字典，例如 {'EU': ExpectedUtilityModel, 'PT': ProspectTheoryModel}
        
    Returns:
        潜变量矩阵 [n_samples, n_vars]
    """
    variables = []
    
    for ctx in contexts:
        var_row = []
        
        # 从期望效用模型提取
        if 'EU' in models:
            gamble_a = ctx['gamble_a']['outcomes']
            gamble_b = ctx['gamble_b']['outcomes']
            eu_a = models['EU'].expected_utility(gamble_a)
            eu_b = models['EU'].expected_utility(gamble_b)
            var_row.append(eu_b - eu_a)  # EU_diff
        
        # 从前景理论模型提取
        if 'PT' in models:
            gamble_a = ctx['gamble_a']['outcomes']
            gamble_b = ctx['gamble_b']['outcomes']
            pt_a = models['PT'].prospect_value(gamble_a)
            pt_b = models['PT'].prospect_value(gamble_b)
            var_row.append(pt_b - pt_a)  # PT_diff
        
        # 添加风险差异
        risk_a = ctx['context']['features'].get('risk_A', 0)
        risk_b = ctx['context']['features'].get('risk_B', 0)
        var_row.append(risk_b - risk_a)  # Risk_diff
        
        variables.append(var_row)
    
    return np.array(variables)


# ==================== 5. 可视化工具 ====================

def plot_predictions_vs_true(y_true: np.ndarray, y_pred: np.ndarray,
                            model_name: str = "Model", save_path: Optional[str] = None):
    """
    绘制预测值 vs 真实值散点图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        model_name: 模型名称
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # 对角线
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # 计算 R²
    r2 = r2_score(y_true, y_pred)
    plt.title(f'{model_name}: Predictions vs True (R² = {r2:.3f})')
    plt.xlabel('True bRate')
    plt.ylabel('Predicted bRate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_parameter_space_analysis(analysis_result: Dict, save_path: Optional[str] = None):
    """
    绘制参数空间分析图
    
    Args:
        analysis_result: parameter_space_analysis 的返回结果
        save_path: 保存路径（可选）
    """
    true_by_bin = analysis_result['true_by_bin']
    pred_by_bin = analysis_result['pred_by_bin']
    bin_centers = analysis_result['bin_centers']
    
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, true_by_bin['mean'], 'o-', label='True', linewidth=2, markersize=8)
    plt.plot(bin_centers, pred_by_bin['mean'], 's-', label='Predicted', linewidth=2, markersize=8)
    
    # 误差棒
    plt.errorbar(bin_centers, true_by_bin['mean'], yerr=true_by_bin['std'], 
                fmt='o', capsize=5, alpha=0.5)
    plt.errorbar(bin_centers, pred_by_bin['mean'], yerr=pred_by_bin['std'], 
                fmt='s', capsize=5, alpha=0.5)
    
    plt.xlabel(analysis_result['param_col'])
    plt.ylabel('Mean bRate')
    plt.title(f'Parameter Space Analysis (Correlation = {analysis_result["correlation"]:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_conditional_distributions(conditional_results: Dict, save_path: Optional[str] = None):
    """
    绘制条件分布比较图
    
    Args:
        conditional_results: conditional_distribution_check 的返回结果
        save_path: 保存路径（可选）
    """
    n_conditions = len(conditional_results)
    fig, axes = plt.subplots(1, n_conditions, figsize=(6*n_conditions, 6))
    
    if n_conditions == 1:
        axes = [axes]
    
    for idx, (col, values) in enumerate(conditional_results.items()):
        ax = axes[idx]
        
        conditions = list(values.keys())
        true_means = [values[v]['mean_true'] for v in conditions]
        pred_means = [values[v]['mean_pred'] for v in conditions]
        
        x = np.arange(len(conditions))
        width = 0.35
        
        ax.bar(x - width/2, true_means, width, label='True', alpha=0.8)
        ax.bar(x + width/2, pred_means, width, label='Predicted', alpha=0.8)
        
        ax.set_xlabel(col)
        ax.set_ylabel('Mean bRate')
        ax.set_title(f'Conditional Analysis: {col}')
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in conditions])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==================== 6. 综合评估函数 ====================

def comprehensive_evaluation(y_true: np.ndarray, y_pred: np.ndarray,
                           df: pd.DataFrame, contexts: List[Dict],
                           symbolic_vars: Optional[np.ndarray] = None,
                           neural_embeddings: Optional[np.ndarray] = None,
                           llm_embeddings: Optional[np.ndarray] = None,
                           n_subjects: Optional[np.ndarray] = None,
                           model_name: str = "Model",
                           output_dir: Optional[str] = None) -> Dict:
    """
    综合评估函数，执行所有评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        df: 包含条件列的 DataFrame
        contexts: 上下文列表
        symbolic_vars: 符号模型潜变量（用于 RSA）
        neural_embeddings: 神经网络表征（用于 RSA）
        llm_embeddings: LLM 表征（用于 RSA）
        n_subjects: 被试数量（用于 Beta NLL）
        model_name: 模型名称
        output_dir: 输出目录（用于保存图表）
        
    Returns:
        综合评估结果字典
    """
    results = {}
    
    # 1. 拟合优度
    print(f"评估 {model_name}: 拟合优度...")
    results['goodness_of_fit'] = goodness_of_fit(y_true, y_pred, n_subjects)
    
    # 2. 分布匹配
    print(f"评估 {model_name}: 分布匹配...")
    results['distribution_match'] = distribution_match(y_true, y_pred)
    
    # 3. 条件分布检查
    print(f"评估 {model_name}: 条件分布检查...")
    condition_cols = ['Feedback', 'Amb', 'Corr'] if all(c in df.columns for c in ['Feedback', 'Amb', 'Corr']) else []
    if condition_cols:
        results['conditional'] = conditional_distribution_check(
            df, y_pred, condition_cols
        )
    
    # 4. 参数空间分析
    print(f"评估 {model_name}: 参数空间分析...")
    if 'EV_diff' in df.columns:
        results['parameter_space'] = parameter_space_analysis(
            df, y_pred, param_col='EV_diff'
        )
    
    # 5. RSA 分析
    if symbolic_vars is not None and neural_embeddings is not None:
        print(f"评估 {model_name}: RSA 分析...")
        results['rsa'] = compute_rsa(
            symbolic_vars, neural_embeddings, llm_embeddings
        )
    
    # 6. 可视化
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        plot_predictions_vs_true(
            y_true, y_pred, model_name,
            save_path=os.path.join(output_dir, f'{model_name}_predictions.png')
        )
        
        if 'parameter_space' in results:
            plot_parameter_space_analysis(
                results['parameter_space'],
                save_path=os.path.join(output_dir, f'{model_name}_parameter_space.png')
            )
        
        if 'conditional' in results:
            plot_conditional_distributions(
                results['conditional'],
                save_path=os.path.join(output_dir, f'{model_name}_conditional.png')
            )
    
    return results
