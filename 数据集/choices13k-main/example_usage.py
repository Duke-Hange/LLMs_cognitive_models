"""
使用示例：完整的项目流程
"""

import numpy as np
import pandas as pd
from data_standardization import Choices13kStandardizer, create_splits
from models import ExpectedUtilityModel, ProspectTheoryModel, NeuralModel, LLMModel, compare_models
from evaluation import comprehensive_evaluation, extract_symbolic_variables

# ==================== 步骤 1: 数据标准化 ====================

print("=" * 60)
print("步骤 1: 数据标准化")
print("=" * 60)

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

# 准备上下文和 Prompt
contexts = [item['context'] for item in standardized_data]
prompts = [item['prompt'] for item in standardized_data]

# 转换为 DataFrame 以便划分
df_standardized = pd.DataFrame([
    {
        'Problem': item['metadata']['problem_id'],
        'Feedback': item['metadata']['feedback_condition'],
        'Block': item['metadata']['block'],
        'bRate': item['action']['bRate'],
        'EV_diff': item['context']['features']['EV_diff'],
        'Amb': item['context']['features']['Amb'],
        'Corr': item['context']['features']['Corr']
    }
    for item in standardized_data
])

# ==================== 步骤 2: 数据划分 ====================

print("\n" + "=" * 60)
print("步骤 2: 数据划分")
print("=" * 60)

# 使用 Feedback-Split（最重要的划分）
train_idx, test_idx = create_splits(df_standardized, split_type='feedback')
print(f"训练集大小: {len(train_idx)}")
print(f"测试集大小: {len(test_idx)}")

# 准备训练和测试数据
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
contexts_train = [contexts[i] for i in train_idx]
contexts_test = [contexts[i] for i in test_idx]
prompts_train = [prompts[i] for i in train_idx]
prompts_test = [prompts[i] for i in test_idx]
df_train = df_standardized.iloc[train_idx]
df_test = df_standardized.iloc[test_idx]

# ==================== 步骤 3: 模型训练 ====================

print("\n" + "=" * 60)
print("步骤 3: 模型训练")
print("=" * 60)

# 层级 1: 符号模型
print("\n训练符号模型...")
eu_model = ExpectedUtilityModel()
eu_model.fit(X_train, y_train, contexts_train)

pt_model = ProspectTheoryModel()
pt_model.fit(X_train, y_train, contexts_train)

# 层级 2: 神经网络
print("\n训练神经网络...")
neural_model = NeuralModel(input_dim=X.shape[1], hidden_dim=64, num_layers=2, dropout=0.3)
neural_model.fit(X_train, y_train, epochs=50, batch_size=64, lr=0.001, verbose=True)

# 层级 3: LLM（示例，需要实际实现）
print("\n准备 LLM...")
llm_model = LLMModel(model_name="llama3-8b", use_few_shot=True)

# 添加 Few-shot 示例
few_shot_examples = [
    {'prompt': prompts_train[0], 'output': y_train[0]},
    {'prompt': prompts_train[1], 'output': y_train[1]},
    {'prompt': prompts_train[2], 'output': y_train[2]},
]
llm_model.add_few_shot_examples(few_shot_examples)

# ==================== 步骤 4: 模型预测 ====================

print("\n" + "=" * 60)
print("步骤 4: 模型预测")
print("=" * 60)

# 符号模型预测
eu_pred = np.array([eu_model.predict(ctx) for ctx in contexts_test])
pt_pred = np.array([pt_model.predict(ctx) for ctx in contexts_test])

# 神经网络预测
neural_pred = neural_model.predict(X_test)

# LLM 预测（示例，需要实际实现）
# llm_pred = llm_model.predict(prompts_test)

# ==================== 步骤 5: 模型比较 ====================

print("\n" + "=" * 60)
print("步骤 5: 模型比较")
print("=" * 60)

models_dict = {
    'Expected Utility': eu_model,
    'Prospect Theory': pt_model,
    'Neural Network': neural_model,
}

# 注意：LLM 需要单独的 prompts，这里先不包含
comparison_results = compare_models(
    models_dict,
    X_test,
    y_test,
    contexts=contexts_test
)

print("\n模型比较结果:")
for model_name, results in comparison_results.items():
    print(f"\n{model_name}:")
    print(f"  MSE: {results['mse']:.6f}")
    print(f"  RMSE: {results['rmse']:.6f}")
    print(f"  R²: {results['r2']:.4f}")
    print(f"  Correlation: {results['correlation']:.4f}")

# ==================== 步骤 6: 综合评估 ====================

print("\n" + "=" * 60)
print("步骤 6: 综合评估")
print("=" * 60)

# 提取符号模型潜变量（用于 RSA）
symbolic_vars = extract_symbolic_variables(
    contexts_test,
    {'EU': eu_model, 'PT': pt_model}
)

# 获取神经网络表征（用于 RSA）
neural_embeddings = neural_model.get_embeddings(X_test)

# 对每个模型进行综合评估
for model_name, model in models_dict.items():
    print(f"\n评估 {model_name}...")
    
    if isinstance(model, (ExpectedUtilityModel, ProspectTheoryModel)):
        predictions = np.array([model.predict(ctx) for ctx in contexts_test])
    elif isinstance(model, NeuralModel):
        predictions = model.predict(X_test)
    else:
        continue
    
    # 获取被试数量
    n_subjects_test = np.array([
        standardized_data[i]['action']['n_subjects'] 
        for i in test_idx
    ])
    
    # 综合评估
    eval_results = comprehensive_evaluation(
        y_true=y_test,
        y_pred=predictions,
        df=df_test,
        contexts=contexts_test,
        symbolic_vars=symbolic_vars if model_name == 'Neural Network' else None,
        neural_embeddings=neural_embeddings if model_name == 'Neural Network' else None,
        n_subjects=n_subjects_test,
        model_name=model_name,
        output_dir=f'results/{model_name}'
    )
    
    print(f"\n{model_name} 评估结果:")
    print(f"  R²: {eval_results['goodness_of_fit']['r2']:.4f}")
    print(f"  Correlation: {eval_results['goodness_of_fit']['correlation']:.4f}")
    if 'rsa' in eval_results:
        print(f"  RSA (Symbolic vs Neural): {eval_results['rsa']['symbolic_vs_neural']['rsa']:.4f}")

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)
