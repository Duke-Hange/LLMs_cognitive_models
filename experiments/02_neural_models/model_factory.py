"""
模型工厂 - 统一创建符号模型和神经网络模型
"""
from typing import Union
from enum import Enum

# 导入符号模型
from enhanced_symbolic_models import (
    create_enhanced_model as create_symbolic_model,
    EnhancedSymbolicModel
)

# 导入神经网络模型包装器
from models import (
    ValueBasedNetWrapper,
    ContextDependentNetWrapper,
    ContextDependentNetSigmoidWrapper
)


class ModelType(Enum):
    """模型类型枚举"""
    # 符号模型
    EV = "ev"  # Expected Value
    EU = "eu"  # Expected Utility
    PT3 = "pt3"  # Prospect Theory 3-Parameter
    PT5 = "pt5"  # Prospect Theory 5-Parameter
    CPT5 = "cpt5"  # Cumulative Prospect Theory 5-Parameter
    # 神经网络模型
    VALUE_BASED = "value_based"
    CONTEXT_DEPENDENT_RELU = "context_dependent_relu"
    CONTEXT_DEPENDENT_SIGMOID = "context_dependent_sigmoid"


def create_model(model_type: Union[ModelType, str], **kwargs):
    """
    根据类型创建模型实例
    
    Args:
        model_type: 模型类型
        **kwargs: 模型构造参数
    """
    # 确保model_type是ModelType枚举值
    if isinstance(model_type, str):
        model_type = ModelType(model_type.lower())
    
    # 根据类型创建模型
    if model_type in [
        ModelType.EV,
        ModelType.EU,
        ModelType.PT3, 
        ModelType.PT5,
        ModelType.CPT5,
    ]:
        # 符号模型
        return create_symbolic_model(model_type.value, **kwargs)
    
    elif model_type == ModelType.VALUE_BASED:
        # Value-Based神经网络模型
        input_dim_per_gamble = kwargs.get('input_dim_per_gamble', 18)
        hidden_dim = kwargs.get('hidden_dim', 64)
        return ValueBasedNetWrapper(input_dim_per_gamble, hidden_dim)
    
    elif model_type == ModelType.CONTEXT_DEPENDENT_RELU:
        # Context-Dependent神经网络模型(ReLU)
        input_dim = kwargs.get('input_dim', 36)
        hidden_dim = kwargs.get('hidden_dim', 32)
        return ContextDependentNetWrapper(input_dim, hidden_dim)
    
    elif model_type == ModelType.CONTEXT_DEPENDENT_SIGMOID:
        # Context-Dependent神经网络模型(Sigmoid)
        input_dim = kwargs.get('input_dim', 36)
        hidden_dim = kwargs.get('hidden_dim', 32)
        return ContextDependentNetSigmoidWrapper(input_dim, hidden_dim)
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def get_all_model_types():
    """获取所有支持的模型类型"""
    return list(ModelType)


def get_model_names():
    """获取所有模型的可读名称"""
    return {
        # 符号模型
        "ev": "Expected Value (期望值模型)",
        "eu": "Expected Utility (期望效用模型)",
        "pt3": "Prospect Theory 3P (前景理论3参数)",
        "pt5": "Prospect Theory 5P (前景理论5参数)",
        "cpt5": "Cumulative Prospect Theory 5P (累积前景理论5参数)",
        # 神经网络模型
        "value_based": "Value-Based Network (价值基础网络)",
        "context_dependent_relu": "Context-Dependent Network (ReLU) (上下文相关网络ReLU)",
        "context_dependent_sigmoid": "Context-Dependent Network (Sigmoid) (上下文相关网络Sigmoid)",
    }


if __name__ == "__main__":
    # 测试模型工厂
    print("测试模型工厂:")
    
    model_names = get_model_names()
    for model_key, model_name in model_names.items():
        try:
            model = create_model(model_key)
            print(f"✓ 成功创建: {model_name} ({model_key})")
        except Exception as e:
            print(f"✗ 创建失败: {model_name} ({model_key}) - 错误: {str(e)}")
    
    print("\n模型创建测试完成!")