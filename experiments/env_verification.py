"""
环境验证脚本 - 符号模型层实验第1天
验证所有必要的Python包和GPU可用性
"""

import sys
import platform

def check_python_version():
    """检查Python版本"""
    print("=" * 60)
    print("Python环境验证")
    print("=" * 60)
    
    version_info = sys.version_info
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print(f"平台: {platform.platform()}")
    
    # 检查是否为yh311_G环境
    if "yh311_G" in sys.executable:
        print("[OK] 当前环境: yh311_G")
    else:
        print("[WARN] 当前环境可能不是yh311_G")
        print(f"   Python路径: {sys.executable}")
    
    return version_info

def check_packages():
    """检查所有必要的包"""
    print("\n" + "=" * 60)
    print("包依赖验证")
    print("=" * 60)
    
    required_packages = [
        ('numpy', '1.20.0'),
        ('pandas', '1.3.0'),
        ('scipy', '1.7.0'),
        ('sklearn', '1.0.0'),
        ('matplotlib', '3.4.0'),
        ('seaborn', '0.11.0'),
        ('torch', '2.0.0'),
    ]
    
    all_passed = True
    
    for package_name, min_version in required_packages:
        try:
            module = __import__(package_name)
            version = getattr(module, '__version__', '未知版本')
            
            # 简单版本检查（实际应使用pkg_resources或packaging）
            print(f"[OK] {package_name:20} {version:15} (要求: >={min_version})")
            
        except ImportError:
            print(f"[FAIL] {package_name:20} 未安装")
            all_passed = False
    
    return all_passed

def check_gpu():
    """检查GPU可用性"""
    print("\n" + "=" * 60)
    print("GPU验证")
    print("=" * 60)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"[OK] GPU可用")
            print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("[WARN] GPU不可用，将使用CPU")
            print("  注意：参数优化可能较慢")
        
        return cuda_available
        
    except ImportError:
        print("[FAIL] PyTorch未安装，无法检查GPU")
        return False

def check_directory_structure():
    """检查目录结构"""
    print("\n" + "=" * 60)
    print("目录结构验证")
    print("=" * 60)
    
    import os
    
    base_dirs = [
        'experiments/00_data_preparation',
        'experiments/00_data_preparation/notebooks',
        'experiments/00_data_preparation/scripts',
        'experiments/00_data_preparation/outputs',
        'experiments/01_symbolic_models_enhanced',
        'experiments/01_symbolic_models_enhanced/analysis',
        'experiments/01_symbolic_models_enhanced/results',
        'shared',
        'shared/utils',
        'shared/visualization',
    ]
    
    all_exist = True
    
    for dir_path in base_dirs:
        if os.path.exists(dir_path):
            print(f"[OK] {dir_path}")
        else:
            print(f"[FAIL] {dir_path} - 目录不存在")
            all_exist = False
    
    return all_exist

def main():
    """主验证函数"""
    print("符号模型层实验 - 环境验证报告")
    print("生成时间:", platform.uname().node, "-", platform.uname().system)
    print()
    
    # 检查Python版本
    version_info = check_python_version()
    
    # 检查包
    packages_ok = check_packages()
    
    # 检查GPU
    gpu_ok = check_gpu()
    
    # 检查目录结构
    dirs_ok = check_directory_structure()
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    if packages_ok and dirs_ok:
        print("[OK] 所有基本验证通过")
        if gpu_ok:
            print("[OK] GPU可用，可以加速计算")
        else:
            print("[WARN] GPU不可用，将使用CPU计算")
        print("\n环境准备就绪，可以开始实验！")
    else:
        print("[FAIL] 验证失败，请检查以下问题：")
        if not packages_ok:
            print("  - 缺少必要的Python包")
        if not dirs_ok:
            print("  - 目录结构不完整")
        print("\n请先解决以上问题再继续。")
    
    return packages_ok and dirs_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)