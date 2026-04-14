"""
Visualization System Functionality Test Script
Used to verify that shared visualization modules work properly
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from pathlib import Path

# Add project root path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Verify visualization module imports
try:
    print("=" * 60)
    print("Testing Visualization Module Import Functionality")
    print("=" * 60)
    
    from shared import (
        setup_chinese_font,
        apply_modern_theme,
        get_model_color,
        get_metric_color,
        plot_learning_curve,
        plot_model_comparison,
        plot_model_performance_comparison,
        plot_cross_entropy_comparison,
        plot_scatter_with_regression
    )
    
    print("OK All visualization modules imported successfully!")
    
except ImportError as e:
    print("FAILED To import:")
    print(e)
    sys.exit(1)

# Test function
def test_font_setup():
    """Test font setup"""
    print("--- Testing Font Setup ---")
    try:
        setup_chinese_font()
        apply_modern_theme()
        print("OK Font and theme setup successful")
    except Exception as e:
        print("FAILED Font setup!")
        print(e)
        return False
    return True

def test_colors():
    """Test color acquisition functionality"""
    print("--- Testing Color Acquisition ---")
    try:
        models = ['ev', 'eu', 'pt3', 'pt5', 'value_based']
        for model in models:
            color = get_model_color(model)
            print(f"  {model}: {color}")
        
        metrics = ['mse', 'r2', 'correlation', 'cross_entropy']  
        for metric in metrics:
            color = get_metric_color(metric)
            print(f"  {metric}: {color}")
        
        print("OK Color acquisition function working")
    except Exception as e:
        print("FAILED Color acquisition!")
        print(e)
        return False
    return True

def test_plots():
    """Test plotting functionality"""
    print("--- Testing Plotting Functionality ---")
    
    # Create output directory
    output_dir = PROJECT_ROOT / "test_output_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Test learning curve
        print("  1. Drawing learning curve...")
        fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        metrics = [
            [0.45, 0.42, 0.40, 0.38, 0.36, 0.35, 0.34, 0.33, 0.32, 0.31],  # EV model
            [0.42, 0.40, 0.38, 0.36, 0.35, 0.34, 0.33, 0.32, 0.31, 0.30],  # EU model
        ]
        model_names = ['EV', 'EU']
        
        fig, ax = plot_learning_curve(
            fractions, metrics, model_names,
            title="Test Learning Curve",
            xlabel="Training Data Portion (%)",
            ylabel="MSE",
            save_path=output_dir / "learning_curve_test.png"
        )
        plt.close(fig)
        print("    Learning curve saved")

        # 2. Test model performance comparison
        print("  2. Drawing model performance comparison...")
        results = {
            'EV': {'mse': 0.32, 'r2': 0.78, 'correlation': 0.83},
            'EU': {'mse': 0.30, 'r2': 0.82, 'correlation': 0.85},
            'PT3': {'mse': 0.28, 'r2': 0.84, 'correlation': 0.86}
        }
        
        metric_names = ['mse', 'r2', 'correlation']
        
        fig, ax = plot_model_performance_comparison(
            results, metric_names,
            title="Model Performance Comparison Test",
            save_path=output_dir / "model_comparison_test.png"
        )
        plt.close(fig)
        print("    Model comparison chart saved")

        # 3. Test cross-entropy comparison
        print("  3. Drawing cross-entropy comparison...")
        ce_data = {
            'EV': [0.65, 0.63, 0.61, 0.60, 0.59, 0.58, 0.57, 0.56, 0.55, 0.54],
            'EU': [0.62, 0.60, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.52, 0.51]
        }
        labels = ['EV', 'EU']
        
        fig, ax = plot_cross_entropy_comparison(
            ce_data, labels,
            title="Cross-Entropy Loss Comparison Test",
            save_path=output_dir / "cross_entropy_test.png"
        )
        plt.close(fig)
        print("    Cross-entropy comparison chart saved")

        # 4. Test scatter plot
        print("  4. Drawing scatter plot...")
        x_data = np.random.randn(100)
        y_data = 2 * x_data + np.random.randn(100) * 0.5
        
        fig, ax = plot_scatter_with_regression(
            x_data, y_data,
            title="Regression Scatter Plot Test",
            xlabel="X Variable", 
            ylabel="Y Variable",
            save_path=output_dir / "scatter_regression_test.png"
        )
        plt.close(fig)
        print("    Scatter regression plot saved")
        
        print("OK All plotting functions passed!")
        
    except Exception as e:
        print("FAILED Plotting functions test!")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Main function"""
    print("Starting Visualization System Functionality Tests...")
    
    tests = [
        test_font_setup,
        test_colors,
        test_plots
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_func in tests:
        if test_func():
            passed_tests += 1
        else:
            print(f"Test {test_func.__name__} failed")
    
    print("=" * 60)
    print(f"Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("All tests passed! Visualization system is working properly.")
        
        # Show generated test charts
        output_dir = PROJECT_ROOT / "test_output_visualizations"
        print(f"Generated test charts saved to: {output_dir}")
        generated_files = list(output_dir.glob("*.png"))
        print(f"   Generated {len(generated_files)} test chart files")
        for f in generated_files[:5]:  # Show only first 5
            print(f"   - {f.name}")
        if len(generated_files) > 5:
            print(f"   ... plus {len(generated_files)-5} more files")
    else:
        print("Some tests failed, please check the issues")
    
    print("=" * 60)

if __name__ == "__main__":
    main()