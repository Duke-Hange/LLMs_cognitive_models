"""
Comprehensive visualization demo script
Demonstrates the project's unified visualization capabilities
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import sys

# Add project path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared import (
    setup_chinese_font, apply_modern_theme,
    plot_learning_curve, plot_model_performance_comparison,
    plot_cross_entropy_comparison, plot_scatter_with_regression,
    plot_model_comparison, create_correlation_heatmap
)

def create_comprehensive_visualization_demo():
    """Create comprehensive visualization demonstration"""
    print("Creating comprehensive visualization demo...")
    
    # Create output directory
    output_dir = PROJECT_ROOT / "comprehensive_viz_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set Chinese font and modern theme
    setup_chinese_font()
    apply_modern_theme()
    
    # Simulate some data for demonstration
    # Model performance data
    model_performance_data = {
        'EV': {'MSE': 0.32, 'R2': 0.78, 'Correlation': 0.83, 'Cross-Entropy': 0.65},
        'EU': {'MSE': 0.28, 'R2': 0.81, 'Correlation': 0.85, 'Cross-Entropy': 0.58},
        'PT3': {'MSE': 0.25, 'R2': 0.84, 'Correlation': 0.87, 'Cross-Entropy': 0.52},
        'PT5': {'MSE': 0.22, 'R2': 0.86, 'Correlation': 0.89, 'Cross-Entropy': 0.48}
    }
    
    print("1. Drawing model performance comparison...")
    # Extract metric names
    metrics = ['MSE', 'R2', 'Correlation']
    fig, ax = plot_model_performance_comparison(
        model_performance_data,
        metrics,
        title="Comprehensive Model Performance Comparison",
        save_path=output_dir / "model_performance_comparison.png"
    )
    plt.close(fig)
    print("   - Model performance comparison chart saved")
    
    print("2. Drawing learning curves...")
    # Learning curve data
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ev_mse = [0.45, 0.42, 0.40, 0.38, 0.36, 0.35, 0.34, 0.33, 0.32, 0.32]
    eu_mse = [0.42, 0.39, 0.37, 0.35, 0.34, 0.33, 0.32, 0.31, 0.30, 0.28]
    pt3_mse = [0.40, 0.37, 0.35, 0.33, 0.32, 0.31, 0.30, 0.29, 0.28, 0.25]
    
    metrics_list = [ev_mse, eu_mse, pt3_mse]
    model_names = ['EV', 'EU', 'PT3']
    
    fig, ax = plot_learning_curve(
        fractions, metrics_list, model_names,
        metric_name="MSE",
        title="Learning Curve Comparison - MSE Performance",
        save_path=output_dir / "learning_curve_comparison.png"
    )
    plt.close(fig)
    print("   - Learning curve chart saved")
    
    print("3. Drawing cross-entropy comparison...")
    # Cross entropy data
    ce_data = {
        'EV': [0.68, 0.66, 0.65, 0.64, 0.63, 0.62, 0.61, 0.60, 0.59, 0.65],
        'EU': [0.62, 0.60, 0.59, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.58],
        'PT3': [0.58, 0.56, 0.55, 0.54, 0.53, 0.52, 0.51, 0.50, 0.49, 0.52]
    }
    
    fig, ax = plot_cross_entropy_comparison(
        ce_data, ['EV', 'EU', 'PT3'],
        title="Cross-Entropy Loss Comparison During Training",
        save_path=output_dir / "cross_entropy_comparison.png"
    )
    plt.close(fig)
    print("   - Cross-entropy comparison chart saved")
    
    print("4. Drawing scatter & regression analysis...")
    # Generate simulated true vs predicted values
    true_values = np.linspace(0.1, 0.9, 100)
    predicted_values = true_values + np.random.normal(0, 0.05, size=true_values.shape)
    
    fig, ax = plot_scatter_with_regression(
        true_values, predicted_values,
        title="Model Prediction Accuracy Analysis",
        xlabel="True Values",
        ylabel="Predicted Values",
        save_path=output_dir / "prediction_accuracy.png"
    )
    plt.close(fig)
    print("   - Prediction accuracy analysis chart saved")
    
    print("5. Drawing model metrics comparison...")
    # Use pandas dataframe for model comparison
    metrics_df = pd.DataFrame.from_dict(model_performance_data, orient='index').T
    fig, ax = plot_model_comparison(
        metrics_df,
        title="Model Metrics Comparison Heatmap",
        save_path=output_dir / "metrics_comparison_heatmap.png"
    )
    plt.close(fig)
    print("   - Model metrics heatmap saved")
    
    print("6. Creating feature correlation heatmap...")
    # Generate some correlated feature data
    np.random.seed(42)
    n_samples = 100
    feat1 = np.random.randn(n_samples)
    feat2 = 0.5 * feat1 + np.random.randn(n_samples) * 0.5
    feat3 = -0.3 * feat1 + 0.4 * feat2 + np.random.randn(n_samples) * 0.3
    feat4 = np.random.randn(n_samples)
    
    correlation_data = pd.DataFrame({
        'Feature_1': feat1,
        'Feature_2': feat2, 
        'Feature_3': feat3,
        'Feature_4': feat4
    })
    
    fig, ax = create_correlation_heatmap(
        correlation_data,
        title="Feature Correlation Matrix",
        save_path=output_dir / "feature_correlation.png"
    )
    plt.close(fig)
    print("   - Feature correlation chart saved")
    
    # Summary info
    generated_files = list(output_dir.glob("*.png"))
    print(f"\nOK Comprehensive visualization demo completed!")
    print(f"Generated {len(generated_files)} visualization charts:")
    for file in generated_files:
        print(f"   {file.name}")
    
    return generated_files


def main():
    """Main function"""
    print("=" * 60)
    print("Comprehensive Visualization Demo for the Project")
    print("=" * 60)
    print()
    
    try:
        generated_files = create_comprehensive_visualization_demo()
        
        print()
        print("=" * 60)
        print("Visualization system implementation summary:")
        print("- Successfully created shared.visualization module")
        print("- Unified visualization functionality across models") 
        print("- Consistent color schemes and themes")
        print("- Standardized plotting methods for model comparisons")
        print("- Professional look and Chinese font support")
        print("- Efficient integration with existing modules")
        print()
        print(f"All demos saved to: {Path.cwd() / 'comprehensive_viz_demo'}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during visualization demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()