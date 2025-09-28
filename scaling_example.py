#!/usr/bin/env python3
"""
Example demonstrating input scaling in ConformalInference.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.synthe.conformal_inference import ConformalInference

def main():
    """Demonstrate input scaling functionality."""
    
    # Generate test data with different scales
    np.random.seed(42)
    n_samples = 500
    
    # Create data with very different scales
    x1 = np.random.normal(0, 1, n_samples)  # Standard scale
    x2 = np.random.normal(1000, 100, n_samples)  # Large scale
    x3 = np.random.normal(0, 0.01, n_samples)  # Small scale
    
    # Combine into multivariate data
    X = np.column_stack([x1, x2, x3])
    
    print("=== ConformalInference with Input Scaling ===")
    print(f"Original data shape: {X.shape}")
    print("Original data statistics:")
    print(f"  Mean: {np.mean(X, axis=0)}")
    print(f"  Std:  {np.std(X, axis=0)}")
    print(f"  Min:  {np.min(X, axis=0)}")
    print(f"  Max:  {np.max(X, axis=0)}")
    
    # Fit the model
    print(f"\nFitting ConformalInference model...")
    model = ConformalInference(
        type_pi="bootstrap",
        objective="crps",
        optimizer="optuna",
        split_ratio=0.5,
        seed=42
    )
    
    model.fit(X)
    
    print(f"Model fitted successfully!")
    print(f"Input scaler fitted: {model.input_scaler_ is not None}")
    if model.input_scaler_ is not None:
        print(f"Scaler mean: {model.input_scaler_.mean_}")
        print(f"Scaler scale: {model.input_scaler_.scale_}")
    
    # Generate synthetic samples
    print(f"\nGenerating synthetic samples...")
    n_synthetic = 200
    synthetic_data = model.sample(n_samples=n_synthetic)
    
    print(f"Generated {len(synthetic_data)} synthetic samples")
    print("Synthetic data statistics:")
    print(f"  Mean: {np.mean(synthetic_data, axis=0)}")
    print(f"  Std:  {np.std(synthetic_data, axis=0)}")
    print(f"  Min:  {np.min(synthetic_data, axis=0)}")
    print(f"  Max:  {np.max(synthetic_data, axis=0)}")
    
    # Compare original vs synthetic
    print(f"\nComparison (Original vs Synthetic):")
    print(f"Mean difference: {np.mean(X, axis=0) - np.mean(synthetic_data, axis=0)}")
    print(f"Std difference:  {np.std(X, axis=0) - np.std(synthetic_data, axis=0)}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Original vs Synthetic Data (with Input Scaling)', fontsize=16)
    
    for i in range(3):
        axes[i].hist(X[:, i], bins=30, alpha=0.5, label='Original', density=True)
        axes[i].hist(synthetic_data[:, i], bins=30, alpha=0.5, label='Synthetic', density=True)
        axes[i].set_title(f'Variable {i+1}')
        axes[i].legend()
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')
    
    plt.tight_layout()
    plt.savefig('scaling_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved as 'scaling_comparison.png'")
    
    # Test with different scales
    print(f"\n=== Testing with Different Scales ===")
    
    # Create data with extreme scales
    X_extreme = np.column_stack([
        np.random.normal(0, 1, n_samples),      # Standard
        np.random.normal(1e6, 1e5, n_samples),  # Very large
        np.random.normal(0, 1e-6, n_samples)     # Very small
    ])
    
    print("Extreme scale data statistics:")
    print(f"  Mean: {np.mean(X_extreme, axis=0)}")
    print(f"  Std:  {np.std(X_extreme, axis=0)}")
    
    # Fit model on extreme scale data
    model_extreme = ConformalInference(
        type_pi="bootstrap",
        objective="crps",
        optimizer="optuna",
        split_ratio=0.5,
        seed=42
    )
    
    model_extreme.fit(X_extreme)
    synthetic_extreme = model_extreme.sample(n_samples=100)
    
    print("Synthetic data from extreme scales:")
    print(f"  Mean: {np.mean(synthetic_extreme, axis=0)}")
    print(f"  Std:  {np.std(synthetic_extreme, axis=0)}")
    
    print(f"\nInput scaling successfully implemented and tested!")

if __name__ == "__main__":
    main()
