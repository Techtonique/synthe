#!/usr/bin/env python3
"""
Example demonstrating Gaussian Mixture Model sampling in EmpiricalCopula.
"""

import numpy as np
import matplotlib.pyplot as plt
from synthe.empirical_copula import EmpiricalCopula

def main():
    """Demonstrate GMM sampling functionality."""
    
    # Generate test data with complex dependence structure
    np.random.seed(42)
    n_samples = 300
    
    # Create correlated data with different marginal distributions
    x1 = np.random.gamma(2, 1, n_samples)
    x2 = 0.7 * x1 + np.random.normal(0, 0.3, n_samples)
    x3 = np.random.beta(2, 5, n_samples)
    x4 = np.random.exponential(1, n_samples)
    
    # Combine into multivariate data
    X = np.column_stack([x1, x2, x3, x4])
    
    print("=== Empirical Copula with GMM Sampling ===")
    print(f"Original data shape: {X.shape}")
    print("Original correlation matrix:")
    print(np.corrcoef(X.T))
    
    # Fit empirical copula
    copula = EmpiricalCopula(smoothing_method="jitter", jitter_scale=0.01)
    copula.fit(X)
    
    print(f"\nCopula fitted successfully!")
    print(f"Number of samples: {copula.n_samples_}")
    print(f"Number of variables: {copula.n_vars_}")
    
    # Test different sampling methods
    methods = ["bootstrap", "kde", "gmm"]
    n_test_samples = 200
    
    print(f"\n=== Comparing Sampling Methods ===")
    
    for method in methods:
        print(f"\n--- {method.upper()} Method ---")
        
        if method == "gmm":
            # Test different GMM configurations
            configs = [
                {"n_components": 3, "covariance_type": "full"},
                {"n_components": 5, "covariance_type": "diag"},
                {"n_components": 2, "covariance_type": "tied"}
            ]
            
            for i, config in enumerate(configs):
                samples = copula.sample(n_samples=n_test_samples, 
                                      method=method, 
                                      random_state=42,
                                      **config)
                print(f"  GMM Config {i+1} ({config}):")
                print(f"    Generated {len(samples)} samples")
                print(f"    Correlation matrix:")
                corr_matrix = np.corrcoef(samples.T)
                print(f"    {corr_matrix}")
        else:
            samples = copula.sample(n_samples=n_test_samples, 
                                  method=method, 
                                  random_state=42)
            print(f"  Generated {len(samples)} samples")
            print(f"  Correlation matrix:")
            corr_matrix = np.corrcoef(samples.T)
            print(f"  {corr_matrix}")
    
    # Demonstrate GMM with different number of components
    print(f"\n=== GMM with Different Number of Components ===")
    
    n_components_list = [2, 3, 5, 8, 10]
    
    for n_comp in n_components_list:
        samples = copula.sample(n_samples=100, 
                              method="gmm", 
                              n_components=n_comp,
                              covariance_type="full",
                              random_state=42)
        
        # Calculate correlation preservation
        orig_corr = np.corrcoef(X.T)
        sim_corr = np.corrcoef(samples.T)
        corr_diff = np.mean(np.abs(orig_corr - sim_corr))
        
        print(f"  {n_comp} components: avg correlation difference = {corr_diff:.4f}")
    
    # Validate the GMM method
    print(f"\n=== Validation Results ===")
    validation_results = copula.validate_fit(verbose=False)
    print(f"Overall model quality: {validation_results['summary']['overall_quality']}")
    print(f"Pass rate: {validation_results['summary']['pass_rate']:.1f}%")
    print(f"Average Spearman difference: {validation_results['summary']['avg_spearman_difference']:.4f}")
    
    # Show model information
    print(f"\n=== Model Information ===")
    info = copula.get_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print(f"\nGMM sampling successfully implemented and tested!")

if __name__ == "__main__":
    main()
