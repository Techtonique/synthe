import numpy as np
from scipy.stats import rankdata, norm, ks_2samp
from synthe.diversity_generator import DiversityGenerator


# Test the controlled diversity approach
def test_controlled_diversity():
    """Test the controlled diversity generation"""
    np.random.seed(42)

    print("CONTROLLED DIVERSITY VIA TARGET CORRELATION MATRIX")
    print("=" * 70)

    # Test with different target correlations
    target_correlations = [0.8, 0.3, 0.0, -0.2, -0.5]

    for target_corr in target_correlations:
        print(f"\n" + "="*50)
        print(f"TARGET CORRELATION: {target_corr}")
        print("="*50)

        # Generate correlated test data
        n_samples = 1000
        n_features = 3

        # Create correlated data
        base = np.random.normal(0, 1, n_samples)
        X = np.column_stack([
            base + 0.1 * np.random.normal(0, 1, n_samples),
            base + 0.2 * np.random.normal(0, 1, n_samples),
            base + 0.3 * np.random.normal(0, 1, n_samples)
        ])

        print(f"Original data correlations:")
        orig_corr = np.corrcoef(X.T)
        print(orig_corr)

        # Apply controlled diversity
        generator = DiversityGenerator(
            target_correlation=target_corr,
            random_state=42
        )

        diverse_samples = generator.generate_diverse_samples(X, n_samples=3)

        # Analyze results
        print(f"\nDiverse samples correlations (average):")
        diverse_corrs = []
        for i in range(len(diverse_samples)):
            corr = np.corrcoef(diverse_samples[i].T)
            diverse_corrs.append(corr)
            if i == 0:
                print(f"Sample {i}:")
                print(corr)


        avg_diverse_corr = np.mean(diverse_corrs, axis=0)
        print(f"\nAverage diverse correlation:")
        print(avg_diverse_corr)

        # Check distribution preservation
        print(f"\nDistribution preservation (KS tests):")
        ks_pvalues = []
        for j in range(n_features):
            ks_stat, pvalue = ks_2samp(X[:, j], diverse_samples[0, :, j])
            ks_pvalues.append(pvalue)
            print(f"  Feature {j}: p-value = {pvalue:.4f} {'✓' if pvalue > 0.05 else '✗'}")

        success_rate = np.mean([p > 0.05 for p in ks_pvalues])
        print(f"Success rate: {success_rate:.1%}")

        # Calculate achieved correlation vs target
        achieved_off_diag = avg_diverse_corr[np.triu_indices(n_features, 1)]
        target_achieved = np.mean(achieved_off_diag)
        print(f"Target correlation: {target_corr:.2f}, Achieved: {target_achieved:.2f}")

# Test with custom correlation matrices
def test_custom_correlation_matrices():
    """Test with custom correlation matrix structures"""
    np.random.seed(42)

    print("\n" + "=" * 70)
    print("CUSTOM CORRELATION MATRIX STRUCTURES")
    print("=" * 70)

    n_features = 4
    n_samples = 1000

    # Generate test data
    X = np.random.normal(0, 1, (n_samples, n_features))

    # Test different correlation structures
    correlation_structures = {
        "Low Positive": np.array([
            [1.0, 0.1, 0.1, 0.1],
            [0.1, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.1],
            [0.1, 0.1, 0.1, 1.0]
        ]),

        "Anti-Correlation": np.array([
            [1.0, -0.3, -0.2, -0.1],
            [-0.3, 1.0, -0.2, -0.1],
            [-0.2, -0.2, 1.0, -0.3],
            [-0.1, -0.1, -0.3, 1.0]
        ]),

        "Block Structure": np.array([
            [1.0, 0.8, 0.1, 0.1],
            [0.8, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.7],
            [0.1, 0.1, 0.7, 1.0]
        ])
    }

    for name, target_corr in correlation_structures.items():
        print(f"\n--- {name} ---")
        print(f"Target correlation matrix:")
        print(target_corr)

        generator = DiversityGenerator(
            target_correlation=target_corr,
            random_state=42
        )

        diverse_samples = generator.generate_diverse_samples(X, n_samples=2)

        # Check achieved correlation
        achieved_corr = np.corrcoef(diverse_samples[0].T)
        print(f"Achieved correlation matrix:")
        print(achieved_corr)

        # Calculate error
        error = np.mean(np.abs(target_corr - achieved_corr))
        print(f"Mean absolute error: {error:.4f}")

# Test with challenging distributions
def test_challenging_distributions():
    """Test with non-Gaussian distributions"""
    np.random.seed(42)

    print("\n" + "=" * 70)
    print("CHALLENGING DISTRIBUTIONS")
    print("=" * 70)

    n_samples = 1000

    # Create mixed non-Gaussian data
    X = np.column_stack([
        np.random.exponential(2, n_samples),      # Exponential
        np.random.chisquare(3, n_samples),        # Chi-squared
        np.random.lognormal(0, 0.5, n_samples),   # Log-normal
        np.random.beta(2, 5, n_samples)           # Beta
    ])

    # Add some correlation
    X[:, 1] = 0.6 * X[:, 0] + 0.4 * X[:, 1]
    X[:, 2] = 0.4 * X[:, 0] + 0.6 * X[:, 2]

    print("Original distributions:")
    for j in range(X.shape[1]):
        print(f"  Feature {j}: mean={np.mean(X[:, j]):.3f}, std={np.std(X[:, j]):.3f}")

    print(f"\nOriginal correlations:")
    print(np.corrcoef(X.T))

    # Apply diversity with negative correlation target
    generator = DiversityGenerator(
        target_correlation=-0.2,  # Slightly negative for diversity
        preserve_moments=True,
        random_state=42
    )

    diverse_samples = generator.generate_diverse_samples(X, n_samples=3)

    print(f"\nDiverse sample correlations (first sample):")
    print(np.corrcoef(diverse_samples[0].T))

    # Check distribution preservation
    print(f"\nDistribution preservation:")
    for j in range(X.shape[1]):
        ks_stat, pvalue = ks_2samp(X[:, j], diverse_samples[0, :, j])
        mean_ratio = np.mean(diverse_samples[0, :, j]) / np.mean(X[:, j])
        std_ratio = np.std(diverse_samples[0, :, j]) / np.std(X[:, j])

        print(f"  Feature {j}: KS p={pvalue:.4f}, mean ratio={mean_ratio:.3f}, std ratio={std_ratio:.3f}")


# Run all tests
if __name__ == "__main__":
    test_controlled_diversity()
    test_custom_correlation_matrices()
    test_challenging_distributions()