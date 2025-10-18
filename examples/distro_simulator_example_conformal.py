"""
DistroSimulator Examples - Converting Colab notebook examples
"""

import numpy as np
import matplotlib.pyplot as plt
from synthe import DistroSimulator  # Assuming your package name is synthe

"""
DistroSimulator Examples - Converting Colab notebook examples
"""

import numpy as np
import matplotlib.pyplot as plt
from synthe import DistroSimulator  # Assuming your package name is synthe

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("EXAMPLE 1: UNIVARIATE NORMAL DISTRIBUTION")
print("=" * 60)

# Generate univariate normal data
n = 500
Y_uni = np.random.normal(0, 1, n).reshape(-1, 1)

# Initialize and fit the simulator
simulator_uni = DistroSimulator(
    kernel='rbf',
    residual_sampling='bootstrap',
    random_state=42,
    conformalize=True
)

# Fit the model
simulator_uni.fit(Y_uni, n_trials=50, metric='energy')

# Generate synthetic samples
Y_sim_uni = simulator_uni.sample(n_samples=500)

print(f"\nBest hyperparameters:")
print(f"  Sigma: {simulator_uni.best_params_['sigma']:.4f}")
print(f"  Lambda: {simulator_uni.best_params_['lambd']:.6f}")
print(f"  Best score: {simulator_uni.best_score_:.6f}")

# Test similarity
results_uni = simulator_uni.test_similarity(Y_uni, Y_sim_uni, n_perm=250)
print("\nSimilarity test results:")
for key, value in results_uni.items():
    if key != 'corr_tests':
        print(f"  {key}: {value}")

# Compare distributions visually
simulator_uni.compare_distributions(Y_uni, Y_sim_uni, save_prefix="univariate")


print("\n" + "=" * 60)
print("EXAMPLE 2: BIVARIATE NORMAL WITH CORRELATION")
print("=" * 60)

# Generate bivariate normal data with correlation
mean = [0, 0]
cov = [[1, 0.7], [0.7, 1]]
Y_multi = np.random.multivariate_normal(mean, cov, n)

# Initialize simulator for multivariate data
simulator_multi = DistroSimulator(
    kernel='rbf',
    residual_sampling='kde',  # Try KDE sampling
    random_state=42,
    conformalize=True
)

# Fit the model
simulator_multi.fit(Y_multi, n_trials=50, metric='mmd')

# Generate synthetic samples
Y_sim_multi = simulator_multi.sample(n_samples=1000)

print(f"\nBest hyperparameters:")
print(f"  Sigma: {simulator_multi.best_params_['sigma']:.4f}")
print(f"  Lambda: {simulator_multi.best_params_['lambd']:.6f}")
print(f"  Best score: {simulator_multi.best_score_:.6f}")

# Test similarity
results_multi = simulator_multi.test_similarity(Y_multi, Y_sim_multi, n_perm=250)
print("\nSimilarity test results:")
for key, value in results_multi.items():
    if key == 'corr_tests':
        print("  Correlation tests:")
        for pair, stats in value.items():
            print(f"    {pair}: r_orig={stats[0]:.3f}, r_sim={stats[1]:.3f}, p={stats[3]:.4f}")
    else:
        print(f"  {key}: {value}")

# Compare distributions visually
simulator_multi.compare_distributions(Y_multi, Y_sim_multi, save_prefix="bivariate")


print("\n" + "=" * 60)
print("EXAMPLE 3: 3D MIXTURE OF GAUSSIANS")
print("=" * 60)

# Generate 3D mixture of multivariate normals
n_samples = 800

cov_matrix1 = np.array([[1.0, 0.5, 0.3],
                        [0.5, 1.5, 0.4],
                        [0.3, 0.4, 0.8]])
component1 = np.random.multivariate_normal([0, 1, -0.5], cov_matrix1, int(0.6 * n_samples))

cov_matrix2 = np.array([[0.8, -0.3, 0.1],
                        [-0.3, 1.2, -0.2],
                        [0.1, -0.2, 1.0]])
component2 = np.random.multivariate_normal([2, -1, 1], cov_matrix2, int(0.4 * n_samples))

Y_mixture = np.vstack([component1, component2])
np.random.shuffle(Y_mixture)

print(f"Sample size: {len(Y_mixture)}")
print("\nOriginal correlation matrix:")
print(np.corrcoef(Y_mixture.T))

# Initialize simulator with GMM residual sampling
simulator_mixture = DistroSimulator(
    kernel='rbf',
    residual_sampling='gmm',
    gmm_components=5,
    use_rff='auto',  # Auto-enable RFF for large datasets
    random_state=42
)

# Fit the model
simulator_mixture.fit(Y_mixture, n_trials=50, metric='energy')

# Generate synthetic samples
Y_sim_mixture = simulator_mixture.sample(n_samples=1000)

print(f"\nBest hyperparameters:")
print(f"  Sigma: {simulator_mixture.best_params_['sigma']:.4f}")
print(f"  Lambda: {simulator_mixture.best_params_['lambd']:.6f}")
print(f"  Best score: {simulator_mixture.best_score_:.6f}")

# Test similarity
results_mixture = simulator_mixture.test_similarity(Y_mixture, Y_sim_mixture, n_perm=250)
print("\nSimilarity test results:")
for key, value in results_mixture.items():
    if key == 'corr_tests':
        print("  Correlation tests:")
        for pair, stats in value.items():
            print(f"    {pair}: r_orig={stats[0]:.3f}, r_sim={stats[1]:.3f}, p={stats[3]:.4f}")
    else:
        print(f"  {key}: {value}")

# Compare distributions visually
simulator_mixture.compare_distributions(Y_mixture, Y_sim_mixture, save_prefix="mixture")


print("\n" + "=" * 60)
print("EXAMPLE 4: COMPARE RESIDUAL SAMPLING METHODS")
print("=" * 60)

# Generate test data
Y_test = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 500)

# Test different residual sampling methods
sampling_methods = ['bootstrap', 'kde', 'gmm']
results_comparison = {}

for method in sampling_methods:
    print(f"\nTesting {method.upper()} sampling...")
    
    sim = DistroSimulator(
        residual_sampling=method,
        random_state=42,
        conformalize=True
    )
    
    sim.fit(Y_test, n_trials=30, metric='energy')
    Y_sim_test = sim.sample(n_samples=1000)
    
    results_comparison[method] = {
        'best_score': sim.best_score_,
        'sigma': sim.best_params_['sigma'],
        'lambda': sim.best_params_['lambd']
    }
    
    print(f"  Best score: {sim.best_score_:.6f}")

print("\nComparison of sampling methods:")
for method, results in results_comparison.items():
    print(f"{method.upper()}:")
    print(f"  Score: {results['best_score']:.6f}")
    print(f"  Sigma: {results['sigma']:.4f}")
    print(f"  Lambda: {results['lambda']:.6f}")

# Visualize residual sampling comparison
simulator_multi.compare_residual_sampling(n_samples=1000)


print("\n" + "=" * 60)
print("EXAMPLE 5: COMPARE KERNEL APPROXIMATION METHODS")
print("=" * 60)

# Generate larger dataset to see RFF benefits
Y_large = np.random.multivariate_normal([0, 0, 0], 
                                        [[1, 0.3, 0.2], 
                                         [0.3, 1, 0.4], 
                                         [0.2, 0.4, 1]], 
                                        1500)

simulator_approx = DistroSimulator(random_state=42)

# Compare approximation methods
comparison_results = simulator_approx.compare_approximation_methods(
    Y_large, 
    n_train=750,
    n_trials=20
)

print("\nApproximation method comparison complete!")
print("See printed output above for detailed results.")


print("\n" + "=" * 60)
print("EXAMPLE 6: DIGITS DATASET (requires sklearn)")
print("=" * 60)

try:
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    
    # Load and transform digits data
    digits = load_digits()
    pca = PCA(n_components=15, whiten=False)
    Y_digits = pca.fit_transform(digits.data)
    
    print(f"Transformed digits shape: {Y_digits.shape}")
    
    # Initialize simulator
    simulator_digits = DistroSimulator(
        use_rff='auto',
        residual_sampling='kde',
        random_state=42,
        conformalize=True
    )
    
    # Fit the model
    simulator_digits.fit(Y_digits, n_trials=30, metric='mmd')
    
    # Generate synthetic samples
    Y_sim_digits = simulator_digits.sample(n_samples=len(Y_digits))
    
    print(f"\nBest hyperparameters:")
    print(f"  Sigma: {simulator_digits.best_params_['sigma']:.4f}")
    print(f"  Lambda: {simulator_digits.best_params_['lambd']:.6f}")
    print(f"  Best score: {simulator_digits.best_score_:.6f}")
    
    # Test similarity
    results_digits = simulator_digits.test_similarity(Y_digits, Y_sim_digits, n_perm=100)
    print("\nKey similarity metrics:")
    print(f"  Energy distance p-value: {results_digits['energy_perm'][1]:.4f}")
    print(f"  MMD p-value: {results_digits['mmd_perm'][1]:.4f}")
    print(f"  Min marginal KS p-value: {results_digits['min_marginal_ks_p']:.4f}")
    
    # Visualize a few dimensions
    simulator_digits.compare_distributions(Y_digits, Y_sim_digits, save_prefix="digits")
    
    print("\nDigits example complete!")
    
except ImportError:
    print("sklearn not available, skipping digits example")


print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)