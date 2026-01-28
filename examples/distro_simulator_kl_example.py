import numpy as np
from synthe import DistroSimulator

# ============================================================================
# EXAMPLE: Using KL divergence metric for distribution matching
# ============================================================================

print("=" * 70)
print("TESTING KL DIVERGENCE METRIC IN DISTROSIMULATOR")
print("=" * 70)

# Generate synthetic multivariate data
np.random.seed(42)
n_samples = 300
X_true = np.random.randn(n_samples, 2)
Y_true = np.column_stack([
    X_true[:, 0] + 0.5 * X_true[:, 1] + np.random.randn(n_samples) * 0.1,
    X_true[:, 1]**2 + np.random.randn(n_samples) * 0.1
])

print(f"\nOriginal data shape: {Y_true.shape}")
print(f"Original data mean: {Y_true.mean(axis=0)}")
print(f"Original data std: {Y_true.std(axis=0)}")

# Test different metrics
print("\n" + "-" * 70)
print("COMPARING DIFFERENT METRICS")
print("-" * 70)

metrics = ["energy", "mmd", "kl"]
results = {}

for metric_name in metrics:
    print(f"\nFitting with metric: {metric_name.upper()}")
    
    sim = DistroSimulator(
        random_state=42, 
        use_rff=False, 
        n_clusters=3,
        residual_sampling="kde"
    )
    
    sim.fit(Y_true, n_trials=15, metric=metric_name)
    samples = sim.sample(200)
    
    # Compute KL divergence between original and synthetic
    kl_value = sim._kl_divergence(Y_true, samples)
    
    results[metric_name] = {
        'best_params': sim.best_params_,
        'best_score': sim.best_score_,
        'kl_to_original': kl_value,
        'mean': samples.mean(axis=0),
        'std': samples.std(axis=0)
    }
    
    print(f"  Best score ({metric_name}): {sim.best_score_:.6f}")
    print(f"  KL divergence to original: {kl_value:.6f}")
    print(f"  Synthetic mean: {samples.mean(axis=0)}")
    print(f"  Synthetic std: {samples.std(axis=0)}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nOriginal - Mean: {Y_true.mean(axis=0)}, Std: {Y_true.std(axis=0)}")
print("\nBest KL divergence to original:")
for metric_name in metrics:
    kl_val = results[metric_name]['kl_to_original']
    print(f"  {metric_name.upper():10s}: {kl_val:.6f}")

print("\n" + "=" * 70)
print("NOTES:")
print("- KL divergence measures how much information is lost when using")
print("  the synthetic distribution Q to approximate the observed distribution P")
print("- KL(P||Q) is always non-negative, with 0 meaning identical distributions")
print("- Lower KL divergence indicates better distribution matching")
print("=" * 70)
