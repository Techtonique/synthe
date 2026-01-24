import numpy as np
from synthe import DistroSimulator

# ============================================================================
# EXAMPLE: Testing reproducibility with different seeds
# ============================================================================

print("=" * 70)
print("TESTING REPRODUCIBILITY WITH DIFFERENT RANDOM SEEDS")
print("=" * 70)

# Generate synthetic multivariate data
np.random.seed(42)
n_samples = 200
X_true = np.random.randn(n_samples, 2)
Y_true = np.column_stack([
    X_true[:, 0] + 0.5 * X_true[:, 1] + np.random.randn(n_samples) * 0.1,
    X_true[:, 1]**2 + np.random.randn(n_samples) * 0.1
])

print(f"\nOriginal data shape: {Y_true.shape}")
print(f"Original data mean: {Y_true.mean(axis=0)}")
print(f"Original data std: {Y_true.std(axis=0)}")

# Test 1: Same seed should give identical results
print("\n" + "-" * 70)
print("TEST 1: Same seed (42) - Should produce identical results")
print("-" * 70)

sim1 = DistroSimulator(random_state=42, use_rff=False, n_clusters=3)
sim1.fit(Y_true, n_trials=10)
samples1_a = sim1.sample(100)

sim2 = DistroSimulator(random_state=42, use_rff=False, n_clusters=3)
sim2.fit(Y_true, n_trials=10)
samples1_b = sim2.sample(100)

print(f"\nRun 1 - Best params: {sim1.best_params_}")
print(f"Run 2 - Best params: {sim2.best_params_}")
print(f"\nAre best params identical? {sim1.best_params_ == sim2.best_params_}")
print(f"Are samples identical? {np.allclose(samples1_a, samples1_b)}")
print(f"Max difference in samples: {np.max(np.abs(samples1_a - samples1_b)):.10f}")

# Test 2: Different seeds should give different results
print("\n" + "-" * 70)
print("TEST 2: Different seeds (42 vs 123) - Should produce different results")
print("-" * 70)

sim3 = DistroSimulator(random_state=42, use_rff=False, n_clusters=3)
sim3.fit(Y_true, n_trials=10)
samples2_a = sim3.sample(100)

sim4 = DistroSimulator(random_state=123, use_rff=False, n_clusters=3)
sim4.fit(Y_true, n_trials=10)
samples2_b = sim4.sample(100)

print(f"\nSeed 42 - Best params: {sim3.best_params_}")
print(f"Seed 123 - Best params: {sim4.best_params_}")
print(f"\nAre best params identical? {sim3.best_params_ == sim4.best_params_}")
print(f"Are samples identical? {np.allclose(samples2_a, samples2_b)}")
print(f"Max difference in samples: {np.max(np.abs(samples2_a - samples2_b)):.6f}")

# Test 3: Different residual sampling methods with same seed
print("\n" + "-" * 70)
print("TEST 3: Different sampling methods with same seed (42)")
print("-" * 70)

for method in ["bootstrap", "kde", "gmm"]:
    sim = DistroSimulator(
        random_state=42, 
        use_rff=False, 
        residual_sampling=method,
        n_clusters=3
    )
    sim.fit(Y_true, n_trials=5)
    samples = sim.sample(100)
    print(f"\n{method.upper():12s} - Mean: {samples.mean(axis=0)}, "
            f"Std: {samples.std(axis=0)}")

# Test 4: Test reproducibility with RFF
print("\n" + "-" * 70)
print("TEST 4: RFF approximation with same seed - Should be reproducible")
print("-" * 70)

sim5 = DistroSimulator(random_state=42, use_rff=True, rff_components=50, n_clusters=3)
sim5.fit(Y_true, n_trials=10)
samples3_a = sim5.sample(100)

sim6 = DistroSimulator(random_state=42, use_rff=True, rff_components=50, n_clusters=3)
sim6.fit(Y_true, n_trials=10)
samples3_b = sim6.sample(100)

print(f"\nRun 1 - Best params: {sim5.best_params_}")
print(f"Run 2 - Best params: {sim6.best_params_}")
print(f"\nAre best params identical? {sim5.best_params_ == sim6.best_params_}")
print(f"Are samples identical? {np.allclose(samples3_a, samples3_b)}")
print(f"Max difference in samples: {np.max(np.abs(samples3_a - samples3_b)):.10f}")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETED")
print("=" * 70)