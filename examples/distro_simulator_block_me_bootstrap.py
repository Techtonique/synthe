"""
Example: Time series and DistroSimulator residual sampling methods
Demonstrates block-bootstrap and maximum-entropy bootstrap (me-bootstrap)
for time series residuals.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from synthe.distro_simulator import DistroSimulator


def generate_ar1(n, phi=0.8, sigma=1.0, seed=None):
    rng = np.random.RandomState(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(scale=sigma)
    return x


if __name__ == "__main__":
    np.random.seed(42)

    # Create an AR(1) time series to emulate temporal dependence
    n = 1000
    ts = generate_ar1(n, phi=0.8, sigma=1.0, seed=42)
    Y = ts.reshape(-1, 1)  # DistroSimulator expects 2D array (n_samples, n_features)

    # Instantiate simulator with a valid initial sampling method and seed.
    # We'll change `residual_sampling` attribute later to use block-bootstrap / me-bootstrap.
    sim = DistroSimulator(random_state=42, residual_sampling="bootstrap", use_rff=False)

    # Fit model quickly (use small number of optimization trials for example speed)
    print("Fitting simulator (quick demo, n_trials=5)...")
    sim.fit(Y, n_train=500, n_trials=5)

    # Original residuals (after fit)
    orig_res = sim.residuals_.flatten()

    def acf(x, nlags=10):
        x = x - np.mean(x)
        acf_vals = [1.0]
        for lag in range(1, nlags + 1):
            acf_vals.append(np.corrcoef(x[:-lag], x[lag:])[0, 1])
        return np.array(acf_vals)

    print("Original residual ACF (lags 0..5):", np.round(acf(orig_res, nlags=5)[:6], 3))

    # Compare residual sampling methods that preserve temporal structure
    methods = ["block-bootstrap", "me-bootstrap"]

    for m in methods:
        sim.residual_sampling = m
        sim.block_size = 10

        # Check residuals before sampling
        if sim.residuals_ is None or sim.residuals_.size == 0:
            print(f"Residuals are empty for method {m}, skipping.")
            continue

        try:
            sampled = sim._sample_residuals(500)
            if sampled.ndim > 1:
                sampled = sampled.flatten()
        except Exception as e:
            print(f"Error with method {m}: {e}")
            continue

        print(f"\nMethod: {m}")
        print("Sampled residual ACF (lags 0..5):", np.round(acf(sampled, nlags=5)[:6], 3))

        # KS test
        ks_stat, ks_p = ks_2samp(orig_res, sampled)
        print(f"KS test: statistic={ks_stat:.4f}, p-value={ks_p:.4g}")

        # Quick visual comparison
        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.plot(orig_res[:200], label="orig residuals")
        plt.title("Original residuals (first 200)")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(sampled[:200], label=f"sampled ({m})")
        plt.title(f"Sampled residuals ({m}) (first 200)")
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.hist([orig_res, sampled], bins=30, label=["orig", "sampled"], density=True)
        plt.title("Residual distribution")
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Generate synthetic time-series using the fitted simulator
    sim.residual_sampling = "block-bootstrap"
    sim.block_size = 10
    Y_sim = sim.sample(300)
    print("\nGenerated synthetic time-series shape:", Y_sim.shape)

    plt.figure(figsize=(8, 3))
    plt.plot(Y[:200], label="original (first 200)")
    plt.plot(Y_sim[:200], label="simulated (first 200)")
    plt.legend()
    plt.title("Original vs Simulated (first 200 observations)")
    plt.tight_layout()
    plt.show()
