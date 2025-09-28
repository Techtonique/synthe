import numpy as np
import pandas as pd
from synthe.conformal_inference import ConformalInference
from scipy.stats import ks_2samp
from scipy.spatial.distance import cdist

np.random.seed(1234)

# Parameters for the mixture
n_components = 3  # Number of Gaussian components
n_samples = 500  # Total number of samples

# Define means, standard deviations, and mixing weights for the Gaussians
means = [0, 5, 10]
std_devs = [1, 1.5, 2]
weights = [0.3, 0.4, 0.3]  # Weights of each component

# Make sure the weights sum to 1
weights /= np.sum(weights)

# Generate the mixture model
# Step 1: Randomly choose the Gaussian component for each sample
component_choices = np.random.choice(n_components, size=n_samples, p=weights)

# Step 2: Sample from the corresponding Gaussian distribution
samples = np.zeros(n_samples)
for i in range(n_components):
    # Select the indices of the samples assigned to this component
    component_indices = component_choices == i
    # Sample from the Gaussian distribution for these indices
    samples[component_indices] = np.random.normal(means[i], std_devs[i], component_indices.sum())

for method in ['optuna', 'gpopt']:

    print(f"\nUsing optimizer: {method}")

    ci = ConformalInference(optimizer=method, objective="crps")

    ci.fit(samples)
    synthetic = ci.sample(n_samples=n_samples).ravel()
    # Test de Kolmogorov-Smirnov
    ks_stat, ks_pvalue = ks_2samp(samples, synthetic)
    print(f"KS test: statistic={ks_stat:.4f}, p-value={ks_pvalue:.4g}")

    # Energy distance (scipy >= 1.2.0)
    try:
        from scipy.stats import energy_distance
        edist = energy_distance(samples, synthetic)
        print(f"Energy distance: {edist:.4f}")
    except ImportError:
        # Fallback: simple mean Euclidean distance
        edist = np.mean(cdist(samples.reshape(-1,1), synthetic.reshape(-1,1)))
        print(f"Mean Euclidean distance: {edist:.4f}")

