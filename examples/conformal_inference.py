import numpy as np
import pandas as pd
from synthe.conformal_inference import ConformalInference

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

ci = ConformalInference(optimizer='gpopt')

ci.fit(samples)
print(ci.sample(n_samples=100))

ci = ConformalInference(optimizer='optuna')

ci.fit(samples)
print(ci.sample(n_samples=100))