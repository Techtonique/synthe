import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from synthe.conformal_inference import ConformalInference

# Parameters for the mixture
n_components = 3  # Number of Gaussian components
n_samples = 1000  # Total number of samples

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

# Plot the histogram of the samples
plt.hist(samples, bins=30, density=True, alpha=0.6, color='g')

# Plot the individual Gaussian components
x = np.linspace(min(samples), max(samples), 1000)
for i in range(n_components):
    plt.plot(x, weights[i] * (1 / (std_devs[i] * np.sqrt(2 * np.pi))) *
             np.exp(-0.5 * ((x - means[i]) / std_devs[i])**2), label=f'Component {i+1}')

plt.title('Univariate Mixture of Gaussians')
plt.legend()
plt.show()

ci = ConformalInference()

ci.fit(samples)
