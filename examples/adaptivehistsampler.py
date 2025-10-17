import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from synthe.adaptivehistsampler import AdaptiveHistogramSampler

# Generate data
X_train = np.random.randn(500, 2)

# Fit sampler on training data
sampler = AdaptiveHistogramSampler(n_bins=10, method="quantile")
sampler.fit(X_train)

# Sample new data
X_sampled = sampler.sample(1000, oversample=True, oversample_method="jitter")

# Generate FRESH test data (completely independent)
X_test = np.random.randn(1000, 2)  # Same distribution, different points

# Proper evaluation - test if sampler captures the TRUE distribution
sampler.plot_comparison(X_test)  # Compare with true distribution
results = sampler.goodness_of_fit(X_test)  # Test against independent data

print("Goodness-of-fit results against independent test data:")
for key, value in results.items():
    print(f"  {key}: {value}")