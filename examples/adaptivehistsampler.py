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
sampler.plot_comparison(X_sampled)  # NOT X_test!
results = sampler.goodness_of_fit(X_sampled)  # NOT X_test!
print("Goodness-of-fit results:", results)
