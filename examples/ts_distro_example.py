import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from synthe.meboot import MaximumEntropyBootstrap

# Set style for better plots
plt.style.use('seaborn-v0_8')

# Example 1: Basic usage with economic data simulation
print("=== Example 1: Basic Hypothesis Testing ===")
np.random.seed(42)

# Simulate economic time series (like consumption data)
t = np.arange(100)
trend = 0.05 * t
seasonal = 10 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 2, 100)
economic_data = 100 + trend + seasonal + noise

# Fit ME bootstrap to the economic time series
diff_data = np.diff(economic_data)
meb = MaximumEntropyBootstrap(random_state=42)
meb.fit(diff_data)

# Plot bootstrap ensemble
fig = meb.plot_bootstrap_ensemble(reps=50)
plt.show()

# Hypothesis test: Is the mean of the differenced series zero?
test_result = meb.test_mean(null_value=0)
print("\nME Bootstrap Mean Test:")
print(f"  Statistic: {test_result.statistic:.4f}")
print(f"  p-value: {test_result.p_value:.4f}")
print(f"  95% CI: [{test_result.ci_lower:.4f}, {test_result.ci_upper:.4f}]")
print(f"  Reject null: {test_result.reject_null}")

# Plot sampling distribution of the mean
fig, stats_boot = meb.plot_sampling_distribution(np.mean)
plt.show()

# Summary statistics
df_summary = meb.summary()
print("\nSummary statistics:")
print(df_summary)

