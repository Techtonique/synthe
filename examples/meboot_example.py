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

meb = MaximumEntropyBootstrap(random_state=42)
meb.fit(economic_data)

# sample bootstrap replicates
bootstrap_samples = meb.sample(reps=50)
print(f"Generated {bootstrap_samples} bootstrap samples.")

# Test if mean is different from 100
result = meb.test_mean(null_value=100, alternative='two-sided')
print(f"Test for mean = 100:")
print(f"  Statistic: {result.statistic:.3f}")
print(f"  P-value: {result.p_value:.4f}")
print(f"  95% CI: ({result.ci_lower:.3f}, {result.ci_upper:.3f})")
print(f"  Reject null: {result.reject_null}")

# Test if variance is greater than 4
result_var = meb.test_variance(null_value=4, alternative='greater')
print(f"\nTest for variance > 4:")
print(f"  Statistic: {result_var.statistic:.3f}")
print(f"  P-value: {result_var.p_value:.4f}")
print(f"  Reject null: {result_var.reject_null}")

# Generate plots
print("\nGenerating plots...")
fig1 = meb.plot_me_density()
fig1.suptitle('Maximum Entropy Density Analysis - Economic Data', fontsize=14)

fig2 = meb.plot_bootstrap_ensemble(reps=500)
fig2.suptitle('Bootstrap Ensemble - Economic Data', fontsize=14)

# Plot sampling distribution of mean
def custom_statistic(x):
    return np.mean(x)

fig3, boot_stats = meb.plot_sampling_distribution(custom_statistic, reps=999)
fig3.suptitle('Sampling Distribution of Mean - Economic Data', fontsize=14)

# Example 2: Correlation test (like in the consumption function)
print("\n=== Example 2: Correlation Testing ===")

# Simulate income data correlated with consumption
income_data = economic_data * 1.1 + np.random.normal(0, 5, 100)

# Test if correlation is different from 0
result_corr = meb.test_correlation(income_data, null_value=0)
print(f"Test for correlation = 0:")
print(f"  Correlation: {result_corr.statistic:.3f}")
print(f"  P-value: {result_corr.p_value:.4f}")
print(f"  95% CI: ({result_corr.ci_lower:.3f}, {result_corr.ci_upper:.3f})")
print(f"  Reject null: {result_corr.reject_null}")

# Example 3: Compare with traditional t-test
print("\n=== Example 3: Comparison with Traditional Methods ===")

# Traditional t-test
t_stat, t_pval = stats.ttest_1samp(economic_data, 100)
print(f"Traditional t-test for mean = 100:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  P-value: {t_pval:.4f}")

# Bootstrap test for comparison
boot_result = meb.test_mean(null_value=100)
print(f"Bootstrap test for mean = 100:")
print(f"  P-value: {boot_result.p_value:.4f}")

# Summary statistics
print("\n=== Summary Statistics ===")
summary_df = meb.summary()
print(summary_df.T)

plt.show()
