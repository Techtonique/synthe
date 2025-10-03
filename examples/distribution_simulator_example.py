import numpy as np
import pandas as pd
from synthe.distribution_simulator import DistributionSimulator
from scipy.stats import ks_2samp, anderson_ksamp

# Example 1: Univariate Normal
np.random.seed(42)
Y = np.random.normal(loc=0, scale=1, size=300)

sim = DistributionSimulator(model_type='krr', random_state=42, verbose=True)
sim.fit(Y)
samples = sim.sample(500)

print("Univariate simulation:")
print("Original mean/std:", np.mean(Y), np.std(Y))
print("Simulated mean/std:", np.mean(samples), np.std(samples))

# Statistical tests
ks_stat, ks_p = ks_2samp(Y, samples)
print(f"KS test: statistic={ks_stat:.4f}, p-value={ks_p:.4g}")

ad_result = anderson_ksamp([Y, samples])
print(f"Anderson-Darling test: statistic={ad_result.statistic:.4f}, "
      f"p-value={ad_result.significance_level:.4g}")

# Example 2: Multivariate Normal
Y_multi = np.random.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], size=300)

sim_multi = DistributionSimulator(model_type='krr', random_state=123, verbose=True)
sim_multi.fit(Y_multi)
samples_multi = sim_multi.sample(500)

print("\nMultivariate simulation:")
print("Original mean/std:", np.mean(Y_multi, axis=0), np.std(Y_multi, axis=0))
print("Simulated mean/std:", np.mean(samples_multi, axis=0), np.std(samples_multi, axis=0))

# KS and AD tests for each marginal
for i in range(Y_multi.shape[1]):
    ks_stat, ks_p = ks_2samp(Y_multi[:, i], samples_multi[:, i])
    ad_result = anderson_ksamp([Y_multi[:, i], samples_multi[:, i]])
    print(f"Marginal {i}: KS statistic={ks_stat:.4f}, p-value={ks_p:.4g}; "
          f"AD statistic={ad_result.statistic:.4f}, "
          f"p-value={ad_result.significance_level:.4g}")

# Example 3: Stocks 

log_returns = pd.read_csv("https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/time_series/multivariate/log_returns.csv")
log_returns.drop(columns=["Unnamed: 0"], inplace=True)
log_returns.index = pd.date_range(start="2024-04-24", 
                                  periods=len(log_returns), 
                                  freq="B")
Y = log_returns.values
sim_stocks = DistributionSimulator(model_type='krr', random_state=42, verbose=True)
sim_stocks.fit(Y)
samples_stocks = sim_stocks.sample(500)
print("\nStocks simulation KRR:")
print("Original mean/std:", np.mean(Y, axis=0), np.std(Y, axis=0))
print("Simulated mean/std:", np.mean(samples_stocks, axis=0), np.std(samples_stocks, axis=0))
# KS and AD tests for each marginal
for i in range(Y.shape[1]):
    ks_stat, ks_p = ks_2samp(Y[:, i], samples_stocks[:, i])
    ad_result = anderson_ksamp([Y[:, i], samples_stocks[:, i]])
    print(f"Stock {i}: KS statistic={ks_stat:.4f}, p-value={ks_p:.4g}; "
          f"AD statistic={ad_result.statistic:.4f}, "
          f"p-value={ad_result.significance_level:.4g}")
    

sim_stocks = DistributionSimulator(model_type='xgboost', random_state=42, verbose=True)
sim_stocks.fit(Y)
samples_stocks = sim_stocks.sample(500)
print("\nStocks simulation XGB:")
print("Original mean/std:", np.mean(Y, axis=0), np.std(Y, axis=0))
print("Simulated mean/std:", np.mean(samples_stocks, axis=0),
      np.std(samples_stocks, axis=0))
# KS and AD tests for each marginal
for i in range(Y.shape[1]):
    ks_stat, ks_p = ks_2samp(Y[:, i], samples_stocks[:, i])
    ad_result = anderson_ksamp([Y[:, i], samples_stocks[:, i]])
    print(f"Stock {i}: KS statistic={ks_stat:.4f}, p-value={ks_p:.4g}; "
          f"AD statistic={ad_result.statistic:.4f}, "
          f"p-value={ad_result.significance_level:.4g}")

sim_stocks = DistributionSimulator(model_type='xgboost', random_state=42, verbose=True, metric='mmd')
sim_stocks.fit(Y)
samples_stocks = sim_stocks.sample(500)
print("\nStocks simulation XGB:")
print("Original mean/std:", np.mean(Y, axis=0), np.std(Y, axis=0))
print("Simulated mean/std:", np.mean(samples_stocks, axis=0),
      np.std(samples_stocks, axis=0))
# KS and AD tests for each marginal
for i in range(Y.shape[1]):
    ks_stat, ks_p = ks_2samp(Y[:, i], samples_stocks[:, i])
    ad_result = anderson_ksamp([Y[:, i], samples_stocks[:, i]])
    print(f"Stock {i}: KS statistic={ks_stat:.4f}, p-value={ks_p:.4g}; "
          f"AD statistic={ad_result.statistic:.4f}, "
          f"p-value={ad_result.significance_level:.4g}")


sim_stocks = DistributionSimulator(model_type='xgboost', random_state=42, verbose=True, metric='energy')
sim_stocks.fit(Y)
samples_stocks = sim_stocks.sample(500)
print("\nStocks simulation XGB:")
print("Original mean/std:", np.mean(Y, axis=0), np.std(Y, axis=0))
print("Simulated mean/std:", np.mean(samples_stocks, axis=0),
      np.std(samples_stocks, axis=0))
# KS and AD tests for each marginal
for i in range(Y.shape[1]):
    ks_stat, ks_p = ks_2samp(Y[:, i], samples_stocks[:, i])
    ad_result = anderson_ksamp([Y[:, i], samples_stocks[:, i]])
    print(f"Stock {i}: KS statistic={ks_stat:.4f}, p-value={ks_p:.4g}; "
          f"AD statistic={ad_result.statistic:.4f}, "
          f"p-value={ad_result.significance_level:.4g}")


# # Optional: Plot for visual inspection
# try:
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.hist(Y, bins=30, alpha=0.5, label='Original')
#     plt.hist(samples, bins=30, alpha=0.5, label='Simulated')
#     plt.title('Univariate')
#     plt.legend()
#
#     plt.subplot(1, 2, 2)
#     plt.scatter(Y_multi[:, 0], Y_multi[:, 1], alpha=0.4, label='Original')
#     plt.scatter(samples_multi[:, 0], samples_multi[:, 1], alpha=0.4, label='Simulated')
#     plt.title('Multivariate')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
# except ImportError:
#     pass