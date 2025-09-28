import numpy as np
import pandas as pd

from synthe.empirical_copula import EmpiricalCopula

log_returns = pd.read_csv("log_returns.csv")
log_returns.drop(columns=["Unnamed: 0"], inplace=True)
log_returns.index = pd.date_range(start="2024-04-24", 
                                  periods=len(log_returns), 
                                  freq="B")
print(log_returns.head())
print(log_returns.tail())
print(log_returns.shape)

# Test different empirical copula methods
methods = ["none", "jitter"]

for method in methods:
    print(f"\n\n Testing empirical copula with smoothing method: {method} ======")
    copula = EmpiricalCopula(smoothing_method=method)
    copula.fit(log_returns.values)
    simulated_data = copula.sample(n_samples=500)
    copula.validate_fit(log_returns.values)
    copula.plot_pairwise_pseudo()
    #copula.plot_marginals()
    copula.estimate_tail_dependence()

    
