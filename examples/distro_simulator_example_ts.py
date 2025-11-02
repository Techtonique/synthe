"""
DistroSimulator Examples - Converting Colab notebook examples
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from synthe import DistroSimulator  # Assuming your package name is synthe

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("EXAMPLE 1: UNIVARIATE NORMAL DISTRIBUTION")
print("=" * 60)

#url = "https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/time_series/univariate/USAccDeaths.csv"
url = "https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/time_series/univariate/AirPassengers.csv"
#Y_uni = pd.read_csv(url)["value"].diff().dropna().values.reshape(-1, 1)
Y_uni = pd.read_csv(url)["value"].values.reshape(-1, 1)  # DistroSimulator expects 2D array


for method in ['block-bootstrap', 'me-bootstrap']:
    print(f"\n--- Residual Sampling Method: {method} ---")
    # Initialize and fit the simulator
    simulator_uni = DistroSimulator(
        kernel='rbf',
        residual_sampling=method,
        random_state=42
    )

    # Fit the model
    simulator_uni.fit(Y_uni, n_trials=100, metric='energy')

    # Generate synthetic samples
    Y_sim_uni = simulator_uni.sample(n_samples=100)

    print(f"\nBest hyperparameters:")
    print(f"  Sigma: {simulator_uni.best_params_['sigma']:.4f}")
    print(f"  Lambda: {simulator_uni.best_params_['lambd']:.6f}")
    print(f"  Best score: {simulator_uni.best_score_:.6f}")

    # Test similarity
    results_uni = simulator_uni.test_similarity(Y_uni, Y_sim_uni, n_perm=250)
    print("\nSimilarity test results:")
    for key, value in results_uni.items():
        if key != 'corr_tests':
            print(f"  {key}: {value}")

    # Compare distributions visually
    simulator_uni.compare_distributions(Y_uni, Y_sim_uni, save_prefix="univariate")

    # # plot cumulative sum of original vs simulated time series
    # plt.figure(figsize=(12, 6))
    # plt.plot(np.cumsum(Y_uni), label='Original Time Series (Cumsum)', alpha=0.7)
    # plt.plot(np.cumsum(Y_sim_uni), label='Simulated Time Series (Cumsum)', alpha=0.7)
    # plt.title('Cumulative Sum: Original vs Simulated Time Series (Univariate)')
    # plt.xlabel('Time')
    # plt.ylabel('Cumulative Value')
    # plt.legend()
    # plt.show()

    # plot cumulative sum of original vs simulated time series
    plt.figure(figsize=(12, 6))
    plt.plot(Y_uni, label='Original Time Series (Cumsum)', alpha=0.7)
    plt.plot(Y_sim_uni, label='Simulated Time Series (Cumsum)', alpha=0.7)
    plt.title('Cumulative Sum: Original vs Simulated Time Series (Univariate)')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Value')
    plt.legend()
    plt.show()

