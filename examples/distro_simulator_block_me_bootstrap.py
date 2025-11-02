"""
Example: Time series and DistroSimulator residual sampling methods
Demonstrates block-bootstrap and maximum-entropy bootstrap (me-bootstrap)
for time series residuals.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp
from synthe.distro_simulator import DistroSimulator

if __name__ == "__main__":
    np.random.seed(42)    
    Y = pd.read_csv('https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/time_series/univariate/AirPassengers.csv')["value"]

    Y = Y.values.reshape(-1, 1)  # DistroSimulator expects 2D array (n_samples, n_features)

    
