import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import properscoring as ps

from scipy.stats import gaussian_kde, norm
from scipy.interpolate import interp1d

"""Tools for MTS."""

# Authors: Thierry Moudiki <thierry.moudiki@gmail.com>
#
# License: BSD 3 Clear

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import mean_pinball_loss
except ImportError:
    pass


# (block) bootstrap
def bootstrap(x, h, block_size=None, seed=123):
    """
    Generates block bootstrap indices for a given time series.

    Parameters:
    - x: numpy array, the original time series (univariate or multivariate).
    - h: int, output length
    - block_size: int, the size of the blocks to resample (if None, independent bootstrap).
    - seed: int, reproducibility seed.

    Returns:
    - numpy arrays containing resampled time series.
    """
    if len(x.shape) == 1:
        time_series_length = len(x)
        ndim = 1
    else:
        time_series_length = x.shape[0]
        ndim = x.shape[1]

    if block_size is not None:
        num_blocks = (time_series_length + block_size - 1) // block_size
        all_indices = np.arange(time_series_length)

        indices = []
        for i in range(num_blocks):
            np.random.seed(seed + i * 100)
            start_index = np.random.randint(
                0, time_series_length - block_size + 1
            )
            block_indices = all_indices[start_index : start_index + block_size]
            indices.extend(block_indices)

    else:  # block_size is None
        indices = np.random.choice(
            range(time_series_length), size=h, replace=True
        )

    if ndim == 1:
        return x[np.array(indices[:h])]
    else:
        return x[np.array(indices[:h]), :]


def mmd_rbf(y_true, y_synthetic, bandwidth=None):
    """MMD with RBF kernel (vectorized version)"""
    X, Y = np.asarray(y_true), np.asarray(y_synthetic)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n, m = len(X), len(Y)
    # Compute pairwise squared Euclidean distance between points in X and Y
    try:
        XX_dists = (
            np.sum(X**2, axis=1)[:, None]
            + np.sum(X**2, axis=1)
            - 2 * np.dot(X, X.T)
        )
        YY_dists = (
            np.sum(Y**2, axis=1)[:, None]
            + np.sum(Y**2, axis=1)
            - 2 * np.dot(Y, Y.T)
        )
        XY_dists = (
            np.sum(X**2, axis=1)[:, None]
            + np.sum(Y**2, axis=1)
            - 2 * np.dot(X, Y.T)
        )
    except Exception as e:
        print("Error in distance computation:", e)
        XX_dists = (
            np.sum(X**2, axis=1)[:, None]
            + np.sum(X**2, axis=1)
            - 2 * np.dot(X, X.T)
        )
        YY_dists = (
            np.sum(Y**2, axis=1)[:, None]
            + np.sum(Y**2, axis=1)
            - 2 * np.dot(Y, Y.T)
        )
        XY_dists = (
            np.sum(X**2, axis=1)[:, None]
            + np.sum(Y**2, axis=1)
            - 2 * np.dot(X, Y.T)
        )
    if bandwidth is None:
        # Median heuristic for bandwidth
        bandwidth = np.median(XX_dists[XX_dists > 0]) / 2
        if bandwidth == 0:
            bandwidth = 1.0
    # Compute the kernel matrices using broadcasting
    K_XX = np.exp(-XX_dists / (2 * bandwidth**2))
    K_YY = np.exp(-YY_dists / (2 * bandwidth**2))
    K_XY = np.exp(-XY_dists / (2 * bandwidth**2))
    # MMD computation
    mmd_sq = (
        (np.sum(K_XX) - np.trace(K_XX)) / (n * (n - 1))
        + (np.sum(K_YY) - np.trace(K_YY)) / (m * (m - 1))
        - 2 * np.sum(K_XY) / (n * m)
    )
    return max(0, np.sqrt(mmd_sq))


def mmd_matern52(y_true, y_synthetic, bandwidth=None):
    """MMD with Matérn 5/2 kernel (vectorized version)"""
    X, Y = np.asarray(y_true), np.asarray(y_synthetic)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n, m = len(X), len(Y)

    if n < 2 or m < 2:
        raise ValueError("MMD requires at least 2 samples in each dataset.")
    # Compute squared Euclidean distances
    X_sq = np.sum(X**2, axis=1)
    Y_sq = np.sum(Y**2, axis=1)

    XX_dists = X_sq[:, None] + X_sq - 2 * np.dot(X, X.T)
    YY_dists = Y_sq[:, None] + Y_sq - 2 * np.dot(Y, Y.T)
    XY_dists = X_sq[:, None] + Y_sq - 2 * np.dot(X, Y.T)
    # Ensure non-negative (numerical stability)
    XX_dists = np.clip(XX_dists, 0, None)
    YY_dists = np.clip(YY_dists, 0, None)
    XY_dists = np.clip(XY_dists, 0, None)
    # Bandwidth selection (median heuristic on Euclidean distances)
    if bandwidth is None:
        sqrt_XX = np.sqrt(XX_dists)
        # Use upper triangle (excluding diagonal) to get unique pairwise distances
        iu = np.triu_indices(n, k=1)
        if len(iu[0]) == 0:
            bandwidth = 1.0
        else:
            non_zero_dists = sqrt_XX[iu]
            bandwidth = np.median(non_zero_dists)
            if bandwidth == 0:
                bandwidth = 1.0
    # Precompute sqrt distances
    sqrt_d_XX = np.sqrt(XX_dists)
    sqrt_d_YY = np.sqrt(YY_dists)
    sqrt_d_XY = np.sqrt(XY_dists)
    # Matérn 5/2 kernel: k(d) = exp(-√5 d / ℓ) * (1 + √5 d / ℓ + 5 d² / (3 ℓ²))
    sqrt_5 = np.sqrt(5)
    K_XX = np.exp(-sqrt_5 * sqrt_d_XX / bandwidth) * (
        1 + sqrt_5 * sqrt_d_XX / bandwidth + 5 * XX_dists / (3 * bandwidth**2)
    )
    K_YY = np.exp(-sqrt_5 * sqrt_d_YY / bandwidth) * (
        1 + sqrt_5 * sqrt_d_YY / bandwidth + 5 * YY_dists / (3 * bandwidth**2)
    )
    K_XY = np.exp(-sqrt_5 * sqrt_d_XY / bandwidth) * (
        1 + sqrt_5 * sqrt_d_XY / bandwidth + 5 * XY_dists / (3 * bandwidth**2)
    )
    # MMD computation
    mmd_sq = (
        (np.sum(K_XX) - np.trace(K_XX)) / (n * (n - 1))
        + (np.sum(K_YY) - np.trace(K_YY)) / (m * (m - 1))
        - 2 * np.sum(K_XY) / (n * m)
    )
    return max(0, np.sqrt(mmd_sq))


def energy_distance(y_true, y_synthetic):
    """Energy distance (vectorized version)"""
    X, Y = y_true, y_synthetic
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        return np.inf
    # Compute pairwise squared Euclidean distances
    XX_dists = (
        np.sum(X**2, axis=1)[:, None]
        + np.sum(X**2, axis=1)
        - 2 * np.dot(X, X.T)
    )
    YY_dists = (
        np.sum(Y**2, axis=1)[:, None]
        + np.sum(Y**2, axis=1)
        - 2 * np.dot(Y, Y.T)
    )
    XY_dists = (
        np.sum(X**2, axis=1)[:, None]
        + np.sum(Y**2, axis=1)
        - 2 * np.dot(X, Y.T)
    )
    # Compute the Energy Distance
    XX = np.sum(XX_dists) / (n * (n - 1))
    YY = np.sum(YY_dists) / (m * (m - 1))
    XY = np.sum(XY_dists) / (n * m)
    return max(0.0, 2 * XY - XX - YY)


def crps(y_true, y_synthetic):
    """Compute CRPS using properscoring package"""
    y_true = np.asarray(y_true)
    y_synthetic = np.asarray(y_synthetic)
    return np.median(ps.crps_ensemble(y_true, y_synthetic))


def simulate_distribution(
    data, method="bootstrap", num_samples=250, seed=123, **kwargs
):
    """
    Simulate the distribution of an input vector using various methods.

    Parameters:
        data (array-like): Input vector of data.
        method (str): Method for simulation:
                      - 'bootstrap': Bootstrap resampling.
                      - 'kde': Kernel Density Estimation.
                      - 'normal': Normal distribution.
                      - 'ecdf': Empirical CDF-based sampling.
                      - 'permutation': Permutation resampling.
                      - 'smooth-bootstrap': Smoothed bootstrap with added noise.
        num_samples (int): Number of samples to generate.
        seed (int): Random seed for reproducibility.
        kwargs: Additional parameters for specific methods:
                - kde_bandwidth (str or float): Bandwidth for KDE ('scott', 'silverman', or float).
                - dist (str): Parametric distribution type ('normal').
                - noise_std (float): Noise standard deviation for smoothed bootstrap.

    Returns:
        np.ndarray: Simulated distribution samples.
    """
    assert method in [
        "bootstrap",
        "kde",
        "parametric",
        "ecdf",
        "permutation",
        "smooth-bootstrap",
    ], f"Unknown method '{method}'. Choose from 'bootstrap', 'kde', 'parametric', 'ecdf', 'permutation', or 'smooth_bootstrap'."

    data = np.array(data)

    np.random.seed(seed)

    if method == "bootstrap":
        simulated_data = np.random.choice(data, size=num_samples, replace=True)

    elif method == "kde":
        kde_bandwidth = kwargs.get("kde_bandwidth", "scott")
        kde = gaussian_kde(data, bw_method=kde_bandwidth)
        simulated_data = kde.resample(num_samples).flatten()

    elif method == "normal":
        mean, std = np.mean(data), np.std(data)
        simulated_data = np.random.normal(mean, std, size=num_samples)

    elif method == "ecdf":
        data = np.sort(data)
        ecdf_y = np.arange(1, len(data) + 1) / len(data)
        inverse_cdf = interp1d(
            ecdf_y, data, bounds_error=False, fill_value=(data[0], data[-1])
        )
        random_uniform = np.random.uniform(0, 1, size=num_samples)
        simulated_data = inverse_cdf(random_uniform)

    elif method == "permutation":
        simulated_data = np.random.permutation(data)
        while len(simulated_data) < num_samples:
            simulated_data = np.concatenate(
                [simulated_data, np.random.permutation(data)]
            )
        simulated_data = simulated_data[:num_samples]

    elif method == "smooth_bootstrap":
        noise_std = kwargs.get("noise_std", 0.1)
        bootstrap_samples = np.random.choice(
            data, size=num_samples, replace=True
        )
        noise = np.random.normal(0, noise_std, size=num_samples)
        simulated_data = bootstrap_samples + noise

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from 'bootstrap', 'kde', 'parametric', 'ecdf', 'permutation', or 'smooth_bootstrap'."
        )

    return simulated_data


def simulate_replications(
    data, method="kde", num_replications=10, n_obs=None, seed=123, **kwargs
):
    """
    Create multiple replications of the input's distribution using a specified simulation method.

    Parameters:
        data (array-like): Input vector of data.
        method (str): Method for simulation:
                      - 'bootstrap': Bootstrap resampling.
                      - 'kde': Kernel Density Estimation.
                      - 'normal': Parametric distribution fitting.
                      - 'ecdf': Empirical CDF-based sampling.
                      - 'permutation': Permutation resampling.
                      - 'smooth_bootstrap': Smoothed bootstrap with added noise.
        num_samples (int): Number of samples in each replication.
        num_replications (int): Number of replications to generate.
        n_obs (int): Number of observations to generate for each replication.
        seed (int): Random seed for reproducibility.
        kwargs: Additional parameters for specific methods.

    Returns:
        pd.DataFrame: A DataFrame where each column represents a replication.
    """

    num_samples = len(data)

    replications = []

    for _ in range(num_replications):
        simulated_data = simulate_distribution(
            data, method=method, num_samples=num_samples, seed=seed, **kwargs
        )
        replications.append(simulated_data)

    # Combine replications into a DataFrame
    replications_df = pd.DataFrame(replications).transpose()
    replications_df.columns = [
        f"Replication_{i+1}" for i in range(num_replications)
    ]

    # If n_obs is specified, sample n_obs from each replication
    if n_obs is not None:
        replications_df = replications_df.sample(
            n=n_obs, replace=True, random_state=42
        ).reset_index(drop=True)
        return replications_df.values

    return replications_df.values
