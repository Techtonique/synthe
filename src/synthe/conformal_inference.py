
import nnetsauce as ns
import numpy as np

from collections import namedtuple
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from .utils import simulate_replications


class ConformalInference:
    """GAN-like Synthetic data simulation using Conformal Inference.

    Attributes:

        type_pi: a string;
            type of prediction interval: currently `None`
            (split conformal without simulation)
            for type_pi in: 
                - 'bootstrap': Bootstrap resampling.
                - 'kde': Kernel Density Estimation.

        type_split: a string;
            "random" (random split of data) or "sequential" (sequential split of data)
        
        replications: an integer;
            Number of replications for simulated conformal (default is 250)
        
        kernel: a string;
            Kernel to be used in local conformal inference (default is 'gaussian')
            if type_pi is 'kde', this kernel will also be used in the simulation of residuals
        
        split_ratio: a float;
            Ratio of calibration data (default is 0.5)

        seed: an integer;
            Reproducibility of fit (there's a random split between fitting and calibration data)
    """

    def __init__(
        self,
        type_pi=None,
        type_split="random",
        replications=250,
        kernel="gaussian",  
        split_ratio=0.5,     
        seed=123,
    ):

        self.type_pi = type_pi
        self.type_split = type_split
        self.replications = replications
        self.kernel = kernel
        self.split_ratio = split_ratio
        self.seed = seed
        self.calibrated_residuals_ = None
        self.scaled_calibrated_residuals_ = None
        self.calibrated_residuals_scaler_ = None
        self.kde_ = None

    def fit(self, X, **kwargs):
        """Fit the `method` to training data (X, y).

        Args:

            X: array-like, shape = [n_samples, n_features];
                Training set vectors, where n_samples is the number
                of samples and n_features is the number of features.
                Can be a 1d array.

        """

        np.random.seed(self.seed)

        if self.type_split == "random":

            if len(X.shape) == 1: # 1d array                
                X_train, X_calibration = ns.SubSampler(y=X, 
                                                       row_sample=self.split_ratio, 
                                                       seed=self.seed, 
                                                       n_jobs=None)                
                random_covariate = np.random.randn(len(X)).reshape(-1, 1)                            
            else: # multi-dim array
                X_train, X_calibration, _, _ = train_test_split(
                    X, np.random.randn(X.shape[0], 1), test_size=0.5, 
                    random_state=self.seed
                )
                random_covariate = np.random.randn(X.shape[0], 1)                        
                
        elif self.type_split == "sequential":
            
            if len(X.shape) == 1: # 1d array
                n_x = len(X)
                random_covariate = np.random.randn(len(X)).reshape(-1, 1)
            else: # multi-dim array
                n_x = X.shape[0]
                random_covariate = np.random.randn(X.shape[0], 1)                        

            n_x_half = n_x // 2
            first_half_idx = range(0, n_x_half)
            second_half_idx = range(n_x_half, n_x)
            X_train = X[first_half_idx]
            X_calibration = X[second_half_idx]
        
        random_covariate_train = random_covariate[X_train.index]
        random_covariate_calibration = random_covariate[X_calibration.index]

        self.obj.fit(random_covariate, X_train)
        preds_calibration = self.obj.predict(X_calibration)
        self.calibrated_residuals_ = y_calibration - preds_calibration
        absolute_residuals = np.abs(self.calibrated_residuals_)
        self.calibrated_residuals_scaler_ = StandardScaler(
            with_mean=True, with_std=True
        )
        self.scaled_calibrated_residuals_ = (
            self.calibrated_residuals_scaler_.fit_transform(
                self.calibrated_residuals_.reshape(-1, 1)
            ).ravel()
        )
        try:
            # numpy version >= 1.22
            self.quantile_ = np.quantile(
                a=absolute_residuals, q=self.level / 100, method="higher"
            )
        except Exception:
            # numpy version < 1.22
            self.quantile_ = np.quantile(
                a=absolute_residuals,
                q=self.level / 100,
                interpolation="higher",
            )

        preds = self.obj.predict(X_calibration)

        return self

    def sample(self, n_samples=100):
        """Obtain predictions and prediction intervals

        Args:

            n_samples: an integer;
                Number of samples to be generated

        """
                
        if self.type_pi is None:
            preds = self.obj.predict(np.random.randn(n_samples, 1))
            lower_bounds = preds - self.quantile_
            upper_bounds = preds + self.quantile_
            return namedtuple(
                "ConformalPrediction",
                ["predictions", "lower_bounds", "upper_bounds"],
            )(preds, lower_bounds, upper_bounds)

        elif self.type_pi in ["bootstrap", "kde"]:
            simulated_residuals = simulate_replications(
                data=self.scaled_calibrated_residuals_,
                method=self.type_pi,
                num_replications=self.replications,
                n_obs=n_samples,
                kernel=self.kernel,
            )
            simulated_residuals = simulated_residuals * np.std(
                self.calibrated_residuals_
            ) + np.mean(self.calibrated_residuals_)
            preds = self.obj.predict(np.random.randn(n_samples, 1))
            lower_bounds = preds - np.quantile(
                np.abs(simulated_residuals), q=self.level / 100, method="higher"
            )
            upper_bounds = preds + np.quantile(
                np.abs(simulated_residuals), q=self.level / 100, method="higher"
            )
            return namedtuple(
                "ConformalPrediction",
                ["predictions", "lower_bounds", "upper_bounds"],
            )(preds, lower_bounds, upper_bounds)
        else:
            raise ValueError(
                f"Unknown type_pi '{self.type_pi}'. Choose from None, 'bootstrap', or 'kde'."
            )


        """MMD with RBF kernel"""
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        if bandwidth is None:
            # Median heuristic
            XX_dists = cdist(X, X, 'sqeuclidean')
            bandwidth = np.median(XX_dists[XX_dists > 0]) / 2
            if bandwidth == 0:
                bandwidth = 1.0
        
        def rbf_kernel(a, b, sigma):
            return np.exp(-cdist(a, b, 'sqeuclidean') / (2 * sigma**2))
        
        K_XX = rbf_kernel(X, X, bandwidth)
        K_YY = rbf_kernel(Y, Y, bandwidth)
        K_XY = rbf_kernel(X, Y, bandwidth)
        
        n, m = len(X), len(Y)
        mmd_sq = (np.sum(K_XX) - np.trace(K_XX)) / (n * (n - 1)) + \
                 (np.sum(K_YY) - np.trace(K_YY)) / (m * (m - 1)) - \
                 2 * np.sum(K_XY) / (n * m)
        
        return max(0, np.sqrt(mmd_sq))

    def _mmd_rbf(self, X, Y, bandwidth=None):
        """MMD with RBF kernel (vectorized version)"""
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        n, m = len(X), len(Y)
        
        # Compute pairwise squared Euclidean distance between points in X and Y
        XX_dists = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
        YY_dists = np.sum(Y**2, axis=1)[:, None] + np.sum(Y**2, axis=1) - 2 * np.dot(Y, Y.T)
        XY_dists = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
        
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
        mmd_sq = (np.sum(K_XX) - np.trace(K_XX)) / (n * (n - 1)) + \
                (np.sum(K_YY) - np.trace(K_YY)) / (m * (m - 1)) - \
                2 * np.sum(K_XY) / (n * m)
        
        return max(0, np.sqrt(mmd_sq))

    def _energy_distance(self, X, Y):
        """Energy distance (vectorized version)"""
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
        XX_dists = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
        YY_dists = np.sum(Y**2, axis=1)[:, None] + np.sum(Y**2, axis=1) - 2 * np.dot(Y, Y.T)
        XY_dists = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
        
        # Compute the Energy Distance
        XX = np.sum(XX_dists) / (n * (n - 1))
        YY = np.sum(YY_dists) / (m * (m - 1))
        XY = np.sum(XY_dists) / (n * m)
        
        return max(0.0, 2 * XY - XX - YY)
    
    def _compute_objective(self, y_true, y_synthetic):
        """Compute chosen objective"""
        if self.objective == 'mmd':
            return self._mmd_rbf(y_true, y_synthetic, self.kernel_bandwidth)
        elif self.objective == 'energy':
            return self._energy_distance(y_true, y_synthetic)
        else:
            return self._mmd_rbf(y_true, y_synthetic, self.kernel_bandwidth)
