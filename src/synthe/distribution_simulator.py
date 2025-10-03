"""
Distribution Simulator using Residual Resampling
A model-agnostic method for simulating unconditional distributions
"""

import numpy as np
from typing import Optional, Callable, Dict, Tuple, Union
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from empirical_copula import EmpiricalCopula
import warnings

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    warnings.warn("Optuna not installed. Using grid search for hyperparameters.")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class DistributionSimulator:
    """
    Model-agnostic simulator for unconditional distributions using 
    residual resampling and surrogate modeling.
    
    This method generates samples by combining predictions from a surrogate
    model trained on auxiliary variables with resampled residuals. For 
    multivariate distributions, offers optional copula-based dependence modeling.
    
    Parameters
    ----------
    model_type : str, default='krr'
        Type of surrogate model: 'krr' (Kernel Ridge Regression) or 'xgboost'
    use_copula : bool, default=False
        For multivariate data, use empirical copula for dependence structure
    X_dist : callable, optional
        Function that generates auxiliary X samples: X_dist(n_samples) -> array
        If None, uses standard normal
    metric : str, default='auto'
        Discrepancy metric: 'wasserstein' (1D), 'mmd' (multivariate), 
        'energy', or 'auto'
    n_trials : int, default=50
        Number of optimization trials for hyperparameter tuning
    test_size : float, default=0.5
        Fraction of data to use for validation
    random_state : int, optional
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to print optimization progress
    
    Attributes
    ----------
    is_fitted_ : bool
        Whether the simulator has been fitted
    best_params_ : dict
        Optimal hyperparameters found during tuning
    best_metric_ : float
        Best discrepancy metric achieved
    model_ : object
        Fitted surrogate model (or None if using copula)
    residuals_ : ndarray
        Fitted residuals for resampling (or None if using copula)
    copula_ : object
        Fitted empirical copula (if use_copula=True)
    n_dims_ : int
        Dimensionality of target distribution
    
    Notes
    -----
    The auxiliary variable X serves as a source of randomness rather than
    a predictor. For univariate Y, the surrogate captures location/scale
    patterns that vary across the X space, with residuals adding distributional
    details. For multivariate Y with copula=True, only copula is used.
    
    Examples
    --------
    >>> import numpy as np
    >>> Y = np.random.normal(0, 1, 200)
    >>> sim = DistributionSimulator(model_type='krr', random_state=42)
    >>> sim.fit(Y)
    >>> samples = sim.sample(500)
    """
    
    def __init__(
        self,
        model_type: str = 'krr',
        use_copula: bool = False,
        X_dist: Optional[Callable] = None,
        metric: str = 'auto',
        n_trials: int = 50,
        test_size: float = 0.5,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        if model_type not in ['krr', 'xgboost']:
            raise ValueError(f"model_type must be 'krr' or 'xgboost', got '{model_type}'")
        
        if model_type == 'xgboost' and not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        self.model_type = model_type
        self.use_copula = use_copula
        self.X_dist = X_dist if X_dist is not None else self._default_X_dist
        self.metric = metric
        self.n_trials = n_trials
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        
        # Fitted attributes
        self.is_fitted_ = False
        self.best_params_ = None
        self.best_metric_ = None
        self.model_ = None
        self.residuals_ = None
        self.copula_ = None
        self.n_dims_ = None
        self.Y_train_ = None
        
    def _default_X_dist(self, n_samples: int) -> np.ndarray:
        """Default auxiliary distribution: standard normal"""
        rng = np.random.RandomState(self.random_state)
        return rng.normal(0, 1, (n_samples, 1))
    
    def _ensure_2d(self, Y: np.ndarray) -> np.ndarray:
        """Ensure Y is 2D array"""
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        return Y
    
    def _compute_mmd(self, u: np.ndarray, v: np.ndarray, 
                     sigma: float = 1.0) -> float:
        """Compute Maximum Mean Discrepancy with Gaussian kernel"""
        u = self._ensure_2d(u)
        v = self._ensure_2d(v)
        
        def kernel_matrix(A, B):
            dist_sq = np.sum(A**2, axis=1)[:, None] + \
                      np.sum(B**2, axis=1)[None, :] - \
                      2 * np.dot(A, B.T)
            return np.exp(-dist_sq / (2 * sigma**2))
        
        K_uu = kernel_matrix(u, u)
        K_vv = kernel_matrix(v, v)
        K_uv = kernel_matrix(u, v)
        
        return np.mean(K_uu) + np.mean(K_vv) - 2 * np.mean(K_uv)
    
    def _compute_energy_distance(self, u: np.ndarray, 
                                 v: np.ndarray) -> float:
        """Compute energy distance"""
        u = self._ensure_2d(u)
        v = self._ensure_2d(v)
        
        dist_uu = cdist(u, u, metric='euclidean')
        dist_vv = cdist(v, v, metric='euclidean')
        dist_uv = cdist(u, v, metric='euclidean')
        
        n, m = len(u), len(v)
        term1 = 2 * np.sum(dist_uv) / (n * m)
        term2 = np.sum(dist_uu) / (n * n)
        term3 = np.sum(dist_vv) / (m * m)
        
        return term1 - term2 - term3
    
    def _compute_metric(self, Y_true: np.ndarray, Y_sim: np.ndarray) -> float:
        """Compute discrepancy metric between distributions"""
        Y_true = self._ensure_2d(Y_true)
        Y_sim = self._ensure_2d(Y_sim)
        d = Y_true.shape[1]
        
        metric = self.metric
        if metric == 'auto':
            metric = 'wasserstein' if d == 1 else 'mmd'
        
        if metric == 'wasserstein' and d == 1:
            return stats.wasserstein_distance(Y_true.flatten(), 
                                             Y_sim.flatten())
        elif metric == 'mmd':
            return self._compute_mmd(Y_true, Y_sim)
        elif metric == 'energy':
            return self._compute_energy_distance(Y_true, Y_sim)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _generate_pseudo(self, model, residuals: np.ndarray, 
                        n_samples: int) -> np.ndarray:
        """Generate pseudo-samples using fitted model and residuals"""
        X_new = self.X_dist(n_samples)
        preds = model.predict(X_new)
        preds = self._ensure_2d(preds)
        
        # Resample residuals with replacement
        rng = np.random.RandomState(self.random_state)
        n_res = residuals.shape[0]
        idx = rng.choice(n_res, n_samples, replace=True)
        e_star = residuals[idx]
        
        return preds + e_star
    
    def _fit_krr(self, X_train, Y_train, sigma, lambd):
        """Fit Kernel Ridge Regression model"""
        gamma = 1 / (2 * sigma**2)
        model = KernelRidge(kernel='rbf', gamma=gamma, alpha=lambd)
        model.fit(X_train, Y_train)
        return model
    
    def _fit_xgboost(self, X_train, Y_train, max_depth, learning_rate, 
                     n_estimators, reg_lambda):
        """Fit XGBoost model"""
        model = xgb.XGBRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            reg_lambda=reg_lambda,
            random_state=self.random_state,
            verbosity=0
        )
        model.fit(X_train, Y_train)
        return model
    
    def _objective_optuna(self, trial, Y_train, Y_test, X_train):
        """Objective function for Optuna optimization"""
        if self.model_type == 'krr':
            sigma = trial.suggest_float('sigma', 0.01, 10, log=True)
            lambd = trial.suggest_float('lambda', 1e-5, 1, log=True)
            model = self._fit_krr(X_train, Y_train, sigma, lambd)
            
        elif self.model_type == 'xgboost':
            max_depth = trial.suggest_int('max_depth', 3, 10)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            reg_lambda = trial.suggest_float('reg_lambda', 1e-3, 10, log=True)
            model = self._fit_xgboost(X_train, Y_train, max_depth, 
                                      learning_rate, n_estimators, reg_lambda)
        
        # Compute residuals
        preds = model.predict(X_train)
        preds = self._ensure_2d(preds)
        residuals = Y_train - preds
        
        # Generate pseudo-samples
        Y_sim = self._generate_pseudo(model, residuals, len(Y_test))
        
        # Compute metric
        metric_val = self._compute_metric(Y_test, Y_sim)
        
        return metric_val
    
    def fit(self, Y: np.ndarray) -> 'DistributionSimulator':
        """
        Fit the simulator to data Y
        
        Parameters
        ----------
        Y : array-like of shape (n_samples,) or (n_samples, n_features)
            Target distribution samples
            
        Returns
        -------
        self : DistributionSimulator
            Fitted simulator
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        Y = self._ensure_2d(Y)
        n, d = Y.shape
        self.n_dims_ = d
        
        # For multivariate with copula option
        if d >= 2 and self.use_copula:
            if self.verbose:
                print(f"Fitting {d}D distribution using Empirical Copula")
            
            # Import here to avoid circular dependency
            try:
                from scipy.interpolate import interp1d
                from sklearn.neighbors import KernelDensity
                from sklearn.mixture import GaussianMixture
                # Use the EmpiricalCopula class provided
                exec(open('empirical_copula.py').read(), globals())
                self.copula_ = EmpiricalCopula(smoothing_method="none")
                self.copula_.fit(Y)
                self.is_fitted_ = True
                if self.verbose:
                    print("Copula fitted successfully")
                return self
            except Exception as e:
                warnings.warn(f"Copula fitting failed: {e}. Falling back to surrogate method.")
                self.use_copula = False
        
        # Standard surrogate + residual method
        Y_train, Y_test = train_test_split(
            Y, test_size=self.test_size, random_state=self.random_state
        )
        self.Y_train_ = Y_train
        
        # Generate auxiliary variables
        X_train = self.X_dist(len(Y_train))
        
        if self.verbose:
            print(f"Fitting {d}D distribution with {len(Y_train)} training samples")
            print(f"Using {self.model_type.upper()} surrogate model")
        
        # Optimize hyperparameters
        if HAS_OPTUNA:
            study = optuna.create_study(direction='minimize')
            study.optimize(
                lambda trial: self._objective_optuna(trial, Y_train, Y_test, X_train),
                n_trials=self.n_trials,
                show_progress_bar=self.verbose
            )
            self.best_params_ = study.best_params
            self.best_metric_ = study.best_value
        else:
            # Simple grid search fallback for KRR only
            if self.model_type != 'krr':
                raise ValueError("Grid search only supports KRR. Install Optuna for XGBoost.")
            
            best_metric = np.inf
            best_params = None
            
            for sigma in np.logspace(-2, 1, 10):
                for lambd in np.logspace(-5, -1, 10):
                    model = self._fit_krr(X_train, Y_train, sigma, lambd)
                    preds = self._ensure_2d(model.predict(X_train))
                    residuals = Y_train - preds
                    Y_sim = self._generate_pseudo(model, residuals, len(Y_test))
                    metric_val = self._compute_metric(Y_test, Y_sim)
                    
                    if metric_val < best_metric:
                        best_metric = metric_val
                        best_params = {'sigma': sigma, 'lambda': lambd}
            
            self.best_params_ = best_params
            self.best_metric_ = best_metric
        
        # Fit final model with best parameters
        if self.model_type == 'krr':
            sigma = self.best_params_['sigma']
            lambd = self.best_params_['lambda']
            self.model_ = self._fit_krr(X_train, Y_train, sigma, lambd)
        elif self.model_type == 'xgboost':
            self.model_ = self._fit_xgboost(
                X_train, Y_train,
                self.best_params_['max_depth'],
                self.best_params_['learning_rate'],
                self.best_params_['n_estimators'],
                self.best_params_['reg_lambda']
            )
        
        # Compute and store residuals
        preds = self._ensure_2d(self.model_.predict(X_train))
        self.residuals_ = Y_train - preds
        
        self.is_fitted_ = True
        
        if self.verbose:
            print(f"\nBest parameters: {self.best_params_}")
            print(f"Best metric: {self.best_metric_:.6f}")
        
        return self
    
    def sample(self, n_samples: int = 100) -> np.ndarray:
        """
        Generate samples from the fitted distribution
        
        Parameters
        ----------
        n_samples : int, default=100
            Number of samples to generate
            
        Returns
        -------
        samples : ndarray of shape (n_samples,) or (n_samples, n_features)
            Generated samples
        """
        if not self.is_fitted_:
            raise ValueError("Simulator must be fitted before sampling. Call fit() first.")
        
        # Use copula if fitted
        if self.copula_ is not None:
            samples = self.copula_.sample(n_samples=n_samples, 
                                         method="gmm",
                                         random_state=self.random_state)
            if self.n_dims_ == 1:
                return samples.flatten()
            return samples
        
        # Otherwise use surrogate + residuals
        samples = self._generate_pseudo(self.model_, self.residuals_, n_samples)
        
        if self.n_dims_ == 1:
            return samples.flatten()
        return samples
    
    def get_params(self) -> Dict:
        """Get fitted parameters and diagnostics"""
        if not self.is_fitted_:
            raise ValueError("Simulator not fitted yet")
        
        result = {
            'n_dims': self.n_dims_,
            'uses_copula': self.copula_ is not None
        }
        
        if self.copula_ is None:
            result.update({
                'model_type': self.model_type,
                'best_params': self.best_params_,
                'best_metric': self.best_metric_,
                'metric_type': self.metric if self.metric != 'auto' 
                              else ('wasserstein' if self.n_dims_ == 1 else 'mmd')
            })
        
        return result
    
    def __repr__(self) -> str:
        if self.is_fitted_:
            method = "Copula" if self.copula_ is not None else self.model_type.upper()
            return (f"DistributionSimulator(method='{method}', "
                   f"n_dims={self.n_dims_}, fitted=True)")
        return f"DistributionSimulator(model_type='{self.model_type}', fitted=False)"

    """
    Model-agnostic simulator for unconditional distributions using 
    residual resampling and surrogate modeling.
    
    Parameters
    ----------
    model_type : str, default='krr'
        Type of surrogate model: 'krr' (Kernel Ridge Regression) or 'xgboost'
    X_dist : callable, optional
        Function that generates auxiliary X samples: X_dist(n_samples) -> array
        If None, uses standard normal
    metric : str, default='auto'
        Discrepancy metric: 'wasserstein' (1D), 'mmd' (multivariate), 
        'energy', or 'auto'
    n_trials : int, default=50
        Number of optimization trials for hyperparameter tuning
    test_size : float, default=0.5
        Fraction of data to use for validation
    random_state : int, optional
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to print optimization progress
    
    Attributes
    ----------
    is_fitted_ : bool
        Whether the simulator has been fitted
    best_params_ : dict
        Optimal hyperparameters found during tuning
    best_metric_ : float
        Best discrepancy metric achieved
    model_ : object
        Fitted surrogate model
    residuals_ : ndarray
        Fitted residuals for resampling
    n_dims_ : int
        Dimensionality of target distribution
    
    Examples
    --------
    >>> import numpy as np
    >>> Y = np.random.normal(0, 1, 200)
    >>> sim = DistributionSimulator(model_type='krr', random_state=42)
    >>> sim.fit(Y)
    >>> samples = sim.sample(500)
    """
    
    def __init__(
        self,
        model_type: str = 'krr',
        X_dist: Optional[Callable] = None,
        metric: str = 'auto',
        n_trials: int = 50,
        test_size: float = 0.5,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        if model_type not in ['krr', 'xgboost']:
            raise ValueError(f"model_type must be 'krr' or 'xgboost', got '{model_type}'")
        
        if model_type == 'xgboost' and not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        self.model_type = model_type
        self.X_dist = X_dist if X_dist is not None else self._default_X_dist
        self.metric = metric
        self.n_trials = n_trials
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        
        # Fitted attributes
        self.is_fitted_ = False
        self.best_params_ = None
        self.best_metric_ = None
        self.model_ = None
        self.residuals_ = None
        self.n_dims_ = None
        self.Y_train_ = None
        
    def _default_X_dist(self, n_samples: int) -> np.ndarray:
        """Default auxiliary distribution: standard normal"""
        rng = np.random.RandomState(self.random_state)
        return rng.normal(0, 1, (n_samples, 1))
    
    def _ensure_2d(self, Y: np.ndarray) -> np.ndarray:
        """Ensure Y is 2D array"""
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        return Y
    
    def _compute_mmd(self, u: np.ndarray, v: np.ndarray, 
                     sigma: float = 1.0) -> float:
        """Compute Maximum Mean Discrepancy with Gaussian kernel"""
        u = self._ensure_2d(u)
        v = self._ensure_2d(v)
        
        def kernel_matrix(A, B):
            dist_sq = np.sum(A**2, axis=1)[:, None] + \
                      np.sum(B**2, axis=1)[None, :] - \
                      2 * np.dot(A, B.T)
            return np.exp(-dist_sq / (2 * sigma**2))
        
        K_uu = kernel_matrix(u, u)
        K_vv = kernel_matrix(v, v)
        K_uv = kernel_matrix(u, v)
        
        return np.mean(K_uu) + np.mean(K_vv) - 2 * np.mean(K_uv)
    
    def _compute_energy_distance(self, u: np.ndarray, 
                                 v: np.ndarray) -> float:
        """Compute energy distance"""
        u = self._ensure_2d(u)
        v = self._ensure_2d(v)
        
        dist_uu = cdist(u, u, metric='euclidean')
        dist_vv = cdist(v, v, metric='euclidean')
        dist_uv = cdist(u, v, metric='euclidean')
        
        n, m = len(u), len(v)
        term1 = 2 * np.sum(dist_uv) / (n * m)
        term2 = np.sum(dist_uu) / (n * n)
        term3 = np.sum(dist_vv) / (m * m)
        
        return term1 - term2 - term3
    
    def _compute_metric(self, Y_true: np.ndarray, Y_sim: np.ndarray) -> float:
        """Compute discrepancy metric between distributions"""
        Y_true = self._ensure_2d(Y_true)
        Y_sim = self._ensure_2d(Y_sim)
        d = Y_true.shape[1]
        
        metric = self.metric
        if metric == 'auto':
            metric = 'wasserstein' if d == 1 else 'mmd'
        
        if metric == 'wasserstein' and d == 1:
            return stats.wasserstein_distance(Y_true.flatten(), 
                                             Y_sim.flatten())
        elif metric == 'mmd':
            return self._compute_mmd(Y_true, Y_sim)
        elif metric == 'energy':
            return self._compute_energy_distance(Y_true, Y_sim)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _generate_pseudo(self, model, residuals: np.ndarray, 
                        n_samples: int) -> np.ndarray:
        """Generate pseudo-samples using fitted model and residuals"""
        X_new = self.X_dist(n_samples)
        preds = model.predict(X_new)
        preds = self._ensure_2d(preds)
        
        # Resample residuals with replacement
        rng = np.random.RandomState(self.random_state)
        n_res = residuals.shape[0]
        idx = rng.choice(n_res, n_samples, replace=True)
        e_star = residuals[idx]
        
        return preds + e_star
    
    def _fit_krr(self, X_train, Y_train, sigma, lambd):
        """Fit Kernel Ridge Regression model"""
        gamma = 1 / (2 * sigma**2)
        model = KernelRidge(kernel='rbf', gamma=gamma, alpha=lambd)
        model.fit(X_train, Y_train)
        return model
    
    def _fit_xgboost(self, X_train, Y_train, max_depth, learning_rate, 
                     n_estimators, reg_lambda):
        """Fit XGBoost model"""
        model = xgb.XGBRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            reg_lambda=reg_lambda,
            random_state=self.random_state,
            verbosity=0
        )
        model.fit(X_train, Y_train)
        return model
    
    def _objective_optuna(self, trial, Y_train, Y_test, X_train):
        """Objective function for Optuna optimization"""
        if self.model_type == 'krr':
            sigma = trial.suggest_float('sigma', 0.01, 10, log=True)
            lambd = trial.suggest_float('lambda', 1e-5, 1, log=True)
            model = self._fit_krr(X_train, Y_train, sigma, lambd)
            
        elif self.model_type == 'xgboost':
            max_depth = trial.suggest_int('max_depth', 3, 10)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            reg_lambda = trial.suggest_float('reg_lambda', 1e-3, 10, log=True)
            model = self._fit_xgboost(X_train, Y_train, max_depth, 
                                      learning_rate, n_estimators, reg_lambda)
        
        # Compute residuals
        preds = model.predict(X_train)
        preds = self._ensure_2d(preds)
        residuals = Y_train - preds
        
        # Generate pseudo-samples
        Y_sim = self._generate_pseudo(model, residuals, len(Y_test))
        
        # Compute metric
        metric_val = self._compute_metric(Y_test, Y_sim)
        
        return metric_val
    
    def fit(self, Y: np.ndarray) -> 'DistributionSimulator':
        """
        Fit the simulator to data Y
        
        Parameters
        ----------
        Y : array-like of shape (n_samples,) or (n_samples, n_features)
            Target distribution samples
            
        Returns
        -------
        self : DistributionSimulator
            Fitted simulator
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        Y = self._ensure_2d(Y)
        n, d = Y.shape
        self.n_dims_ = d
        
        # Split data
        Y_train, Y_test = train_test_split(
            Y, test_size=self.test_size, random_state=self.random_state
        )
        self.Y_train_ = Y_train
        
        # Generate auxiliary variables
        X_train = self.X_dist(len(Y_train))
        
        if self.verbose:
            print(f"Fitting {d}D distribution with {len(Y_train)} training samples")
            print(f"Using {self.model_type.upper()} surrogate model")
        
        # Optimize hyperparameters
        if HAS_OPTUNA:
            study = optuna.create_study(direction='minimize')
            study.optimize(
                lambda trial: self._objective_optuna(trial, Y_train, Y_test, X_train),
                n_trials=self.n_trials,
                show_progress_bar=self.verbose
            )
            self.best_params_ = study.best_params
            self.best_metric_ = study.best_value
        else:
            # Simple grid search fallback for KRR only
            if self.model_type != 'krr':
                raise ValueError("Grid search only supports KRR. Install Optuna for XGBoost.")
            
            best_metric = np.inf
            best_params = None
            
            for sigma in np.logspace(-2, 1, 10):
                for lambd in np.logspace(-5, -1, 10):
                    model = self._fit_krr(X_train, Y_train, sigma, lambd)
                    preds = self._ensure_2d(model.predict(X_train))
                    residuals = Y_train - preds
                    Y_sim = self._generate_pseudo(model, residuals, len(Y_test))
                    metric_val = self._compute_metric(Y_test, Y_sim)
                    
                    if metric_val < best_metric:
                        best_metric = metric_val
                        best_params = {'sigma': sigma, 'lambda': lambd}
            
            self.best_params_ = best_params
            self.best_metric_ = best_metric
        
        # Fit final model with best parameters
        if self.model_type == 'krr':
            sigma = self.best_params_['sigma']
            lambd = self.best_params_['lambda']
            self.model_ = self._fit_krr(X_train, Y_train, sigma, lambd)
        elif self.model_type == 'xgboost':
            self.model_ = self._fit_xgboost(
                X_train, Y_train,
                self.best_params_['max_depth'],
                self.best_params_['learning_rate'],
                self.best_params_['n_estimators'],
                self.best_params_['reg_lambda']
            )
        
        # Compute and store residuals
        preds = self._ensure_2d(self.model_.predict(X_train))
        self.residuals_ = Y_train - preds
        
        self.is_fitted_ = True
        
        if self.verbose:
            print(f"\nBest parameters: {self.best_params_}")
            print(f"Best metric: {self.best_metric_:.6f}")
        
        return self
    
    def sample(self, n_samples: int = 100) -> np.ndarray:
        """
        Generate samples from the fitted distribution
        
        Parameters
        ----------
        n_samples : int, default=100
            Number of samples to generate
            
        Returns
        -------
        samples : ndarray of shape (n_samples,) or (n_samples, n_features)
            Generated samples
        """
        if not self.is_fitted_:
            raise ValueError("Simulator must be fitted before sampling. Call fit() first.")
        
        samples = self._generate_pseudo(self.model_, self.residuals_, n_samples)
        
        # Return 1D array for univariate case
        if self.n_dims_ == 1:
            return samples.flatten()
        return samples
    
    def get_params(self) -> Dict:
        """Get fitted parameters and diagnostics"""
        if not self.is_fitted_:
            raise ValueError("Simulator not fitted yet")
        
        return {
            'model_type': self.model_type,
            'best_params': self.best_params_,
            'best_metric': self.best_metric_,
            'n_dims': self.n_dims_,
            'metric_type': self.metric if self.metric != 'auto' 
                          else ('wasserstein' if self.n_dims_ == 1 else 'mmd')
        }
    
    def __repr__(self) -> str:
        if self.is_fitted_:
            return (f"DistributionSimulator(model_type='{self.model_type}', "
                   f"n_dims={self.n_dims_}, fitted=True)")
        return f"DistributionSimulator(model_type='{self.model_type}', fitted=False)"