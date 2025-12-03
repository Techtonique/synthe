import numpy as np
import optuna
import scipy.stats as stats
import warnings

from scipy.spatial.distance import cdist
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.kernel_ridge import KernelRidge
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from .meboot import MaximumEntropyBootstrap
from .utils import bootstrap

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from time import time


class DistroSimulator:
    def __init__(
        self,
        kernel="rbf",
        backend="numpy",
        n_clusters=5,
        clustering_method="kmeans",
        kde_kernel="gaussian",
        random_state=None,
        conformalize=False,
        residual_sampling="bootstrap",
        block_size=None,
        gmm_components=3,
        category_encoder=None,
        use_rff="auto",
        rff_components="auto",
        rff_gamma=None,
        kernel_approximation="rff",
        force_rff_threshold=1000,
    ):
        """
        Initialize the multivariate data generator.

        Parameters:
        -----------
        kernel : str, default='rbf'
            Kernel type for KernelRidge regression
        backend : str, default='numpy'
            Backend for distance calculations ('numpy', 'gpu', 'tpu')
        n_clusters : int, default=5
            Number of clusters for stratified splitting
        clustering_method : str, default='kmeans'
            Clustering method for stratification ('kmeans' or 'gmm')
        random_state : int, default=None
            Random seed for reproducibility
        conformalize : bool
            Use split conformal prediction or not
        residual_sampling : str, default='bootstrap'
            Method for sampling residuals ('bootstrap', 'kde', 'gmm', 'block-bootstrap', 'me-bootstrap').
            Where 'me-bootstrap' refers to Maximum Entropy Bootstrap.
        block_size : int, default=None
            Block size for block bootstrap (if applicable)
        gmm_components : int, default=3
            Number of components for GMM sampling
        category_encoder: object, default=None
            Category encoder
        use_rff : bool or 'auto', default='auto'
            Whether to use kernel approximation. 'auto' enables for large datasets
        rff_components : int or 'auto', default='auto'
            Number of approximation components. 'auto' chooses based on data size
        rff_gamma : float, default=None
            Gamma parameter for approximation. If None, will be tuned.
        kernel_approximation : str, default='rff'
            Approximation method ('rff' or 'nystroem')
        force_rff_threshold : int, default=1000
            Auto-enable RFF when n_samples exceeds this threshold
        """
        self.kernel = kernel
        self.backend = backend
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.random_state = random_state
        self.conformalize = conformalize
        self.residual_sampling = residual_sampling
        self.block_size = block_size
        self.gmm_components = gmm_components
        self.category_encoder = category_encoder
        self.use_rff = use_rff
        self.rff_components = rff_components
        self.rff_gamma = rff_gamma
        self.kernel_approximation = kernel_approximation
        self.force_rff_threshold = force_rff_threshold
        self.kde_kernel = kde_kernel

        if random_state is not None:
            np.random.seed(random_state)
            if JAX_AVAILABLE:
                key = jax.random.PRNGKey(random_state)
        # Validate sampling method
        valid_sampling_methods = [
            "bootstrap",
            "kde",
            "gmm",
            "block-bootstrap",
            "me-bootstrap",
        ]
        if residual_sampling not in valid_sampling_methods:
            raise ValueError(
                f"residual_sampling must be one of {valid_sampling_methods}"
            )
        # Validate approximation method
        valid_approximations = ["rff", "nystroem"]
        if kernel_approximation not in valid_approximations:
            raise ValueError(
                f"kernel_approximation must be one of {valid_approximations}"
            )
        # Initialize JAX if using GPU/TPU backend
        if backend in ["gpu", "tpu"] and JAX_AVAILABLE:
            self._setup_jax_backend()
        elif backend in ["gpu", "tpu"] and not JAX_AVAILABLE:
            print("JAX not available. Falling back to NumPy backend.")
            self.backend = "numpy"
        # Initialize attributes that will be set during fitting
        self.model = None
        self.residuals_ = None
        self.X_dist = None
        self.is_fitted = False
        self.best_params_ = None
        self.best_score_ = None
        self.cluster_labels_ = None
        self.cluster_model_ = None
        self.kde_model_ = None
        self.gmm_model_ = None
        self.scaler_ = None
        self.actual_rff_components_ = None
        self.actual_use_rff_ = None

    def _setup_jax_backend(self):
        """Setup JAX backend for GPU/TPU acceleration."""
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for GPU/TPU backend")

        # JIT compiled distance functions
        @jit
        def pairwise_sq_dists_jax(X1, X2):
            X1_sq = jnp.sum(X1**2, axis=1)[:, jnp.newaxis]
            X2_sq = jnp.sum(X2**2, axis=1)[jnp.newaxis, :]
            return X1_sq + X2_sq - 2 * X1 @ X2.T

        @jit
        def cdist_jax(X1, X2):
            return vmap(
                lambda x: vmap(lambda y: jnp.sqrt(jnp.sum((x - y) ** 2)))(X2)
            )(X1)

        self._pairwise_sq_dists_jax = pairwise_sq_dists_jax
        self._cdist_jax = cdist_jax

    def _determine_components(self, n_samples):
        """Automatically determine optimal number of components."""
        if self.rff_components == "auto":
            # Optimized heuristic based on performance results
            if n_samples < 500:
                return min(50, n_samples)
            elif n_samples < 2000:
                return min(100, n_samples // 2)
            elif n_samples < 5000:
                return min(150, n_samples // 3)
            elif n_samples < 10000:
                return min(200, n_samples // 4)
            else:
                return min(300, n_samples // 5)
        else:
            return self.rff_components

    def _create_model(self, gamma, alpha, use_rff=None):
        """Create the appropriate model based on RFF setting."""
        if use_rff is None:
            use_rff = self.actual_use_rff_

        if use_rff:
            # Use kernel approximation with Ridge regression
            if self.rff_gamma is not None:
                effective_gamma = self.rff_gamma
            else:
                effective_gamma = gamma
            # Determine number of components
            n_components = self.actual_rff_components_

            if self.kernel_approximation == "rff":
                approximator = RBFSampler(
                    gamma=effective_gamma,
                    n_components=n_components,
                    random_state=self.random_state,
                )
            else:  # nystroem
                approximator = Nystroem(
                    kernel="rbf",
                    gamma=effective_gamma,
                    n_components=n_components,
                    random_state=self.random_state,
                )
            # Create pipeline with scaling, approximation, and Ridge
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("approx", approximator),
                    ("ridge", Ridge(alpha=alpha)),
                ]
            )
        # Standard KernelRidge
        return KernelRidge(kernel=self.kernel, gamma=gamma, alpha=alpha)

    def _fit_residual_sampler(self, **kwargs):
        """Fit the chosen residual sampling model."""
        if self.residuals_ is None or len(self.residuals_) == 0:
            raise ValueError("No residuals available for fitting sampler")

        if self.residual_sampling == "kde":
            kernel_bandwidths = {"bandwidth": np.logspace(-6, 6, 150)}
            grid = GridSearchCV(
                KernelDensity(kernel=self.kde_kernel, **kwargs),
                param_grid=kernel_bandwidths,
            )
            grid.fit(self.residuals_)
            self.kde_model_ = grid.best_estimator_
            self.kde_model_.fit(self.residuals_)

        elif self.residual_sampling == "gmm":
            self.gmm_model_ = GaussianMixture(
                n_components=min(self.gmm_components, len(self.residuals_)),
                random_state=self.random_state,
                covariance_type="full",
            )
            self.gmm_model_.fit(self.residuals_)

    def _sample_residuals(self, num_samples):
        """Sample residuals using the chosen method."""
        if self.residuals_ is None:
            raise ValueError("No residuals available for sampling")

        if self.residual_sampling == "bootstrap":
            # Original bootstrap method
            n = self.residuals_.shape[0]
            idx = np.random.choice(n, num_samples, replace=True)
            return self.residuals_[idx]

        elif self.residual_sampling == "kde":
            # Kernel Density Estimation sampling
            if self.kde_model_ is None:
                raise ValueError(
                    "KDE model not fitted. Call _fit_residual_sampler first."
                )
            # Sample from KDE
            return self.kde_model_.sample(num_samples)

        elif self.residual_sampling == "gmm":
            # Gaussian Mixture Model sampling
            if self.gmm_model_ is None:
                raise ValueError(
                    "GMM model not fitted. Call _fit_residual_sampler first."
                )
            # Sample from GMM
            return self.gmm_model_.sample(num_samples)[0]

        elif self.residual_sampling == "me-bootstrap":
            meb = MaximumEntropyBootstrap(random_state=self.random_state)
            # If residuals are shorter than num_samples, repeat or tile them
            residuals = self.residuals_.flatten()
            if residuals.shape[0] < num_samples:
                # Repeat residuals to reach num_samples
                repeats = int(np.ceil(num_samples / residuals.shape[0]))
                residuals = np.tile(residuals, repeats)[:num_samples]
            else:
                residuals = residuals[:num_samples]
            meb.fit(residuals)
            return meb.sample(1)[:, 0].reshape(-1, 1)

        elif self.residual_sampling == "block-bootstrap":
            # Block Bootstrap sampling
            return bootstrap(
                self.residuals_, num_samples, block_size=self.block_size
            )

        else:
            # Should not reach here due to validation in __init__
            raise ValueError(
                f"Unknown sampling method: {self.residual_sampling}"
            )

    def _pairwise_sq_dists(self, X1, X2):
        """Compute pairwise squared Euclidean distances."""
        if self.backend in ["gpu", "tpu"] and JAX_AVAILABLE:
            X1_jax = jnp.array(X1)
            X2_jax = jnp.array(X2)
            result = self._pairwise_sq_dists_jax(X1_jax, X2_jax)
            return np.array(result)
        else:
            X1 = np.atleast_2d(X1)
            X2 = np.atleast_2d(X2)
            return (
                np.sum(X1**2, axis=1)[:, np.newaxis]
                + np.sum(X2**2, axis=1)[np.newaxis, :]
                - 2 * X1 @ X2.T
            )

    def _compute_clusters(self, Y):
        """Compute cluster labels for stratified splitting."""
        if self.clustering_method == "kmeans":
            self.cluster_model_ = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
        elif self.clustering_method == "gmm":
            self.cluster_model_ = GaussianMixture(
                n_components=self.n_clusters, random_state=self.random_state
            )
        else:
            raise ValueError("clustering_method must be 'kmeans' or 'gmm'")
        self.cluster_model_.fit(Y)
        return self.cluster_model_.predict(Y)

    def _train_test_split(self, Y, n_train, sequential: bool = False):
        """Create train-test split. Stratified by clusters or sequential if specified."""
        try:
            n_samples = len(Y)
        except Exception:
            n_samples = Y.shape[0]

        if sequential:
            # --- Sequential split (no shuffling, preserves temporal order)
            train_idx = np.arange(n_train)
            test_idx = np.arange(n_train, n_samples)
            return train_idx, test_idx

        # --- Stratified split (default)
        self.cluster_labels_ = self._compute_clusters(Y)
        return train_test_split(
            np.arange(n_samples),
            train_size=n_train,
            stratify=self.cluster_labels_,
            random_state=self.random_state,
        )

    def _mmd(self, u, v, kernel_sigma=1):
        """Maximum Mean Discrepancy between two distributions."""
        if u.ndim == 1:
            u = u.reshape(-1, 1)
        if v.ndim == 1:
            v = v.reshape(-1, 1)

        def kmat(A, B):
            return np.exp(
                -self._pairwise_sq_dists(A, B) / (2 * kernel_sigma**2)
            )

        return (
            np.mean(kmat(u, u)) + np.mean(kmat(v, v)) - 2 * np.mean(kmat(u, v))
        )

    def _custom_energy_distance(self, u, v):
        """Energy distance between two distributions."""
        if u.ndim == 1:
            u = u.reshape(-1, 1)
        if v.ndim == 1:
            v = v.reshape(-1, 1)

        n, d = u.shape
        m = v.shape[0]

        if self.backend in ["gpu", "tpu"] and JAX_AVAILABLE:
            # JAX implementation
            u_jax = jnp.array(u)
            v_jax = jnp.array(v)
            dist_xx = self._cdist_jax(u_jax, u_jax)
            dist_yy = self._cdist_jax(v_jax, v_jax)
            dist_xy = self._cdist_jax(u_jax, v_jax)
            term1 = 2 * jnp.sum(dist_xy) / (n * m)
            term2 = jnp.sum(dist_xx) / (n * n)
            term3 = jnp.sum(dist_yy) / (m * m)
            return float(term1 - term2 - term3)
        else:
            # NumPy implementation
            dist_xx = cdist(u, u, metric="euclidean")
            dist_yy = cdist(v, v, metric="euclidean")
            dist_xy = cdist(u, v, metric="euclidean")
            term1 = 2 * np.sum(dist_xy) / (n * m)
            term2 = np.sum(dist_xx) / (n * n)
            term3 = np.sum(dist_yy) / (m * m)
            return term1 - term2 - term3

    def _generate_pseudo(self, num_samples):
        """Generate synthetic data using the fitted model and residuals."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        X_new = self.X_dist[:num_samples]
        # Handle prediction based on model type
        if self.actual_use_rff_:
            # For RFF pipeline
            preds = self.model.predict(X_new)
        else:
            # For standard KernelRidge
            preds = self.model.predict(X_new)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        # Sample residuals using the chosen method
        return preds + self._sample_residuals(preds.shape[0])

    def fit(self, Y, n_train=None, metric="energy", n_trials=50, **kwargs):
        """
        Fit the data generator to match the distribution of Y.

        Parameters:
        -----------
        Y : array-like, shape (n_samples, n_features)
            Target multivariate data to emulate
        n_train : int, default=None
            Number of training samples (default: n_samples // 2)
        metric : str, default='energy'
            Distance metric for optimization ('energy', 'mmd', or 'wasserstein')
        n_trials : int, default=50
            Number of Optuna optimization trials
        **kwargs : dict
            Additional arguments for Optuna optimization

        Returns:
        --------
        self : object
            Returns self
        """
        if self.category_encoder is not None:
            Y = self.category_encoder.fit_transform(Y)
            try:
                Y = Y.values
            except Exception as e:
                pass

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n, d = Y.shape
        self.n_features_ = d
        # Determine whether to use RFF
        if self.use_rff == "auto":
            self.actual_use_rff_ = n >= self.force_rff_threshold
        else:
            self.actual_use_rff_ = self.use_rff
        # Auto-enable RFF for large datasets with component determination
        if self.actual_use_rff_:
            self.actual_rff_components_ = self._determine_components(n)
            if self.use_rff == "auto":
                print(
                    f"Large dataset detected (n={n}). Auto-enabling {self.kernel_approximation.upper()} for scalability."
                )

        if n_train is None:
            n_train = n // 2
        # Store the input distribution function
        self.X_dist = np.random.normal(0, 1, (n, d))
        # Create stratified train-test split
        if self.residual_sampling in ("block-bootstrap", "me-bootstrap"):
            train_idx, test_idx = self._train_test_split(
                Y, n_train, sequential=True
            )
        else:
            train_idx, test_idx = self._train_test_split(
                Y, n_train, sequential=False
            )
        Y_train = Y[train_idx]
        Y_test = Y[test_idx]
        X_train = self.X_dist[:n_train]

        if self.conformalize:

            def objective(trial):
                sigma = trial.suggest_float("sigma", 0.01, 10, log=True)
                lambd = trial.suggest_float("lambd", 1e-5, 1, log=True)
                gamma = 1 / (2 * sigma**2)
                # Determine proper training set size (50% of training data)
                n_proper_train = int(0.5 * len(Y_train))
                # Use stratified split for proper training and calibration sets
                proper_train_idx, calib_idx = self._train_test_split(
                    Y_train, n_proper_train
                )
                # Split the data
                X_proper_train = X_train[proper_train_idx]
                Y_proper_train = Y_train[proper_train_idx]
                X_calib = X_train[calib_idx]
                Y_calib = Y_train[calib_idx]
                # Standardize the response (Y) using proper training set statistics
                if not hasattr(self, "y_scaler_"):
                    self.y_scaler_ = StandardScaler()
                    Y_proper_train_scaled = self.y_scaler_.fit_transform(
                        Y_proper_train
                    )
                else:
                    Y_proper_train_scaled = self.y_scaler_.transform(
                        Y_proper_train
                    )
                # Create model with current parameters and fit on standardized proper training set
                model = self._create_model(gamma, lambd)
                model.fit(X_proper_train, Y_proper_train_scaled)
                # Get predictions on calibration set and transform back to original scale
                preds_calib_scaled = model.predict(X_calib)
                if preds_calib_scaled.ndim == 1:
                    preds_calib_scaled = preds_calib_scaled.reshape(-1, 1)
                # Transform predictions back to original scale
                preds_calib = self.y_scaler_.inverse_transform(
                    preds_calib_scaled
                )
                # Calculate residuals on calibration set in original scale
                res_calib = Y_calib - preds_calib
                # Standardize residuals using calibration set statistics
                if res_calib.ndim == 1:
                    res_mean = np.mean(res_calib)
                    res_std = np.std(res_calib, ddof=1)
                    # Avoid division by zero
                    res_std = res_std if res_std > 1e-10 else 1.0
                    res_calib_standardized = (res_calib - res_mean) / res_std
                else:
                    res_mean = np.mean(res_calib, axis=0)
                    res_std = np.std(res_calib, axis=0, ddof=1)
                    # Avoid division by zero
                    res_std = np.where(res_std > 1e-10, res_std, 1.0)
                    res_calib_standardized = (res_calib - res_mean) / res_std

                # Store the calibrated standardized residuals for use in the generation method
                calibrated_residuals = (
                    res_calib_standardized * res_std + res_mean
                )

                # Use the existing method to generate pseudo samples with conformal prediction
                Y_sim = self._generate_pseudo_with_model(
                    model, calibrated_residuals, len(Y_test)
                )

                # Calculate distance metric
                if metric == "energy":
                    dist_val = self._custom_energy_distance(Y_test, Y_sim)
                elif metric == "mmd":
                    dist_val = self._mmd(Y_test, Y_sim)
                elif metric == "wasserstein" and d == 1:
                    dist_val = stats.wasserstein_distance(
                        Y_test.flatten(), Y_sim.flatten()
                    )
                else:
                    raise ValueError("Invalid metric for dimension")

                return dist_val

        else:

            def objective(trial):
                sigma = trial.suggest_float("sigma", 0.01, 10, log=True)
                lambd = trial.suggest_float("lambd", 1e-5, 1, log=True)
                gamma = 1 / (2 * sigma**2)
                # Create model with current parameters
                model = self._create_model(gamma, lambd)
                model.fit(X_train, Y_train)
                preds_train = model.predict(X_train)
                if preds_train.ndim == 1:
                    preds_train = preds_train.reshape(-1, 1)
                res = Y_train - preds_train
                Y_sim = self._generate_pseudo_with_model(
                    model, res, len(Y_test)
                )
                if metric == "energy":
                    dist_val = self._custom_energy_distance(Y_test, Y_sim)
                elif metric == "mmd":
                    dist_val = self._mmd(Y_test, Y_sim)
                elif metric == "wasserstein" and d == 1:
                    dist_val = stats.wasserstein_distance(
                        Y_test.flatten(), Y_sim.flatten()
                    )
                else:
                    raise ValueError("Invalid metric for dimension")
                return dist_val

        # Optimize hyperparameters
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, **kwargs)
        # Store best parameters and fit final model
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        sigma = self.best_params_["sigma"]
        lambd = self.best_params_["lambd"]
        gamma = 1 / (2 * sigma**2)
        # Fit final model with best parameters
        self.model = self._create_model(gamma, lambd)
        self.model.fit(X_train, Y_train)
        # Compute residuals
        preds_train = self.model.predict(X_train)
        if preds_train.ndim == 1:
            preds_train = preds_train.reshape(-1, 1)
        self.residuals_ = Y_train - preds_train
        # Fit the residual sampler
        self._fit_residual_sampler()
        self.is_fitted = True
        # Print final configuration
        if self.actual_use_rff_:
            print(
                f"  Using {self.kernel_approximation.upper()} with {self.actual_rff_components_} components"
            )
        else:
            print(f"  Using standard kernel method")
        return self

    def _generate_pseudo_with_model(self, model, residuals, num_samples):
        """Helper method to generate data with a specific model."""
        X_new = self.X_dist[:num_samples]

        # Handle prediction based on model type
        if hasattr(model, "named_steps"):
            # Pipeline (RFF or Nystroem)
            preds = model.predict(X_new)
        else:
            # Standard model
            preds = model.predict(X_new)

        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        # Temporarily store original state
        original_residuals = self.residuals_
        original_kde = self.kde_model_
        original_gmm = self.gmm_model_

        # Set residuals for this model
        self.residuals_ = residuals

        # Fit sampler with the new residuals
        self._fit_residual_sampler()

        # Sample residuals
        sampled_residuals = self._sample_residuals(num_samples)

        # Restore original state
        self.residuals_ = original_residuals
        self.kde_model_ = original_kde
        self.gmm_model_ = original_gmm

        return preds + sampled_residuals

    def sample(self, n_samples=1):
        """
        Generate synthetic samples.

        Parameters:
        -----------
        n_samples : int, default=1
            Number of samples to generate

        Returns:
        --------
        Y_sim : array, shape (n_samples, n_features)
            Generated synthetic data
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._generate_pseudo(n_samples)

    def compare_approximation_methods(self, Y, n_train=None, n_trials=20):
        """
        Compare different kernel approximation methods.

        Parameters:
        -----------
        Y : array-like
            Target data
        n_train : int, default=None
            Number of training samples
        n_trials : int, default=20
            Number of optimization trials

        Returns:
        --------
        comparison_results : dict
            Comparison results
        """
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        print("Comparing Kernel Approximation Methods...")

        # Store original settings
        original_use_rff = self.use_rff
        original_approximation = self.kernel_approximation
        original_is_fitted = self.is_fitted

        methods = ["rff", "nystroem"]
        results = {}

        for method in methods:
            print(f"\nTesting {method.upper()}...")
            self.use_rff = True
            self.kernel_approximation = method

            start_time = time()
            self.fit(Y, n_train=n_train, n_trials=n_trials)
            method_time = time() - start_time
            method_score = self.best_score_
            method_params = self.best_params_

            results[method] = {
                "time": method_time,
                "score": method_score,
                "params": method_params,
                "components": self.actual_rff_components_,
            }

        # Test standard method for comparison
        print(f"\nTesting Standard Kernel...")
        self.use_rff = False
        start_time = time()
        self.fit(Y, n_train=n_train, n_trials=n_trials)
        standard_time = time() - start_time
        standard_score = self.best_score_
        standard_params = self.best_params_

        results["standard"] = {
            "time": standard_time,
            "score": standard_score,
            "params": standard_params,
            "components": "N/A",
        }

        # Restore original settings
        self.use_rff = original_use_rff
        self.kernel_approximation = original_approximation
        self.is_fitted = original_is_fitted

        # Print comparison
        print("\n" + "=" * 60)
        print("KERNEL APPROXIMATION COMPARISON RESULTS")
        print("=" * 60)

        for method in ["standard"] + methods:
            data = results[method]
            print(f"\n{method.upper()}:")
            print(f"  Time: {data['time']:.2f}s")
            print(f"  Score: {data['score']:.6f}")
            print(f"  Components: {data['components']}")
            if method != "standard":
                speedup = standard_time / data["time"]
                score_ratio = data["score"] / standard_score
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Score Ratio: {score_ratio:.4f}")

        return results

    def compare_residual_sampling(self, n_samples=1000):
        """
        Compare different residual sampling methods visually.

        Parameters:
        -----------
        n_samples : int, default=1000
            Number of samples to generate for comparison
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        # Store original sampling method
        original_sampling = self.residual_sampling
        # Generate samples with different methods
        sampling_methods = ["bootstrap", "kde", "gmm"]
        samples = {}

        for method in sampling_methods:
            self.residual_sampling = method
            if method == "kde":
                self._fit_residual_sampler()
            elif method == "gmm":
                self._fit_residual_sampler()
            samples[method] = self._sample_residuals(n_samples)
        # Restore original method
        self.residual_sampling = original_sampling
        self._fit_residual_sampler()
        # Plot comparison
        n_dims = self.residuals_.shape[1]
        fig, axes = plt.subplots(
            n_dims,
            len(sampling_methods) + 1,
            figsize=(5 * (len(sampling_methods) + 1), 4 * n_dims),
        )

        if n_dims == 1:
            axes = axes.reshape(1, -1)

        for dim in range(n_dims):
            # Original residuals
            axes[dim, 0].hist(
                self.residuals_[:, dim], bins=30, alpha=0.7, density=True
            )
            axes[dim, 0].set_title(f"Original Residuals\nDim {dim+1}")
            axes[dim, 0].set_xlabel("Residual Value")
            axes[dim, 0].set_ylabel("Density")

            # Sampled residuals
            for j, method in enumerate(sampling_methods):
                col = j + 1
                axes[dim, col].hist(
                    samples[method][:, dim], bins=30, alpha=0.7, density=True
                )
                axes[dim, col].set_title(
                    f"{method.upper()} Sampling\nDim {dim+1}"
                )
                axes[dim, col].set_xlabel("Residual Value")
                axes[dim, col].set_ylabel("Density")

        plt.tight_layout()
        plt.show()

        return samples

    def _perm_test(self, Y_orig, Y_sim, stat_func, n_perm=1000):
        """Permutation test for distribution comparison."""
        if Y_orig.ndim == 1:
            Y_orig = Y_orig.reshape(-1, 1)
        if Y_sim.ndim == 1:
            Y_sim = Y_sim.reshape(-1, 1)

        obs = stat_func(Y_orig, Y_sim)
        combined = np.vstack((Y_orig, Y_sim))
        n1 = Y_orig.shape[0]
        perms = np.zeros(n_perm)

        for i in range(n_perm):
            idx = np.random.permutation(combined.shape[0])
            p1 = combined[idx[:n1]]
            p2 = combined[idx[n1:]]
            perms[i] = stat_func(p1, p2)

        pval = (np.sum(perms >= obs) + 1) / (n_perm + 1)
        return obs, pval

    def _fisher_z_test(self, r1, r2, n1, n2):
        """Fisher z-test for comparing correlation coefficients."""
        z1 = np.arctanh(r1)
        z2 = np.arctanh(r2)
        z = (z1 - z2) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
        p = 2 * (1 - stats.norm.cdf(np.abs(z)))
        return z, p

    def test_similarity(self, Y_orig, Y_sim, n_perm=1000):
        """
        Test statistical similarity between original and synthetic data.

        Parameters:
        -----------
        Y_orig : array-like
            Original data
        Y_sim : array-like
            Synthetic data
        n_perm : int, default=1000
            Number of permutations for permutation tests

        Returns:
        --------
        results : dict
            Dictionary containing test results
        """
        if Y_orig.ndim == 1:
            Y_orig = Y_orig.reshape(-1, 1)
        if Y_sim.ndim == 1:
            Y_sim = Y_sim.reshape(-1, 1)

        d = Y_orig.shape[1]
        results = {}
        # Test 1: Perm with energy
        results["energy_perm"] = self._perm_test(
            Y_orig, Y_sim, self._custom_energy_distance, n_perm
        )
        # Test 2: Perm with MMD
        results["mmd_perm"] = self._perm_test(
            Y_orig, Y_sim, lambda u, v: self._mmd(u, v), n_perm
        )

        # Test 3: Perm with avg Wasserstein on margins
        def avg_wass(u, v):
            return np.mean(
                [stats.wasserstein_distance(u[:, i], v[:, i]) for i in range(d)]
            )

        results["avg_wass_perm"] = self._perm_test(
            Y_orig, Y_sim, avg_wass, n_perm
        )
        # Test 4: Min p-value from marginal KS tests
        ps_ks = [
            stats.ks_2samp(Y_orig[:, i], Y_sim[:, i]).pvalue for i in range(d)
        ]
        results["min_marginal_ks_p"] = min(ps_ks)
        # Test 5: Min p-value from marginal Anderson-Darling tests
        ps_ad = [
            stats.anderson_ksamp([Y_orig[:, i], Y_sim[:, i]]).significance_level
            for i in range(d)
        ]
        results["min_marginal_ad_p"] = min(ps_ad)
        # Test 6: Min p-value from marginal Cramer-von Mises tests
        ps_cvm = [
            stats.cramervonmises_2samp(Y_orig[:, i], Y_sim[:, i]).pvalue
            for i in range(d)
        ]
        results["min_marginal_cvm_p"] = min(ps_cvm)
        # Correlation test: Compare all pairwise correlations
        corr_results = {}
        pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
        for i, j in pairs:
            r_orig = stats.pearsonr(Y_orig[:, i], Y_orig[:, j])[0]
            r_sim = stats.pearsonr(Y_sim[:, i], Y_sim[:, j])[0]
            z, p = self._fisher_z_test(r_orig, r_sim, len(Y_orig), len(Y_sim))
            corr_results[f"corr_dim{i+1}_dim{j+1}"] = (r_orig, r_sim, z, p)
        results["corr_tests"] = corr_results
        return results

    def compare_distributions(self, Y_orig, Y_sim, save_prefix=""):
        """
        Visual comparison of original and synthetic distributions.

        Parameters:
        -----------
        Y_orig : array-like
            Original data
        Y_sim : array-like
            Synthetic data
        save_prefix : str, default=''
            Prefix for saving plots
        """
        if Y_orig.ndim == 1:
            Y_orig = Y_orig.reshape(-1, 1)
        if Y_sim.ndim == 1:
            Y_sim = Y_sim.reshape(-1, 1)

        n, d = Y_orig.shape

        # Create a figure with subplots for statistical tests
        fig, axes = plt.subplots(2, d, figsize=(6 * d, 10))
        if d == 1:
            axes = axes.reshape(2, 1)

        # Statistical test results storage
        ks_results = []
        ad_results = []

        for i in range(d):
            # Top row: Histograms with statistical test annotations
            ax_hist = axes[0, i]

            # Plot histograms
            ax_hist.hist(
                Y_orig[:, i],
                alpha=0.5,
                label="Original",
                density=True,
                bins=20,
                color="blue",
            )
            ax_hist.hist(
                Y_sim[:, i],
                alpha=0.5,
                label="Simulated",
                density=True,
                bins=20,
                color="red",
            )

            # Perform statistical tests
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(Y_orig[:, i], Y_sim[:, i])
            ks_results.append((ks_stat, ks_pvalue))

            # Anderson-Darling test
            ad_result = stats.anderson_ksamp([Y_orig[:, i], Y_sim[:, i]])
            ad_stat = ad_result.statistic
            ad_critical = ad_result.critical_values
            ad_significance = ad_result.significance_level
            ad_results.append((ad_stat, ad_significance))

            # Add test results to histogram plot
            textstr = "\n".join(
                (
                    f"KS test: p = {ks_pvalue:.4f}",
                    f"AD test: p < {ad_significance:.3f}",
                    f"AD stat: {ad_stat:.4f}",
                )
            )
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            ax_hist.text(
                0.05,
                0.95,
                textstr,
                transform=ax_hist.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=props,
            )

            ax_hist.legend()
            ax_hist.set_title(
                f"Dimension {i+1} - Histograms with Statistical Tests"
            )
            ax_hist.set_xlabel("Value")
            ax_hist.set_ylabel("Density")

            # Bottom row: ECDFs with KS test visualization
            ax_ecdf = axes[1, i]

            # Compute ECDFs
            sorted_orig = np.sort(Y_orig[:, i])
            ecdf_orig = np.arange(1, len(sorted_orig) + 1) / len(sorted_orig)
            sorted_sim = np.sort(Y_sim[:, i])
            ecdf_sim = np.arange(1, len(sorted_sim) + 1) / len(sorted_sim)

            # Plot ECDFs
            ax_ecdf.step(
                sorted_orig,
                ecdf_orig,
                label="Original",
                color="blue",
                linewidth=2,
            )
            ax_ecdf.step(
                sorted_sim,
                ecdf_sim,
                label="Simulated",
                color="red",
                linewidth=2,
            )

            # Find the point of maximum difference for KS test
            # Combine and sort all values
            all_values = np.sort(np.concatenate([sorted_orig, sorted_sim]))
            # Compute ECDFs at all points
            ecdf_orig_all = np.searchsorted(
                sorted_orig, all_values, side="right"
            ) / len(sorted_orig)
            ecdf_sim_all = np.searchsorted(
                sorted_sim, all_values, side="right"
            ) / len(sorted_sim)
            # Find maximum difference
            diff = np.abs(ecdf_orig_all - ecdf_sim_all)
            max_idx = np.argmax(diff)
            max_x = all_values[max_idx]
            max_y1 = ecdf_orig_all[max_idx]
            max_y2 = ecdf_sim_all[max_idx]

            # Mark the maximum difference point
            ax_ecdf.plot(
                [max_x, max_x],
                [max_y1, max_y2],
                "k-",
                linewidth=3,
                label=f"KS stat: {ks_stat:.4f}",
            )
            ax_ecdf.plot(max_x, max_y1, "ko", markersize=8)
            ax_ecdf.plot(max_x, max_y2, "ko", markersize=8)

            ax_ecdf.legend()
            ax_ecdf.set_title(f"Dimension {i+1} - ECDFs with KS Statistic")
            ax_ecdf.set_xlabel("Value")
            ax_ecdf.set_ylabel("ECDF")

        plt.tight_layout()
        if save_prefix:
            plt.savefig(
                f"{save_prefix}_statistical_comparison.png",
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

        # Print comprehensive test results
        print("\n" + "=" * 60)
        print("COMPREHENSIVE STATISTICAL TEST RESULTS")
        print("=" * 60)

        for i in range(d):
            ks_stat, ks_pvalue = ks_results[i]
            ad_stat, ad_significance = ad_results[i]

            print(f"\nDimension {i+1}:")
            print(f"  Kolmogorov-Smirnov Test:")
            print(f"    Statistic: {ks_stat:.6f}")
            print(f"    p-value: {ks_pvalue:.6f}")
            print(
                f"    Significance: {'Not Significant' if ks_pvalue > 0.05 else 'SIGNIFICANT'}"
            )

            print(f"  Anderson-Darling Test:")
            print(f"    Statistic: {ad_stat:.6f}")
            print(f"    Significance level: {ad_significance:.3f}")
            print(
                f"    Interpretation: {'Distributions differ' if ad_stat > ad_result.critical_values[2] else 'Distributions similar'}"
            )

        # Create summary plot for all dimensions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # KS test p-values across dimensions
        ks_pvalues = [result[1] for result in ks_results]
        dimensions = list(range(1, d + 1))

        bars = ax1.bar(
            dimensions,
            ks_pvalues,
            color=["red" if p < 0.05 else "green" for p in ks_pvalues],
        )
        ax1.axhline(
            y=0.05, color="black", linestyle="--", alpha=0.7, label="α = 0.05"
        )
        ax1.set_xlabel("Dimension")
        ax1.set_ylabel("KS Test p-value")
        ax1.set_title("Kolmogorov-Smirnov Test Results\nby Dimension")
        ax1.set_xticks(dimensions)
        ax1.legend()

        # Add value labels on bars
        for bar, pvalue in zip(bars, ks_pvalues):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{pvalue:.3f}",
                ha="center",
                va="bottom",
            )

        # AD test statistics across dimensions
        ad_stats = [result[0] for result in ad_results]

        bars = ax2.bar(dimensions, ad_stats, color="skyblue")
        ax2.set_xlabel("Dimension")
        ax2.set_ylabel("AD Test Statistic")
        ax2.set_title("Anderson-Darling Test Statistics\nby Dimension")
        ax2.set_xticks(dimensions)

        # Add value labels on bars
        for bar, stat in zip(bars, ad_stats):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{stat:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        if save_prefix:
            plt.savefig(
                f"{save_prefix}_test_summary.png", dpi=300, bbox_inches="tight"
            )
        plt.show()

        # Additional: Q-Q plots for each dimension
        fig, axes = plt.subplots(1, d, figsize=(5 * d, 5))
        if d == 1:
            axes = [axes]

        for i in range(d):
            # Sort data for Q-Q plot
            orig_sorted = np.sort(Y_orig[:, i])
            sim_sorted = np.sort(Y_sim[:, i])

            # Generate theoretical quantiles
            n_orig = len(orig_sorted)
            n_sim = len(sim_sorted)

            # Use smaller set for quantiles to avoid interpolation issues
            n_points = min(n_orig, n_sim, 1000)
            quantiles = np.linspace(0, 1, n_points)

            orig_quantiles = np.quantile(orig_sorted, quantiles)
            sim_quantiles = np.quantile(sim_sorted, quantiles)

            axes[i].plot(
                orig_quantiles, sim_quantiles, "o", alpha=0.6, markersize=3
            )
            min_val = min(orig_quantiles.min(), sim_quantiles.min())
            max_val = max(orig_quantiles.max(), sim_quantiles.max())
            axes[i].plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                alpha=0.8,
                linewidth=2,
            )
            axes[i].set_xlabel("Original Data Quantiles")
            axes[i].set_ylabel("Simulated Data Quantiles")
            axes[i].set_title(f"Dimension {i+1} - Q-Q Plot")

            # Add correlation coefficient
            corr = np.corrcoef(orig_quantiles, sim_quantiles)[0, 1]
            axes[i].text(
                0.05,
                0.95,
                f"Corr: {corr:.4f}",
                transform=axes[i].transAxes,
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="white", alpha=0.8
                ),
                verticalalignment="top",
            )

        plt.tight_layout()
        if save_prefix:
            plt.savefig(
                f"{save_prefix}_qq_plots.png", dpi=300, bbox_inches="tight"
            )
        plt.show()

        return {
            "ks_results": ks_results,
            "ad_results": ad_results,
            "dimensions": d,
        }
