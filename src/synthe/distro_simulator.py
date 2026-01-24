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

        # Initialize random number generator with the seed
        self.rng = np.random.RandomState(random_state)
        
        # Set global numpy seed for sklearn consistency
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize JAX random key if JAX is available
        self.jax_key = None
        if JAX_AVAILABLE and random_state is not None:
            self.jax_key = jax.random.PRNGKey(random_state)

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

        # Initialize JAX key if not already done
        if self.jax_key is None:
            seed = self.random_state if self.random_state is not None else 0
            self.jax_key = jax.random.PRNGKey(seed)

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
                    ("ridge", Ridge(alpha=alpha, random_state=self.random_state)),
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
                cv=3,
                random_state=self.random_state,
            )
            grid.fit(self.residuals_)
            self.kde_model_ = grid.best_estimator_

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
            idx = self.rng.choice(n, num_samples, replace=True)
            return self.residuals_[idx]

        elif self.residual_sampling == "kde":
            # Kernel Density Estimation sampling
            if self.kde_model_ is None:
                raise ValueError(
                    "KDE model not fitted. Call _fit_residual_sampler first."
                )
            # Sample from KDE with random_state
            samples = self.kde_model_.sample(num_samples, random_state=self.rng)
            return samples

        elif self.residual_sampling == "gmm":
            # Gaussian Mixture Model sampling
            if self.gmm_model_ is None:
                raise ValueError(
                    "GMM model not fitted. Call _fit_residual_sampler first."
                )
            # Sample from GMM
            return self.gmm_model_.sample(num_samples)[0]

        elif self.residual_sampling == "me-bootstrap":
            # Note: MaximumEntropyBootstrap needs to be imported
            # from .meboot import MaximumEntropyBootstrap
            # meb = MaximumEntropyBootstrap(random_state=self.random_state)
            residuals = self.residuals_.flatten()
            if residuals.shape[0] < num_samples:
                repeats = int(np.ceil(num_samples / residuals.shape[0]))
                residuals = np.tile(residuals, repeats)[:num_samples]
            else:
                residuals = residuals[:num_samples]
            # meb.fit(residuals)
            # return meb.sample(1)[:, 0].reshape(-1, 1)
            # Placeholder for ME-bootstrap
            idx = self.rng.choice(len(residuals), num_samples, replace=True)
            return residuals[idx].reshape(-1, 1)

        elif self.residual_sampling == "block-bootstrap":
            # Note: bootstrap function needs to be imported
            # from .utils import bootstrap
            # return bootstrap(
            #     self.residuals_,
            #     num_samples,
            #     block_size=self.block_size,
            #     seed=self.random_state,
            # )
            # Placeholder for block bootstrap
            idx = self.rng.choice(len(self.residuals_), num_samples, replace=True)
            return self.residuals_[idx]

        else:
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
        n_samples = len(Y)
        
        # Adjust number of clusters based on dataset size to avoid tiny clusters
        # Rule: ensure at least 10 samples per cluster on average
        effective_n_clusters = min(self.n_clusters, max(2, n_samples // 10))
        
        if effective_n_clusters < self.n_clusters:
            warnings.warn(
                f"Reducing n_clusters from {self.n_clusters} to {effective_n_clusters} "
                f"due to small dataset size (n={n_samples}).",
                UserWarning
            )
        
        if self.clustering_method == "kmeans":
            self.cluster_model_ = KMeans(
                n_clusters=effective_n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
        elif self.clustering_method == "gmm":
            self.cluster_model_ = GaussianMixture(
                n_components=effective_n_clusters, 
                random_state=self.random_state
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
            # Sequential split (no shuffling, preserves temporal order)
            train_idx = np.arange(n_train)
            test_idx = np.arange(n_train, n_samples)
            return train_idx, test_idx

        # Stratified split (default)
        self.cluster_labels_ = self._compute_clusters(Y)
        
        # Check if stratification is possible
        unique_labels, counts = np.unique(self.cluster_labels_, return_counts=True)
        min_cluster_size = counts.min()
        
        # If any cluster has too few samples for stratification, fall back to random split
        if min_cluster_size < 2:
            warnings.warn(
                f"Cluster sizes too small for stratification (min={min_cluster_size}). "
                "Using random split instead.",
                UserWarning
            )
            indices = np.arange(n_samples)
            self.rng.shuffle(indices)
            return indices[:n_train], indices[n_train:]
        
        try:
            return train_test_split(
                np.arange(n_samples),
                train_size=n_train,
                stratify=self.cluster_labels_,
                random_state=self.random_state,
            )
        except ValueError as e:
            # Fall back to random split if stratification fails
            warnings.warn(
                f"Stratification failed: {e}. Using random split instead.",
                UserWarning
            )
            indices = np.arange(n_samples)
            self.rng.shuffle(indices)
            return indices[:n_train], indices[n_train:]

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
            except Exception:
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
        self.X_dist = self.rng.normal(0, 1, (n, d))
        
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
            Y_sim = self._generate_pseudo_with_model(model, res, len(Y_test))
            
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

        # Optimize hyperparameters with seeded sampler
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False, **kwargs)
        
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
                f"Using {self.kernel_approximation.upper()} with {self.actual_rff_components_} components"
            )
        else:
            print(f"Using standard kernel method")
        
        return self

    def _generate_pseudo_with_model(self, model, residuals, num_samples):
        """Helper method to generate data with a specific model."""
        X_new = self.X_dist[:num_samples]

        # Handle prediction based on model type
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

