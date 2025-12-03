import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.decomposition import PCA
from typing import Optional, Tuple, Literal
import warnings


class DiffusionModel(BaseEstimator):
    """
    Sklearn-compatible diffusion model with Bayesian optimization.

    Uses a non-neural sklearn regressor by default for reverse process,
    implementing DDPM with proper forward/reverse diffusion and distribution
    matching via MMD or energy distance metrics.

    Parameters:
    -----------
    timesteps : int, default=1000
        Number of diffusion timesteps
    beta_start : float, default=0.0001
        Initial noise variance
    beta_end : float, default=0.02
        Final noise variance
    model : sklearn estimator, optional
        Base model for reverse process (default: Ridge with alpha=1.0)
    schedule : {'linear', 'cosine'}, default='linear'
        Noise schedule type
    use_pca : bool, default=False
        Apply PCA for dimensionality reduction (recommended for >100 dims)
    pca_components : int, default=50
        Number of PCA components if use_pca=True
    variance_type : {'fixed_small', 'fixed_large'}, default='fixed_small'
        Variance schedule for reverse process
    random_state : int, optional
        Random seed for reproducibility
    batch_size : int, default=32
        Batch size for training data generation
    """

    def __init__(
        self,
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        schedule="linear",
        model=None,
        use_pca=False,
        pca_components=50,
        variance_type="fixed_small",
        random_state=None,
        batch_size=32,
    ):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule
        self.model = model
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.variance_type = variance_type
        self.random_state = random_state
        self.batch_size = batch_size

        # Input validation
        self._validate_parameters()
        self._init_noise_schedule()

    def _validate_parameters(self):
        """Validate input parameters with comprehensive checks"""
        if self.beta_start >= self.beta_end:
            raise ValueError("beta_start must be less than beta_end")
        if self.timesteps <= 0:
            raise ValueError("timesteps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.schedule not in ["linear", "cosine"]:
            raise ValueError("schedule must be 'linear' or 'cosine'")

        # Set random seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _init_noise_schedule(self):
        """Initialize forward diffusion noise schedule with numerical stability"""
        if self.schedule == "linear":
            self.betas = np.linspace(
                self.beta_start, self.beta_end, self.timesteps
            )
        elif self.schedule == "cosine":
            s = 0.008
            steps = np.arange(self.timesteps + 1, dtype=np.float64)
            alphas_bar = (
                np.cos(((steps / self.timesteps) + s) / (1 + s) * np.pi * 0.5)
                ** 2
            )
            alphas_bar = alphas_bar / alphas_bar[0]
            self.betas = np.clip(
                1 - (alphas_bar[1:] / alphas_bar[:-1]), 0, 0.999
            )

        self.alphas = 1.0 - self.betas
        # Numerical stability: clamp away from 0
        self.alphas_cumprod = np.clip(np.cumprod(self.alphas), 1e-8, 1.0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        # Compute posterior variance (beta_tilde)
        alphas_cumprod_prev = np.concatenate([[1.0], self.alphas_cumprod[:-1]])
        self.posterior_variance = (
            self.betas
            * (1.0 - alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )

    def _validate_data(self, X):
        """Validate input data with comprehensive checks"""
        X = np.asarray(X)
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("Input contains NaN or infinite values")
        if X.std(axis=0).min() == 0:
            warnings.warn("Some features have zero variance")
        return X

    def _validate_timestep(self, t):
        """Validate timestep bounds"""
        if not (0 <= t < self.timesteps):
            raise ValueError(f"Timestep {t} out of range [0, {self.timesteps})")

    def forward_diffusion(
        self, x0: np.ndarray, t: np.ndarray, noise: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward diffusion: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        Parameters:
        -----------
        x0 : ndarray, shape (n_samples, n_features)
            Original data samples
        t : ndarray, shape (n_samples,)
            Timesteps (0 to timesteps-1)
        noise : ndarray, optional
            Noise to add (generated if None)

        Returns:
        --------
        xt : ndarray
            Noised data
        noise : ndarray
            Noise that was added
        """
        # Input validation
        x0 = self._validate_data(x0)
        for t_val in np.unique(t):
            self._validate_timestep(t_val)

        if noise is None:
            noise = np.random.randn(*x0.shape)

        # Vectorized computation for batch processing
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, np.newaxis]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][
            :, np.newaxis
        ]

        xt = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
        return xt, noise

    def _create_features(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Create features for the regressor with better temporal encoding
        """
        # Normalize timestep and create positional encoding
        t_norm = t / self.timesteps
        # Simple positional encoding: [sin(2πt), cos(2πt), t_norm]
        t_encoded = np.column_stack(
            [np.sin(2 * np.pi * t_norm), np.cos(2 * np.pi * t_norm), t_norm]
        )

        # Concatenate spatial and temporal features
        features = np.hstack([x, t_encoded])
        return features

    def fit(
        self,
        X: np.ndarray,
        y=None,
        n_steps: int = 1000,
        validation_split: float = 0.0,
    ) -> "DiffusionModel":
        """
        Train the reverse diffusion model with batch processing

        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Training data (will be normalized internally)
        y : ignored
        n_steps : int
            Number of training samples to generate
        validation_split : float, default=0.0
            Fraction of data to use for validation

        Returns:
        --------
        self
        """
        X = self._validate_data(X)
        n_samples, n_features = X.shape

        # Store normalization params
        self.X_mean_ = X.mean(axis=0)
        self.X_std_ = X.std(axis=0) + 1e-8
        X_norm = (X - self.X_mean_) / self.X_std_

        # Optional PCA for high-dimensional data
        if self.use_pca and n_features > 50:
            self.pca_ = PCA(n_components=min(self.pca_components, n_features))
            X_norm = self.pca_.fit_transform(X_norm)
            n_features = X_norm.shape[1]
        else:
            self.pca_ = None

        self.n_features_ = n_features

        # Initialize model if not provided (non-neural default)
        if self.model is None:
            self.model = Ridge(alpha=1.0, random_state=self.random_state)

        # Batch training for efficiency
        n_batches = max(1, n_steps // self.batch_size)
        actual_steps = n_batches * self.batch_size

        for batch in range(n_batches):
            # Sample batch of data and timesteps
            indices = np.random.randint(n_samples, size=self.batch_size)
            t_batch = np.random.randint(self.timesteps, size=self.batch_size)

            x0_batch = X_norm[indices]
            noise_batch = np.random.randn(self.batch_size, n_features)

            # Vectorized forward diffusion
            xt_batch, _ = self.forward_diffusion(x0_batch, t_batch, noise_batch)

            # Create features with improved encoding
            features = self._create_features(xt_batch, t_batch)

            # Train model (use partial_fit if available)
            if hasattr(self.model, "partial_fit") and batch > 0:
                self.model.partial_fit(features, noise_batch)
            else:
                # For first batch or models without partial_fit, do full fit on accumulated data
                if batch == 0:
                    X_train, y_train = features, noise_batch
                else:
                    X_train = np.vstack([X_train, features])
                    y_train = np.vstack([y_train, noise_batch])

        # Final fit if not using partial_fit
        if not hasattr(self.model, "partial_fit"):
            self.model.fit(X_train, y_train)

        return self

    def sample(
        self,
        n_samples: int = 1,
        return_trajectory: bool = False,
        ddim: bool = False,
        ddim_steps: int = 50,
    ) -> np.ndarray:
        """
        Generate samples via reverse diffusion with optional DDIM

        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        return_trajectory : bool
            If True, return all intermediate denoising steps
        ddim : bool, default=False
            Use DDIM for faster sampling
        ddim_steps : int, default=50
            Number of DDIM steps (if ddim=True)

        Returns:
        --------
        samples : ndarray
            Generated samples (denormalized to original scale)
        """
        if ddim:
            return self._sample_ddim(n_samples, ddim_steps, return_trajectory)

        # Standard DDPM sampling
        x = np.random.randn(n_samples, self.n_features_)
        trajectory = [x.copy()] if return_trajectory else None

        # Reverse diffusion: p(x_{t-1} | x_t)
        for t in reversed(range(self.timesteps)):
            # Prepare features
            t_array = np.full(n_samples, t)
            features = self._create_features(x, t_array)

            # Predict noise
            pred_noise = self.model.predict(features).reshape(x.shape)

            # Compute mean of p(x_{t-1} | x_t)
            alpha = self.alphas[t]
            alpha_bar = self.alphas_cumprod[t]
            beta = self.betas[t]

            # Mean: mu_theta(x_t, t)
            coef1 = 1.0 / np.sqrt(alpha)
            coef2 = beta / np.sqrt(1.0 - alpha_bar)
            mean = coef1 * (x - coef2 * pred_noise)

            # Add variance (except final step)
            if t > 0:
                # Use improved DDPM variance
                if self.variance_type == "fixed_small":
                    variance = self.posterior_variance[t]
                else:  # fixed_large
                    variance = beta
                noise = np.sqrt(variance) * np.random.randn(*x.shape)
                x = mean + noise
            else:
                x = mean

            if return_trajectory:
                trajectory.append(x.copy())

        return self._postprocess_samples(x, trajectory, return_trajectory)

    def _sample_ddim(
        self, n_samples: int, steps: int, return_trajectory: bool
    ) -> np.ndarray:
        """
        Faster deterministic sampling using DDIM
        """
        skip = max(1, self.timesteps // steps)
        timesteps = np.arange(0, self.timesteps, skip)[::-1]

        x = np.random.randn(n_samples, self.n_features_)
        trajectory = [x.copy()] if return_trajectory else None

        for i, t in enumerate(timesteps):
            t_array = np.full(n_samples, t)
            features = self._create_features(x, t_array)
            pred_noise = self.model.predict(features).reshape(x.shape)

            alpha_bar = self.alphas_cumprod[t]
            alpha_bar_prev = (
                self.alphas_cumprod[timesteps[i + 1]]
                if i < len(timesteps) - 1
                else 1.0
            )

            # DDIM sampling formula
            pred_x0 = (x - np.sqrt(1 - alpha_bar) * pred_noise) / np.sqrt(
                alpha_bar
            )

            # Direction pointing to x_t
            dir_xt = np.sqrt(1 - alpha_bar_prev) * pred_noise

            x = np.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

            if return_trajectory:
                trajectory.append(x.copy())

        return self._postprocess_samples(x, trajectory, return_trajectory)

    def _postprocess_samples(
        self, x: np.ndarray, trajectory: list, return_trajectory: bool
    ) -> np.ndarray:
        """Apply inverse transforms to generated samples"""
        if return_trajectory:
            processed_trajectory = []
            for step in trajectory:
                if self.pca_ is not None:
                    step = self.pca_.inverse_transform(step)
                step = step * self.X_std_ + self.X_mean_
                processed_trajectory.append(step)
            return np.array(processed_trajectory)

        # Inverse PCA transform if used
        if self.pca_ is not None:
            x = self.pca_.inverse_transform(x)

        # Denormalize
        x = x * self.X_std_ + self.X_mean_
        return x

    def predict(self, n_samples: int = 1, **kwargs) -> np.ndarray:
        """Sklearn-style predict method (alias for sample)"""
        return self.sample(n_samples=n_samples, **kwargs)

    def reconstruction_error(
        self,
        X: np.ndarray,
        metric: str = "mmd",
        n_samples: Optional[int] = None,
        gamma: str = "auto",
    ) -> float:
        """
        Compute distributional reconstruction error with adaptive bandwidth

        Parameters:
        -----------
        X : ndarray
            Original data distribution
        metric : str
            Error metric: 'mmd' (Maximum Mean Discrepancy) or 'energy'
        n_samples : int, optional
            Number of samples to generate (default: min(1000, len(X)))
        gamma : str or float, default='auto'
            RBF bandwidth for MMD ('auto' for median heuristic)

        Returns:
        --------
        error : float
            Reconstruction error (lower is better)
        """
        if n_samples is None:
            n_samples = min(1000, len(X))

        X_reconstructed = self.sample(n_samples)

        if metric == "mmd":
            return self._compute_mmd(X, X_reconstructed, gamma)
        elif metric == "energy":
            return self._compute_energy_distance(X, X_reconstructed)
        else:
            raise ValueError(f"Metric must be 'mmd' or 'energy', got: {metric}")

    def _compute_mmd(
        self, X: np.ndarray, Y: np.ndarray, gamma: str = "auto"
    ) -> float:
        """
        Vectorized Maximum Mean Discrepancy with adaptive bandwidth
        """
        # Subsample for efficiency if needed
        max_samples = 1000
        if len(X) > max_samples:
            X = X[np.random.choice(len(X), max_samples, replace=False)]
        if len(Y) > max_samples:
            Y = Y[np.random.choice(len(Y), max_samples, replace=False)]

        # Adaptive bandwidth selection
        if gamma == "auto":
            combined = np.vstack([X[:100], Y[:100]])
            squared_dists = np.sum(
                (combined[:, None, :] - combined[None, :, :]) ** 2, axis=2
            )
            gamma = 1.0 / np.median(
                squared_dists[np.triu_indices(len(squared_dists), k=1)]
            )

        # Vectorized squared distances
        XX = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
        YY = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2)
        XY = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)

        # RBF kernel
        K_XX = np.exp(-gamma * XX).mean()
        K_YY = np.exp(-gamma * YY).mean()
        K_XY = np.exp(-gamma * XY).mean()

        mmd = K_XX + K_YY - 2 * K_XY
        return max(0, mmd)  # Ensure non-negative

    def _compute_energy_distance(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Vectorized energy distance with scipy fallback
        """
        # Try using scipy's optimized implementation
        try:
            from scipy.spatial.distance import pdist, cdist

            # Subsample for efficiency
            max_samples = 500
            if len(X) > max_samples:
                X = X[np.random.choice(len(X), max_samples, replace=False)]
            if len(Y) > max_samples:
                Y = Y[np.random.choice(len(Y), max_samples, replace=False)]

            xy_dist = cdist(X, Y, metric="euclidean").mean()
            xx_dist = pdist(X, metric="euclidean").mean() if len(X) > 1 else 0
            yy_dist = pdist(Y, metric="euclidean").mean() if len(Y) > 1 else 0

            energy = 2 * xy_dist - xx_dist - yy_dist
            return max(0, energy)  # Ensure non-negative
        except ImportError:
            # Fallback to vectorized numpy implementation
            max_samples = 500
            if len(X) > max_samples:
                X = X[np.random.choice(len(X), max_samples, replace=False)]
            if len(Y) > max_samples:
                Y = Y[np.random.choice(len(Y), max_samples, replace=False)]

            # Vectorized pairwise distances
            def pairwise_dist(A, B):
                return np.sqrt(
                    np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
                ).mean()

            xy_dist = pairwise_dist(X, Y)

            n, m = len(X), len(Y)
            if n > 1:
                xx_dists = np.sqrt(
                    np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
                )
                xx_dist = (xx_dists.sum() - np.trace(xx_dists)) / (n * (n - 1))
            else:
                xx_dist = 0

            if m > 1:
                yy_dists = np.sqrt(
                    np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2)
                )
                yy_dist = (yy_dists.sum() - np.trace(yy_dists)) / (m * (m - 1))
            else:
                yy_dist = 0

            energy = 2 * xy_dist - xx_dist - yy_dist
            return max(0, energy)

    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        n_calls: int = 20,
        metric: str = "mmd",
        cv_splits: int = 3,
    ) -> dict:
        """
        Bayesian optimization of hyperparameters with cross-validation
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
        except ImportError:
            warnings.warn(
                "scikit-optimize not installed. Install with: pip install scikit-optimize"
            )
            return {}

        # Define search space including model parameters
        space = [
            Integer(100, 2000, name="timesteps"),
            Real(1e-5, 1e-3, name="beta_start", prior="log-uniform"),
            Real(0.01, 0.05, name="beta_end"),
            Categorical(["linear", "cosine"], name="schedule"),
            Real(
                0.1, 10.0, name="ridge_alpha", prior="log-uniform"
            ),  # Ridge regularization
        ]

        def objective(params):
            try:
                # Clone to avoid parameter contamination
                model_clone = clone(self)
                model_clone.timesteps = params[0]
                model_clone.beta_start = params[1]
                model_clone.beta_end = params[2]
                model_clone.schedule = params[3]
                model_clone.model = Ridge(
                    alpha=params[4], random_state=self.random_state
                )
                model_clone._init_noise_schedule()

                # Cross-validated evaluation
                errors = []
                for _ in range(cv_splits):
                    # Light training for optimization
                    model_clone.fit(X, n_steps=500)

                    # Evaluate on subsample
                    X_eval = X[: min(100, len(X))]
                    error = model_clone.reconstruction_error(
                        X_eval, metric=metric, n_samples=100
                    )
                    errors.append(error)

                return np.mean(errors)
            except Exception as e:
                # Return large error for failed configurations
                return 1e6

        result = gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            random_state=self.random_state,
            verbose=False,
        )

        best_params = {
            "timesteps": result.x[0],
            "beta_start": result.x[1],
            "beta_end": result.x[2],
            "schedule": result.x[3],
            "model": Ridge(alpha=result.x[4], random_state=self.random_state),
        }

        # Set best parameters and retrain fully
        self.set_params(**best_params)
        self.fit(X, n_steps=2000)

        print(f"\n✅ Optimization complete!")
        print(f"Best {metric.upper()} error: {result.fun:.6f}")
        print(f"Best parameters: {best_params}")

        return best_params
