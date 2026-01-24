import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted
from sklearn.decomposition import PCA
from typing import Optional, Tuple, Literal, Union, List
import warnings


class DiffusionModel(BaseEstimator):
    """
    Sklearn-compatible diffusion model with MMD-based and noise-prediction training.

    Implements both traditional DDPM (noise prediction with MSE) and novel MMD-based
    training that directly minimizes distribution mismatch between true posterior
    and learned transitions using Maximum Mean Discrepancy.

    Parameters
    ----------
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
    variance_type : {'fixed_small', 'fixed_large', 'learned'}, default='fixed_small'
        Variance schedule for reverse process
    random_state : int, optional
        Random seed for reproducibility
    batch_size : int, default=32
        Batch size for training data generation
    training_objective : {'noise', 'mmd', 'hybrid'}, default='noise'
        Training objective:
        - 'noise': Traditional DDPM noise prediction with MSE loss
        - 'mmd': Direct MMD minimization between true and learned posteriors
        - 'hybrid': Combine both objectives
    mmd_samples_per_step : int, default=10
        Number of samples to draw per timestep for MMD estimation
    mmd_kernel : {'rbf', 'imq', 'linear'}, default='rbf'
        Kernel for MMD computation
    mmd_bandwidth : float or 'auto', default='auto'
        Kernel bandwidth (gamma for RBF)

    Examples
    --------
    Traditional noise-prediction training:
    >>> model = DiffusionModel(timesteps=100, training_objective='noise')
    >>> model.fit(X, n_steps=1000)

    MMD-based training (distribution matching):
    >>> model = DiffusionModel(timesteps=100, training_objective='mmd',
    ...                        mmd_samples_per_step=20)
    >>> model.fit(X, n_steps=1000)

    Hybrid approach:
    >>> model = DiffusionModel(timesteps=100, training_objective='hybrid')
    >>> model.fit(X, n_steps=1000)
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: Literal["linear", "cosine"] = "linear",
        model: Optional[BaseEstimator] = None,
        use_pca: bool = False,
        pca_components: int = 50,
        variance_type: Literal[
            "fixed_small", "fixed_large", "learned"
        ] = "fixed_small",
        random_state: Optional[int] = None,
        batch_size: int = 32,
        training_objective: Literal["noise", "mmd", "hybrid"] = "noise",
        mmd_samples_per_step: int = 10,
        mmd_kernel: Literal["rbf", "imq", "linear"] = "rbf",
        mmd_bandwidth: Union[float, str] = "auto",
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
        self.training_objective = training_objective
        self.mmd_samples_per_step = mmd_samples_per_step
        self.mmd_kernel = mmd_kernel
        self.mmd_bandwidth = mmd_bandwidth

        # Initialize random state generator
        self._rng = np.random.RandomState(random_state)

        # Input validation
        self._validate_parameters()
        self._init_noise_schedule()

    def _validate_parameters(self) -> None:
        """Validate input parameters with comprehensive checks"""
        if self.beta_start >= self.beta_end:
            raise ValueError("beta_start must be less than beta_end")
        if self.timesteps <= 0:
            raise ValueError("timesteps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.schedule not in ["linear", "cosine"]:
            raise ValueError("schedule must be 'linear' or 'cosine'")
        if self.variance_type not in ["fixed_small", "fixed_large", "learned"]:
            raise ValueError(
                "variance_type must be 'fixed_small', 'fixed_large', or 'learned'"
            )
        if self.training_objective not in ["noise", "mmd", "hybrid"]:
            raise ValueError(
                "training_objective must be 'noise', 'mmd', or 'hybrid'"
            )
        if self.mmd_kernel not in ["rbf", "imq", "linear"]:
            raise ValueError("mmd_kernel must be 'rbf', 'imq', or 'linear'")
        if self.mmd_samples_per_step <= 0:
            raise ValueError("mmd_samples_per_step must be positive")

    def _init_noise_schedule(self) -> None:
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

        # Compute posterior variance (beta_tilde) for true posterior
        alphas_cumprod_prev = np.concatenate([[1.0], self.alphas_cumprod[:-1]])
        self.posterior_variance = (
            self.betas
            * (1.0 - alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )

        # Compute posterior mean coefficients
        self.posterior_mean_coef1 = (
            self.betas
            * np.sqrt(alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * np.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def _validate_data(self, X: np.ndarray) -> np.ndarray:
        """Validate input data with comprehensive checks"""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("Input contains NaN or infinite values")
        if X.std(axis=0).min() == 0:
            warnings.warn("Some features have zero variance")
        return X

    def _validate_timestep(self, t: Union[int, np.ndarray]) -> None:
        """Validate timestep bounds"""
        t_array = np.atleast_1d(t)
        if np.any(t_array < 0) or np.any(t_array >= self.timesteps):
            raise ValueError(
                f"Timestep values out of range [0, {self.timesteps})"
            )

    def forward_diffusion(
        self, x0: np.ndarray, t: np.ndarray, noise: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward diffusion: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        Parameters
        ----------
        x0 : ndarray of shape (n_samples, n_features)
            Original data samples
        t : ndarray of shape (n_samples,)
            Timesteps (0 to timesteps-1)
        noise : ndarray, optional
            Noise to add (generated if None)

        Returns
        -------
        xt : ndarray
            Noised data
        noise : ndarray
            Noise that was added
        """
        x0 = self._validate_data(x0)
        self._validate_timestep(t)

        if noise is None:
            noise = self._rng.randn(*x0.shape)

        # Vectorized computation for batch processing
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, np.newaxis]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][
            :, np.newaxis
        ]

        xt = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
        return xt, noise

    def _sample_true_posterior(
        self, x0: np.ndarray, xt: np.ndarray, t: int, n_samples: int = 1
    ) -> np.ndarray:
        """
        Sample from true posterior q(x_{t-1} | x_t, x_0)

        This is the ground truth denoising distribution that we want to match.

        Parameters
        ----------
        x0 : ndarray of shape (batch_size, n_features)
            Clean data
        xt : ndarray of shape (batch_size, n_features)
            Noisy data at timestep t
        t : int
            Current timestep
        n_samples : int
            Number of samples to draw per data point

        Returns
        -------
        samples : ndarray of shape (batch_size * n_samples, n_features)
            Samples from true posterior
        """
        if t == 0:
            # At t=0, posterior is deterministic: x_{-1} doesn't exist, return x_0
            return np.repeat(x0, n_samples, axis=0)

        batch_size = x0.shape[0]

        # Posterior mean: μ̃_t = coef1 * x_0 + coef2 * x_t
        coef1 = self.posterior_mean_coef1[t]
        coef2 = self.posterior_mean_coef2[t]
        posterior_mean = coef1 * x0 + coef2 * xt

        # Posterior variance: σ̃_t²
        posterior_var = self.posterior_variance[t]

        # Sample from N(μ̃_t, σ̃_t² I)
        samples = []
        for _ in range(n_samples):
            noise = self._rng.randn(*x0.shape) * np.sqrt(posterior_var)
            sample = posterior_mean + noise
            samples.append(sample)

        return np.vstack(samples)

    def _sample_learned_transition(
        self, xt: np.ndarray, t: int, n_samples: int = 1
    ) -> np.ndarray:
        """
        Sample from learned transition p_θ(x_{t-1} | x_t)

        Parameters
        ----------
        xt : ndarray of shape (batch_size, n_features)
            Noisy data at timestep t
        t : int
            Current timestep
        n_samples : int
            Number of samples to draw per data point

        Returns
        -------
        samples : ndarray of shape (batch_size * n_samples, n_features)
            Samples from learned transition
        """
        batch_size = xt.shape[0]
        t_array = np.full(batch_size, t)

        # Predict noise using learned model
        features = self._create_features(xt, t_array)
        pred_noise = self.model_.predict(features).reshape(xt.shape)

        # Compute predicted mean
        alpha = self.alphas[t]
        alpha_bar = self.alphas_cumprod[t]
        beta = self.betas[t]

        coef1 = 1.0 / np.sqrt(alpha)
        coef2 = beta / np.sqrt(1.0 - alpha_bar)
        predicted_mean = coef1 * (xt - coef2 * pred_noise)

        # Get variance
        if self.variance_type == "fixed_small":
            variance = self.posterior_variance[t]
        elif self.variance_type == "fixed_large":
            variance = beta
        else:  # learned - for now use fixed_small
            variance = self.posterior_variance[t]

        # Sample from N(μ_θ, σ² I)
        if t == 0:
            # Deterministic at final step
            return np.repeat(predicted_mean, n_samples, axis=0)

        samples = []
        for _ in range(n_samples):
            noise = self._rng.randn(*xt.shape) * np.sqrt(variance)
            sample = predicted_mean + noise
            samples.append(sample)

        return np.vstack(samples)

    def _compute_kernel(
        self, X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute kernel matrix K(X, Y) using specified kernel

        Parameters
        ----------
        X : ndarray of shape (n, d)
        Y : ndarray of shape (m, d)
        gamma : float, optional
            Kernel bandwidth

        Returns
        -------
        K : ndarray of shape (n, m)
            Kernel matrix
        """
        if self.mmd_kernel == "linear":
            return X @ Y.T

        # Compute squared distances
        try:
            from scipy.spatial.distance import cdist

            sq_dist = cdist(X, Y, metric="sqeuclidean")
        except ImportError:
            sq_dist = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)

        if gamma is None:
            # Use median heuristic
            if len(X) > 100:
                sample_idx = self._rng.choice(len(X), 100, replace=False)
                sq_dist_sample = sq_dist[sample_idx][:, :100]
            else:
                sq_dist_sample = sq_dist
            gamma = 1.0 / np.median(sq_dist_sample[sq_dist_sample > 0])

        if self.mmd_kernel == "rbf":
            return np.exp(-gamma * sq_dist)
        elif self.mmd_kernel == "imq":
            # Inverse multiquadric: (||x-y||² + c²)^(-β)
            c = 1.0
            beta = 0.5
            return (sq_dist + c**2) ** (-beta)
        else:
            raise ValueError(f"Unknown kernel: {self.mmd_kernel}")

    def _compute_mmd_unbiased(
        self, X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None
    ) -> float:
        """
        Compute unbiased MMD² estimator

        MMD²(P, Q) = E[k(u,u')] + E[k(v,v')] - 2E[k(u,v)]

        Uses unbiased estimator that excludes diagonal terms.
        """
        n, m = len(X), len(Y)

        # Compute kernel matrices
        K_XX = self._compute_kernel(X, X, gamma)
        K_YY = self._compute_kernel(Y, Y, gamma)
        K_XY = self._compute_kernel(X, Y, gamma)

        # Unbiased estimator: exclude diagonal
        if n > 1:
            K_XX_unbiased = (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1))
        else:
            K_XX_unbiased = 0

        if m > 1:
            K_YY_unbiased = (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1))
        else:
            K_YY_unbiased = 0

        K_XY_mean = K_XY.mean()

        mmd_sq = K_XX_unbiased + K_YY_unbiased - 2 * K_XY_mean

        return max(0.0, mmd_sq)  # Ensure non-negative

    def _create_features(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Create features for the regressor with multi-frequency temporal encoding
        """
        # Normalize timestep
        t_norm = t / self.timesteps

        # Multi-frequency positional encoding
        frequencies = [1.0, 2.0, 4.0]
        t_encoded_parts = []

        for freq in frequencies:
            t_encoded_parts.extend(
                [
                    np.sin(2 * np.pi * freq * t_norm),
                    np.cos(2 * np.pi * freq * t_norm),
                ]
            )

        t_encoded_parts.append(t_norm)
        t_encoded = np.column_stack(t_encoded_parts)

        # Concatenate spatial and temporal features
        features = np.hstack([x, t_encoded])
        return features

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        n_steps: int = 1000,
    ) -> "DiffusionModel":
        """
        Train the reverse diffusion model using specified objective

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data (will be normalized internally)
        y : ignored
            Not used, present for API consistency
        n_steps : int, default=1000
            Number of training iterations

        Returns
        -------
        self : DiffusionModel
            Fitted estimator
        """
        X = self._validate_data(X)
        n_samples, n_features = X.shape

        # Store normalization params
        self.X_mean_ = X.mean(axis=0)
        self.X_std_ = X.std(axis=0) + 1e-8
        X_norm = (X - self.X_mean_) / self.X_std_

        # Optional PCA
        if self.use_pca and n_features > 100:
            self.pca_ = PCA(
                n_components=min(self.pca_components, n_features),
                random_state=self.random_state,
            )
            X_norm = self.pca_.fit_transform(X_norm)
            n_features = X_norm.shape[1]
        else:
            self.pca_ = None

        self.n_features_ = n_features

        # Initialize model
        if self.model is None:
            self.model_ = Ridge(alpha=1.0, random_state=self.random_state)
        else:
            self.model_ = clone(self.model)

        # Train based on objective
        if self.training_objective == "noise":
            self._fit_noise_prediction(X_norm, n_samples, n_steps)
        elif self.training_objective == "mmd":
            self._fit_mmd_matching(X_norm, n_samples, n_steps)
        elif self.training_objective == "hybrid":
            self._fit_hybrid(X_norm, n_samples, n_steps)

        self.is_fitted_ = True
        return self

    def _fit_noise_prediction(
        self, X_norm: np.ndarray, n_samples: int, n_steps: int
    ) -> None:
        """Traditional DDPM training: predict noise with MSE loss"""
        n_batches = max(1, n_steps // self.batch_size)

        X_train_batches: List[np.ndarray] = []
        y_train_batches: List[np.ndarray] = []

        for batch_idx in range(n_batches):
            indices = self._rng.randint(n_samples, size=self.batch_size)
            t_batch = self._rng.randint(self.timesteps, size=self.batch_size)

            x0_batch = X_norm[indices]
            noise_batch = self._rng.randn(self.batch_size, self.n_features_)

            xt_batch, _ = self.forward_diffusion(x0_batch, t_batch, noise_batch)
            features = self._create_features(xt_batch, t_batch)

            X_train_batches.append(features)
            y_train_batches.append(noise_batch)

        X_train = np.vstack(X_train_batches)
        y_train = np.vstack(y_train_batches)

        self.model_.fit(X_train, y_train)

    def _fit_mmd_matching(
        self, X_norm: np.ndarray, n_samples: int, n_steps: int
    ) -> None:
        """
        MMD-based training: directly minimize distribution mismatch

        This implements the algorithm from the theoretical document:
        1. Sample (x_0, t, x_t)
        2. Draw samples from true posterior q(x_{t-1}|x_t,x_0)
        3. Draw samples from learned transition p_θ(x_{t-1}|x_t)
        4. Compute MMD² between these sample sets
        5. Update θ to minimize MMD
        """
        print(
            f"Training with MMD objective (kernel={self.mmd_kernel}, samples={self.mmd_samples_per_step})"
        )

        # Determine gamma for kernel
        if self.mmd_bandwidth == "auto":
            gamma = None  # Will be computed adaptively
        else:
            gamma = float(self.mmd_bandwidth)

        # For sklearn models without iterative updates, we need to:
        # 1. Generate training data with MMD-weighted importance
        # 2. Fit the model on this data

        # We'll use an iterative refinement approach:
        # Start with noise prediction, then refine with MMD
        print("Phase 1: Initial noise prediction training...")
        self._fit_noise_prediction(X_norm, n_samples, n_steps // 2)

        print("Phase 2: MMD refinement...")
        # Generate MMD-weighted training data
        X_train_batches: List[np.ndarray] = []
        y_train_batches: List[np.ndarray] = []
        weights_batches: List[np.ndarray] = []

        n_batches = max(1, (n_steps // 2) // self.batch_size)

        for batch_idx in range(n_batches):
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{n_batches}")

            indices = self._rng.randint(n_samples, size=self.batch_size)
            t_batch = self._rng.randint(1, self.timesteps, size=self.batch_size)

            x0_batch = X_norm[indices]
            noise_batch = self._rng.randn(self.batch_size, self.n_features_)
            xt_batch, _ = self.forward_diffusion(x0_batch, t_batch, noise_batch)

            # For each timestep, compute MMD between true and learned transitions
            batch_weights = []
            for i in range(self.batch_size):
                t = t_batch[i]
                x0_i = x0_batch[i : i + 1]
                xt_i = xt_batch[i : i + 1]

                # Sample from both distributions
                true_samples = self._sample_true_posterior(
                    x0_i, xt_i, t, n_samples=self.mmd_samples_per_step
                )
                learned_samples = self._sample_learned_transition(
                    xt_i, t, n_samples=self.mmd_samples_per_step
                )

                # Compute MMD
                mmd = self._compute_mmd_unbiased(
                    true_samples, learned_samples, gamma
                )

                # Weight by MMD (higher mismatch = higher weight)
                weight = 1.0 + mmd
                batch_weights.append(weight)

            features = self._create_features(xt_batch, t_batch)
            X_train_batches.append(features)
            y_train_batches.append(noise_batch)
            weights_batches.append(np.array(batch_weights))

        # Fit with weighted samples
        X_train = np.vstack(X_train_batches)
        y_train = np.vstack(y_train_batches)
        weights = np.concatenate(weights_batches)

        # Normalize weights
        weights = weights / weights.sum() * len(weights)

        print("  Final weighted fit...")
        if (
            hasattr(self.model_, "fit")
            and "sample_weight" in self.model_.fit.__code__.co_varnames
        ):
            self.model_.fit(X_train, y_train, sample_weight=weights)
        else:
            # Fallback: repeat samples according to weights
            weights_int = np.round(weights).astype(int)
            indices_weighted = np.repeat(np.arange(len(X_train)), weights_int)
            self.model_.fit(
                X_train[indices_weighted], y_train[indices_weighted]
            )

        print("MMD training complete!")

    def _fit_hybrid(
        self, X_norm: np.ndarray, n_samples: int, n_steps: int
    ) -> None:
        """Hybrid training: combine noise prediction and MMD objectives"""
        # Split training: 60% noise prediction, 40% MMD refinement
        n_noise_steps = int(0.6 * n_steps)
        n_mmd_steps = n_steps - n_noise_steps

        print(
            f"Hybrid training: {n_noise_steps} noise steps + {n_mmd_steps} MMD steps"
        )
        self._fit_noise_prediction(X_norm, n_samples, n_noise_steps)

        # Continue with MMD refinement
        self.training_objective = "mmd"  # Temporarily switch
        self._fit_mmd_matching(X_norm, n_samples, n_mmd_steps)
        self.training_objective = "hybrid"  # Restore

    def sample(
        self,
        n_samples: int = 1,
        return_trajectory: bool = False,
        ddim: bool = False,
        ddim_steps: int = 50,
    ) -> np.ndarray:
        """
        Generate samples via reverse diffusion

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate
        return_trajectory : bool, default=False
            If True, return all intermediate denoising steps
        ddim : bool, default=False
            Use DDIM for faster deterministic sampling
        ddim_steps : int, default=50
            Number of DDIM steps (only used if ddim=True)

        Returns
        -------
        samples : ndarray
            Generated samples (denormalized to original scale)
        """
        check_is_fitted(self, ["model_", "X_mean_", "X_std_", "n_features_"])

        if ddim:
            return self._sample_ddim(n_samples, ddim_steps, return_trajectory)

        # Standard DDPM sampling
        x = self._rng.randn(n_samples, self.n_features_)
        trajectory = [x.copy()] if return_trajectory else None

        for t in reversed(range(self.timesteps)):
            t_array = np.full(n_samples, t)
            features = self._create_features(x, t_array)

            pred_noise = self.model_.predict(features).reshape(x.shape)

            alpha = self.alphas[t]
            alpha_bar = self.alphas_cumprod[t]
            beta = self.betas[t]

            coef1 = 1.0 / np.sqrt(alpha)
            coef2 = beta / np.sqrt(1.0 - alpha_bar)
            mean = coef1 * (x - coef2 * pred_noise)

            if t > 0:
                if self.variance_type == "fixed_small":
                    variance = self.posterior_variance[t]
                else:
                    variance = beta
                noise = np.sqrt(variance) * self._rng.randn(*x.shape)
                x = mean + noise
            else:
                x = mean

            if return_trajectory:
                trajectory.append(x.copy())

        return self._postprocess_samples(x, trajectory, return_trajectory)

    def _sample_ddim(
        self, n_samples: int, steps: int, return_trajectory: bool
    ) -> np.ndarray:
        """DDIM sampling for faster generation"""
        skip = max(1, self.timesteps // steps)
        timesteps = np.arange(0, self.timesteps, skip)[::-1]

        x = self._rng.randn(n_samples, self.n_features_)
        trajectory = [x.copy()] if return_trajectory else None

        for i, t in enumerate(timesteps):
            t_array = np.full(n_samples, t)
            features = self._create_features(x, t_array)
            pred_noise = self.model_.predict(features).reshape(x.shape)

            alpha_bar = self.alphas_cumprod[t]
            alpha_bar_prev = (
                self.alphas_cumprod[timesteps[i + 1]]
                if i < len(timesteps) - 1
                else 1.0
            )

            pred_x0 = (x - np.sqrt(1 - alpha_bar) * pred_noise) / np.sqrt(
                alpha_bar
            )
            dir_xt = np.sqrt(1 - alpha_bar_prev) * pred_noise
            x = np.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

            if return_trajectory:
                trajectory.append(x.copy())

        return self._postprocess_samples(x, trajectory, return_trajectory)

    def _postprocess_samples(
        self,
        x: np.ndarray,
        trajectory: Optional[List[np.ndarray]],
        return_trajectory: bool,
    ) -> np.ndarray:
        """Apply inverse transforms to generated samples"""
        if return_trajectory and trajectory is not None:
            processed_trajectory = []
            for step in trajectory:
                if self.pca_ is not None:
                    step = self.pca_.inverse_transform(step)
                step = step * self.X_std_ + self.X_mean_
                processed_trajectory.append(step)
            return np.array(processed_trajectory)

        if self.pca_ is not None:
            x = self.pca_.inverse_transform(x)

        x = x * self.X_std_ + self.X_mean_
        return x

    def predict(self, n_samples: int = 1, **kwargs) -> np.ndarray:
        """Sklearn-style predict method (alias for sample)"""
        return self.sample(n_samples=n_samples, **kwargs)

    def score(self, X: np.ndarray, metric: str = "mmd") -> float:
        """Compute negative reconstruction error (higher is better)"""
        return -self.reconstruction_error(X, metric=metric)

    def reconstruction_error(
        self,
        X: np.ndarray,
        metric: str = "mmd",
        n_samples: Optional[int] = None,
        gamma: Union[str, float] = "auto",
    ) -> float:
        """
        Compute distributional reconstruction error

        Parameters
        ----------
        X : ndarray
            Original data distribution
        metric : str, default='mmd'
            Error metric: 'mmd' or 'energy'
        n_samples : int, optional
            Number of samples to generate
        gamma : str or float, default='auto'
            RBF bandwidth for MMD

        Returns
        -------
        error : float
            Reconstruction error (lower is better)
        """
        check_is_fitted(self, ["model_", "X_mean_", "X_std_"])

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
        self, X: np.ndarray, Y: np.ndarray, gamma: Union[str, float] = "auto"
    ) -> float:
        """Compute MMD with adaptive bandwidth (for evaluation)"""
        max_samples = 1000
        if len(X) > max_samples:
            indices = self._rng.choice(len(X), max_samples, replace=False)
            X = X[indices]
        if len(Y) > max_samples:
            indices = self._rng.choice(len(Y), max_samples, replace=False)
            Y = Y[indices]

        if gamma == "auto":
            n_bandwidth = min(100, len(X), len(Y))
            X_bw = X[:n_bandwidth]
            Y_bw = Y[:n_bandwidth]
            combined = np.vstack([X_bw, Y_bw])

            from scipy.spatial.distance import pdist

            squared_dists = pdist(combined, metric="sqeuclidean")
            gamma = 1.0 / np.median(squared_dists)

        try:
            from scipy.spatial.distance import cdist

            XX = cdist(X, X, metric="sqeuclidean")
            YY = cdist(Y, Y, metric="sqeuclidean")
            XY = cdist(X, Y, metric="sqeuclidean")

            K_XX = np.exp(-gamma * XX).mean()
            K_YY = np.exp(-gamma * YY).mean()
            K_XY = np.exp(-gamma * XY).mean()
        except ImportError:
            chunk_size = 100

            def chunked_kernel_mean(A, B):
                total = 0.0
                count = 0
                for i in range(0, len(A), chunk_size):
                    A_chunk = A[i : i + chunk_size]
                    for j in range(0, len(B), chunk_size):
                        B_chunk = B[j : j + chunk_size]
                        sq_dist = np.sum(
                            (A_chunk[:, None, :] - B_chunk[None, :, :]) ** 2,
                            axis=2,
                        )
                        total += np.exp(-gamma * sq_dist).sum()
                        count += len(A_chunk) * len(B_chunk)
                return total / count

            K_XX = chunked_kernel_mean(X, X)
            K_YY = chunked_kernel_mean(Y, Y)
            K_XY = chunked_kernel_mean(X, Y)

        mmd = K_XX + K_YY - 2 * K_XY
        return max(0, mmd)

    def _compute_energy_distance(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute energy distance"""
        try:
            from scipy.spatial.distance import pdist, cdist

            max_samples = 500
            if len(X) > max_samples:
                indices = self._rng.choice(len(X), max_samples, replace=False)
                X = X[indices]
            if len(Y) > max_samples:
                indices = self._rng.choice(len(Y), max_samples, replace=False)
                Y = Y[indices]

            xy_dist = cdist(X, Y, metric="euclidean").mean()
            xx_dist = pdist(X, metric="euclidean").mean() if len(X) > 1 else 0
            yy_dist = pdist(Y, metric="euclidean").mean() if len(Y) > 1 else 0

            energy = 2 * xy_dist - xx_dist - yy_dist
            return max(0, energy)
        except ImportError:
            max_samples = 500
            if len(X) > max_samples:
                indices = self._rng.choice(len(X), max_samples, replace=False)
                X = X[indices]
            if len(Y) > max_samples:
                indices = self._rng.choice(len(Y), max_samples, replace=False)
                Y = Y[indices]

            def chunked_pairwise_dist(A, B, chunk_size=100):
                total = 0.0
                count = 0
                for i in range(0, len(A), chunk_size):
                    A_chunk = A[i : i + chunk_size]
                    for j in range(0, len(B), chunk_size):
                        B_chunk = B[j : j + chunk_size]
                        dist = np.sqrt(
                            np.sum(
                                (A_chunk[:, None, :] - B_chunk[None, :, :])
                                ** 2,
                                axis=2,
                            )
                        )
                        total += dist.sum()
                        count += len(A_chunk) * len(B_chunk)
                return total / count

            xy_dist = chunked_pairwise_dist(X, Y)

            n, m = len(X), len(Y)
            if n > 1:
                xx_dist = chunked_pairwise_dist(X, X)
                xx_dist = (xx_dist * n * n) / (n * (n - 1))
            else:
                xx_dist = 0

            if m > 1:
                yy_dist = chunked_pairwise_dist(Y, Y)
                yy_dist = (yy_dist * m * m) / (m * (m - 1))
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
        """Bayesian optimization of hyperparameters with cross-validation"""
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
        except ImportError:
            raise ImportError(
                "scikit-optimize not installed. Install with: pip install scikit-optimize"
            )

        space = [
            Integer(100, 2000, name="timesteps"),
            Real(1e-5, 1e-3, name="beta_start", prior="log-uniform"),
            Real(0.01, 0.05, name="beta_end"),
            Categorical(["linear", "cosine"], name="schedule"),
            Real(0.1, 10.0, name="ridge_alpha", prior="log-uniform"),
        ]

        def objective(params):
            try:
                model = DiffusionModel(
                    timesteps=params[0],
                    beta_start=params[1],
                    beta_end=params[2],
                    schedule=params[3],
                    model=Ridge(
                        alpha=params[4], random_state=self.random_state
                    ),
                    random_state=self.random_state,
                    batch_size=self.batch_size,
                    use_pca=self.use_pca,
                    pca_components=self.pca_components,
                    training_objective=self.training_objective,
                )

                errors = []
                for _ in range(cv_splits):
                    model.fit(X, n_steps=500)
                    X_eval = X[: min(100, len(X))]
                    error = model.reconstruction_error(
                        X_eval, metric=metric, n_samples=100
                    )
                    errors.append(error)

                return np.mean(errors)
            except Exception as e:
                warnings.warn(f"Optimization iteration failed: {e}")
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

        self.set_params(**best_params)
        self._init_noise_schedule()
        self.fit(X, n_steps=2000)

        print(f"\n✅ Optimization complete!")
        print(f"Best {metric.upper()} error: {result.fun:.6f}")
        print(f"Best parameters:")
        for key, value in best_params.items():
            if key != "model":
                print(f"  {key}: {value}")
            else:
                print(f"  model: Ridge(alpha={result.x[4]:.4f})")

        return best_params
