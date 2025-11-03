import numpy as np
from scipy.stats import rankdata, norm, ks_2samp
import warnings


class DiversityGenerator:
    """
    Three-step Gaussian Copula transformation for controlled diversity generation
    while preserving marginal distributions.
    """

    def __init__(
        self, target_correlation=0.1, preserve_moments=True, random_state=None
    ):
        self.target_correlation = target_correlation
        self.preserve_moments = preserve_moments
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
        self.fitted_ = False  # Initialize fitted_ attribute

    def fit(self, X):
        """
        STEP 1: Learn ECDFs and create target correlation matrix
        """
        X = np.asarray(X)
        self.n_samples_, self.n_features_ = X.shape
        self.original_dtype_ = X.dtype

        # Store original statistics for moment preservation
        self.original_means_ = np.mean(X, axis=0)
        self.original_stds_ = np.std(X, axis=0)

        # Store ECDF information for inverse transformation
        self.sorted_columns_ = [
            np.sort(X[:, j]) for j in range(self.n_features_)
        ]
        self.quantile_positions_ = (np.arange(1, self.n_samples_ + 1)) / (
            self.n_samples_ + 1
        )

        # Create target correlation matrix
        self.target_corr_matrix_ = self._create_target_correlation_matrix()

        # Precompute Cholesky decomposition for correlation application
        try:
            self.cholesky_factor_ = np.linalg.cholesky(self.target_corr_matrix_)
        except np.linalg.LinAlgError:
            self.target_corr_matrix_ = self._nearest_positive_definite(
                self.target_corr_matrix_
            )
            self.cholesky_factor_ = np.linalg.cholesky(self.target_corr_matrix_)

        self.fitted_ = True
        return self

    def _create_target_correlation_matrix(self):
        """Create valid target correlation matrix"""
        if isinstance(self.target_correlation, (int, float)):
            corr_val = float(self.target_correlation)
            corr_val = np.clip(corr_val, -1.0 / (self.n_features_ - 1), 1.0)

            corr_matrix = np.full(
                (self.n_features_, self.n_features_), corr_val
            )
            np.fill_diagonal(corr_matrix, 1.0)

        elif isinstance(self.target_correlation, np.ndarray):
            corr_matrix = self.target_correlation.copy()
            np.fill_diagonal(corr_matrix, 1.0)
        else:
            raise ValueError("target_correlation must be scalar or matrix")

        return corr_matrix

    def _nearest_positive_definite(self, matrix):
        """Ensure matrix is positive definite"""
        n = matrix.shape[0]
        matrix = (matrix + matrix.T) / 2

        min_eigval = np.min(np.linalg.eigvals(matrix))
        if min_eigval > 0:
            return matrix

        identity = np.eye(n)
        for k in range(1, 1000):
            candidate = matrix + k * 1e-8 * identity
            if np.min(np.linalg.eigvals(candidate)) > 0:
                return candidate

        return np.eye(n)

    def transform_to_gaussian(self, X):
        """
        STEP 2: Transform X → ranks → uniform → Gaussian
        """
        X = np.asarray(X)
        n_new = X.shape[0]
        Y = np.zeros((n_new, self.n_features_), dtype=float)

        for j in range(self.n_features_):
            sorted_vals = self.sorted_columns_[j]

            # X → ranks → uniform
            empirical_cdf = np.searchsorted(
                sorted_vals, X[:, j], side="right"
            ) / (self.n_samples_ + 1)
            empirical_cdf = np.clip(empirical_cdf, 0.001, 0.999)

            # uniform → Gaussian (probit transform)
            Y[:, j] = norm.ppf(empirical_cdf)

        return Y

    def apply_target_correlation(self, Y):
        """
        STEP 2 (continued): Apply target correlation to Gaussian data
        """
        return Y @ self.cholesky_factor_.T

    def transform_from_gaussian(self, Y_transformed):
        """
        STEP 3: Transform Gaussian → uniform → inverse ECDF → X_diverse
        """
        n_new = Y_transformed.shape[0]
        Z = np.zeros((n_new, self.n_features_), dtype=float)

        for j in range(self.n_features_):
            sorted_vals = self.sorted_columns_[j]

            # Gaussian → uniform
            U_transformed = norm.cdf(Y_transformed[:, j])

            # uniform → inverse ECDF → X_diverse
            Z[:, j] = np.interp(
                U_transformed, self.quantile_positions_, sorted_vals
            )

            # Optional moment preservation
            if self.preserve_moments:
                Z[:, j] = self._preserve_moments(Z[:, j], j)

        return Z.astype(self.original_dtype_)

    def _preserve_moments(self, values, feature_idx):
        """Preserve mean and standard deviation if needed"""
        current_mean = np.mean(values)
        current_std = np.std(values)

        target_mean = self.original_means_[feature_idx]
        target_std = self.original_stds_[feature_idx]

        mean_ratio = abs(current_mean - target_mean) / (
            abs(target_mean) + 1e-10
        )
        std_ratio = abs(current_std - target_std) / (target_std + 1e-10)

        if mean_ratio > 0.02 or std_ratio > 0.05:
            values_centered = values - current_mean
            if current_std > 1e-10:
                values_scaled = values_centered * (target_std / current_std)
            else:
                values_scaled = values_centered
            return values_scaled + target_mean

        return values

    def generate_diverse_samples(self, X, n_samples=5):
        """Generate diverse samples using the three-step pipeline"""
        if not self.fitted_:
            self.fit(X)

        diverse_samples = []

        for i in range(n_samples):
            if i == 0:
                # First sample: transform original data
                Y_gaussian = self.transform_to_gaussian(X)
            else:
                # Additional samples: generate new Gaussian data
                Y_gaussian = np.random.normal(
                    0, 1, (self.n_samples_, self.n_features_)
                )

            # Apply target correlation
            Y_diverse = self.apply_target_correlation(Y_gaussian)

            # Transform back to original distributions
            X_diverse = self.transform_from_gaussian(Y_diverse)
            diverse_samples.append(X_diverse)

        return np.array(diverse_samples)

    def fit_transform(self, X, n_samples=5):
        """Fit and generate diverse samples in one call"""
        return self.generate_diverse_samples(X, n_samples)
