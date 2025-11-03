import nnetsauce as ns
import numpy as np
import optuna
import scipy.stats as stats
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
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


class TsDistroSimulator:
    def __init__(
        self,
        kernel="rbf",
        backend="numpy",
        kde_kernel="gaussian",
        random_state=None,
        residual_sampling="bootstrap",
        block_size=None,
        gmm_components=3,
    ):
        self.kernel = kernel
        self.backend = backend
        self.random_state = random_state
        self.residual_sampling = residual_sampling
        self.block_size = block_size
        self.gmm_components = gmm_components
        self.kde_kernel = kde_kernel
        self.Y_ = None
        self.n_samples_ = None

        if random_state is not None:
            np.random.seed(random_state)
            if JAX_AVAILABLE:
                key = jax.random.PRNGKey(random_state)

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

        if backend in ["gpu", "tpu"] and JAX_AVAILABLE:
            self._setup_jax_backend()
        elif backend in ["gpu", "tpu"] and not JAX_AVAILABLE:
            print("JAX not available. Falling back to NumPy backend.")
            self.backend = "numpy"

        self.model = None
        self.residuals_ = None
        self.X_dist = None
        self.is_fitted = False
        self.best_params_ = None
        self.best_score_ = None
        self.kde_model_ = None
        self.gmm_model_ = None

    def _setup_jax_backend(self):
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for GPU/TPU backend")

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

    def _create_model(self, gamma, alpha, lags=20, n_hidden_features=5):
        return ns.MTS(
            obj=KernelRidge(kernel=self.kernel, gamma=gamma, alpha=alpha),
            lags=lags,
            n_hidden_features=n_hidden_features,
        )

    def _fit_residual_sampler(self, **kwargs):
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

    def _sample_residuals(self, num_samples, random_state=123):
        if self.residuals_ is None:
            raise ValueError("No residuals available for sampling")

        n = len(self.residuals_)

        if self.residual_sampling == "bootstrap":
            np.random.seed(random_state)
            if num_samples <= n:
                idx = np.random.choice(n, num_samples, replace=True)
                return self.residuals_[idx]
            else:
                n_repeats = (num_samples // n) + 1
                tiled = np.tile(self.residuals_, (n_repeats, 1))
                idx = np.random.choice(len(tiled), num_samples, replace=False)
                return tiled[idx]

        elif self.residual_sampling == "kde":
            if self.kde_model_ is None:
                raise ValueError(
                    "KDE model not fitted. Call _fit_residual_sampler first."
                )

            samples = self.kde_model_.sample(
                num_samples, random_state=random_state
            )

            if samples.ndim == 1:
                samples = samples.reshape(-1, 1)
            return samples

        elif self.residual_sampling == "gmm":
            if self.gmm_model_ is None:
                raise ValueError(
                    "GMM model not fitted. Call _fit_residual_sampler first."
                )

            # Set random state before sampling
            np.random.seed(random_state)
            samples = self.gmm_model_.sample(num_samples)[0]

            if samples.ndim == 1:
                samples = samples.reshape(-1, 1)
            return samples

        elif self.residual_sampling == "me-bootstrap":
            meb = MaximumEntropyBootstrap(random_state=random_state)
            residuals = self.residuals_.flatten()
            if residuals.shape[0] < num_samples:
                repeats = int(np.ceil(num_samples / residuals.shape[0]))
                residuals = np.tile(residuals, repeats)[:num_samples]
            else:
                residuals = residuals[:num_samples]
            meb.fit(residuals)
            samples = meb.sample(1)[:, 0].reshape(-1, 1)
            # Ensure we have exactly num_samples
            if len(samples) < num_samples:
                n_repeats = (num_samples // len(samples)) + 1
                samples = np.tile(samples, (n_repeats, 1))[:num_samples]
            return samples

        elif self.residual_sampling == "block-bootstrap":
            samples = bootstrap(
                self.residuals_,
                num_samples,
                block_size=self.block_size,
                seed=random_state,
            )
            # Ensure correct shape
            if samples.ndim == 1:
                samples = samples.reshape(-1, 1)
            return samples

        else:
            raise ValueError(
                f"Unknown sampling method: {self.residual_sampling}"
            )

    def _pairwise_sq_dists(self, X1, X2):
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

    def _mmd(self, u, v, kernel_sigma=1):
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
        if u.ndim == 1:
            u = u.reshape(-1, 1)
        if v.ndim == 1:
            v = v.reshape(-1, 1)

        n, d = u.shape
        m = v.shape[0]

        if self.backend in ["gpu", "tpu"] and JAX_AVAILABLE:
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
            dist_xx = cdist(u, u, metric="euclidean")
            dist_yy = cdist(v, v, metric="euclidean")
            dist_xy = cdist(u, v, metric="euclidean")
            term1 = 2 * np.sum(dist_xy) / (n * m)
            term2 = np.sum(dist_xx) / (n * n)
            term3 = np.sum(dist_yy) / (m * m)
            return term1 - term2 - term3

    def _generate_pseudo_single(self, random_state=123):
        """
        Generate a single synthetic realization.

        Returns original data (structure) + resampled residuals (noise)
        Each call produces a different realization due to different residual samples.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        n_rows = self.n_samples_

        # Base: original time series structure
        base = self.Y_.copy()
        if base.ndim == 1:
            base = base.reshape(-1, 1)

        # Noise: sample new residuals from learned distribution
        residuals = self._sample_residuals(n_rows, random_state)

        # Ensure residuals match the size of base
        # This is needed because model residuals may be shorter due to lags
        if residuals.shape[0] < n_rows:
            n_repeats = (n_rows // residuals.shape[0]) + 1
            residuals = np.tile(residuals, (n_repeats, 1))[:n_rows]
        elif residuals.shape[0] > n_rows:
            residuals = residuals[:n_rows]

        # Return: structure + new noise realization
        return base + residuals

    def fit(self, Y, metric="energy", n_trials=50, **kwargs):
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n, d = Y.shape
        self.n_features_ = d
        self.n_samples_ = n
        self.Y_ = Y  # Store once before optimization

        self.X_dist = np.random.normal(0, 1, (n, d))

        def objective(trial):
            sigma = trial.suggest_float("sigma", 0.01, 10, log=True)
            lambd = trial.suggest_float("lambd", 1e-5, 1, log=True)
            lags = trial.suggest_int("lags", 1, 50)
            n_hidden_features = trial.suggest_int("n_hidden_features", 1, 20)
            gamma = 1 / (2 * sigma**2)

            model = self._create_model(gamma, lambd, lags, n_hidden_features)
            model.fit(Y)

            # Generate synthetic sample using this model's residuals
            Y_sim = self._generate_pseudo_with_model(
                model, model.residuals_, n, random_state=trial.number
            )

            if metric == "energy":
                dist_val = self._custom_energy_distance(Y, Y_sim)
            elif metric == "mmd":
                dist_val = self._mmd(Y, Y_sim)
            elif metric == "wasserstein" and d == 1:
                dist_val = stats.wasserstein_distance(
                    Y.flatten(), Y_sim.flatten()
                )
            else:
                raise ValueError("Invalid metric for dimension")

            return dist_val

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, **kwargs)

        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        sigma = self.best_params_["sigma"]
        lambd = self.best_params_["lambd"]
        lags = self.best_params_["lags"]
        n_hidden_features = self.best_params_["n_hidden_features"]
        gamma = 1 / (2 * sigma**2)

        self.model = self._create_model(gamma, lambd, lags, n_hidden_features)
        self.model.fit(Y)

        self.residuals_ = self.model.residuals_

        self._fit_residual_sampler()
        self.is_fitted = True

        print(f"  Best energy distance: {self.best_score_:.6f}")
        print(f"  Best lags: {lags}, n_hidden_features: {n_hidden_features}")

        return self

    def _generate_pseudo_with_model(
        self, model, residuals, num_samples, random_state=None
    ):
        """Helper function for optimization - temporarily uses different residuals"""
        # Temporarily store and swap residual models
        original_residuals = self.residuals_
        original_kde = self.kde_model_
        original_gmm = self.gmm_model_

        self.residuals_ = residuals

        # Only fit if using kde or gmm
        if self.residual_sampling in ["kde", "gmm"]:
            self._fit_residual_sampler()

        # Length of actual residuals from the model
        residual_len = len(residuals)

        # Use provided random state or generate one
        if random_state is None:
            random_state = np.random.randint(0, 10000)

        # Sample residuals matching the residual length
        sampled_residuals = self._sample_residuals(
            residual_len, random_state=random_state
        )

        # Restore original state
        self.residuals_ = original_residuals
        self.kde_model_ = original_kde
        self.gmm_model_ = original_gmm

        # The model with lags produces residuals shorter than original data
        # We need to align: use only the portion of Y that corresponds to residuals
        # Typically, if lags=L, residuals start from index L
        y_slice = self.Y_[-residual_len:]  # Take the last residual_len points

        # Return: aligned data + resampled residuals
        return y_slice + sampled_residuals

    def sample(self, n_samples=1):
        """
        Generate synthetic samples via distribution matching.

        Each sample is: original structure + resampled residuals

        Parameters:
        -----------
        n_samples : int, default=1
            Number of synthetic realizations to generate

        Returns:
        --------
        samples : ndarray
            - If Y was univariate (n_rows, 1): returns shape (n_rows, n_samples)
            - If Y was multivariate (n_rows, n_features): returns shape (n_features, n_rows, n_samples)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Generate n_samples realizations, each with shape (n_rows, n_features)
        samples_list = []
        for i, _ in enumerate(range(n_samples)):
            sample = self._generate_pseudo_single(
                random_state=1000 + i
            )  # Shape: (n_rows, n_features)
            samples_list.append(sample)

        # Stack to get shape (n_samples, n_rows, n_features)
        stacked = np.stack(samples_list, axis=0)

        # If univariate (n_features == 1), return (n_rows, n_samples)
        if self.n_features_ == 1:
            result = stacked.squeeze(
                axis=2
            ).T  # (n_samples, n_rows) -> (n_rows, n_samples)
        else:
            # If multivariate, return (n_features, n_rows, n_samples)
            result = stacked.transpose(
                2, 1, 0
            )  # (n_samples, n_rows, n_features) -> (n_features, n_rows, n_samples)

        return result

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
            ks_stat, ks_pvalue = stats.ks_2samp(Y_orig[:, i], Y_sim[:, i])
            ks_results.append((ks_stat, ks_pvalue))

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
            all_values = np.sort(np.concatenate([sorted_orig, sorted_sim]))
            ecdf_orig_all = np.searchsorted(
                sorted_orig, all_values, side="right"
            ) / len(sorted_orig)
            ecdf_sim_all = np.searchsorted(
                sorted_sim, all_values, side="right"
            ) / len(sorted_sim)
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

        # Q-Q plots for each dimension
        fig, axes = plt.subplots(1, d, figsize=(5 * d, 5))
        if d == 1:
            axes = [axes]

        for i in range(d):
            orig_sorted = np.sort(Y_orig[:, i])
            sim_sorted = np.sort(Y_sim[:, i])

            n_orig = len(orig_sorted)
            n_sim = len(sim_sorted)

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


# # ============================================================================
# # EXAMPLE USAGE
# # ============================================================================

# if __name__ == "__main__":
#     print("=" * 70)
#     print("TsDistroSimulator Example: Distribution Matching for Time Series")
#     print("=" * 70)

#     # Generate synthetic time series data
#     np.random.seed(42)
#     n_points = 200
#     t = np.linspace(0, 4*np.pi, n_points)

#     # Example 1: Univariate time series
#     print("\n" + "="*70)
#     print("Example 1: Univariate Time Series (sine wave + noise)")
#     print("="*70)

#     y_univariate = np.sin(t) + 0.3 * np.random.randn(n_points)
#     y_univariate = y_univariate.reshape(-1, 1)

#     print(f"\nOriginal data shape: {y_univariate.shape}")

#     # Fit the simulator
#     sim = TsDistroSimulator(
#         kernel="rbf",
#         residual_sampling="kde",  # Use KDE for residual distribution
#         random_state=42
#     )

#     print("\nFitting model...")
#     sim.fit(y_univariate, metric="energy", n_trials=10, show_progress_bar=False)

#     # Generate synthetic samples
#     print("\nGenerating 5 synthetic realizations...")
#     synthetic_samples = sim.sample(n_samples=5)
#     print(f"Synthetic samples shape: {synthetic_samples.shape}")
#     print(f"Expected shape: ({n_points}, 5) ✓" if synthetic_samples.shape == (n_points, 5) else "Shape mismatch!")

#     # Visualize
#     fig, axes = plt.subplots(2, 1, figsize=(12, 8))

#     # Plot original and synthetic samples
#     axes[0].plot(t, y_univariate, 'b-', linewidth=2, label='Original', alpha=0.8)
#     for i in range(5):
#         axes[0].plot(t, synthetic_samples[:, i], alpha=0.4, label=f'Sample {i+1}')
#     axes[0].set_title('Original vs Synthetic Realizations (Univariate)')
#     axes[0].set_xlabel('Time')
#     axes[0].set_ylabel('Value')
#     axes[0].legend()
#     axes[0].grid(True, alpha=0.3)

#     # Distribution comparison
#     axes[1].hist(y_univariate.flatten(), bins=30, alpha=0.5, density=True,
#                  label='Original', color='blue')
#     axes[1].hist(synthetic_samples.flatten(), bins=30, alpha=0.5, density=True,
#                  label='Synthetic (all samples)', color='red')
#     axes[1].set_title('Distribution Comparison')
#     axes[1].set_xlabel('Value')
#     axes[1].set_ylabel('Density')
#     axes[1].legend()
#     axes[1].grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig('univariate_example.png', dpi=150, bbox_inches='tight')
#     plt.show()

#     # Example 2: Multivariate time series
#     print("\n" + "="*70)
#     print("Example 2: Multivariate Time Series (2 features)")
#     print("="*70)

#     y_multivariate = np.column_stack([
#         np.sin(t) + 0.2 * np.random.randn(n_points),      # Feature 1
#         np.cos(t) + 0.2 * np.random.randn(n_points)       # Feature 2
#     ])

#     print(f"\nOriginal data shape: {y_multivariate.shape}")

#     # Fit the simulator
#     sim_multi = TsDistroSimulator(
#         kernel="rbf",
#         residual_sampling="bootstrap",  # Use bootstrap for residuals
#         random_state=42
#     )

#     print("\nFitting model...")
#     sim_multi.fit(y_multivariate, metric="energy", n_trials=10, show_progress_bar=False)

#     # Generate synthetic samples
#     print("\nGenerating 3 synthetic realizations...")
#     synthetic_multi = sim_multi.sample(n_samples=3)
#     print(f"Synthetic samples shape: {synthetic_multi.shape}")
#     print(f"Expected shape: (2, {n_points}, 3) ✓" if synthetic_multi.shape == (2, n_points, 3) else "Shape mismatch!")

#     # Visualize multivariate
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))

#     # Feature 1
#     axes[0, 0].plot(t, y_multivariate[:, 0], 'b-', linewidth=2, label='Original', alpha=0.8)
#     for i in range(3):
#         axes[0, 0].plot(t, synthetic_multi[0, :, i], alpha=0.5, label=f'Sample {i+1}')
#     axes[0, 0].set_title('Feature 1: Original vs Synthetic')
#     axes[0, 0].set_xlabel('Time')
#     axes[0, 0].set_ylabel('Value')
#     axes[0, 0].legend()
#     axes[0, 0].grid(True, alpha=0.3)

#     # Feature 2
#     axes[0, 1].plot(t, y_multivariate[:, 1], 'b-', linewidth=2, label='Original', alpha=0.8)
#     for i in range(3):
#         axes[0, 1].plot(t, synthetic_multi[1, :, i], alpha=0.5, label=f'Sample {i+1}')
#     axes[0, 1].set_title('Feature 2: Original vs Synthetic')
#     axes[0, 1].set_xlabel('Time')
#     axes[0, 1].set_ylabel('Value')
#     axes[0, 1].legend()
#     axes[0, 1].grid(True, alpha=0.3)

#     # Distribution comparison - Feature 1
#     axes[1, 0].hist(y_multivariate[:, 0], bins=30, alpha=0.5, density=True,
#                     label='Original', color='blue')
#     axes[1, 0].hist(synthetic_multi[0, :, :].flatten(), bins=30, alpha=0.5,
#                     density=True, label='Synthetic', color='red')
#     axes[1, 0].set_title('Feature 1 Distribution')
#     axes[1, 0].set_xlabel('Value')
#     axes[1, 0].set_ylabel('Density')
#     axes[1, 0].legend()
#     axes[1, 0].grid(True, alpha=0.3)

#     # Distribution comparison - Feature 2
#     axes[1, 1].hist(y_multivariate[:, 1], bins=30, alpha=0.5, density=True,
#                     label='Original', color='blue')
#     axes[1, 1].hist(synthetic_multi[1, :, :].flatten(), bins=30, alpha=0.5,
#                     density=True, label='Synthetic', color='red')
#     axes[1, 1].set_title('Feature 2 Distribution')
#     axes[1, 1].set_xlabel('Value')
#     axes[1, 1].set_ylabel('Density')
#     axes[1, 1].legend()
#     axes[1, 1].grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig('multivariate_example.png', dpi=150, bbox_inches='tight')
#     plt.show()

#     # Example 3: Using compare_distributions method
#     print("\n" + "="*70)
#     print("Example 3: Detailed Distribution Comparison (Univariate)")
#     print("="*70)

#     # Generate more samples for better distribution comparison
#     synthetic_for_comparison = sim.sample(n_samples=1)  # Single realization

#     print("\nPerforming statistical tests...")
#     results = sim.compare_distributions(
#         y_univariate,
#         synthetic_for_comparison,
#         save_prefix="univariate"
#     )

#     # Example 4: Different residual sampling methods
#     print("\n" + "="*70)
#     print("Example 4: Comparing Different Residual Sampling Methods")
#     print("="*70)

#     methods = ["bootstrap", "kde", "gmm"]
#     fig, axes = plt.subplots(len(methods), 2, figsize=(14, 4*len(methods)))

#     for idx, method in enumerate(methods):
#         print(f"\nTesting method: {method}")

#         sim_method = TsDistroSimulator(
#             kernel="rbf",
#             residual_sampling=method,
#             random_state=42
#         )

#         sim_method.fit(y_univariate, metric="energy", n_trials=5,
#                       show_progress_bar=False)
#         synthetic_method = sim_method.sample(n_samples=3)

#         # Plot time series
#         axes[idx, 0].plot(t, y_univariate, 'b-', linewidth=2,
#                          label='Original', alpha=0.8)
#         for i in range(3):
#             axes[idx, 0].plot(t, synthetic_method[:, i], alpha=0.4)
#         axes[idx, 0].set_title(f'Method: {method.upper()} - Time Series')
#         axes[idx, 0].set_xlabel('Time')
#         axes[idx, 0].set_ylabel('Value')
#         axes[idx, 0].legend()
#         axes[idx, 0].grid(True, alpha=0.3)

#         # Plot distributions
#         axes[idx, 1].hist(y_univariate.flatten(), bins=30, alpha=0.5,
#                          density=True, label='Original', color='blue')
#         axes[idx, 1].hist(synthetic_method.flatten(), bins=30, alpha=0.5,
#                          density=True, label='Synthetic', color='red')
#         axes[idx, 1].set_title(f'Method: {method.upper()} - Distribution')
#         axes[idx, 1].set_xlabel('Value')
#         axes[idx, 1].set_ylabel('Density')
#         axes[idx, 1].legend()
#         axes[idx, 1].grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig('method_comparison.png', dpi=150, bbox_inches='tight')
#     plt.show()

#     # Summary statistics
#     print("\n" + "="*70)
#     print("Summary Statistics")
#     print("="*70)

#     print("\nUnivariate Example:")
#     print(f"  Original - Mean: {y_univariate.mean():.4f}, Std: {y_univariate.std():.4f}")
#     print(f"  Synthetic - Mean: {synthetic_samples.mean():.4f}, Std: {synthetic_samples.std():.4f}")

#     print("\nMultivariate Example:")
#     print(f"  Original Feature 1 - Mean: {y_multivariate[:, 0].mean():.4f}, Std: {y_multivariate[:, 0].std():.4f}")
#     print(f"  Synthetic Feature 1 - Mean: {synthetic_multi[0, :, :].mean():.4f}, Std: {synthetic_multi[0, :, :].std():.4f}")
#     print(f"  Original Feature 2 - Mean: {y_multivariate[:, 1].mean():.4f}, Std: {y_multivariate[:, 1].std():.4f}")
#     print(f"  Synthetic Feature 2 - Mean: {synthetic_multi[1, :, :].mean():.4f}, Std: {synthetic_multi[1, :, :].std():.4f}")

#     print("\n" + "="*70)
#     print("Examples Complete!")
#     print("="*70)
#     print("\nKey Takeaways:")
#     print("1. Univariate: sample(n) returns shape (n_rows, n)")
#     print("2. Multivariate: sample(n) returns shape (n_features, n_rows, n)")
#     print("3. Each sample is: original structure + resampled residuals")
#     print("4. Different residual sampling methods affect noise characteristics")
#     print("5. Distribution matching preserves statistical properties")
#     print("="*70)
