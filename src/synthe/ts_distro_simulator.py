import nnetsauce as ns 
import numpy as np
import optuna
import scipy.stats as stats
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.kernel_ridge import KernelRidge
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    def __init__(self, kernel="rbf", backend="numpy", kde_kernel="gaussian",
                 random_state=None, residual_sampling="bootstrap", block_size=None,
                 gmm_components=3, use_rff="auto", rff_components="auto",
                 rff_gamma=None, kernel_approximation="rff", force_rff_threshold=1000):
        self.kernel = kernel
        self.backend = backend
        self.random_state = random_state
        self.residual_sampling = residual_sampling
        self.block_size = block_size
        self.gmm_components = gmm_components
        self.use_rff = use_rff
        self.rff_components = rff_components
        self.rff_gamma = rff_gamma
        self.kernel_approximation = kernel_approximation
        self.force_rff_threshold = force_rff_threshold
        self.kde_kernel = kde_kernel
        self.Y_ = None 

        if random_state is not None:
            np.random.seed(random_state)
            if JAX_AVAILABLE:
                key = jax.random.PRNGKey(random_state)
        
        valid_sampling_methods = ["bootstrap", "kde", "gmm", "block-bootstrap", "me-bootstrap"]
        if residual_sampling not in valid_sampling_methods:
            raise ValueError(f"residual_sampling must be one of {valid_sampling_methods}")
        
        valid_approximations = ["rff", "nystroem"]
        if kernel_approximation not in valid_approximations:
            raise ValueError(f"kernel_approximation must be one of {valid_approximations}")
        
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
        self.actual_rff_components_ = None
        self.actual_use_rff_ = None

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
            return vmap(lambda x: vmap(lambda y: jnp.sqrt(jnp.sum((x - y) ** 2)))(X2))(X1)

        self._pairwise_sq_dists_jax = pairwise_sq_dists_jax
        self._cdist_jax = cdist_jax

    def _determine_components(self, n_samples):
        if self.rff_components == "auto":
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

    def _create_model(self, gamma, alpha, lags=20, n_hidden_features=5, use_rff=None):
        if use_rff is None:
            use_rff = self.actual_use_rff_

        if use_rff:
            effective_gamma = self.rff_gamma if self.rff_gamma is not None else gamma
            n_components = self.actual_rff_components_

            if self.kernel_approximation == "rff":
                approximator = RBFSampler(gamma=effective_gamma, n_components=n_components,
                                         random_state=self.random_state)
            else:
                approximator = Nystroem(kernel="rbf", gamma=effective_gamma,
                                       n_components=n_components, random_state=self.random_state)
            
            return Pipeline([("scaler", StandardScaler()), ("approx", approximator),
                           ("ridge", Ridge(alpha=alpha))])
        
        return ns.MTS(obj=KernelRidge(kernel=self.kernel, gamma=gamma, alpha=alpha), 
                      lags=lags, n_hidden_features=n_hidden_features)

    def _fit_residual_sampler(self, **kwargs):
        if self.residuals_ is None or len(self.residuals_) == 0:
            raise ValueError("No residuals available for fitting sampler")

        if self.residual_sampling == "kde":
            kernel_bandwidths = {"bandwidth": np.logspace(-6, 6, 150)}
            grid = GridSearchCV(KernelDensity(kernel=self.kde_kernel, **kwargs),
                               param_grid=kernel_bandwidths)
            grid.fit(self.residuals_)
            self.kde_model_ = grid.best_estimator_
            self.kde_model_.fit(self.residuals_)

        elif self.residual_sampling == "gmm":
            self.gmm_model_ = GaussianMixture(
                n_components=min(self.gmm_components, len(self.residuals_)),
                random_state=self.random_state, covariance_type="full")
            self.gmm_model_.fit(self.residuals_)

    def _sample_residuals(self, n, num_samples):
        if self.residuals_ is None:
            raise ValueError("No residuals available for sampling")

        if self.residual_sampling == "bootstrap":
            idx = np.random.choice(n, num_samples, replace=True)
            return self.residuals_[idx]

        elif self.residual_sampling == "kde":
            if self.kde_model_ is None:
                raise ValueError("KDE model not fitted. Call _fit_residual_sampler first.")
            return self.kde_model_.sample(num_samples)

        elif self.residual_sampling == "gmm":
            if self.gmm_model_ is None:
                raise ValueError("GMM model not fitted. Call _fit_residual_sampler first.")
            return self.gmm_model_.sample(num_samples)[0]
        
        elif self.residual_sampling == "me-bootstrap":
            meb = MaximumEntropyBootstrap(random_state=self.random_state)
            residuals = self.residuals_.flatten()
            if residuals.shape[0] < num_samples:
                repeats = int(np.ceil(num_samples / residuals.shape[0]))
                residuals = np.tile(residuals, repeats)[:num_samples]
            else:
                residuals = residuals[:num_samples]
            meb.fit(residuals)
            return meb.sample(1)[:, 0].reshape(-1, 1)
                
        elif self.residual_sampling == "block-bootstrap":
            return bootstrap(self.residuals_, num_samples, block_size=self.block_size)

        else:
            raise ValueError(f"Unknown sampling method: {self.residual_sampling}")

    def _pairwise_sq_dists(self, X1, X2):
        if self.backend in ["gpu", "tpu"] and JAX_AVAILABLE:
            X1_jax = jnp.array(X1)
            X2_jax = jnp.array(X2)
            result = self._pairwise_sq_dists_jax(X1_jax, X2_jax)
            return np.array(result)
        else:
            X1 = np.atleast_2d(X1)
            X2 = np.atleast_2d(X2)
            return (np.sum(X1**2, axis=1)[:, np.newaxis] + 
                   np.sum(X2**2, axis=1)[np.newaxis, :] - 2 * X1 @ X2.T)

    def _mmd(self, u, v, kernel_sigma=1):
        if u.ndim == 1:
            u = u.reshape(-1, 1)
        if v.ndim == 1:
            v = v.reshape(-1, 1)

        def kmat(A, B):
            return np.exp(-self._pairwise_sq_dists(A, B) / (2 * kernel_sigma**2))

        return np.mean(kmat(u, u)) + np.mean(kmat(v, v)) - 2 * np.mean(kmat(u, v))

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

    def _generate_pseudo(self, num_samples):
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.actual_use_rff_:
            X_new = self.X_dist[:num_samples]
            preds = self.model.predict(X_new)
        else:
            # Instead of predicting, use self.Y_[:num_samples] directly for time series
            preds = self.Y_[:num_samples]

        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        # Pad or trim preds if needed
        if preds.shape[0] < num_samples:
            pad_len = num_samples - preds.shape[0]
            preds = np.vstack([preds, np.zeros((pad_len, preds.shape[1]))])
        elif preds.shape[0] > num_samples:
            preds = preds[:num_samples]

        residuals = self._sample_residuals(preds.shape[0], num_samples)
        # Pad or trim residuals if needed
        if residuals.shape[0] < num_samples:
            pad_len = num_samples - residuals.shape[0]
            residuals = np.vstack([residuals, np.zeros((pad_len, residuals.shape[1]))])
        elif residuals.shape[0] > num_samples:
            residuals = residuals[:num_samples]

        return preds + residuals

    def fit(self, Y, metric="energy", n_trials=50, **kwargs):
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n, d = Y.shape
        self.n_features_ = d
        
        if self.use_rff == "auto":
            self.actual_use_rff_ = n >= self.force_rff_threshold
        else:
            self.actual_use_rff_ = self.use_rff
        
        if self.actual_use_rff_:
            self.actual_rff_components_ = self._determine_components(n)
            if self.use_rff == "auto":
                print(f"Large dataset detected (n={n}). Auto-enabling "
                      f"{self.kernel_approximation.upper()} for scalability.")

        self.X_dist = np.random.normal(0, 1, (n, d))

        def objective(trial):
            sigma = trial.suggest_float("sigma", 0.01, 10, log=True)
            lambd = trial.suggest_float("lambd", 1e-5, 1, log=True)
            lags = trial.suggest_int("lags", 1, 50)
            n_hidden_features = trial.suggest_int("n_hidden_features", 1, 20)
            gamma = 1 / (2 * sigma**2)
            
            model = self._create_model(gamma, lambd, lags, n_hidden_features)
            model.fit(Y)
            self.Y_ = Y 
            
            # ns.MTS expects an integer for prediction horizon            
            Y_sim = self._generate_pseudo_with_model(model, model.residuals_ , n)
            
            if metric == "energy":
                dist_val = self._custom_energy_distance(Y, Y_sim)
            elif metric == "mmd":
                dist_val = self._mmd(Y, Y_sim)
            elif metric == "wasserstein" and d == 1:
                dist_val = stats.wasserstein_distance(Y.flatten(), Y_sim.flatten())
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
        
        # ns.MTS expects an integer for prediction horizon
        self.residuals_ = self.model.residuals_
        
        self._fit_residual_sampler()
        self.is_fitted = True
        
        if self.actual_use_rff_:
            print(f"  Using {self.kernel_approximation.upper()} with "
                  f"{self.actual_rff_components_} components")
        else:
            print(f"  Using standard kernel method")
        
        print(f"  Best energy distance: {self.best_score_:.6f}")
        print(f"  Best lags: {lags}, n_hidden_features: {n_hidden_features}")
        
        return self

    def _generate_pseudo_with_model(self, model, residuals, num_samples):
        # Check if model is RFF/Nystroem Pipeline by checking for 'approx' step
        # ns.MTS is also a Pipeline internally, but doesn't have 'approx' step
       
        # Temporarily store and swap residual models
        original_residuals = model.residuals_
        original_kde = self.kde_model_
        original_gmm = self.gmm_model_

        self.residuals_ = residuals
        
        # Only fit if using kde or gmm
        if self.residual_sampling in ["kde", "gmm"]:
            self._fit_residual_sampler()
        
        sampled_residuals = self._sample_residuals(num_samples, num_samples)

        # Restore original state
        self.residuals_ = original_residuals
        self.kde_model_ = original_kde
        self.gmm_model_ = original_gmm

        # Ensure shapes match for addition
        y_slice = self.Y_[:num_samples]
        res_slice = sampled_residuals
        # Pad or trim residuals if needed
        if res_slice.shape[0] < num_samples:
            pad_len = num_samples - res_slice.shape[0]
            res_slice = np.vstack([res_slice, np.zeros((pad_len, res_slice.shape[1]))])
        elif res_slice.shape[0] > num_samples:
            res_slice = res_slice[:num_samples]
        # Remove debug prints
        return y_slice + res_slice

    def sample(self, n_samples=1):
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._generate_pseudo(n_samples)

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
