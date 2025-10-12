import numpy as np
import matplotlib.pyplot as plt
import warnings

from typing import Optional, Union, Dict, Tuple
from scipy import stats
from scipy.interpolate import interp1d, griddata
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture


class EmpiricalCopula:
    """
    Empirical Copula implementation for multivariate dependence modeling.

    This class implements a non-parametric copula based on the empirical distribution
    of the data. It can fit to multivariate data and generate samples that preserve
    the original dependence structure.

    The empirical copula is defined as:
    C_n(u1, ..., ud) = (1/n) * sum(I(U1i <= u1, ..., Udi <= ud))

    where U_ji are the pseudo-observations (ranks) of the original data.
    """

    def __init__(
        self,
        smoothing_method: str = "none",
        jitter_scale: float = 0.01,
        boundary_correction: bool = True,
    ):
        """
        Initialize the Empirical Copula.

        Parameters:
        -----------
        smoothing_method : str, default "none"
            Smoothing method for the empirical copula:
            - "none": Pure empirical copula (no smoothing)
            - "jitter": Add small random noise to avoid ties
        jitter_scale : float, default 0.0
            Scale of uniform jitter to add to pseudo-observations (0 = no jitter).
        boundary_correction : bool, default True
            Whether to apply boundary correction for kernel methods.
        """
        self.smoothing_method = smoothing_method
        self.jitter_scale = jitter_scale
        self.boundary_correction = boundary_correction
        # Fitted attributes
        self.is_fitted_ = False
        self.n_samples_ = None
        self.n_vars_ = None
        self.pseudo_observations_ = None
        self.original_data_ = None
        self.marginal_cdfs_ = []
        self.marginal_quantiles_ = []
        # For kernel-based methods
        self.kde_model_ = None
        # For Gaussian mixture model
        self.gmm_model_ = None

    def fit(self, X: np.ndarray) -> "EmpiricalCopula":
        """
        Fit the empirical copula to the data.

        Parameters:
        -----------
        X : np.ndarray
            Input data of shape (n_samples, n_features) on original scale.

        Returns:
        --------
        self : EmpiricalCopula
            Returns self for method chaining.

        Raises:
        -------
        ValueError
            If X has inappropriate dimensions.
        """
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] < 2:
            raise ValueError("X must have at least 2 variables")
        if X.shape[0] < 2:
            raise ValueError("X must have at least 2 observations")

        self.n_samples_, self.n_vars_ = X.shape
        self.original_data_ = X.copy()
        # Step 1: Convert to pseudo-observations (ranks)
        self.pseudo_observations_ = self._to_pseudo_observations(X)
        # Step 2: Apply smoothing if requested
        if self.smoothing_method != "none":
            self.pseudo_observations_ = self._apply_smoothing(
                self.pseudo_observations_
            )
        # Step 3: Store marginal information for inverse transformation
        self._fit_marginal_transforms(X)
        self.is_fitted_ = True
        # print(f"Empirical copula fitted successfully using '{self.smoothing_method}' method")
        return self

    def sample(
        self,
        n_samples: int = 50,
        method: str = "bootstrap",
        kernel: str = "gaussian",
        n_components: int = 5,
        covariance_type: str = "full",
        return_pseudo: bool = False,
        random_state: Optional[int] = 123,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate samples from the fitted empirical copula.

        Parameters:
        -----------
        n_samples : int, default 50
            Number of samples to generate.
        method : str, default "bootstrap"
            Sampling method:
            - "bootstrap": Bootstrap resampling from fitted pseudo-observations
            - "kde": Kernel density estimation sampling (if smoothing was used)
            - "gmm": Gaussian mixture model sampling
        kernel : str, default "gaussian"
            Kernel to use if method is "kde" (default is 'gaussian').
            Can also be 'tophat'.
        n_components : int, default 5
            Number of Gaussian components for GMM method.
        covariance_type : str, default "full"
            Type of covariance parameters for GMM method.
            Options: 'full', 'tied', 'diag', 'spherical'.
        return_pseudo : bool, default False
            If True, return samples on [0,1] copula scale.
            If False, return samples transformed to original scale.
        random_state : int, optional
            Random state for reproducible sampling.
        kwargs : additional arguments for specific sampling methods.

        Returns:
        --------
        samples : np.ndarray
            Generated samples of shape (n_samples, n_features).
        """
        if not self.is_fitted_:
            raise ValueError(
                "Copula must be fitted before sampling. Call fit() first."
            )

        if random_state is not None:
            np.random.seed(random_state)
        # Generate pseudo-observations
        if method == "bootstrap":
            pseudo_samples = self._bootstrap_sample(n_samples)
        elif method == "kde":
            pseudo_samples = self._kde_sample(
                n_samples, kernel=kernel, **kwargs
            )
        elif method == "gmm":
            pseudo_samples = self._gmm_sample(
                n_samples,
                n_components=n_components,
                covariance_type=covariance_type,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown sampling method: {method}. "
                f"Supported methods: 'bootstrap', 'kde', 'gmm'"
            )
        if return_pseudo:
            return pseudo_samples
        # Transform back to original scale
        return self._inverse_transform(pseudo_samples)

    def plot_pairwise_pseudo(self):
        if not self.is_fitted_:
            raise ValueError("Copula must be fitted before plotting.")
        plt.figure(figsize=(15, 15))
        for i in range(self.n_vars_):
            for j in range(i + 1, self.n_vars_):
                plt.subplot(
                    self.n_vars_ - 1,
                    self.n_vars_ - 1,
                    i * (self.n_vars_ - 1) + j - i,
                )
                plt.scatter(
                    self.pseudo_observations_[:, i],
                    self.pseudo_observations_[:, j],
                    s=5,
                    alpha=0.5,
                )
                plt.xlabel(f"Variable {i+1}")
                plt.ylabel(f"Variable {j+1}")
                plt.title(
                    f"Var{i+1}-Var{j+1} (ρ={self._calculate_spearman_matrix(self.original_data_)[i,j]:.2f})"
                )
        plt.tight_layout()
        plt.show()

    def estimate_tail_dependence(self, threshold=0.05):
        tail_dep = {}
        for i in range(self.n_vars_):
            for j in range(i + 1, self.n_vars_):
                u = self.pseudo_observations_[:, i]
                v = self.pseudo_observations_[:, j]
                lower_tail = (
                    np.mean((u < threshold) & (v < threshold)) / threshold
                )
                upper_tail = (
                    np.mean((u > 1 - threshold) & (v > 1 - threshold))
                    / threshold
                )
                tail_dep[f"var{i+1}-var{j+1}"] = {
                    "lower": lower_tail,
                    "upper": upper_tail,
                }
        return tail_dep

    def plot_marginals(self, simulated_samples):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 5))
        for j in range(self.n_vars_):
            plt.subplot(2, self.n_vars_ // 2, j + 1)
            plt.hist(
                self.original_data_[:, j],
                bins=30,
                alpha=0.5,
                label="Original",
                density=True,
            )
            plt.hist(
                simulated_samples[:, j],
                bins=30,
                alpha=0.5,
                label="Simulated",
                density=True,
            )
            plt.title(f"Variable {j+1}")
            plt.legend()
        plt.tight_layout()
        plt.show()

    def validate_fit(
        self,
        X_test: Optional[np.ndarray] = None,
        n_bootstrap: int = 250,
        alpha: float = 0.05,
        verbose: bool = True,
    ) -> Dict:
        """
        Validate the fitted empirical copula using comprehensive hypothesis tests.

        This method performs:
        1. Kolmogorov-Smirnov tests on marginal distributions
        2. Anderson-Darling tests for marginal goodness-of-fit
        3. Tests for dependence measures (Spearman rho, Kendall tau, Pearson correlation)
        4. Cramér-von Mises test for copula goodness-of-fit
        5. Tests for uniform distribution of pseudo-observations

        Parameters:
        -----------
        X_test : np.ndarray, optional
            Test data for validation. If None, uses training data.
        n_bootstrap : int, default 1000
            Number of bootstrap samples for validation.
        alpha : float, default 0.05
            Significance level for statistical tests.
        verbose : bool, default True
            Whether to print detailed validation results.

        Returns:
        --------
        validation_results : dict
            Dictionary containing all validation test results.
        """
        if not self.is_fitted_:
            raise ValueError("Copula must be fitted before validation.")

        # Use training data if no test data provided
        if X_test is None:
            X_test = self.original_data_.copy()
            if verbose:
                print("Note: Using training data for validation")

        # Generate bootstrap samples for comparison
        bootstrap_samples = self.sample(n_samples=n_bootstrap, random_state=42)

        results = {
            "marginal_tests": {},
            "dependence_tests": {},
            "copula_tests": {},
            "uniformity_tests": {},
            "summary": {},
        }

        if verbose:
            print("\n=== EMPIRICAL COPULA VALIDATION TESTS ===\n")

        # 1. MARGINAL DISTRIBUTION TESTS
        if verbose:
            print("1. Marginal Distribution Tests:")
            print("-" * 35)

        for j in range(self.n_vars_):
            original_margin = X_test[:, j]
            simulated_margin = bootstrap_samples[:, j]
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(
                original_margin, simulated_margin
            )
            # Anderson-Darling test (if samples are from same distribution)
            try:
                # Combine samples and test if they're from the same distribution
                combined_data = np.concatenate(
                    [original_margin, simulated_margin]
                )
                combined_mean = np.mean(combined_data)
                combined_std = np.std(combined_data)
                # Test both against normal distribution with combined parameters
                ad_orig = stats.anderson(original_margin, dist="norm")
                ad_sim = stats.anderson(simulated_margin, dist="norm")
                # Use the test statistic difference as a measure
                ad_diff = abs(ad_orig.statistic - ad_sim.statistic)
                ad_critical = ad_orig.critical_values[
                    2
                ]  # 5% significance level
                ad_pass = ad_diff < ad_critical * 0.5  # Heuristic threshold
            except:
                ad_diff = np.nan
                ad_pass = None
            # Two-sample t-test for means
            ttest_stat, ttest_pvalue = stats.ttest_ind(
                original_margin, simulated_margin
            )
            # Levene's test for equal variances
            levene_stat, levene_pvalue = stats.levene(
                original_margin, simulated_margin
            )

            results["marginal_tests"][f"variable_{j+1}"] = {
                "ks_statistic": ks_stat,
                "ks_p_value": ks_pvalue,
                "ks_reject_null": ks_pvalue < alpha,
                "ad_difference": ad_diff,
                "ad_pass": ad_pass,
                "ttest_statistic": ttest_stat,
                "ttest_p_value": ttest_pvalue,
                "mean_difference_significant": ttest_pvalue < alpha,
                "levene_statistic": levene_stat,
                "levene_p_value": levene_pvalue,
                "variance_difference_significant": levene_pvalue < alpha,
            }

            if verbose:
                status = "FAIL" if ks_pvalue < alpha else "PASS"
                mean_status = "FAIL" if ttest_pvalue < alpha else "PASS"
                var_status = "FAIL" if levene_pvalue < alpha else "PASS"
                print(f"Variable {j+1}:")
                print(
                    f"  KS test: statistic={ks_stat:.4f}, p-value={ks_pvalue:.4f} [{status}]"
                )
                print(
                    f"  Mean test: p-value={ttest_pvalue:.4f} [{mean_status}]"
                )
                print(
                    f"  Variance test: p-value={levene_pvalue:.4f} [{var_status}]"
                )

        # 2. DEPENDENCE STRUCTURE TESTS
        if verbose:
            print(f"\n2. Dependence Structure Tests:")
            print("-" * 32)

        # Calculate dependence measures
        orig_corr = np.corrcoef(X_test.T)
        orig_spearman = self._calculate_spearman_matrix(X_test)
        orig_kendall = self._calculate_kendall_matrix(X_test)

        sim_corr = np.corrcoef(bootstrap_samples.T)
        sim_spearman = self._calculate_spearman_matrix(bootstrap_samples)
        sim_kendall = self._calculate_kendall_matrix(bootstrap_samples)

        # Statistical tests for dependence measures
        dependence_results = {}

        for i in range(self.n_vars_):
            for j in range(i + 1, self.n_vars_):
                pair_name = f"var{i+1}_var{j+1}"
                # Test correlations using Fisher's z-transform
                r1, r2 = orig_corr[i, j], sim_corr[i, j]
                n1, n2 = len(X_test), len(bootstrap_samples)
                # Fisher's z-transform
                z1 = (
                    0.5 * np.log((1 + r1) / (1 - r1))
                    if abs(r1) < 0.999
                    else np.sign(r1) * 3
                )
                z2 = (
                    0.5 * np.log((1 + r2) / (1 - r2))
                    if abs(r2) < 0.999
                    else np.sign(r2) * 3
                )
                # Test statistic
                se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
                z_stat = (z1 - z2) / se if se > 0 else 0
                corr_pvalue = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                # Spearman and Kendall differences
                spear_diff = abs(orig_spearman[i, j] - sim_spearman[i, j])
                kendall_diff = abs(orig_kendall[i, j] - sim_kendall[i, j])

                dependence_results[pair_name] = {
                    "pearson_original": r1,
                    "pearson_simulated": r2,
                    "pearson_z_statistic": z_stat,
                    "pearson_p_value": corr_pvalue,
                    "pearson_significant_diff": corr_pvalue < alpha,
                    "spearman_difference": spear_diff,
                    "kendall_difference": kendall_diff,
                    "spearman_large_diff": spear_diff > 0.1,
                    "kendall_large_diff": kendall_diff > 0.1,
                }

        results["dependence_tests"] = dependence_results

        if verbose:
            for pair, tests in dependence_results.items():
                corr_status = (
                    "FAIL" if tests["pearson_significant_diff"] else "PASS"
                )
                spear_status = (
                    "WARN" if tests["spearman_large_diff"] else "PASS"
                )
                kendall_status = (
                    "WARN" if tests["kendall_large_diff"] else "PASS"
                )
                print(f"{pair.replace('_', '-')}:")
                print(
                    f"  Pearson: {tests['pearson_original']:.4f} vs {tests['pearson_simulated']:.4f}, "
                    f"p-val={tests['pearson_p_value']:.4f} [{corr_status}]"
                )
                print(
                    f"  Spearman diff: {tests['spearman_difference']:.4f} [{spear_status}]"
                )
                print(
                    f"  Kendall diff: {tests['kendall_difference']:.4f} [{kendall_status}]"
                )
        # 3. UNIFORMITY TESTS FOR PSEUDO-OBSERVATIONS
        if verbose:
            print(f"\n3. Uniformity Tests (Pseudo-Observations):")
            print("-" * 42)
        # Test if pseudo-observations are uniform on [0,1]
        pseudo_test = self._to_pseudo_observations(X_test)

        uniformity_results = {}

        for j in range(self.n_vars_):
            pseudo_margin = pseudo_test[:, j]
            # Kolmogorov-Smirnov test against uniform distribution
            ks_uniform_stat, ks_uniform_pvalue = stats.kstest(
                pseudo_margin, "uniform"
            )
            # Anderson-Darling test for uniformity
            # Transform to standard normal and test
            normal_transformed = stats.norm.ppf(
                np.clip(pseudo_margin, 1e-10, 1 - 1e-10)
            )
            ad_result = stats.anderson(normal_transformed, dist="norm")

            # Cramer-von Mises test for uniformity
            def cvm_uniform(data):
                """Cramér-von Mises test for uniform distribution."""
                n = len(data)
                sorted_data = np.sort(data)
                i = np.arange(1, n + 1)
                T = (1.0 / (12 * n)) + np.sum(
                    ((2 * i - 1) / (2 * n) - sorted_data) ** 2
                )
                return T

            cvm_stat = cvm_uniform(pseudo_margin)
            # Critical value at 5% significance level
            cvm_critical = 0.461 / (
                np.sqrt(len(pseudo_margin))
                + 0.25
                + 0.75 / np.sqrt(len(pseudo_margin))
            )

            uniformity_results[f"variable_{j+1}"] = {
                "ks_uniform_statistic": ks_uniform_stat,
                "ks_uniform_p_value": ks_uniform_pvalue,
                "ks_uniform_reject": ks_uniform_pvalue < alpha,
                "ad_statistic": ad_result.statistic,
                "ad_critical_5pct": ad_result.critical_values[2],
                "ad_reject": ad_result.statistic > ad_result.critical_values[2],
                "cvm_statistic": cvm_stat,
                "cvm_critical": cvm_critical,
                "cvm_reject": cvm_stat > cvm_critical,
            }

            if verbose:
                ks_status = "FAIL" if ks_uniform_pvalue < alpha else "PASS"
                ad_status = (
                    "FAIL"
                    if ad_result.statistic > ad_result.critical_values[2]
                    else "PASS"
                )
                cvm_status = "FAIL" if cvm_stat > cvm_critical else "PASS"
                print(f"Variable {j+1}:")
                print(
                    f"  KS uniform: stat={ks_uniform_stat:.4f}, p-val={ks_uniform_pvalue:.4f} [{ks_status}]"
                )
                print(
                    f"  AD normal: stat={ad_result.statistic:.4f}, crit={ad_result.critical_values[2]:.4f} [{ad_status}]"
                )
                print(
                    f"  CvM uniform: stat={cvm_stat:.4f}, crit={cvm_critical:.4f} [{cvm_status}]"
                )

        results["uniformity_tests"] = uniformity_results

        pseudo_orig = self._to_pseudo_observations(X_test)
        pseudo_sim = self._to_pseudo_observations(bootstrap_samples)
        # 5. SUMMARY ASSESSMENT
        if verbose:
            print(f"\n5. Overall Assessment:")
            print("-" * 22)
        # Count various test failures
        ks_failures = sum(
            1
            for j in range(self.n_vars_)
            if results["marginal_tests"][f"variable_{j+1}"]["ks_reject_null"]
        )

        mean_failures = sum(
            1
            for j in range(self.n_vars_)
            if results["marginal_tests"][f"variable_{j+1}"][
                "mean_difference_significant"
            ]
        )

        var_failures = sum(
            1
            for j in range(self.n_vars_)
            if results["marginal_tests"][f"variable_{j+1}"][
                "variance_difference_significant"
            ]
        )

        corr_failures = sum(
            1
            for tests in dependence_results.values()
            if tests["pearson_significant_diff"]
        )

        uniform_failures = sum(
            1
            for j in range(self.n_vars_)
            if results["uniformity_tests"][f"variable_{j+1}"][
                "ks_uniform_reject"
            ]
        )
        # Calculate average differences
        avg_spear_diff = np.mean(
            [
                tests["spearman_difference"]
                for tests in dependence_results.values()
            ]
        )
        avg_kendall_diff = np.mean(
            [
                tests["kendall_difference"]
                for tests in dependence_results.values()
            ]
        )
        # Overall quality assessment
        total_tests = (
            self.n_vars_ * 3 + len(dependence_results) + self.n_vars_ + 1
        )
        total_failures = (
            ks_failures
            + mean_failures
            + var_failures
            + corr_failures
            + uniform_failures
        )

        pass_rate = (total_tests - total_failures) / total_tests * 100

        if pass_rate >= 85 and avg_spear_diff <= 0.05:
            quality = "Excellent"
        elif pass_rate >= 70 and avg_spear_diff <= 0.10:
            quality = "Good"
        elif pass_rate >= 50 and avg_spear_diff <= 0.15:
            quality = "Fair"
        else:
            quality = "Poor"

        results["summary"] = {
            "ks_failures": ks_failures,
            "mean_failures": mean_failures,
            "variance_failures": var_failures,
            "correlation_failures": corr_failures,
            "uniformity_failures": uniform_failures,
            "total_failures": total_failures,
            "total_tests": total_tests,
            "pass_rate": pass_rate,
            "avg_spearman_difference": avg_spear_diff,
            "avg_kendall_difference": avg_kendall_diff,
            "overall_quality": quality,
        }

        if verbose:
            print(f"Test Summary ({total_tests} total tests):")
            print(f"  Marginal KS failures: {ks_failures}/{self.n_vars_}")
            print(f"  Mean difference failures: {mean_failures}/{self.n_vars_}")
            print(
                f"  Variance difference failures: {var_failures}/{self.n_vars_}"
            )
            print(
                f"  Correlation failures: {corr_failures}/{len(dependence_results)}"
            )
            print(f"  Uniformity failures: {uniform_failures}/{self.n_vars_}")
            print(f"  Overall pass rate: {pass_rate:.1f}%")
            print(f"  Average Spearman difference: {avg_spear_diff:.4f}")
            print(f"  Average Kendall difference: {avg_kendall_diff:.4f}")
            print(f"  Overall model quality: {quality}")

        return results

    def _to_pseudo_observations(self, X: np.ndarray) -> np.ndarray:
        """Convert data to pseudo-observations using empirical CDF."""
        n_samples, n_vars = X.shape
        pseudo_obs = np.zeros_like(X)

        for j in range(n_vars):
            # Rank-based transformation
            ranks = stats.rankdata(X[:, j], method="average")
            # Use (rank - 0.5) / n to avoid boundary values
            pseudo_obs[:, j] = (ranks - 0.5) / n_samples

        return pseudo_obs

    def _apply_smoothing(self, pseudo_obs: np.ndarray) -> np.ndarray:
        """Apply smoothing to pseudo-observations."""
        if self.smoothing_method == "jitter":
            # Add uniform jitter
            jitter = np.random.uniform(
                -self.jitter_scale / 2, self.jitter_scale / 2, pseudo_obs.shape
            )
            smoothed = pseudo_obs + jitter
            # Ensure values stay in [0,1]
            smoothed = np.clip(smoothed, 1e-10, 1 - 1e-10)
            return smoothed
        else:
            return pseudo_obs

    def _fit_marginal_transforms(self, X: np.ndarray) -> None:
        """Fit marginal transformations for inverse sampling."""
        self.marginal_cdfs_ = []
        self.marginal_quantiles_ = []

        for j in range(self.n_vars_):
            data_col = X[:, j]
            sorted_data = np.sort(data_col)
            # Create empirical CDF
            n = len(sorted_data)
            cdf_values = np.arange(1, n + 1) / n
            # Store quantile function (inverse CDF)
            # Add boundary extrapolation
            extended_probs = np.concatenate([[0], cdf_values, [1]])
            extended_data = np.concatenate(
                [
                    [sorted_data[0] - (sorted_data[1] - sorted_data[0])],
                    sorted_data,
                    [sorted_data[-1] + (sorted_data[-1] - sorted_data[-2])],
                ]
            )
            quantile_func = interp1d(
                extended_probs,
                extended_data,
                kind="linear",
                bounds_error=False,
                fill_value=(extended_data[0], extended_data[-1]),
            )
            self.marginal_quantiles_.append(quantile_func)

    def _bootstrap_sample(self, n_samples: int) -> np.ndarray:
        """Generate samples using bootstrap resampling."""
        # Randomly sample indices with replacement
        indices = np.random.choice(
            self.n_samples_, size=n_samples, replace=True
        )
        return self.pseudo_observations_[indices]

    def _kde_sample(
        self, n_samples: int, kernel="gaussian", **kwargs
    ) -> np.ndarray:
        """Generate samples using kernel density estimation."""
        kernel_bandwidths = {"bandwidth": np.logspace(-6, 6, 150)}
        grid = GridSearchCV(
            KernelDensity(kernel=kernel, **kwargs), param_grid=kernel_bandwidths
        )
        grid.fit(self.pseudo_observations_)
        self.kde_model_ = grid.best_estimator_
        return self.kde_model_.sample(n_samples)

    def _gmm_sample(
        self,
        n_samples: int,
        n_components: int = 5,
        covariance_type: str = "full",
        **kwargs,
    ) -> np.ndarray:
        """
        Generate samples using Gaussian mixture model.

        Parameters:
        -----------
        n_samples : int
            Number of samples to generate.
        n_components : int, default 5
            Number of Gaussian components in the mixture.
        covariance_type : str, default "full"
            Type of covariance parameters. Options: 'full', 'tied', 'diag', 'spherical'.
        **kwargs : additional arguments for GaussianMixture.

        Returns:
        --------
        samples : np.ndarray
            Generated samples on [0,1] copula scale.
        """
        # Fit Gaussian mixture model to pseudo-observations
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=kwargs.get("random_state", None),
            **{k: v for k, v in kwargs.items() if k != "random_state"},
        )

        # Fit the model
        gmm.fit(self.pseudo_observations_)

        # Store the fitted model
        self.gmm_model_ = gmm

        # Generate samples
        samples, _ = gmm.sample(n_samples)

        # Ensure samples are in [0,1] range (clip if necessary)
        samples = np.clip(samples, 1e-10, 1 - 1e-10)

        return samples

    def _inverse_transform(self, pseudo_samples: np.ndarray) -> np.ndarray:
        """Transform pseudo-observations back to original scale."""
        n_samples, n_vars = pseudo_samples.shape
        original_samples = np.zeros_like(pseudo_samples)

        for j in range(n_vars):
            u = pseudo_samples[:, j]
            # Ensure values are in valid range
            u = np.clip(u, 1e-10, 1 - 1e-10)
            original_samples[:, j] = self.marginal_quantiles_[j](u)

        return original_samples

    def _calculate_spearman_matrix(self, X: np.ndarray) -> np.ndarray:
        """Calculate Spearman rank correlation matrix."""
        n_vars = X.shape[1]
        spearman_matrix = np.zeros((n_vars, n_vars))

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    spearman_matrix[i, j] = 1.0
                else:
                    spearman_matrix[i, j], _ = stats.spearmanr(X[:, i], X[:, j])

        return spearman_matrix

    def _calculate_kendall_matrix(self, X: np.ndarray) -> np.ndarray:
        """Calculate Kendall's tau correlation matrix."""
        n_vars = X.shape[1]
        kendall_matrix = np.zeros((n_vars, n_vars))

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    kendall_matrix[i, j] = 1.0
                else:
                    kendall_matrix[i, j], _ = stats.kendalltau(X[:, i], X[:, j])

        return kendall_matrix

    def get_info(self) -> Dict:
        """Get information about the fitted empirical copula."""
        if not self.is_fitted_:
            raise ValueError("Copula must be fitted first.")

        return {
            "n_samples": self.n_samples_,
            "n_vars": self.n_vars_,
            "smoothing_method": self.smoothing_method,
            "jitter_scale": self.jitter_scale,
            "boundary_correction": self.boundary_correction,
            "has_kde_model": self.kde_model_ is not None,
            "has_gmm_model": self.gmm_model_ is not None,
        }

    def __repr__(self) -> str:
        if self.is_fitted_:
            return (
                f"EmpiricalCopula(n_samples={self.n_samples_}, n_vars={self.n_vars_}, "
                f"smoothing='{self.smoothing_method}', fitted=True)"
            )
        else:
            return f"EmpiricalCopula(smoothing='{self.smoothing_method}', fitted=False)"
