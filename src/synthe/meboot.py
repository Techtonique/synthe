import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Union, Optional, List, Tuple, Callable
import warnings
from dataclasses import dataclass


@dataclass
class HypothesisTestResult:
    """Container for hypothesis test results."""

    statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    null_value: float
    alternative: str
    reject_null: bool
    test_type: str


class MaximumEntropyBootstrap:
    """
    Maximum Entropy Bootstrap for time series inference with plotting and hypothesis testing.
    """

    def __init__(self, trim: float = 0.10, random_state: Optional[int] = None):
        self.trim = trim
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

        # Storage for intermediate results
        self.order_stats_ = None
        self.ordering_index_ = None
        self.intermediate_points_ = None
        self.interval_means_ = None
        self.limits_ = None
        self.original_series_ = None

    def _calculate_trimmed_mean_diff(self, x: np.ndarray) -> float:
        """Calculate trimmed mean of consecutive differences."""
        diffs = np.diff(x)
        if len(diffs) == 0:
            return 0.0

        lower_bound = np.percentile(diffs, self.trim * 100)
        upper_bound = np.percentile(diffs, (1 - self.trim) * 100)
        trimmed_diffs = diffs[(diffs >= lower_bound) & (diffs <= upper_bound)]

        return (
            np.mean(trimmed_diffs) if len(trimmed_diffs) > 0 else np.mean(diffs)
        )

    def _calculate_interval_means(self, order_stats: np.ndarray) -> np.ndarray:
        """Calculate means for each interval using mean-preserving constraint."""
        T = len(order_stats)
        means = np.zeros(T)

        means[0] = 0.75 * order_stats[0] + 0.25 * order_stats[1]

        for k in range(1, T - 1):
            means[k] = (
                0.25 * order_stats[k - 1]
                + 0.50 * order_stats[k]
                + 0.25 * order_stats[k + 1]
            )

        means[T - 1] = 0.25 * order_stats[T - 2] + 0.75 * order_stats[T - 1]

        return means

    def fit(
        self, x: Union[np.ndarray, List, pd.Series]
    ) -> "MaximumEntropyBootstrap":
        """Fit the ME bootstrap to the time series."""
        x = np.asarray(x)
        if len(x) < 3:
            raise ValueError("Time series must have at least 3 observations")

        self.original_series_ = x.copy()
        T = len(x)

        # Step 1: Sort data and store ordering index
        self.ordering_index_ = np.argsort(x)
        self.order_stats_ = x[self.ordering_index_]

        # Step 2: Compute intermediate points
        self.intermediate_points_ = (
            self.order_stats_[:-1] + self.order_stats_[1:]
        ) / 2

        # Step 3: Compute limits for tails
        m_trim = self._calculate_trimmed_mean_diff(x)
        z0 = self.order_stats_[0] - m_trim
        zT = self.order_stats_[-1] + m_trim

        self.limits_ = (z0, zT)

        # Step 4: Compute interval means
        self.interval_means_ = self._calculate_interval_means(self.order_stats_)

        return self

    def _generate_me_quantiles(self, size: int) -> np.ndarray:
        """Generate quantiles from maximum entropy density."""
        z0, zT = self.limits_
        all_z_points = np.concatenate([[z0], self.intermediate_points_, [zT]])

        u = np.random.uniform(0, 1, size)
        quantiles = np.zeros(size)
        n_intervals = len(all_z_points) - 1

        for i in range(size):
            interval_idx = int(u[i] * n_intervals)
            interval_idx = min(interval_idx, n_intervals - 1)

            interval_start = all_z_points[interval_idx]
            interval_end = all_z_points[interval_idx + 1]
            interval_frac = (u[i] * n_intervals) - interval_idx

            quantiles[i] = interval_start + interval_frac * (
                interval_end - interval_start
            )

        return quantiles

    def sample(self, reps: int = 999) -> np.ndarray:
        """Generate bootstrap replicates."""
        if self.order_stats_ is None:
            raise ValueError("Must call fit() before sample()")

        T = len(self.order_stats_)
        ensemble = np.zeros((T, reps))

        for j in range(reps):
            me_quantiles = self._generate_me_quantiles(T)
            sorted_quantiles = np.sort(me_quantiles)

            original_order_quantiles = np.zeros(T)
            for i, idx in enumerate(self.ordering_index_):
                original_order_quantiles[idx] = sorted_quantiles[i]

            ensemble[:, j] = original_order_quantiles

        return ensemble

    # ==================== PLOTTING METHODS ====================

    def plot_me_density(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Plot the maximum entropy density with intervals."""
        if self.order_stats_ is None:
            raise ValueError("Must call fit() first")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Plot 1: ME Density Intervals
        z0, zT = self.limits_
        all_z_points = np.concatenate([[z0], self.intermediate_points_, [zT]])

        for i in range(len(all_z_points) - 1):
            ax1.axvspan(
                all_z_points[i],
                all_z_points[i + 1],
                alpha=0.3,
                label=f"Interval {i+1}" if i == 0 else "",
            )
            ax1.axvline(all_z_points[i], color="red", linestyle="--", alpha=0.7)

        ax1.axvline(all_z_points[-1], color="red", linestyle="--", alpha=0.7)
        ax1.set_title("Maximum Entropy Density Intervals")
        ax1.set_xlabel("Value")
        ax1.set_ylabel("Intervals")
        ax1.legend()

        # Plot 2: Original vs Order Statistics
        ax2.plot(
            self.original_series_, "o-", label="Original Series", alpha=0.7
        )
        ax2.plot(self.order_stats_, "s-", label="Order Statistics", alpha=0.7)
        ax2.set_title("Original Series vs Order Statistics")
        ax2.set_xlabel("Index")
        ax2.set_ylabel("Value")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_bootstrap_ensemble(
        self, reps: int = 50, figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """Plot multiple bootstrap replicates with original series."""
        ensemble = self.sample(reps)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)

        # Plot 1: All replicates
        time_index = np.arange(len(self.original_series_))
        for j in range(min(reps, 50)):  # Limit to 50 for clarity
            ax1.plot(time_index, ensemble[:, j], alpha=0.1, color="blue")

        ax1.plot(
            time_index,
            self.original_series_,
            "r-",
            linewidth=2,
            label="Original",
        )
        ax1.set_title(f"ME Bootstrap Ensemble ({reps} replicates)")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Mean and confidence intervals
        mean_ensemble = np.mean(ensemble, axis=1)
        ci_lower = np.percentile(ensemble, 2.5, axis=1)
        ci_upper = np.percentile(ensemble, 97.5, axis=1)

        ax2.fill_between(
            time_index, ci_lower, ci_upper, alpha=0.3, label="95% CI"
        )
        ax2.plot(time_index, mean_ensemble, "b-", label="Ensemble Mean")
        ax2.plot(time_index, self.original_series_, "r-", label="Original")
        ax2.set_title("Ensemble Mean and 95% Confidence Intervals")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Value")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Distribution at selected time points
        if len(time_index) >= 5:
            selected_times = np.linspace(0, len(time_index) - 1, 5, dtype=int)
            for i, t in enumerate(selected_times):
                ax3.hist(
                    ensemble[t, :],
                    bins=30,
                    alpha=0.5,
                    label=f"Time {t}",
                    density=True,
                )
            ax3.set_title("Distribution at Selected Time Points")
            ax3.set_xlabel("Value")
            ax3.set_ylabel("Density")
            ax3.legend()

        plt.tight_layout()
        return fig

    def plot_sampling_distribution(
        self,
        statistic: Callable,
        reps: int = 999,
        figsize: Tuple[int, int] = (12, 10),
    ) -> plt.Figure:
        """Plot sampling distribution of a statistic."""
        ensemble = self.sample(reps)

        # Calculate statistic for each bootstrap sample
        stats_boot = np.zeros(reps)
        for j in range(reps):
            stats_boot[j] = statistic(ensemble[:, j])

        original_stat = statistic(self.original_series_)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Histogram with KDE
        ax1.hist(stats_boot, bins=30, density=True, alpha=0.7, color="skyblue")
        ax1.axvline(
            original_stat,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Original: {original_stat:.3f}",
        )
        ax1.axvline(
            np.mean(stats_boot),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Bootstrap Mean: {np.mean(stats_boot):.3f}",
        )
        ax1.set_title("Bootstrap Sampling Distribution")
        ax1.set_xlabel("Statistic Value")
        ax1.set_ylabel("Density")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Q-Q plot for normality check
        stats.probplot(stats_boot, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot for Normality Check")

        plt.tight_layout()
        return fig, stats_boot

    # ==================== HYPOTHESIS TESTING METHODS ====================

    def hypothesis_test(
        self,
        statistic: Callable,
        null_value: float = 0,
        alternative: str = "two-sided",
        reps: int = 999,
        confidence: float = 0.95,
    ) -> HypothesisTestResult:
        """
        Perform hypothesis test using ME bootstrap.

        Parameters
        ----------
        statistic : callable
            Function that computes the test statistic
        null_value : float, default=0
            Value under the null hypothesis
        alternative : str, default='two-sided'
            Alternative hypothesis: 'two-sided', 'less', or 'greater'
        reps : int, default=999
            Number of bootstrap replicates
        confidence : float, default=0.95
            Confidence level for interval

        Returns
        -------
        HypothesisTestResult
        """
        if alternative not in ["two-sided", "less", "greater"]:
            raise ValueError(
                "Alternative must be 'two-sided', 'less', or 'greater'"
            )

        ensemble = self.sample(reps)

        # Calculate statistic for each bootstrap sample
        stats_boot = np.zeros(reps)
        for j in range(reps):
            stats_boot[j] = statistic(ensemble[:, j])

        original_stat = statistic(self.original_series_)

        # Calculate p-value based on alternative hypothesis
        if alternative == "two-sided":
            p_value = 2 * min(
                np.mean(stats_boot <= null_value),
                np.mean(stats_boot >= null_value),
            )
            ci_lower = np.percentile(stats_boot, (1 - confidence) / 2 * 100)
            ci_upper = np.percentile(
                stats_boot, (1 - (1 - confidence) / 2) * 100
            )
        elif alternative == "less":
            p_value = np.mean(stats_boot <= null_value)
            ci_lower = np.percentile(stats_boot, (1 - confidence) * 100)
            ci_upper = np.inf
        else:  # 'greater'
            p_value = np.mean(stats_boot >= null_value)
            ci_lower = -np.inf
            ci_upper = np.percentile(stats_boot, confidence * 100)

        reject_null = p_value < (1 - confidence)

        return HypothesisTestResult(
            statistic=original_stat,
            p_value=p_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            null_value=null_value,
            alternative=alternative,
            reject_null=reject_null,
            test_type="bootstrap",
        )

    def test_mean(
        self,
        null_value: float = 0,
        alternative: str = "two-sided",
        reps: int = 999,
        confidence: float = 0.95,
    ) -> HypothesisTestResult:
        """Test hypothesis about the mean."""
        return self.hypothesis_test(
            statistic=np.mean,
            null_value=null_value,
            alternative=alternative,
            reps=reps,
            confidence=confidence,
        )

    def test_median(
        self,
        null_value: float = 0,
        alternative: str = "two-sided",
        reps: int = 999,
        confidence: float = 0.95,
    ) -> HypothesisTestResult:
        """Test hypothesis about the median."""
        return self.hypothesis_test(
            statistic=np.median,
            null_value=null_value,
            alternative=alternative,
            reps=reps,
            confidence=confidence,
        )

    def test_variance(
        self,
        null_value: float = 1,
        alternative: str = "two-sided",
        reps: int = 999,
        confidence: float = 0.95,
    ) -> HypothesisTestResult:
        """Test hypothesis about the variance."""
        return self.hypothesis_test(
            statistic=np.var,
            null_value=null_value,
            alternative=alternative,
            reps=reps,
            confidence=confidence,
        )

    def test_correlation(
        self,
        y: np.ndarray,
        null_value: float = 0,
        alternative: str = "two-sided",
        reps: int = 999,
        confidence: float = 0.95,
    ) -> HypothesisTestResult:
        """Test hypothesis about correlation with another series."""
        if len(y) != len(self.original_series_):
            raise ValueError("y must have same length as original series")

        def corr_statistic(x):
            return np.corrcoef(x, y)[0, 1]

        return self.hypothesis_test(
            statistic=corr_statistic,
            null_value=null_value,
            alternative=alternative,
            reps=reps,
            confidence=confidence,
        )

    def compare_means(
        self,
        y: np.ndarray,
        null_value: float = 0,
        alternative: str = "two-sided",
        reps: int = 999,
        confidence: float = 0.95,
    ) -> HypothesisTestResult:
        """Test for difference in means between two series."""

        def mean_diff_statistic(x):
            return np.mean(x) - np.mean(y)

        return self.hypothesis_test(
            statistic=mean_diff_statistic,
            null_value=null_value,
            alternative=alternative,
            reps=reps,
            confidence=confidence,
        )

    # ==================== UTILITY METHODS ====================

    def get_params(self) -> dict:
        """Get parameters of the fitted ME bootstrap."""
        return {
            "order_stats": self.order_stats_,
            "ordering_index": self.ordering_index_,
            "intermediate_points": self.intermediate_points_,
            "interval_means": self.interval_means_,
            "limits": self.limits_,
            "trim": self.trim,
        }

    def summary(self) -> pd.DataFrame:
        """Generate summary statistics of the original series."""
        if self.original_series_ is None:
            raise ValueError("Must call fit() first")

        x = self.original_series_
        stats_dict = {
            "n_observations": len(x),
            "mean": np.mean(x),
            "median": np.median(x),
            "std_dev": np.std(x),
            "variance": np.var(x),
            "min": np.min(x),
            "max": np.max(x),
            "skewness": stats.skew(x),
            "kurtosis": stats.kurtosis(x),
        }

        return pd.DataFrame([stats_dict])
