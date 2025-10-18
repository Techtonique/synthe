import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class AdaptiveHistogramSampler:
    def __init__(self, n_bins=10, method="quantile", seed=123):
        self.n_bins = n_bins
        self.method = method
        self.rng = np.random.default_rng(seed)
        self.bin_edges = None
        self.bin_indices = None
        self.unique_bins = None
        self.bin_probs = None
        self.X = None
        self.n = None
        self.d = None

    def fit(self, X):
        self.X = np.asarray(X)
        self.n, self.d = self.X.shape

        self.bin_edges = []
        for j in range(self.d):
            xj = self.X[:, j]
            if self.method == "quantile":
                edges_j = np.quantile(xj, np.linspace(0, 1, self.n_bins + 1))
            else:
                edges_j = np.linspace(xj.min(), xj.max(), self.n_bins + 1)
            self.bin_edges.append(edges_j)

        # Assign points to bins
        bin_idx = np.zeros((self.n, self.d), dtype=int)
        for j in range(self.d):
            bin_idx[:, j] = np.digitize(self.X[:, j], self.bin_edges[j]) - 1
            bin_idx[:, j] = np.clip(bin_idx[:, j], 0, self.n_bins - 1)
        self.bin_indices = bin_idx

        bin_ids = np.ravel_multi_index(
            self.bin_indices.T, (self.n_bins,) * self.d
        )
        unique_bins, counts = np.unique(bin_ids, return_counts=True)
        self.unique_bins = unique_bins
        self.bin_probs = counts / counts.sum()

    def sample(
        self,
        n_samples,
        oversample=False,
        oversample_method="bootstrap",
        jitter_scale=0.05,
    ):
        if self.bin_probs is None:
            raise RuntimeError("You must call `fit` before `sample`.")

        chosen_bins = self.rng.choice(
            self.unique_bins, size=n_samples, p=self.bin_probs
        )

        if not oversample:
            return self._subsample_existing(chosen_bins)

        if oversample_method == "uniform":
            return self._oversample_uniform(chosen_bins)
        elif oversample_method == "bootstrap":
            return self._oversample_bootstrap(chosen_bins)
        elif oversample_method == "jitter":
            return self._oversample_jitter(chosen_bins, jitter_scale)
        else:
            raise ValueError(f"Unknown oversample_method: {oversample_method}")

    # --- Internal helpers ------------------------------------------------
    def _subsample_existing(self, chosen_bins):
        bin_ids = np.ravel_multi_index(
            self.bin_indices.T, (self.n_bins,) * self.d
        )
        X_sampled = []
        for b in chosen_bins:
            idx_in_bin = np.where(bin_ids == b)[0]
            i = self.rng.choice(idx_in_bin)
            X_sampled.append(self.X[i])
        return np.array(X_sampled)

    def _oversample_uniform(self, chosen_bins):
        X_sampled = []
        for b in chosen_bins:
            multi_idx = np.unravel_index(b, (self.n_bins,) * self.d)
            coords = []
            for j, bi in enumerate(multi_idx):
                left = self.bin_edges[j][bi]
                right = self.bin_edges[j][bi + 1]
                coords.append(self.rng.uniform(left, right))
            X_sampled.append(coords)
        return np.array(X_sampled)

    def _oversample_bootstrap(self, chosen_bins):
        return self._subsample_existing(chosen_bins)

    def _oversample_jitter(self, chosen_bins, jitter_scale):
        base_points = self._subsample_existing(chosen_bins)
        noise = self.rng.normal(scale=jitter_scale, size=base_points.shape)
        return base_points + noise

    # --- Visualization ----------------------------------------------------
    def plot_comparison(self, X_sampled, bins=30):
        """Plot joint 2D histogram and marginal distributions (for d=2)."""
        if self.d != 2:
            raise ValueError("plot_comparison only supports 2D currently.")

        fig = plt.figure(figsize=(10, 10))
        grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)

        main_ax = fig.add_subplot(grid[1:, :-1])
        y_hist = fig.add_subplot(grid[0, :-1], sharex=main_ax)
        x_hist = fig.add_subplot(grid[1:, -1], sharey=main_ax)

        # 2D histogram
        main_ax.hist2d(
            self.X[:, 0], self.X[:, 1], bins=bins, alpha=0.5, cmap="Blues"
        )
        main_ax.hist2d(
            X_sampled[:, 0], X_sampled[:, 1], bins=bins, alpha=0.5, cmap="Reds"
        )
        main_ax.set_xlabel("X1")
        main_ax.set_ylabel("X2")
        main_ax.set_title("Joint distribution")

        # Marginals
        y_hist.hist(
            self.X[:, 0], bins=bins, color="blue", alpha=0.5, density=True
        )
        y_hist.hist(
            X_sampled[:, 0], bins=bins, color="red", alpha=0.5, density=True
        )
        x_hist.hist(
            self.X[:, 1],
            bins=bins,
            orientation="horizontal",
            color="blue",
            alpha=0.5,
            density=True,
        )
        x_hist.hist(
            X_sampled[:, 1],
            bins=bins,
            orientation="horizontal",
            color="red",
            alpha=0.5,
            density=True,
        )

        y_hist.axis("off")
        x_hist.axis("off")

        plt.show()

    # --- Goodness of fit tests -------------------------------------------
    def goodness_of_fit(self, X_sampled):
        """
        Compare marginals with Kolmogorov–Smirnov and Anderson–Darling tests.
        Returns dict of test results for each dimension.
        """
        results = {}
        for j in range(self.d):
            x_orig = self.X[:, j]
            x_samp = X_sampled[:, j]

            # KS test
            ks_stat, ks_p = stats.ks_2samp(x_orig, x_samp)

            # Anderson-Darling test
            ad_result = stats.anderson_ksamp([x_orig, x_samp])

            results[f"dim_{j}"] = {
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_p,
                "ad_statistic": ad_result.statistic,
                "ad_significance_level": ad_result.significance_level,
            }
        return results
