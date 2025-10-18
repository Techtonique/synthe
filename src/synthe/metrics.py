import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import stats
from scipy.spatial import distance
from scipy.stats import entropy, wasserstein_distance, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import kl_div


class DistanceMetrics:
    def __init__(self, vector, matrix):
        self.vector = np.array(vector)
        self.matrix = np.array(matrix)

    def euclidean_distance(self):
        """Euclidean (L2) Distance between vector and each row of the matrix."""
        return np.linalg.norm(self.matrix - self.vector, axis=1)

    def manhattan_distance(self):
        """Manhattan (L1) Distance between vector and each row of the matrix."""
        return np.sum(np.abs(self.matrix - self.vector), axis=1)

    def cosine_distance(self):
        """Cosine Distance between vector and each row of the matrix."""
        similarities = cosine_similarity(
            self.vector.reshape(1, -1), self.matrix
        )
        return 1 - similarities.flatten()

    def mahalanobis_distance(self):
        """Mahalanobis Distance between vector and each row of the matrix."""
        cov_matrix = np.cov(self.matrix.T)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        return [
            distance.mahalanobis(self.vector, m, inv_cov_matrix)
            for m in self.matrix
        ]

    def chebyshev_distance(self):
        """Chebyshev Distance (Maximum absolute difference)."""
        return np.max(np.abs(self.matrix - self.vector), axis=1)

    def hamming_distance(self):
        """Hamming Distance between vector and each row of the matrix (for binary data)."""
        return np.sum(self.matrix != self.vector, axis=1)

    def jaccard_distance(self):
        """Jaccard Distance between vector and each row of the matrix (for binary data)."""
        return [
            1 - jaccard_score(self.vector, m, average="binary")
            for m in self.matrix
        ]

    def weighted_euclidean_distance(self, weights):
        """Weighted Euclidean Distance between vector and each row of the matrix."""
        weights = np.array(weights)
        return np.sqrt(
            np.sum(weights * (self.matrix - self.vector) ** 2, axis=1)
        )

    def kullback_leibler_divergence(self, P, Q):
        """Kullback-Leibler Divergence between two distributions P and Q."""
        return entropy(P, Q)

    def wasserstein_distance(self, distribution_1, distribution_2):
        """Wasserstein Distance (Earth Mover's Distance) between two distributions."""
        return wasserstein_distance(distribution_1, distribution_2)

    def pearson_correlation(self):
        """Pearson Correlation between the vector and each row of the matrix."""
        return [pearsonr(self.vector, m)[0] for m in self.matrix]

    def jensen_shannon_divergence(self, P, Q):
        """Jensen-Shannon Divergence between two distributions."""
        M = 0.5 * (P + Q)
        return 0.5 * (kl_div(P, M).sum() + kl_div(Q, M).sum())

    def total_variation_distance(self, P, Q):
        """Total Variation Distance between two distributions."""
        return 0.5 * np.sum(np.abs(P - Q))

    def qqplot_with_summary(
        self, data1, data2, label1="Sample 1", label2="Sample 2"
    ):
        data1 = np.asarray(data1)
        data2 = np.asarray(data2)

        # Remove NaN values
        data1 = data1[~np.isnan(data1)]
        data2 = data2[~np.isnan(data2)]

        # Q–Q plot
        n_quantiles = min(len(data1), len(data2))
        quantiles1 = np.percentile(data1, np.linspace(0, 100, n_quantiles))
        quantiles2 = np.percentile(data2, np.linspace(0, 100, n_quantiles))

        plt.figure(figsize=(6, 6))
        plt.scatter(quantiles1, quantiles2, alpha=0.7)
        min_val = min(quantiles1.min(), quantiles2.min())
        max_val = max(quantiles1.max(), quantiles2.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="y = x")
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.title("Q–Q Plot")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Descriptive stats
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        n1, n2 = len(data1), len(data2)

        # Kolmogorov–Smirnov test
        ks_stat, ks_p = stats.ks_2samp(data1, data2)

        # Anderson–Darling test (two-sample)
        ad_result = stats.anderson_ksamp([data1, data2])
        ad_stat = ad_result.statistic
        ad_p = ad_result.significance_level / 100  # convert % to proportion

        # Quantile correlation
        corr = np.corrcoef(quantiles1, quantiles2)[0, 1]

        # Summary table
        summary = pd.DataFrame(
            {
                "Statistic": [
                    "Sample size",
                    "Mean",
                    "Std. deviation",
                    "KS statistic",
                    "KS p-value",
                    "AD statistic",
                    "AD p-value",
                    "Quantile correlation",
                ],
                label1: [n1, mean1, std1, ks_stat, ks_p, ad_stat, ad_p, corr],
                label2: [n2, mean2, std2, "", "", "", "", ""],
            }
        )

        return summary


# # Example usage:
# vector = [1, 2, 3]
# matrix = [[1, 1, 1], [2, 2, 2], [4, 5, 6]]

# # Initialize the DistanceMetrics class
# dist = DistanceMetrics(vector, matrix)

# # Compute distances
# euclidean = dist.euclidean_distance()
# manhattan = dist.manhattan_distance()
# cosine = dist.cosine_distance()
# mahalanobis = dist.mahalanobis_distance()
# chebyshev = dist.chebyshev_distance()

# # Print results
# print(f"Euclidean Distance: {euclidean}")
# print(f"Manhattan Distance: {manhattan}")
# print(f"Cosine Distance: {cosine}")
# print(f"Mahalanobis Distance: {mahalanobis}")
# print(f"Chebyshev Distance: {chebyshev}")

# # For distribution-based measures
# P = np.array([0.2, 0.5, 0.3])  # Example distribution P
# Q = np.array([0.3, 0.4, 0.3])  # Example distribution Q

# kl_divergence = dist.kullback_leibler_divergence(P, Q)
# wasserstein = dist.wasserstein_distance(P, Q)
# pearson_corr = dist.pearson_correlation()

# print(f"Kullback-Leibler Divergence: {kl_divergence}")
# print(f"Wasserstein Distance: {wasserstein}")
# print(f"Pearson Correlation: {pearson_corr}")
