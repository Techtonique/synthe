import numpy as np
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
        similarities = cosine_similarity(self.vector.reshape(1, -1), self.matrix)
        return 1 - similarities.flatten()
    
    def mahalanobis_distance(self):
        """Mahalanobis Distance between vector and each row of the matrix."""
        cov_matrix = np.cov(self.matrix.T)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        return [distance.mahalanobis(self.vector, m, inv_cov_matrix) for m in self.matrix]
    
    def chebyshev_distance(self):
        """Chebyshev Distance (Maximum absolute difference)."""
        return np.max(np.abs(self.matrix - self.vector), axis=1)
    
    def hamming_distance(self):
        """Hamming Distance between vector and each row of the matrix (for binary data)."""
        return np.sum(self.matrix != self.vector, axis=1)
    
    def jaccard_distance(self):
        """Jaccard Distance between vector and each row of the matrix (for binary data)."""
        return [1 - jaccard_score(self.vector, m, average='binary') for m in self.matrix]
    
    def weighted_euclidean_distance(self, weights):
        """Weighted Euclidean Distance between vector and each row of the matrix."""
        weights = np.array(weights)
        return np.sqrt(np.sum(weights * (self.matrix - self.vector) ** 2, axis=1))
    
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
    

# Example usage:
vector = [1, 2, 3]
matrix = [
    [1, 1, 1],
    [2, 2, 2],
    [4, 5, 6]
]

# Initialize the DistanceMetrics class
dist = DistanceMetrics(vector, matrix)

# Compute distances
euclidean = dist.euclidean_distance()
manhattan = dist.manhattan_distance()
cosine = dist.cosine_distance()
mahalanobis = dist.mahalanobis_distance()
chebyshev = dist.chebyshev_distance()

# Print results
print(f"Euclidean Distance: {euclidean}")
print(f"Manhattan Distance: {manhattan}")
print(f"Cosine Distance: {cosine}")
print(f"Mahalanobis Distance: {mahalanobis}")
print(f"Chebyshev Distance: {chebyshev}")

# For distribution-based measures
P = np.array([0.2, 0.5, 0.3])  # Example distribution P
Q = np.array([0.3, 0.4, 0.3])  # Example distribution Q

kl_divergence = dist.kullback_leibler_divergence(P, Q)
wasserstein = dist.wasserstein_distance(P, Q)
pearson_corr = dist.pearson_correlation()

print(f"Kullback-Leibler Divergence: {kl_divergence}")
print(f"Wasserstein Distance: {wasserstein}")
print(f"Pearson Correlation: {pearson_corr}")
