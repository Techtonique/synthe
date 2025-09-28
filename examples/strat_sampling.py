# Example usage and comparison
def example_usage():
    """Demonstrate usage with both GMM and KMeans."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    data = np.random.randn(n_samples, 2)
    
    print("=== Stratified Clustering Subsampling Demo ===\n")
    
    # Test with GMM
    print("1. Using Gaussian Mixture Model (GMM):")
    sgmm = StratifiedClusteringSubsampling(
        n_components=3, 
        method=ClusterMethod.GMM,
        random_state=42,
        covariance_type='diag'  # GMM-specific parameter
    )
    sgmm.fit(data)
    train_gmm, test_gmm = sgmm.stratified_sample(data, test_size=0.2)
    
    print(f"   Original data shape: {data.shape}")
    print(f"   Train data shape: {train_gmm.shape}")
    print(f"   Test data shape: {test_gmm.shape}")
    print(f"   Cluster proportions: {sgmm.get_cluster_proportions()}")
    print(f"   Cluster centers shape: {sgmm.get_cluster_centers().shape}\n")
    
    # Test with KMeans
    print("2. Using KMeans:")
    skmeans = StratifiedClusteringSubsampling(
        n_components=3,
        method=ClusterMethod.KMEANS,
        random_state=42,
        init='k-means++',  # KMeans-specific parameter
        n_init=10
    )
    skmeans.fit(data)
    train_kmeans, test_kmeans = skmeans.stratified_sample(data, test_size=0.2)
    
    print(f"   Original data shape: {data.shape}")
    print(f"   Train data shape: {train_kmeans.shape}")
    print(f"   Test data shape: {test_kmeans.shape}")
    print(f"   Cluster proportions: {skmeans.get_cluster_proportions()}")
    print(f"   Cluster centers shape: {skmeans.get_cluster_centers().shape}\n")
    
    # Compare cluster assignments
    gmm_labels = sgmm.get_cluster_labels()
    kmeans_labels = skmeans.get_cluster_labels()
    
    print("3. Comparison:")
    print(f"   GMM unique labels: {np.unique(gmm_labels)}")
    print(f"   KMeans unique labels: {np.unique(kmeans_labels)}")
    print(f"   Jaccard similarity between cluster assignments: "
          f"{len(set(gmm_labels) & set(kmeans_labels)) / len(set(gmm_labels) | set(kmeans_labels)):.3f}")
    
    return (train_gmm, test_gmm), (train_kmeans, test_kmeans)

# Simple function for quick usage
def create_stratified_sampler(n_components=3, method='gmm', **kwargs):
    """
    Convenience function to create a stratified sampler.
    
    :param n_components: Number of clusters
    :param method: 'gmm' or 'kmeans'
    :param kwargs: Additional parameters for the clustering algorithm
    :return: StratifiedClusteringSubsampling instance
    """
    return StratifiedClusteringSubsampling(
        n_components=n_components,
        method=method,
        **kwargs
    )
