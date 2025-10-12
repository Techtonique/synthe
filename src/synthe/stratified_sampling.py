import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from enum import Enum


class ClusterMethod(Enum):
    GMM = "gmm"
    KMEANS = "kmeans"


class StratifiedClusteringSubsampling:
    def __init__(
        self,
        n_components=3,
        method=ClusterMethod.GMM,
        random_state=None,
        **kwargs,
    ):
        """
        Initializes the StratifiedClusteringSubsampling class.

        :param n_components: Number of clusters for clustering algorithm. Default is 3.
        :param method: Cluster method - 'gmm' or 'kmeans'. Default is GMM.
        :param random_state: Seed for random number generator.
        :param kwargs: Additional parameters for the clustering algorithms.
        """
        self.n_components = n_components
        self.method = (
            method
            if isinstance(method, ClusterMethod)
            else ClusterMethod(method.lower())
        )
        self.random_state = random_state
        self.kwargs = kwargs

        # Initialize the clustering model based on the chosen method
        if self.method == ClusterMethod.GMM:
            self.cluster_model = GaussianMixture(
                n_components=self.n_components,
                random_state=self.random_state,
                **kwargs,
            )
        elif self.method == ClusterMethod.KMEANS:
            self.cluster_model = KMeans(
                n_clusters=self.n_components,
                random_state=self.random_state,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported method: {method}. Choose 'gmm' or 'kmeans'."
            )

    def fit(self, data):
        """
        Fit the clustering model to the given 2D data.

        :param data: 2D numpy array where each row is a data point and each column is a feature.
        """
        # Input validation
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array")
        if data.ndim != 2:
            raise ValueError("Data must be 2-dimensional")
        if len(data) < self.n_components:
            raise ValueError(
                f"Number of samples ({len(data)}) must be >= n_components ({self.n_components})"
            )

        self.cluster_model.fit(data)

        # Get cluster labels based on the method
        if self.method == ClusterMethod.GMM:
            self.cluster_labels = self.cluster_model.predict(data)
        else:  # KMEANS
            self.cluster_labels = self.cluster_model.labels_

        return self

    def stratified_sample(self, data, test_size=0.3):
        """
        Perform stratified sampling based on cluster labels.

        :param data: 2D numpy array to sample from.
        :param test_size: Proportion of data to be used for testing (default is 30%).
        :return: Tuple of (train_data, test_data) where each is a stratified sample.
        """
        if not hasattr(self, "cluster_labels"):
            raise ValueError("Must call fit() before stratified_sample()")
        if len(data) != len(self.cluster_labels):
            raise ValueError(
                "Data length must match fitted cluster labels length"
            )
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=self.random_state
        )

        for train_index, test_index in sss.split(data, self.cluster_labels):
            train_data = data[train_index]
            test_data = data[test_index]

        return train_data, test_data

    def get_cluster_labels(self):
        """
        Get the cluster labels assigned by the clustering model.

        :return: 1D numpy array of cluster labels for each data point.
        """
        if not hasattr(self, "cluster_labels"):
            raise ValueError("Must call fit() first")
        return self.cluster_labels

    def predict(self, data):
        """
        Predict the cluster labels for new data using the fitted model.

        :param data: 2D numpy array of new data points.
        :return: Cluster labels for the new data points.
        """
        if self.method == ClusterMethod.GMM:
            return self.cluster_model.predict(data)
        else:  # KMEANS
            return self.cluster_model.predict(data)

    def get_cluster_centers(self):
        """Get the cluster centers."""
        if self.method == ClusterMethod.GMM:
            if not hasattr(self.cluster_model, "means_"):
                raise ValueError("GMM not fitted yet")
            return self.cluster_model.means_
        else:  # KMEANS
            if not hasattr(self.cluster_model, "cluster_centers_"):
                raise ValueError("KMeans not fitted yet")
            return self.cluster_model.cluster_centers_

    def get_cluster_proportions(self):
        """Get the proportion of data points in each cluster."""
        if not hasattr(self, "cluster_labels"):
            raise ValueError("Must call fit() first")
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        return counts / len(self.cluster_labels)

    def score_samples(self, data):
        """
        Get the model scores for samples.
        For GMM: log-likelihood of samples
        For KMeans: negative of inertia (distance to closest cluster center)
        """
        if self.method == ClusterMethod.GMM:
            return self.cluster_model.score_samples(data)
        else:
            # For KMeans, return negative distances to cluster centers
            return -self.cluster_model.transform(data).min(axis=1)

    def get_model_params(self):
        """Get the parameters of the fitted model."""
        return self.cluster_model.get_params()

    def set_method(self, method):
        """
        Change the clustering method after initialization (will require re-fitting).

        :param method: New cluster method ('gmm' or 'kmeans')
        """
        old_method = self.method
        self.method = (
            method
            if isinstance(method, ClusterMethod)
            else ClusterMethod(method.lower())
        )

        if self.method != old_method:
            # Reinitialize the model with the new method
            if self.method == ClusterMethod.GMM:
                self.cluster_model = GaussianMixture(
                    n_components=self.n_components,
                    random_state=self.random_state,
                    **self.kwargs,
                )
            else:  # KMEANS
                self.cluster_model = KMeans(
                    n_clusters=self.n_components,
                    random_state=self.random_state,
                    **self.kwargs,
                )

            # Remove fitted attributes to force re-fitting
            if hasattr(self, "cluster_labels"):
                del self.cluster_labels
