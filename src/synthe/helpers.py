# Authors: Thierry Moudiki
#
# License: BSD 3 Clear

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

def qqplot_with_summary(data1, data2, label1="Sample 1", label2="Sample 2"):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    
    # Remove NaN values
    data1 = data1[~np.isnan(data1)]
    data2 = data2[~np.isnan(data2)]
    
    # Q–Q plot
    n_quantiles = min(len(data1), len(data2))
    quantiles1 = np.percentile(data1, np.linspace(0, 100, n_quantiles))
    quantiles2 = np.percentile(data2, np.linspace(0, 100, n_quantiles))
    
    plt.figure(figsize=(6,6))
    plt.scatter(quantiles1, quantiles2, alpha=0.7)
    min_val = min(quantiles1.min(), quantiles2.min())
    max_val = max(quantiles1.max(), quantiles2.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
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
    summary = pd.DataFrame({
        "Statistic": [
            "Sample size", 
            "Mean", 
            "Std. deviation", 
            "KS statistic", 
            "KS p-value",
            "AD statistic",
            "AD p-value",
            "Quantile correlation"
        ],
        label1: [n1, mean1, std1, ks_stat, ks_p, ad_stat, ad_p, corr],
        label2: [n2, mean2, std2, "", "", "", "", ""]
    })
    
    return summary


# flatten list of lists
# flatten = lambda l: [item for sublist in l for item in sublist]
def flatten(x):
    res = []

    for elt1 in x:
        try:
            for elt2 in elt1:
                res.append(elt2)

        except:
            res.append(elt1)

    return res


# check if the response contains only integers
def is_factor(y):
    """
    Determine if the target variable `y` is for classification (True) or regression (False).

    Parameters:
    y : array-like
        Target variable (labels/response variable)

    Returns:
    bool
        True if `y` is categorical (classification), False if numeric (regression)
    """
    y_array = np.asarray(y)

    # Boolean → classification
    if y_array.dtype == bool:
        return True

    # Strings or objects → classification
    if y_array.dtype.kind in ["U", "S", "O"]:
        return True

    # Numeric types (int/float) with few unique values → classification
    if y_array.dtype.kind in ["i", "u", "f"]:
        unique_values = np.unique(y_array[~np.isnan(y_array)])  # Exclude NaNs
        if len(unique_values) <= 10:  # Threshold for classification
            return True

    # Otherwise → regression
    return False


def subsample_indices(
    data,
    method="random",
    fraction=0.1,
    label_col=None,
    n_components=10,
    eps=0.5,
    min_samples=5,
    random_state=42,
):
    """
    Subsample row indices from a dataset while preserving statistical properties.

    Parameters:
        data (pd.DataFrame or np.ndarray): The input data (Pandas DataFrame or NumPy array).
        method (str): Sampling method. Options: 'random', 'stratified', 'gmm', 'dbscan', 'optimal_transport'.
        fraction (float): Fraction of rows to sample (e.g., 0.1 for 10%).
        label_col (str or int): Column name/index for stratified sampling (required if method='stratified').
                                For NumPy arrays, provide the column index.
        n_components (int): Number of components for GMM-based sampling (used if method='gmm').
        eps (float): The maximum distance between two samples for them to be considered in the same neighborhood (used if method='dbscan').
        min_samples (int): The number of samples in a neighborhood for a point to be considered a core point (used if method='dbscan').
        random_state (int): Random seed for reproducibility.

    Returns:
        list: Indices of the subsampled rows.
    """
    # Check if the input is a Pandas DataFrame or NumPy array
    if isinstance(data, pd.DataFrame):
        is_dataframe = True
        if label_col is not None and label_col not in data.columns:
            raise ValueError(
                f"Column '{label_col}' does not exist in the DataFrame."
            )
        X = data.drop(columns=[label_col] if label_col else []).values
        labels = data[label_col].values if label_col else None
    elif isinstance(data, np.ndarray):
        is_dataframe = False
        if label_col is not None:
            if (
                not isinstance(label_col, int)
                or label_col < 0
                or label_col >= data.shape[1]
            ):
                raise ValueError(
                    "For NumPy arrays, 'label_col' must be a valid column index."
                )
            labels = data[:, label_col]
            X = np.delete(data, label_col, axis=1)
        else:
            X = data
            labels = None
    else:
        raise TypeError(
            "Input data must be a Pandas DataFrame or a NumPy array."
        )

    if method == "random":
        # Simple random sampling
        rng = np.random.default_rng(random_state)
        subsample_indices = rng.choice(
            data.shape[0], size=int(data.shape[0] * fraction), replace=False
        )
        return subsample_indices.tolist()

    elif method == "stratified":
        if labels is None:
            raise ValueError(
                "For stratified sampling, 'label_col' must be specified."
            )
        # Stratified sampling
        _, _, _, subsample_indices = train_test_split(
            np.arange(data.shape[0]),
            labels,
            test_size=fraction,
            stratify=labels,
            random_state=random_state,
        )
        return subsample_indices.tolist()

    elif method == "gmm":
        # GMM-based sampling
        gmm = GaussianMixture(
            n_components=n_components, random_state=random_state
        ).fit(X)

        # Assign each row to a cluster
        clusters = gmm.predict(X)

        # Sample proportionally from each cluster
        rng = np.random.default_rng(random_state)
        subsample_indices = []
        for cluster in np.unique(clusters):
            cluster_indices = np.where(clusters == cluster)[0]
            subsample_size = max(1, int(len(cluster_indices) * fraction))
            subsample_indices.extend(
                rng.choice(cluster_indices, size=subsample_size, replace=False)
            )
        return subsample_indices

    elif method == "dbscan":
        # DBSCAN-based sampling
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

        # Assign each row to a cluster (or -1 for noise)
        clusters = dbscan.labels_

        # Exclude noise points (-1) and sample proportionally from each cluster
        valid_clusters = np.where(clusters != -1)[0]
        if len(valid_clusters) == 0:
            raise ValueError(
                "DBSCAN identified only noise points. Adjust `eps` or `min_samples`."
            )

        rng = np.random.default_rng(random_state)
        subsample_indices = []
        for cluster in np.unique(clusters[valid_clusters]):
            cluster_indices = np.where(clusters == cluster)[0]
            subsample_size = max(1, int(len(cluster_indices) * fraction))
            subsample_indices.extend(
                rng.choice(cluster_indices, size=subsample_size, replace=False)
            )
        return subsample_indices

    elif method == "optimal_transport":
        # Optimal transport-based sampling
        distances = cdist(X, X, metric="euclidean")

        # Solve the optimal transport problem
        row_ind, col_ind = linear_sum_assignment(distances)

        # Select the subsample based on the solution
        subsample_size = int(data.shape[0] * fraction)
        subsample_indices = col_ind[:subsample_size]
        return subsample_indices.tolist()

    else:
        raise ValueError(
            "Invalid sampling method. Choose from 'random', 'stratified', 'gmm', 'dbscan', or 'optimal_transport'."
        )


# Example usage:
# df = pd.DataFrame({'feature1': np.random.randn(1000), 'feature2': np.random.randn(1000), 'label': np.random.choice([0, 1], size=1000)})
# indices = subsample_indices(df, method='dbscan', fraction=0.1, eps=0.5, min_samples=5)
# print(indices)


def dosubsample(y, row_sample=0.8, seed=123, n_jobs=None, verbose=False):
    index = []
    assert (row_sample < 1) & (
        row_sample >= 0
    ), "'row_sample' must be < 1 and >= 0"
    n_obs = len(y)
    n_obs_out = np.ceil(n_obs * row_sample)

    # preproc -----
    if is_factor(y):  # classification

        classes, n_elem_classes = np.unique(y, return_counts=True)
        n_classes = len(classes)
        y_as_classes = y.copy()
        freqs_hist = np.zeros_like(n_elem_classes, dtype=float)
        if verbose is True:
            print(f"creating breaks...")
        if n_jobs is None:
            for i in range(len(n_elem_classes)):
                freqs_hist[i] = float(n_elem_classes[i]) / n_obs
        else:

            def get_freqs_hist(i):
                return float(n_elem_classes[i]) / n_obs

            freqs_hist = Parallel(n_jobs=n_jobs)(
                delayed(get_freqs_hist)(i) for i in range(len(n_elem_classes))
            )

    else:  # regression

        h = np.histogram(y, bins="auto")
        n_elem_classes = np.asarray(h[0], dtype=np.int32)
        freqs_hist = np.zeros_like(n_elem_classes, dtype=float)

        if verbose is True:
            print(f"creating breaks...")
        if n_jobs is None:
            for i in range(len(n_elem_classes)):
                freqs_hist[i] = float(n_elem_classes[i]) / n_obs
        else:

            def get_freqs_hist(i):
                return float(n_elem_classes[i]) / n_obs

            freqs_hist = Parallel(n_jobs=n_jobs)(
                delayed(get_freqs_hist)(i) for i in range(len(n_elem_classes))
            )

        breaks = h[1]
        n_breaks_1 = len(breaks) - 1
        classes = range(n_breaks_1)
        n_classes = n_breaks_1
        y_as_classes = np.zeros_like(y, dtype=int)

        for i in classes:
            y_as_classes[(y > breaks[i]) * (y <= breaks[i + 1])] = int(i)

    # main loop ----

    if verbose is True:
        print(f"main loop...")

    if n_jobs is None:

        if verbose is True:
            iterator = tqdm(range(n_classes))
        else:
            iterator = range(n_classes)

        for i in iterator:
            bool_class_i = y_as_classes == classes[i]
            index_class_i = np.asarray(
                np.where(bool_class_i == True)[0], dtype=np.int32
            )
            if np.sum(bool_class_i) > 1:  # at least 2 elements in class  #i
                np.random.seed(seed + i)
                index.extend(
                    np.random.choice(
                        index_class_i,
                        size=int(n_obs_out * freqs_hist[i]),  # output size
                        replace=True,
                    ).tolist()
                )
            else:  # only one element in class
                try:
                    index.append(index_class_i[0])
                except:
                    pass

    else:  # parallel execution

        def get_index(i):
            bool_class_i = y_as_classes == classes[i]
            index_class_i = np.asarray(
                np.where(bool_class_i == True)[0], dtype=np.int32
            )
            if np.sum(bool_class_i) > 1:  # at least 2 elements in class  #i
                np.random.seed(seed + i)
                return np.random.choice(
                    index_class_i,
                    size=int(n_obs_out * freqs_hist[i]),  # output size
                    replace=True,
                ).tolist()
            else:  # only one element in class
                try:
                    return index_class_i[0]
                except:
                    pass

        if verbose is True:
            index = Parallel(n_jobs=n_jobs)(
                delayed(get_index)(i) for i in tqdm(range(n_classes))
            )
        else:
            index = Parallel(n_jobs=n_jobs)(
                delayed(get_index)(i) for i in range(n_classes)
            )

    try:
        return np.asarray(flatten(index))
    except:
        return np.asarray(index)
