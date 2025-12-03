from sklearn.datasets import (
    make_classification,
    make_regression,
    make_multilabel_classification,
)
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd


class SyntheticTabularSampler:
    """
    A class to generate synthetic tabular datasets for various machine learning tasks.

    This class provides methods to create synthetic datasets for classification,
    regression, multi-output regression, and multi-label classification problems
    using scikit-learn's data generation functions.

    Parameters
    ----------
    random_state : int, optional
        Seed for the random number generator to ensure reproducibility.
        Defaults to 42.
    type : str, optional
        The type of synthetic data to generate.
        Must be one of "classification", "regression", "multioutput_regression",
        or "multilabel_classification". Defaults to "classification".

    Attributes
    ----------
    random_state : int
        The seed used for the random number generator.
    rng : numpy.random.Generator
        The NumPy random number generator instance.
    type : str
        The specified type of synthetic data to generate.
    """

    def __init__(self, random_state: int = 42, type: str = "classification"):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.type = type

    # ============================================================
    # INTERNAL COMPONENTS
    # ============================================================

    # ---- Classification
    def _gen_classification(self):
        n_features = self.rng.integers(5, 120)
        # Ensure n_informative + n_redundant + n_repeated < n_features
        max_informative = min(
            30, n_features - 2
        )  # Leave room for at least 1 useless feature
        n_informative = self.rng.integers(2, max_informative + 1)
        max_redundant = min(10, n_features - n_informative - 1)
        n_redundant = (
            self.rng.integers(0, max_redundant + 1) if max_redundant > 0 else 0
        )

        cfg = dict(
            n_samples=500,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=0,
            n_classes=int(self.rng.choice([2, 3, 4, 5, 6])),
            n_clusters_per_class=self.rng.integers(1, 4),
            class_sep=float(self.rng.uniform(0.5, 5.0)),
            flip_y=float(self.rng.uniform(0.0, 0.1)),
            random_state=self.rng.integers(0, 10_000),
        )
        X, y = make_classification(**cfg)
        return X, y, cfg, "classification"

    # ---- Regression
    def _gen_regression(self):
        cfg = dict(
            n_samples=500,
            n_features=self.rng.integers(5, 120),
            n_informative=self.rng.integers(2, 40),
            noise=float(self.rng.uniform(0.1, 25)),
            bias=float(self.rng.uniform(-10, 10)),
            effective_rank=(
                None if self.rng.random() < 0.5 else self.rng.integers(2, 10)
            ),
            tail_strength=float(self.rng.uniform(0.0, 1.0)),
            random_state=self.rng.integers(0, 10_000),
        )
        X, y = make_regression(**cfg)
        return X, y, cfg, "regression"

    # ---- Multi-output regression
    def _gen_multioutput_regression(self):
        n_targets = self.rng.integers(2, 6)
        cfg = dict(
            n_samples=500,
            n_features=self.rng.integers(5, 60),
            n_informative=self.rng.integers(2, 20),
            noise=float(self.rng.uniform(0.1, 10)),
            random_state=self.rng.integers(0, 10_000),
        )
        X, y = make_regression(**cfg)
        # reshape to multi-target
        y_multi = np.stack(
            [
                y + self.rng.normal(0, cfg["noise"], size=len(y))
                for _ in range(n_targets)
            ],
            axis=1,
        )
        return X, y_multi, cfg, "multioutput_regression"

    # ---- Multi-label classification
    def _gen_multilabel_classification(self):
        cfg = dict(
            n_samples=500,
            n_features=self.rng.integers(5, 80),
            n_classes=self.rng.integers(3, 10),
            n_labels=self.rng.integers(1, 5),
            length=self.rng.integers(20, 100),
            allow_unlabeled=False,
            random_state=self.rng.integers(0, 10_000),
        )
        X, y = make_multilabel_classification(**cfg)
        return X, y, cfg, "multilabel_classification"

    # ---- Nonlinear regression (sinusoidal / polynomial)
    def _gen_nonlinear_regression(self):
        n_features = self.rng.integers(3, 10)
        X = self.rng.normal(size=(500, n_features))

        # Nonlinear target
        y = (
            np.sin(X[:, 0] * self.rng.uniform(1, 5))
            + X[:, 1] ** 2 * self.rng.uniform(0.5, 2.0)
            + np.tanh(X[:, 2] * self.rng.uniform(1, 3))
            + self.rng.normal(0, 0.3, size=500)
        )

        cfg = {"type": "nonlinear_regression", "n_features": n_features}
        return X, y, cfg, "nonlinear_regression"

    # ---- Polynomial interactions
    def _gen_polynomial_features(self):
        base_features = self.rng.integers(3, 8)
        degree = self.rng.integers(2, 4)

        X = self.rng.normal(size=(500, base_features))
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)

        # regression target
        coef = self.rng.normal(0, 1, size=X_poly.shape[1])
        y = X_poly @ coef + self.rng.normal(0, 0.3, 500)

        cfg = {
            "base_features": base_features,
            "degree": degree,
            "expanded_dim": X_poly.shape[1],
        }
        return X_poly, y, cfg, "polynomial_regression"

    # ---- Sparse high-dimensional regression
    def _gen_sparse_regression(self):
        n_features = self.rng.integers(500, 2000)
        X = self.rng.normal(size=(500, n_features))

        # sparse coefficients
        coef = np.zeros(n_features)
        k = self.rng.integers(5, 20)
        idx = self.rng.choice(n_features, size=k, replace=False)
        coef[idx] = self.rng.normal(0, 5, size=k)

        y = X @ coef + self.rng.normal(0, 0.2, size=500)

        cfg = {"n_features": n_features, "nonzero": k}
        return X, y, cfg, "sparse_regression"

    # ---- Time-series-like AR regression
    def _gen_ar_regression(self):
        n_features = self.rng.integers(3, 10)
        X = self.rng.normal(size=(500, n_features))

        # AR(3)-like structure for y
        y = np.zeros(500)
        for t in range(3, 500):
            y[t] = (
                0.6 * y[t - 1]
                - 0.2 * y[t - 2]
                + 0.1 * y[t - 3]
                + X[t] @ self.rng.normal(0, 1, n_features)
                + self.rng.normal(0, 0.5)
            )

        cfg = {"n_features": n_features, "AR_order": 3}
        return X, y, cfg, "autoregressive_regression"

    # ---- Mixture categorical + numerical
    def _gen_mixed_features(self):
        n_num = self.rng.integers(3, 10)
        n_cat = self.rng.integers(1, 5)

        X_num = self.rng.normal(size=(500, n_num))
        X_cat = self.rng.integers(0, 5, size=(500, n_cat))

        X = np.concatenate([X_num, X_cat], axis=1)

        coef = self.rng.normal(0, 1, n_num)
        y = X_num @ coef + self.rng.normal(0, 1, 500)

        cfg = {"numerical": n_num, "categorical": n_cat}
        return X, y, cfg, "mixed_regression"

    # ---- Simple causal DAG-like dataset
    def _gen_causal_style(self):
        X1 = self.rng.normal(size=500)
        X2 = 2 * X1 + self.rng.normal(0, 0.2, 500)
        X3 = -0.7 * X1 + self.rng.normal(0, 0.2, 500)
        X4 = 1.5 * X2 + X3 + self.rng.normal(0, 0.2, 500)
        X = np.column_stack([X1, X2, X3, X4])

        y = 3 * X4 + self.rng.normal(0, 1, 500)

        cfg = {"DAG": "X1→X2/X3→X4→y"}
        return X, y, cfg, "causal_regression"

    # ============================================================
    # PUBLIC METHOD
    # ============================================================
    def sample(self, n_sets: int = 10):
        """
        Generate n_sets diverse datasets with n_samples=500.
        """
        if self.type == "classification":
            generators = [self._gen_classification]
        elif self.type == "regression":
            generators = [
                self._gen_regression,
                self._gen_nonlinear_regression,
                self._gen_sparse_regression,
                self._gen_ar_regression,
            ]
        else:
            raise ValueError(f"Unknown type: {self.type}")

        datasets = []
        for _ in range(n_sets):
            gen = self.rng.choice(generators)
            X, y, cfg, task = gen()

            X_df = pd.DataFrame(
                X, columns=[f"feature_{i}" for i in range(X.shape[1])]
            )
            y_df = (
                pd.DataFrame(y) if y.ndim > 1 else pd.Series(y, name="target")
            )

            datasets.append(
                {
                    "task": task,
                    "config": cfg,
                    "X": X_df,
                    "y": y_df,
                }
            )

        return datasets
