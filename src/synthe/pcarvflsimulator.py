"""
pca_rvfl_simulator.py
=====================
Self-contained PCARVFLSimulator: a GAN-like tabular data synthesiser that
uses PCA scores as the RVFL input and bootstrap residuals as the noise model.

Algorithm
---------
1. StandardScaler normalises Y internally (optional, on by default)
2. Fit PCA on scaled Y  →  Z  (PCA scores; latent representation)
3. Fit RVFL:  Z_train → Y_train_scaled  (closed-form Ridge on random features)
4. Residuals ε = Y_train_scaled - RVFL(Z_train)
5. Sample: draw Z rows from training pool, predict, add bootstrap ε,
           inverse-transform back to original scale

Optuna tunes: n_nodes (50–1000, log) and alpha (1e-5–10, log).

Key design notes
----------------
- Internal StandardScaler (scale=True by default) makes the simulator
  robust to raw, unscaled data.  PCA and residuals both operate on the
  standardised space; samples are inverse-transformed before being returned.
- MMD uses the *biased* estimator (Kxx.mean + Kyy.mean - 2*Kxy.mean),
  which is always ≥ 0.  The unbiased estimator subtracts diagonal terms
  and goes negative for small n — avoid it for optimisation and reporting.

Adequacy tests  (adequacy_report)
----------------------------------
  Distributional distance
    MMD        Biased MMD² (RBF kernel, ≥ 0)           ↓ better
    Energy     Energy distance                          ↓ better

  Per-feature univariate tests
    KS_stat    Mean KS statistic across features        ↓ better
    KS_p_bonf  Bonferroni combined p-value              ↑ better (want > 0.05)
    KS_reject  Fraction of features H₀ rejected         ↓ better (want 0)
    AD_stat    Mean Anderson-Darling statistic           ↓ better
    AD_p_bonf  Bonferroni combined p-value               ↑ better
    AD_reject  Fraction of features H₀ rejected          ↓ better

  Moment matching
    mean_mae   MAE of per-feature means                 ↓ better
    std_mae    MAE of per-feature std devs               ↓ better
    std_ratio  Mean synth std / mean real std            want ≈ 1
    corr_frob  Frobenius norm of ΔPearson corr matrix    ↓ better

  Random-projection sweep
    KS_proj    Mean KS stat over 50 random projections  ↓ better
               Catches inter-feature dependency shifts
               that per-feature marginal tests miss.

Dependencies
------------
    pip install numpy scikit-learn optuna scipy
"""

# ── standard library ─────────────────────────────────────────────────────────
import warnings

warnings.filterwarnings("ignore")

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp, anderson_ksamp


# ─────────────────────────────────────────────────────────────────────────────
# Distance / divergence helpers
# ─────────────────────────────────────────────────────────────────────────────


def _rbf_kernel(
    X: np.ndarray, Y: np.ndarray, sigma: float | None = None
) -> np.ndarray:
    XX = np.sum(X**2, axis=1, keepdims=True)
    YY = np.sum(Y**2, axis=1, keepdims=True)
    D2 = XX + YY.T - 2 * X @ Y.T
    if sigma is None:
        sigma = float(np.median(np.sqrt(np.maximum(D2, 0)))) + 1e-8
    return np.exp(-D2 / (2 * sigma**2))


def mmd_biased(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Biased MMD² estimator — always ≥ 0.

    mmd² = E[k(x,x')] + E[k(y,y')] - 2·E[k(x,y)]

    The unbiased version zeros out diagonal entries of Kxx and Kyy,
    which introduces negative bias for small n and can yield mmd < 0.
    The biased version does not zero diagonals and is always non-negative.
    """
    Kxx = _rbf_kernel(X, X)
    Kyy = _rbf_kernel(Y, Y)
    Kxy = _rbf_kernel(X, Y)
    return float(max(0.0, Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()))


def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Energy distance between two empirical distributions."""

    def _md(A, B):
        d = A[:, None, :] - B[None, :, :]
        return float(np.sqrt((d**2).sum(-1)).mean())

    return float(max(0.0, 2 * _md(X, Y) - _md(X, X) - _md(Y, Y)))


# ─────────────────────────────────────────────────────────────────────────────
# Adequacy report
# ─────────────────────────────────────────────────────────────────────────────


def adequacy_report(
    X_real: np.ndarray,
    X_syn: np.ndarray,
    n_proj: int = 50,
    alpha: float = 0.05,
    cap: int = 500,
    random_state: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Comprehensive adequacy report comparing real and synthetic samples.

    Parameters
    ----------
    X_real       : (n, d) real data — in *original* (unscaled) space
    X_syn        : (m, d) synthetic data — in *original* space
    n_proj       : random projections for the projection sweep (default 50)
    alpha        : significance level for hypothesis tests (default 0.05)
    cap          : max samples for O(n²) metrics MMD / energy (default 500)
    random_state : RNG seed
    verbose      : print formatted summary

    Returns
    -------
    dict of scalar metrics (plus raw per-feature arrays prefixed with '_')
    """
    X_real = np.asarray(X_real, dtype=float)
    X_syn = np.asarray(X_syn, dtype=float)
    if X_real.shape[1] != X_syn.shape[1]:
        raise ValueError(
            "X_real and X_syn must have the same number of features."
        )
    d = X_real.shape[1]

    rng = np.random.RandomState(random_state)

    # Standardise both sets with real-data statistics before computing
    # distance metrics — keeps MMD / Energy on a comparable scale regardless
    # of raw feature magnitudes.
    sc = StandardScaler().fit(X_real)
    Xr_s = sc.transform(X_real)
    Xs_s = sc.transform(X_syn)

    # ── cap samples for O(n²) metrics ────────────────────────────────────────
    def _sub(A):
        return A[rng.choice(len(A), cap, replace=False)] if len(A) > cap else A

    Xr_cap = _sub(Xr_s)
    Xs_cap = _sub(Xs_s)

    # ── distributional distances (on standardised data) ───────────────────────
    mmd_val = mmd_biased(Xr_cap, Xs_cap)
    energy_val = energy_distance(Xr_cap, Xs_cap)

    # ── per-feature tests (on original scale — easier to interpret) ───────────
    ks_stats, ks_pvals, ad_stats, ad_pvals = [], [], [], []
    for j in range(d):
        r = ks_2samp(X_real[:, j], X_syn[:, j])
        ks_stats.append(r.statistic)
        ks_pvals.append(r.pvalue)
        try:
            r2 = anderson_ksamp([X_real[:, j], X_syn[:, j]])
            ad_stats.append(r2.statistic)
            ad_pvals.append(r2.significance_level)
        except Exception:
            ad_stats.append(np.nan)
            ad_pvals.append(np.nan)

    ks_stats = np.array(ks_stats)
    ks_pvals = np.array(ks_pvals)
    ad_stats = np.array(ad_stats)
    ad_pvals = np.array(ad_pvals)

    ks_bonf = float(np.minimum(d * ks_pvals.min(), 1.0))
    ks_reject = float((ks_pvals < alpha).mean())

    valid_adp = ad_pvals[~np.isnan(ad_pvals)]
    ad_bonf = (
        float(np.minimum(d * valid_adp.min(), 1.0))
        if len(valid_adp)
        else np.nan
    )
    ad_reject = float((valid_adp < alpha).mean()) if len(valid_adp) else np.nan

    # ── moment matching (original scale) ─────────────────────────────────────
    mean_real = X_real.mean(0)
    mean_syn = X_syn.mean(0)
    std_real = X_real.std(0)
    std_syn = X_syn.std(0)

    mean_mae = float(np.abs(mean_real - mean_syn).mean())
    std_mae = float(np.abs(std_real - std_syn).mean())

    nz = std_real > 1e-10
    std_ratio = (
        float(std_syn[nz].mean() / std_real[nz].mean()) if nz.any() else np.nan
    )

    # Pearson correlation Frobenius distance (standardised space for fairness)
    nz2 = (Xr_s.std(0) > 1e-10) & (Xs_s.std(0) > 1e-10)
    if nz2.sum() > 1:
        Cr = np.corrcoef(Xr_s[:, nz2].T)
        Cs = np.corrcoef(Xs_s[:, nz2].T)
        corr_frob = float(np.linalg.norm(Cr - Cs, "fro"))
    else:
        corr_frob = np.nan

    # ── random-projection sweep (standardised) ────────────────────────────────
    proj_ks = []
    for _ in range(n_proj):
        v = rng.randn(d)
        v /= np.linalg.norm(v) + 1e-12
        proj_ks.append(ks_2samp(Xr_s @ v, Xs_s @ v).statistic)
    ks_proj = float(np.mean(proj_ks))

    # ── assemble ──────────────────────────────────────────────────────────────
    report = dict(
        MMD=mmd_val,
        Energy=energy_val,
        KS_stat=float(ks_stats.mean()),
        KS_p_bonf=ks_bonf,
        KS_reject=ks_reject,
        AD_stat=float(np.nanmean(ad_stats)),
        AD_p_bonf=ad_bonf,
        AD_reject=ad_reject,
        mean_mae=mean_mae,
        std_mae=std_mae,
        std_ratio=std_ratio,
        corr_frob=corr_frob,
        KS_proj=ks_proj,
        # raw arrays for downstream plotting
        _ks_per_feature=ks_stats,
        _ks_pvals=ks_pvals,
        _ad_per_feature=ad_stats,
        _ad_pvals=ad_pvals,
        n_real=len(X_real),
        n_syn=len(X_syn),
        d=d,
    )

    if verbose:
        _print_report(report, alpha, n_proj)

    return report


def _print_report(r: dict, alpha: float, n_proj: int) -> None:
    W = 56

    def _row(label, value, fmt=".5f", note=""):
        v = (
            "   n/a"
            if (isinstance(value, float) and np.isnan(value))
            else f"{value:{fmt}}"
        )
        print(f"  {label:<24} {v:>10}  {note}")

    def _ok(p):
        return "✓ ok" if (not np.isnan(p) and p > alpha) else "✗ reject"

    print(f"\n{'─' * W}")
    print(
        f"  Adequacy Report  "
        f"(n_real={r['n_real']}, n_syn={r['n_syn']}, d={r['d']})"
    )
    print(f"{'─' * W}")
    print("  ── Distributional distance  [standardised space]")
    _row("MMD (biased, ≥0)", r["MMD"])
    _row("Energy distance", r["Energy"])
    print("  ── Per-feature KS tests  [original scale]")
    _row("KS stat (mean ↓)", r["KS_stat"])
    _row("KS p Bonferroni ↑", r["KS_p_bonf"], note=_ok(r["KS_p_bonf"]))
    _row("KS reject rate ↓", r["KS_reject"], fmt=".3f", note=f"(α={alpha})")
    print("  ── Per-feature AD tests  [original scale]")
    _row("AD stat (mean ↓)", r["AD_stat"])
    _row("AD p Bonferroni ↑", r["AD_p_bonf"], note=_ok(r["AD_p_bonf"]))
    _row("AD reject rate ↓", r["AD_reject"], fmt=".3f", note=f"(α={alpha})")
    print("  ── Moment matching  [original scale]")
    _row("Mean MAE ↓", r["mean_mae"])
    _row("Std  MAE ↓", r["std_mae"])
    _row("Std  ratio", r["std_ratio"], fmt=".4f", note="(want ≈ 1.00)")
    _row("Corr Frobenius ↓", r["corr_frob"], fmt=".4f")
    print("  ── Random-projection sweep  [standardised space]")
    _row("KS proj (mean ↓)", r["KS_proj"], note=f"over {n_proj} directions")
    print(f"{'─' * W}")


# ─────────────────────────────────────────────────────────────────────────────
# RVFL Layer
# ─────────────────────────────────────────────────────────────────────────────


class RVFLLayer:
    """
    Random Vector Functional-Link (RVFL) network.

    H     = activation(X @ W + b)   # random hidden projection (frozen)
    H_aug = [H | X]                 # optional skip connection
    Y_hat = H_aug @ beta            # closed-form Ridge solution
    """

    def __init__(
        self,
        n_nodes=200,
        alpha=1e-3,
        direct_link=True,
        activation="tanh",
        random_state=42,
    ):
        self.n_nodes = n_nodes
        self.alpha = alpha
        self.direct_link = direct_link
        self.activation = activation
        self.rng = np.random.RandomState(random_state)
        self.W_ = self.b_ = self.beta_ = None

    def _act(self, x):
        return {
            "tanh": np.tanh,
            "sigmoid": lambda z: 1.0 / (1.0 + np.exp(-z)),
            "relu": lambda z: np.maximum(0.0, z),
        }.get(self.activation, np.tanh)(x)

    def _features(self, X):
        H = self._act(X @ self.W_ + self.b_)
        return np.hstack([X, H]) if self.direct_link else H

    def fit(self, X, Y):
        self.W_ = self.rng.randn(X.shape[1], self.n_nodes)
        self.b_ = self.rng.randn(self.n_nodes)
        Phi = self._features(X)
        A = Phi.T @ Phi + self.alpha * np.eye(Phi.shape[1])
        self.beta_ = np.linalg.solve(A, Phi.T @ Y)
        return self

    def predict(self, X):
        return self._features(X) @ self.beta_


# ─────────────────────────────────────────────────────────────────────────────
# Base mixin
# ─────────────────────────────────────────────────────────────────────────────


class _RVFLBase:
    def _build(self, n_nodes, alpha):
        return RVFLLayer(
            n_nodes=n_nodes,
            alpha=alpha,
            direct_link=self.direct_link,
            activation=self.activation,
            random_state=self.random_state,
        )

    def _tune(self, objective, n_trials):
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return study.best_params, study.best_value


# ─────────────────────────────────────────────────────────────────────────────
# PCARVFLSimulator
# ─────────────────────────────────────────────────────────────────────────────


class PCARVFLSimulator(_RVFLBase):
    """
    GAN-like synthesiser: PCA scores → RVFL → Ŷ + bootstrap residuals.

    Parameters
    ----------
    n_pca_components        : int or 'auto' (default).  'auto' picks the
                              minimum components to explain
                              ``pca_variance_threshold`` of variance.
    pca_variance_threshold  : float in (0, 1].  Default 0.95.
    scale                   : bool.  If True (default), internally fit a
                              StandardScaler on the training data so that PCA
                              and the RVFL operate on z-scored features.
                              Samples are inverse-transformed before return.
                              Set to False only if your data is already scaled.
    activation              : 'tanh' (default) | 'relu' | 'sigmoid'
    direct_link             : skip connection in RVFL.  Default True.
    random_state            : int seed.

    Attributes (after fit)
    ----------------------
    scaler_       : fitted StandardScaler (or None if scale=False)
    pca_          : fitted sklearn PCA  (operates on scaled data)
    model_        : fitted RVFLLayer
    residuals_    : (n_train, d) training residuals  (scaled space)
    Z_train_      : (n_train, nc) PCA scores
    best_params_  : {'n_nodes', 'alpha'} from Optuna
    is_fitted_    : bool

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> X, _ = load_iris(return_X_y=True)
    >>> sim = PCARVFLSimulator(random_state=0)
    >>> sim.fit(X, n_trials=20)
    >>> X_syn = sim.sample(200)          # returned in original scale
    >>> report = adequacy_report(X, X_syn)
    """

    def __init__(
        self,
        n_pca_components="auto",
        pca_variance_threshold=0.95,
        scale=True,
        activation="tanh",
        direct_link=True,
        random_state=42,
    ):
        self.n_pca_components = n_pca_components
        self.pca_variance_threshold = pca_variance_threshold
        self.scale = scale
        self.activation = activation
        self.direct_link = direct_link
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.scaler_ = None
        self.pca_ = None
        self.model_ = None
        self.residuals_ = None
        self.Z_train_ = None
        self.best_params_ = None
        self.is_fitted_ = False

    # ── helpers ───────────────────────────────────────────────────────────────

    def _auto_nc(self, Y):
        pca_f = PCA(random_state=self.random_state).fit(Y)
        cumv = np.cumsum(pca_f.explained_variance_ratio_)
        return max(
            1,
            min(
                int(np.searchsorted(cumv, self.pca_variance_threshold) + 1),
                Y.shape[1],
            ),
        )

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, Y, n_train=None, metric="mmd", n_trials=50):
        """
        Fit the simulator on data matrix Y.

        Parameters
        ----------
        Y        : (n, d) real samples  (raw, unscaled OK)
        n_train  : training rows (default n // 2)
        metric   : 'mmd' (default) or 'energy'
        n_trials : Optuna trials (default 50)
        """
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        n, d = Y.shape

        # ── optional standardisation ──────────────────────────────────────────
        if self.scale:
            self.scaler_ = StandardScaler().fit(Y)
            Ys = self.scaler_.transform(Y)
        else:
            self.scaler_ = None
            Ys = Y

        # ── PCA latent space  (on scaled data) ────────────────────────────────
        nc = (
            self._auto_nc(Ys)
            if self.n_pca_components == "auto"
            else int(self.n_pca_components)
        )
        self.pca_ = PCA(n_components=nc, random_state=self.random_state).fit(Ys)
        Z_all = self.pca_.transform(Ys)
        evr = self.pca_.explained_variance_ratio_.sum()
        print(f"  [PCARVFL] {nc} components ({100 * evr:.1f}% var)")

        # ── train / test split ────────────────────────────────────────────────
        if n_train is None:
            n_train = n // 2
        idx = self.rng.permutation(n)
        tr, te = idx[:n_train], idx[n_train:]
        Z_train, Y_train = Z_all[tr], Ys[tr]  # everything in scaled space
        Y_test = Ys[te]
        self.Z_train_ = Z_train

        # ── Optuna: tune n_nodes and alpha ────────────────────────────────────
        def objective(trial):
            nn = trial.suggest_int("n_nodes", 50, 1000, log=True)
            a = trial.suggest_float("alpha", 1e-5, 10.0, log=True)
            m = self._build(nn, a)
            m.fit(Z_train, Y_train)
            res = Y_train - m.predict(Z_train)
            zi = self.rng.choice(len(Z_train), len(te), replace=True)
            sim = m.predict(Z_train[zi])
            sim = sim + res[self.rng.choice(len(res), len(te), replace=True)]
            # use the module-level biased MMD — no inline reimplementation
            return (
                mmd_biased(Y_test, sim)
                if metric == "mmd"
                else energy_distance(Y_test, sim)
            )

        self.best_params_, best_val = self._tune(objective, n_trials)

        # ── refit with best params ────────────────────────────────────────────
        self.model_ = self._build(**self.best_params_)
        self.model_.fit(Z_train, Y_train)
        self.residuals_ = Y_train - self.model_.predict(Z_train)
        self.is_fitted_ = True
        print(
            f"  [PCARVFL] nodes={self.best_params_['n_nodes']} "
            f"α={self.best_params_['alpha']:.2e} "
            f"{metric}={best_val:.5f}"
        )
        return self

    # ── sample ────────────────────────────────────────────────────────────────

    def sample(self, n_samples=1):
        """
        Draw n_samples synthetic rows.

        Returns
        -------
        (n_samples, d) array in the *original* (unscaled) feature space.
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before sample().")
        zi = self.rng.choice(len(self.Z_train_), n_samples, replace=True)
        preds = self.model_.predict(self.Z_train_[zi])
        eps = self.residuals_[
            self.rng.choice(len(self.residuals_), n_samples, replace=True)
        ]
        Ys_syn = preds + eps
        # inverse-transform to original scale
        if self.scaler_ is not None:
            return self.scaler_.inverse_transform(Ys_syn)
        return Ys_syn

    def __repr__(self):
        status = "fitted" if self.is_fitted_ else "not fitted"
        return (
            f"PCARVFLSimulator(n_pca_components={self.n_pca_components!r}, "
            f"pca_variance_threshold={self.pca_variance_threshold}, "
            f"scale={self.scale}, activation={self.activation!r}, "
            f"direct_link={self.direct_link}, "
            f"random_state={self.random_state}) [{status}]"
        )
