
import nnetsauce as ns
import numpy as np
import optuna
import GPopt as gp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C

from collections import namedtuple
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from .empirical_copula import EmpiricalCopula
from .row_subsampling import SubSampler
from .utils import simulate_replications, mmd_rbf, mmd_matern52, energy_distance, crps, simulate_distribution


class ConformalInference:
    """GAN-like Synthetic data simulation using Conformal Inference.

    This class implements conformal inference for synthetic data generation with automatic
    input scaling for improved numerical stability and convergence.

    Attributes:

        type_pi: a string;
            type of prediction interval: currently `None`
            (split conformal without simulation)
            for type_pi in: 
                - 'bootstrap': Bootstrap resampling.
                - 'kde': Kernel Density Estimation.

        type_split: a string;
            "random" (random split of data) or "sequential" (sequential split of data)
        
        replications: an integer;
            Number of replications for simulated conformal (default is 250)
        
        kernel: a string;
            Kernel to be used in local conformal inference (default is 'gaussian')
            if type_pi is 'kde', this kernel will also be used in the simulation of residuals
        
        objective: a string;
            Objective function to be used in the optimization of the synthetic data generator
            (default is 'mmd_rbf'). Options are:
                - 'mmd_rbf': Maximum Mean Discrepancy with RBF kernel
                - 'mmd_matern52': Maximum Mean Discrepancy with Matérn 5/2 kernel
                - 'crps': Continuous Ranked Probability Score
                - 'energy': Energy Distance
        
        optimizer: a string;
            Optimizer to be used in the optimization of the synthetic data generator
            ('gpopt' or 'optuna', default is 'optuna')
        
        split_ratio: a float;
            Ratio of calibration data (default is 0.5)

        seed: an integer;
            Reproducibility of fit (there's a random split between fitting and calibration data)
            
    Notes:
        Inputs are automatically scaled using StandardScaler for better numerical stability.
        Generated samples are automatically rescaled back to the original scale.
    """

    def __init__(
        self,
        type_pi="bootstrap",
        type_split="random",
        replications=250,
        kernel="gaussian",  
        objective='crps',
        optimizer='optuna',
        split_ratio=0.5,     
        partition=False,
        seed=123,
    ):
        self.type_pi = type_pi
        self.type_split = type_split
        self.replications = replications
        self.kernel = kernel
        self.objective = objective
        self.optimizer = optimizer
        self.split_ratio = split_ratio
        self.partition = partition
        self.seed = seed
        self.calibrated_residuals_ = None
        self.scaled_calibrated_residuals_ = None
        self.calibrated_residuals_scaler_ = None
        self.random_covariates_ = None 
        self.random_covariates_train_ = None 
        self.random_covariates_calibration_ = None 
        self.kde_ = None
        self.res_opt_ = None
        self.copula = None
        # Input scaler for standardization
        self.input_scaler_ = None
        self.obj = ns.Ridge2Regressor(n_hidden_features=5,
        dropout=0,
        n_clusters=2,
        lambda1=0.1,
        lambda2=0.1,)

    def fit(self, X, **kwargs):
        """Fit the `method` to training data (X, y).

        Args:

            X: array-like, shape = [n_samples, n_features];
                Training set vectors, where n_samples is the number
                of samples and n_features is the number of features.
                Can be a 1d array.

        """

        if self.type_split == "random":

            if len(X.shape) == 1: # 1d array   

                if self.partition == False:             
                    X_train, X_calibration = train_test_split(X, 
                                                              test_size=self.split_ratio, 
                                                              random_state=self.seed)          
                else: 
                    X_ = X.reshape(-1, 1)
                    gmm = GMM(n_components=2, random_state=self.seed).fit(X_)
                    labels = gmm.predict(X_)
                    X_train, X_calibration, _, _ = train_test_split(
                        X_, labels, test_size=self.split_ratio, 
                        random_state=self.seed,
                        stratify=labels)
                    
            else: # multi-dim array

                if self.partition == False:             
                    gmm = GMM(n_components=2, random_state=self.seed).fit(X)
                    labels = gmm.predict(X)
                    sub_train = SubSampler(y=labels, 
                                           n_samples=X_train.shape[0],
                                    seed=self.seed, 
                                    n_jobs=None)                
                    sub_calib = SubSampler(y=labels, 
                                           n_samples=X_calibration.shape[0],
                                    seed=self.seed+1000, 
                                    n_jobs=None)      
                    train_idx = sub_train.subsample()
                    calib_idx = sub_calib.subsample()
                    X_train = X[train_idx]
                    X_calibration = X[calib_idx]
                else: 
                    gmm = GMM(n_components=2, random_state=self.seed).fit(X)
                    labels = gmm.predict(X)
                    X_train, X_calibration, _, _ = train_test_split(
                        X, labels, test_size=self.split_ratio, 
                        random_state=self.seed,
                        stratify=labels)
                                  
        elif self.type_split == "sequential":

            raise NotImplementedError(
                "'sequential' split not implemented yet for 1d arrays"
            )            
        
        # Scale inputs for better numerical stability
        self.input_scaler_ = StandardScaler()
        X_scaled = self.input_scaler_.fit_transform(X)
        X_train_scaled = self.input_scaler_.transform(X_train)
        X_calibration_scaled = self.input_scaler_.transform(X_calibration)
        
        np.random.seed(self.seed)
        self.random_covariates_ = np.random.randn(X.shape[0], 2)                                        
        self.random_covariates_train_ = self.random_covariates_[:X_train.shape[0],:]
        self.random_covariates_calibration_ = self.random_covariates_[X_train.shape[0]:,:]
        self.calibrated_residuals_ = None        
        self.synthetic_data_ = None

        def calibrate():
            """Calibrate the model on calibration data."""

            # Define the objective function            
            def objective_func_optuna(trial):
                # Suggest values for the hyperparameters in the search space using Optuna
                n_hidden_features = trial.suggest_int("n_hidden_features", 3, 250)
                log10_lambda1 = trial.suggest_float("log10_lambda1", -4, 5)
                log10_lambda2 = trial.suggest_float("log10_lambda2", -4, 5)
                dropout = trial.suggest_float("dropout", 0.0, 0.5)
                n_clusters = trial.suggest_int("n_clusters", 0, 5)                
                # Create the Ridge2Regressor model with the suggested hyperparameters
                obj = ns.Ridge2Regressor(n_hidden_features=n_hidden_features,
                                        lambda1=10 ** log10_lambda1,
                                        lambda2=10 ** log10_lambda2,
                                        dropout=max(min(dropout, 0.5), 0),  # ensure dropout is within bounds
                                        n_clusters=n_clusters)                
                # Train the model on scaled inputs
                obj.fit(self.random_covariates_train_, X_train_scaled)                
                # Make predictions on the calibration data
                preds_calibration = obj.predict(self.random_covariates_calibration_)                
                # Compute the residuals
                calibrated_residuals = X_calibration_scaled - preds_calibration                
                # Handle the case for univariate or multivariate residuals
                if len(X_calibration.shape) == 1:
                    calibrated_residuals_sims = simulate_replications(
                        calibrated_residuals, 
                        method=self.type_pi,
                        kernel=self.kernel,
                        num_replications=250,
                        seed=self.seed
                    )
                    synthetic_data = calibrated_residuals_sims + preds_calibration[:, np.newaxis]
                else:
                    # For multivariate residuals, use copula sampling
                    copula = EmpiricalCopula()
                    copula.fit(calibrated_residuals)
                    synthetic_data = copula.sample(n_samples=250) + preds_calibration[:, np.newaxis]
                # Compute the objective loss (use original scale for comparison)
                loss = self._compute_objective(X_calibration, synthetic_data)                
                # Return the computed loss which Optuna will try to minimize
                return loss

            def objective_func(xx):
                self.obj = ns.Ridge2Regressor(n_hidden_features=int(xx[0]),
                                              lambda1=10**xx[1],
                                              lambda2=10**xx[2],
                                              dropout=max(min(xx[3], 0.5), 0),
                                              n_clusters=int(xx[4]))            
                self.obj.fit(self.random_covariates_train_, X_train_scaled)
                preds_calibration = self.obj.predict(self.random_covariates_calibration_)                
                self.calibrated_residuals_ = X_calibration_scaled - preds_calibration                
                if len(X.shape) == 1:
                    self.calibrated_residuals_sims_ = simulate_replications(
                        self.calibrated_residuals_, 
                        method=self.type_pi,
                        kernel=self.kernel,
                        num_replications=250,
                        seed=self.seed)
                    synthetic_data = self.calibrated_residuals_sims_ + preds_calibration[:, np.newaxis]
                else:
                    self.copula = EmpiricalCopula()
                    self.copula.fit(self.calibrated_residuals_)
                    synthetic_data = self.copula.sample(n_samples=250) + preds_calibration[:, np.newaxis]
                return self._compute_objective(X_calibration, synthetic_data)

            if self.optimizer == 'optuna':
                study = optuna.create_study(direction="minimize", 
                                            study_name="synthe_conformal_inference")
                study.optimize(objective_func_optuna, n_trials=100, n_jobs=1, show_progress_bar=True)
                self.res_opt_ = study.best_trial
                return self.res_opt_
            elif self.optimizer == 'gpopt':
            gp_opt = gp.GPOpt(objective_func=objective_func,
                        lower_bound = np.array([   3, -4, -4,   0, 0]),
                        upper_bound = np.array([ 250,  5,  5, 0.5, 5]),
                        params_names=["n_hidden_features", "log10_lambda1", "log10_lambda2", "dropout", "n_clusters"],
                            surrogate_obj = GaussianProcessRegressor(
                            kernel=Matern(nu=2.5),
                            alpha=1e-6,
                            normalize_y=True,
                            n_restarts_optimizer=25,
                            random_state=42,
                        ),
                            n_init=10, n_iter=90, seed=3137,
                            n_jobs=1,
                            )
                self.res_opt_ = gp_opt.optimize(verbose=1)
                return self.res_opt_
        
        calibrate()

        return self       


    def sample(self, n_samples=100):
        """Generate synthetic samples using the fitted model.

        Args:
            n_samples: an integer;
                Number of samples to be generated
                
        Returns:
            np.ndarray: Generated synthetic samples in original scale
        """
        if self.input_scaler_ is None:
            raise ValueError("Model must be fitted before sampling. Call fit() first.")
        
        # Generate random covariates for sampling
        np.random.seed(self.seed)
        random_covariates_new = np.random.randn(n_samples, 2)
        
        # Generate predictions on scaled space
        preds_scaled = self.obj.predict(random_covariates_new)
        
        # Generate residuals using the fitted residual model
        if hasattr(self, 'calibrated_residuals_') and self.calibrated_residuals_ is not None:
            if len(self.calibrated_residuals_.shape) == 1:
                # Univariate residuals
                residual_sims = simulate_replications(
                    self.calibrated_residuals_, 
                    method=self.type_pi,
                    kernel=self.kernel,
                    num_replications=n_samples,
                    seed=self.seed
                )
                synthetic_data_scaled = residual_sims + preds_scaled[:, np.newaxis]
            else:
                # Multivariate residuals - use copula
                if self.copula is not None:
                    residual_sims = self.copula.sample(n_samples=n_samples)
                    synthetic_data_scaled = residual_sims + preds_scaled[:, np.newaxis]
                else:
                    # Fallback to bootstrap if no copula
                    n_residuals = len(self.calibrated_residuals_)
                    indices = np.random.choice(n_residuals, size=n_samples, replace=True)
                    residual_sims = self.calibrated_residuals_[indices]
                    synthetic_data_scaled = residual_sims + preds_scaled
        else:
            # If no residuals available, just return predictions
            synthetic_data_scaled = preds_scaled
        
        # Rescale back to original scale
        synthetic_data_original = self.input_scaler_.inverse_transform(synthetic_data_scaled)
        
        return synthetic_data_original 
        
    def _compute_objective(self, y_true, y_synthetic):
        """Compute chosen objective"""
        if self.objective == 'mmd_rbf':
            return mmd_rbf(y_true, y_synthetic)
        elif self.objective == 'mmd_matern52':
            return mmd_matern52(y_true, y_synthetic)
        elif self.objective == 'crps':
            return crps(y_true, y_synthetic)
        elif self.objective == 'energy':
            return energy_distance(y_true, y_synthetic)
        else: # default
            return mmd_rbf(y_true, y_synthetic)
