
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
        self.X_train_ = None
        self.X_calib_ = None
        self.res_opt_ = None
        self.copula = None
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

                X_train, X_calibration = train_test_split(X, 
                                                            test_size=self.split_ratio, 
                                                            random_state=self.seed)   
                self.X_train_ = X_train
                self.X_calib_ = X_calibration
                    
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
                    self.X_train_ = X_train
                    self.X_calib_ = X_calibration
                else: 
                    gmm = GMM(n_components=2, random_state=self.seed).fit(X)
                    labels = gmm.predict(X)
                    X_train, X_calibration, _, _ = train_test_split(
                        X, labels, test_size=self.split_ratio, 
                        random_state=self.seed,
                        stratify=labels)
                    self.X_train_ = X_train 
                    self.X_calib_ = X_calibration
                                  
        elif self.type_split == "sequential":

            raise NotImplementedError(
                "'sequential' split not implemented yet for 1d arrays"
            )            
        
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
                                        dropout=max(min(dropout, 1), 0.5),  # ensure dropout is within bounds
                                        n_clusters=n_clusters)                
                # Train the model
                obj.fit(self.random_covariates_train_, X_train)                
                # Make predictions on the calibration data
                preds_calibration = obj.predict(self.random_covariates_calibration_)                
                # Compute the residuals
                calibrated_residuals = X_calibration - preds_calibration 
                if len(X.shape) == 1:
                    calibrated_residuals = calibrated_residuals.ravel()                                               
                # Handle the case for univariate or multivariate residuals
                if len(X.shape) == 1:
                    calibrated_residuals_sims = simulate_replications(
                        calibrated_residuals, 
                        method=self.type_pi,
                        kernel=self.kernel,
                        num_replications=275,
                        seed=self.seed
                    )
                    synthetic_data = calibrated_residuals_sims + preds_calibration.ravel()[:, np.newaxis]
                else:
                    # For multivariate residuals, use copula sampling
                    copula = EmpiricalCopula()
                    copula.fit(calibrated_residuals)
                    synthetic_data = copula.sample(n_samples=250) + preds_calibration[:, np.newaxis]
                # Compute the objective loss
                loss = self._compute_objective(X_calibration, synthetic_data)                
                # Return the computed loss which Optuna will try to minimize
                return loss

            def objective_func(xx):
                self.obj = ns.Ridge2Regressor(n_hidden_features=int(xx[0]),
                                              lambda1=10**xx[1],
                                              lambda2=10**xx[2],
                                              dropout=max(min(xx[2], 1), 0.5),
                                              n_clusters=int(xx[3]))            
                self.obj.fit(self.random_covariates_train_, X_train)
                preds_calibration = self.obj.predict(self.random_covariates_calibration_)                
                self.calibrated_residuals_ = X_calibration - preds_calibration                
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
                self.res_opt_ = gp_opt.optimize(verbose=2)
                return self.res_opt_
        
        calibrate()

        return self       


    def sample(self, n_samples=100, seed=123):
        """Obtain predictions and prediction intervals

        Args:

            n_samples: an integer;
                Number of samples to be generated
        """    
        if self.optimizer == 'gpopt':
            self.obj = ns.Ridge2Regressor(n_hidden_features=int(self.res_opt_.best_params['n_hidden_features']),
                                          lambda1=10 ** self.res_opt_.best_params['log10_lambda1'],
                                          lambda2=10 ** self.res_opt_.best_params['log10_lambda2'],
                                          dropout=max(min(self.res_opt_.best_params['dropout'], 1), 0.5),
                                          n_clusters=int(self.res_opt_.best_params['n_clusters']))
        elif self.optimizer == 'optuna':
            best_params = self.res_opt_.params
            self.obj = ns.Ridge2Regressor(
                n_hidden_features=int(best_params["n_hidden_features"]),
                lambda1=10 ** best_params["log10_lambda1"],
                lambda2=10 ** best_params["log10_lambda2"],
                dropout=max(min(best_params["dropout"], 1), 0.5),
                n_clusters=int(best_params["n_clusters"])
            )
        np.random.seed(seed)
        random_covariates_calib = np.random.randn(self.X_calib_.shape[0], 2)
        random_covariates_new = np.random.randn(n_samples, 2)
        self.obj.fit(self.random_covariates_train_, self.X_train_)
        preds_calibration = self.obj.predict(random_covariates_calib)                
        calibrated_residuals = self.X_calib_ - preds_calibration                
        if len(self.X_train_.shape) == 1:
            calibrated_residuals_sims = simulate_replications(
                calibrated_residuals, 
                method=self.type_pi,
                kernel=self.kernel,
                num_replications=n_samples,
                seed=self.seed)
            return calibrated_residuals_sims.T + self.obj.predict(random_covariates_new)[:, np.newaxis]                    
        else:
            copula = EmpiricalCopula()
            copula.fit(calibrated_residuals)
            return self.copula.sample(n_samples=n_samples) + preds_calibration[:, np.newaxis]
        
        
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
