import pandas as pd
import numpy as np
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings


class BayesianTweedieSalesGenerator:
    """
    A Bayesian-inspired sales data generator using TweedieRegressor as a generative model.
    Merges with the previous M5-like generator but uses Tweedie posterior sampling.
    """

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        self._initialize_parameters()
        self._initialize_hierarchies()
        self.tweedie_models = {}
        self.posterior_samples = {}

    def _initialize_parameters(self):
        """Initialize default parameters for data generation."""
        self.params = {
            # Time parameters
            "start_date": "2023-01-01",
            "end_date": "2024-12-31",
            "freq": "D",
            # Scale parameters
            "num_products": 50,
            "num_stores": 5,
            "num_states": 3,
            # Tweedie prior parameters
            "power_prior": (
                1.5,
                0.2,
            ),  # (mean, std) for normal prior on power parameter
            "alpha_prior": (
                1.0,
                0.5,
            ),  # (mean, std) for normal prior on regularization
            "dispersion_prior": (1.0, 0.3),  # (mean, std) for dispersion
            # Feature parameters
            "n_features": 8,
            "n_informative": 5,
        }

    def _initialize_hierarchies(self):
        """Initialize hierarchical structures for products and locations."""
        self.hierarchies = {
            "categories": {
                "FOODS": ["FOODS_1", "FOODS_2", "FOODS_3"],
                "HOBBIES": ["HOBBIES_1", "HOBBIES_2"],
                "HOUSEHOLD": ["HOUSEHOLD_1", "HOUSEHOLD_2"],
            },
            "states": ["CA", "TX", "NY"],
            "store_types": ["Supercenter", "Discount", "Neighborhood"],
        }

    def generate_covariates(self, n_samples, include_temporal=True):
        """
        Generate realistic covariates for sales data.

        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        include_temporal : bool
            Whether to include temporal features

        Returns:
        --------
        pd.DataFrame
            DataFrame with generated covariates
        """
        features = {}

        # Basic features
        features["price"] = np.random.lognormal(2.5, 0.8, n_samples)
        features["promotion"] = np.random.binomial(1, 0.1, n_samples)
        features["competitor_price"] = np.random.lognormal(2.6, 0.7, n_samples)
        features["holiday"] = np.random.binomial(1, 0.05, n_samples)

        # Store traffic (seasonal)
        if include_temporal:
            t = np.arange(n_samples)
            features["store_traffic"] = (
                100
                + 20 * np.sin(2 * np.pi * t / 30)  # Monthly seasonality
                + 10 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
                + np.random.normal(0, 5, n_samples)
            )

            # Day of week effects
            features["day_of_week"] = t % 7
            features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)

        # Economic indicators
        features["unemployment_rate"] = np.random.normal(5.0, 1.0, n_samples)
        features["consumer_confidence"] = np.random.normal(100, 15, n_samples)

        return pd.DataFrame(features)

    def sample_tweedie_parameters(
        self, n_samples=1000, prior_type="informative"
    ):
        """
        Sample Tweedie parameters from Bayesian priors.

        Parameters:
        -----------
        n_samples : int
            Number of parameter samples
        prior_type : str
            Type of prior distribution

        Returns:
        --------
        dict
            Sampled parameters
        """
        if prior_type == "informative":
            # Informed priors based on retail sales data characteristics
            power_samples = np.random.normal(
                1.6, 0.2, n_samples
            )  # Compound Poisson-Gamma
            alpha_samples = np.random.gamma(2, 0.5, n_samples)  # Regularization
            dispersion_samples = np.random.gamma(
                3, 0.3, n_samples
            )  # Dispersion

        elif prior_type == "weakly_informative":
            power_samples = np.random.uniform(1.1, 1.9, n_samples)
            alpha_samples = np.random.gamma(1, 1, n_samples)
            dispersion_samples = np.random.gamma(2, 0.5, n_samples)

        else:  # non_informative
            power_samples = np.random.uniform(1.01, 1.99, n_samples)
            alpha_samples = np.random.gamma(0.1, 0.1, n_samples)
            dispersion_samples = np.random.gamma(1, 1, n_samples)

        # Clip power to valid range
        power_samples = np.clip(power_samples, 1.01, 1.99)

        self.posterior_samples = {
            "power": power_samples,
            "alpha": alpha_samples,
            "dispersion": dispersion_samples,
        }

        return self.posterior_samples

    def fit_tweedie_generative_model(self, X, y, n_models=10):
        """
        Fit multiple TweedieRegressor models to capture parameter uncertainty.

        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target sales
        n_models : int
            Number of models to fit with different parameters

        Returns:
        --------
        dict
            Fitted models and their parameters
        """
        print(
            f"Fitting {n_models} TweedieRegressor models for generative sampling..."
        )

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Sample parameters from priors
        param_samples = self.sample_tweedie_parameters(n_models)

        models = {}
        for i in range(n_models):
            try:
                model = TweedieRegressor(
                    power=param_samples["power"][i],
                    alpha=param_samples["alpha"][i],
                    link="log",
                    max_iter=1000,
                    tol=1e-4,
                )

                model.fit(X_scaled, y)

                models[f"model_{i}"] = {
                    "model": model,
                    "power": param_samples["power"][i],
                    "alpha": param_samples["alpha"][i],
                    "dispersion": param_samples["dispersion"][i],
                }

            except Exception as e:
                print(f"Model {i} failed: {e}")
                continue

        self.tweedie_models = models
        return models

    def sample_from_tweedie_posterior(
        self, X, n_samples=1000, model_weights=None
    ):
        """
        Generate posterior predictive samples using fitted Tweedie models.

        Parameters:
        -----------
        X : array-like
            Features for prediction
        n_samples : int
            Number of posterior samples to generate
        model_weights : array-like
            Weights for each model in the ensemble

        Returns:
        --------
        pd.DataFrame
            Posterior predictive samples
        """
        if not self.tweedie_models:
            raise ValueError(
                "No models fitted. Call fit_tweedie_generative_model first."
            )

        X_scaled = self.scaler.transform(X)
        model_names = list(self.tweedie_models.keys())

        if model_weights is None:
            model_weights = np.ones(len(model_names)) / len(model_names)

        # Generate samples
        posterior_samples = []

        for _ in range(n_samples):
            # Sample a model according to weights
            model_idx = np.random.choice(len(model_names), p=model_weights)
            model_name = model_names[model_idx]
            model_info = self.tweedie_models[model_name]

            # Get prediction (mean of Tweedie distribution)
            mu = model_info["model"].predict(X_scaled)

            # Sample from Tweedie distribution
            power = model_info["power"]
            dispersion = model_info["dispersion"]

            # Sample from compound Poisson-Gamma distribution
            samples = self._sample_tweedie_distribution(mu, power, dispersion)
            posterior_samples.append(samples)

        return pd.DataFrame(
            np.array(posterior_samples).T,
            columns=[f"sample_{i}" for i in range(n_samples)],
        )

    def _sample_tweedie_distribution(self, mu, power, dispersion):
        """
        Sample from Tweedie compound Poisson-Gamma distribution.

        Parameters:
        -----------
        mu : array-like
            Mean parameters
        power : float
            Tweedie power parameter
        dispersion : float
            Dispersion parameter

        Returns:
        --------
        array
            Samples from Tweedie distribution
        """
        samples = []

        for mean_val in mu:
            if power == 1:  # Poisson
                sample = np.random.poisson(mean_val)
            elif power == 2:  # Gamma
                sample = np.random.gamma(mean_val / dispersion, dispersion)
            else:  # Compound Poisson-Gamma (1 < power < 2)
                # Parameterization for compound Poisson-Gamma
                lambda_val = (mean_val ** (2 - power)) / (
                    (2 - power) * dispersion
                )
                phi = dispersion
                p = (power - 1) / (2 - power)

                # Number of claims (Poisson)
                N = np.random.poisson(lambda_val)

                if N > 0:
                    # Size of claims (Gamma)
                    claim_size = np.random.gamma(
                        -p * N, phi * (mean_val ** (power - 1))
                    )
                    sample = claim_size
                else:
                    sample = 0

            samples.append(max(0, sample))  # Ensure non-negative

        return np.array(samples)

    def generate_sales_timeseries(
        self, product_id, store_id, state, n_days=730
    ):
        """
        Generate sales time series using Bayesian Tweedie generative model.

        Parameters:
        -----------
        product_id : int
            Product identifier
        store_id : int
            Store identifier
        state : str
            State location
        n_days : int
            Number of days to generate

        Returns:
        --------
        pd.DataFrame
            Generated sales time series with posterior samples
        """
        # Generate covariates
        covariates = self.generate_covariates(n_days, include_temporal=True)

        # Add product/store specific features
        covariates["product_id"] = product_id
        covariates["store_id"] = store_id
        covariates["state"] = hash(state) % 100  # Encode state as numeric

        # Generate dates
        dates = pd.date_range(
            start=self.params["start_date"], periods=n_days, freq="D"
        )
        covariates["date"] = dates

        # If we have fitted models, use them for generation
        if self.tweedie_models:
            # Use only feature columns for prediction
            feature_cols = [
                col
                for col in covariates.columns
                if col not in ["date", "product_id", "store_id", "state"]
            ]
            X_features = covariates[feature_cols]

            # Generate posterior predictive samples
            posterior_samples = self.sample_from_tweedie_posterior(
                X_features, n_samples=100
            )

            # Use median as point estimate
            covariates["sales"] = posterior_samples.median(axis=1)
            covariates["sales_std"] = posterior_samples.std(axis=1)

            # Store samples
            for i in range(
                min(10, posterior_samples.shape[1])
            ):  # Store first 10 samples
                covariates[f"sales_sample_{i}"] = posterior_samples.iloc[:, i]

        else:
            # Fallback: simple generative model
            print("No fitted models. Using simple generative approach.")
            base_demand = np.random.gamma(2, 2)
            seasonality = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_days) / 365)
            covariates["sales"] = np.random.poisson(base_demand * seasonality)

        # Add product hierarchy
        category, subcategory = self._generate_product_hierarchy(product_id)
        covariates["category"] = category
        covariates["subcategory"] = subcategory
        covariates["store_id"] = f"STORE_{store_id:02d}"
        covariates["product_id"] = f"PROD_{product_id:03d}"
        covariates["state"] = state

        return covariates

    def _generate_product_hierarchy(self, product_id):
        """Generate hierarchical product information."""
        category = np.random.choice(list(self.hierarchies["categories"].keys()))
        subcategory = np.random.choice(self.hierarchies["categories"][category])
        return category, subcategory

    def generate_complete_dataset(
        self, use_bayesian=True, n_samples_per_series=100
    ):
        """
        Generate complete sales dataset using Bayesian Tweedie approach.

        Parameters:
        -----------
        use_bayesian : bool
            Whether to use Bayesian Tweedie sampling
        n_samples_per_series : int
            Number of posterior samples per time series

        Returns:
        --------
        pd.DataFrame
            Complete generated dataset
        """
        print("Generating complete sales dataset...")

        all_data = []

        # Create store-state mapping
        store_state_mapping = {}
        for store_id in range(1, self.params["num_stores"] + 1):
            store_state_mapping[store_id] = np.random.choice(
                self.hierarchies["states"]
            )

        # If using Bayesian approach, first fit models on some synthetic data
        if use_bayesian:
            print("Fitting Bayesian Tweedie models...")
            # Generate some training data to fit models
            X_train = self.generate_covariates(1000, include_temporal=True)
            y_train = self._generate_simple_sales(X_train)
            self.fit_tweedie_generative_model(X_train, y_train, n_models=20)

        # Generate data for all products across all stores
        for product_id in range(1, self.params["num_products"] + 1):
            for store_id in range(1, self.params["num_stores"] + 1):
                state = store_state_mapping[store_id]

                sales_data = self.generate_sales_timeseries(
                    product_id, store_id, state, n_days=n_samples_per_series
                )

                all_data.append(sales_data)

        complete_data = pd.concat(all_data, ignore_index=True)

        print(f"Generated dataset with {complete_data.shape[0]} rows")
        return complete_data

    def _generate_simple_sales(self, covariates):
        """Generate simple sales for model training."""
        # Simple linear model with noise
        base_effect = (
            10
            + 2 * covariates["price"]
            + 5 * covariates["promotion"]
            - 1 * covariates["competitor_price"]
            + 0.1 * covariates["store_traffic"]
            + 3 * covariates["holiday"]
        )

        # Add noise and ensure positive
        sales = np.random.poisson(np.maximum(1, base_effect))
        return sales
