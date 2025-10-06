from synthe.bayesiantweediesalesgenerator import BayesianTweedieSalesGenerator

# Initialize generator
generator = BayesianTweedieSalesGenerator(seed=42)

# Generate complete dataset with Bayesian Tweedie sampling
dataset = generator.generate_complete_dataset(use_bayesian=True, n_samples_per_series=365)

print("Dataset columns:", dataset.columns.tolist())
print("\nDataset summary:")
print(dataset[['sales', 'sales_std', 'price', 'promotion']].describe())

# Show posterior samples
sample_cols = [col for col in dataset.columns if 'sales_sample' in col]
if sample_cols:
    print(f"\nGenerated {len(sample_cols)} posterior samples per observation")
    print("Posterior sample correlations:")
    print(dataset[sample_cols[:5] + ['sales']].corr())