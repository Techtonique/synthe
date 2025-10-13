import numpy as np
from synthe.distro_simulator import DistroSimulator
from time import time

# Create some example data
np.random.seed(42)
n_samples = 500

def example_X_dist(n):
      return np.random.normal(0, 1, (n, 2))

X = example_X_dist(n_samples)
Y = np.column_stack([
      X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.normal(0, 1, n_samples),
      X[:, 0] - 0.3 * X[:, 1] + 0.1 * np.random.normal(0, 1, n_samples)
])

# Use auto configuration - the class will choose the best approach
print("Testing with auto configuration...")
generator = DistroSimulator(
      kernel='rbf', 
      random_state=42
)

start = time()
generator.fit(Y, n_trials=10, metric='energy')
print(f"Auto configuration fitting took {time() - start:.2f} seconds.")

# Generate and test synthetic data
Y_synth = generator.sample(1000)
results = generator.test_similarity(Y[:1000], Y_synth)

print("\nFinal similarity results:")
for key, value in results.items():
      if key != 'corr_tests':
            print(f"{key}: {value}")

results = generator.compare_distributions(Y[:1000], Y_synth)

print("\nFinal distribution comparison results:")
for key, value in results.items():
      if key != 'corr_tests':
            print(f"{key}: {value}")