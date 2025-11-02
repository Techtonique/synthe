import numpy as np
import matplotlib.pyplot as plt
from synthe.ts_distro_simulator import TsDistroSimulator

# Set style for better plots
plt.style.use('seaborn-v0_8')

# Simulate a time series (trend + seasonality + noise)
t = np.arange(120)
trend = 0.1 * t
seasonal = 5 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 1.5, 120)
time_series = 50 + trend + seasonal + noise
Y = time_series.reshape(-1, 1)

# Initialize and fit the TsDistroSimulator
simulator = TsDistroSimulator(
    kernel='rbf',
    residual_sampling='block-bootstrap',
    block_size=12,
    random_state=42
)
simulator.fit(Y, n_trials=50, metric='energy')

# Generate synthetic samples
Y_sim = simulator.sample(n_samples=120)
print("Y", Y)
print("Y_sim", Y_sim)

# Plot original vs synthetic time series
plt.figure(figsize=(14, 6))
plt.plot(Y, label='Original Time Series', color='navy', linewidth=2)
plt.plot(Y_sim, label='Synthetic Time Series', color='crimson', linewidth=2)
plt.title('Original vs Synthetic Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Compare distributions visually and statistically
simulator.compare_distributions(Y, Y_sim, save_prefix="ts_distro")
