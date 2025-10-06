import matplotlib.pyplot as plt
from synthe.healthsims import SmartHealthSimulator

# Example usage
if __name__ == "__main__":
    # Create simulator
    sim = SmartHealthSimulator(days=180, seed=42)
    
    # Display first few rows
    print("First 5 rows of data:")
    print(sim.data.head())
    
    print(f"\nDataset shape: {sim.data.shape}")
    print(f"\nData types:\n{sim.data.dtypes}")
    
    # Generate plots
    sim.plot_time_series()
    plt.show()
    
    sim.plot_mood_sleep()
    plt.show()
    
    sim.plot_activity_distribution()
    plt.show()
    
    wordcloud_fig = sim.plot_mood_wordcloud()
    if