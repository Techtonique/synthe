"""
DiffusionModel Examples - Converting Colab notebook examples
"""

import numpy as np
import matplotlib.pyplot as plt
from synthe import DiffusionModel  # Adjust import

"""
Apply DiffusionModel from synthe package to the examples from the PDF:
1. Univariate normal data
2. Bivariate normal with correlation
3. 3D mixture of Gaussians
4. MNIST digits (with PCA)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import scipy.stats as stats
from tqdm import tqdm

# Assuming DiffusionModel is available from your first document
# from synthe import DiffusionModel
# For this example, we'll import it from the provided code
import sys
sys.path.insert(0, '.')  # Adjust path as needed

# Example usage
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

# Generate toy data
print("Generating Swiss roll data...")
X, _ = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)
X = X[:, [0, 2]]  # Use 2D for visualization

print(f"Data shape: {X.shape}")

# Train diffusion model with improved parameters
print("\nTraining diffusion model with batch processing...")
model = DiffusionModel(
    timesteps=200, 
    beta_start=0.0001, 
    beta_end=0.02,
    schedule='cosine',
    use_pca=False,
    random_state=42,
    batch_size=64
)
model.fit(X, n_steps=2000)

# Generate samples using both methods
print("\nGenerating samples (DDPM)...")
samples_ddpm = model.sample(n_samples=500, ddim=False)

print("Generating samples (DDIM - faster)...")
samples_ddim = model.sample(n_samples=500, ddim=True, ddim_steps=50)

# Compute distributional errors
print("\nDistributional reconstruction errors:")
mmd_error_ddpm = model.reconstruction_error(X, metric='mmd', n_samples=500, gamma='auto')
mmd_error_ddim = model.reconstruction_error(X, metric='mmd', n_samples=500, gamma='auto')

print(f"  DDPM MMD:  {mmd_error_ddpm:.6f}")
print(f"  DDIM MMD:  {mmd_error_ddim:.6f}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(X[:, 0], X[:, 1], c='blue', alpha=0.3, s=10, label='Training data')
axes[0].set_title('Training Data (Swiss Roll)')
axes[0].axis('equal')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].scatter(samples_ddpm[:, 0], samples_ddpm[:, 1], c='red', alpha=0.3, s=10, label='DDPM')
axes[1].set_title(f'DDPM Samples (MMD={mmd_error_ddpm:.4f})')
axes[1].axis('equal')
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].scatter(samples_ddim[:, 0], samples_ddim[:, 1], c='green', alpha=0.3, s=10, label='DDIM')
axes[2].set_title(f'DDIM Samples (MMD={mmd_error_ddim:.4f})')
axes[2].axis('equal')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('diffusion_results_improved.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved as 'diffusion_results_improved.png'")
plt.show()


def test_similarity(Y_orig, Y_sim):
    """Comprehensive similarity tests from the PDF"""
    if Y_orig.ndim == 1:
        Y_orig = Y_orig.reshape(-1, 1)
    if Y_sim.ndim == 1:
        Y_sim = Y_sim.reshape(-1, 1)
    
    d = Y_orig.shape[1]
    results = {}
    
    # KS tests on marginals
    if d == 1:
        ks_res = stats.ks_2samp(Y_orig.flatten(), Y_sim.flatten())
        results['ks'] = (ks_res.statistic, ks_res.pvalue)
        print(f"  KS test: stat={ks_res.statistic:.4f}, p={ks_res.pvalue:.4f}")
        
        # Anderson-Darling
        ad_res = stats.anderson_ksamp([Y_orig.flatten(), Y_sim.flatten()])
        results['ad'] = (ad_res.statistic, ad_res.significance_level)
        print(f"  Anderson-Darling: stat={ad_res.statistic:.4f}, sig={ad_res.significance_level:.4f}")
        
        # Cramer-von Mises
        cvm_res = stats.cramervonmises_2samp(Y_orig.flatten(), Y_sim.flatten())
        results['cvm'] = (cvm_res.statistic, cvm_res.pvalue)
        print(f"  Cramer-von Mises: stat={cvm_res.statistic:.4f}, p={cvm_res.pvalue:.4f}")
    else:
        # Marginal tests for multivariate
        ps_ks = []
        for i in range(d):
            ks_res = stats.ks_2samp(Y_orig[:, i], Y_sim[:, i])
            ps_ks.append(ks_res.pvalue)
            print(f"  Dim {i+1} KS: stat={ks_res.statistic:.4f}, p={ks_res.pvalue:.4f}")
        results['min_marginal_ks_p'] = min(ps_ks)
        
        # Correlation tests for pairs
        if d == 2:
            r_orig = stats.pearsonr(Y_orig[:, 0], Y_orig[:, 1])[0]
            r_sim = stats.pearsonr(Y_sim[:, 0], Y_sim[:, 1])[0]
            
            # Fisher's z-transform test
            z1 = np.arctanh(r_orig)
            z2 = np.arctanh(r_sim)
            n1, n2 = len(Y_orig), len(Y_sim)
            z_stat = (z1 - z2) / np.sqrt(1/(n1-3) + 1/(n2-3))
            p_val = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
            
            results['corr_test'] = (r_orig, r_sim, z_stat, p_val)
            print(f"  Correlation: orig={r_orig:.4f}, sim={r_sim:.4f}, p={p_val:.4f}")
        
        elif d == 3:
            # Test all pairwise correlations
            print("  Pairwise correlations:")
            corr_orig = np.corrcoef(Y_orig.T)
            corr_sim = np.corrcoef(Y_sim.T)
            for i in range(d):
                for j in range(i+1, d):
                    print(f"    Dim{i+1}-Dim{j+1}: orig={corr_orig[i,j]:.4f}, sim={corr_sim[i,j]:.4f}")
    
    return results


def visualize_comparison(Y_orig, Y_sim, title=""):
    """Visualize original vs simulated data"""
    if Y_orig.ndim == 1:
        Y_orig = Y_orig.reshape(-1, 1)
    if Y_sim.ndim == 1:
        Y_sim = Y_sim.reshape(-1, 1)
    
    d = Y_orig.shape[1]
    
    if d == 1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Histogram
        axes[0].hist(Y_orig.flatten(), bins=30, alpha=0.5, label='Original', density=True)
        axes[0].hist(Y_sim.flatten(), bins=30, alpha=0.5, label='Simulated', density=True)
        axes[0].legend()
        axes[0].set_title('Histogram')
        axes[0].grid(alpha=0.3)
        
        # QQ plot
        qq_orig = np.percentile(Y_orig.flatten(), np.linspace(0, 100, 100))
        qq_sim = np.percentile(Y_sim.flatten(), np.linspace(0, 100, 100))
        axes[1].plot(qq_orig, qq_sim, 'o', alpha=0.5)
        minv = min(qq_orig.min(), qq_sim.min())
        maxv = max(qq_orig.max(), qq_sim.max())
        axes[1].plot([minv, maxv], [minv, maxv], 'r--', label='y=x')
        axes[1].legend()
        axes[1].set_title('QQ Plot')
        axes[1].grid(alpha=0.3)
        
        # ECDF
        sorted_orig = np.sort(Y_orig.flatten())
        ecdf_orig = np.arange(1, len(sorted_orig)+1) / len(sorted_orig)
        sorted_sim = np.sort(Y_sim.flatten())
        ecdf_sim = np.arange(1, len(sorted_sim)+1) / len(sorted_sim)
        axes[2].step(sorted_orig, ecdf_orig, label='Original', alpha=0.7)
        axes[2].step(sorted_sim, ecdf_sim, label='Simulated', alpha=0.7)
        axes[2].legend()
        axes[2].set_title('ECDF')
        axes[2].grid(alpha=0.3)
        
    elif d == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].scatter(Y_orig[:, 0], Y_orig[:, 1], alpha=0.5, s=10, label='Original')
        axes[0].set_title('Original Data')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].axis('equal')
        
        axes[1].scatter(Y_sim[:, 0], Y_sim[:, 1], alpha=0.5, s=10, label='Simulated', color='red')
        axes[1].set_title('Simulated Data')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].axis('equal')
        
    elif d == 3:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 6))
        
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(Y_orig[:, 0], Y_orig[:, 1], Y_orig[:, 2], 
                   alpha=0.5, s=10, label='Original')
        ax1.set_title('Original Data')
        ax1.legend()
        
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(Y_sim[:, 0], Y_sim[:, 1], Y_sim[:, 2], 
                   alpha=0.5, s=10, label='Simulated', color='red')
        ax2.set_title('Simulated Data')
        ax2.legend()
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Example 1: Univariate Normal (from PDF Section 1)
# =============================================================================
print("="*70)
print("EXAMPLE 1: UNIVARIATE NORMAL DISTRIBUTION")
print("="*70)

np.random.seed(42)
n = 200
Y_uni = np.random.normal(0, 1, n)

print(f"\nData: {n} samples from N(0, 1)")
print(f"Sample mean: {Y_uni.mean():.4f}, Sample std: {Y_uni.std():.4f}")

# Import and fit DiffusionModel

model_uni = DiffusionModel(
    timesteps=100,
    beta_start=0.0001,
    beta_end=0.02,
    schedule='cosine',
    use_pca=False
)

print("\nFitting DiffusionModel...")
model_uni.fit(Y_uni.reshape(-1, 1), n_steps=1000)

print("\nGenerating samples...")
Y_sim_uni = model_uni.sample(n_samples=500).flatten()

print("\n--- Hypothesis Tests ---")
results_uni = test_similarity(Y_uni, Y_sim_uni)

print("\n--- Reconstruction Error ---")
mmd_error = model_uni.reconstruction_error(Y_uni.reshape(-1, 1), metric='mmd', n_samples=500)
energy_error = model_uni.reconstruction_error(Y_uni.reshape(-1, 1), metric='energy', n_samples=500)
print(f"MMD: {mmd_error:.6f}")
print(f"Energy Distance: {energy_error:.6f}")

visualize_comparison(Y_uni, Y_sim_uni, "Univariate Normal: DiffusionModel")


# =============================================================================
# Example 2: Bivariate Normal with Correlation (from PDF Section 1)
# =============================================================================
print("\n" + "="*70)
print("EXAMPLE 2: BIVARIATE NORMAL WITH CORRELATION")
print("="*70)

np.random.seed(42)
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]
Y_bi = np.random.multivariate_normal(mean, cov, n)

print(f"\nData: {n} samples from N([0,0], cov)")
print("Covariance matrix:")
print(cov)
print(f"\nSample correlation: {np.corrcoef(Y_bi.T)[0,1]:.4f}")

model_bi = DiffusionModel(
    timesteps=100,
    beta_start=0.0001,
    beta_end=0.02,
    schedule='cosine',
    use_pca=False
)

print("\nFitting DiffusionModel...")
model_bi.fit(Y_bi, n_steps=1500)

print("\nGenerating samples...")
Y_sim_bi = model_bi.sample(n_samples=500)

print("\n--- Hypothesis Tests ---")
results_bi = test_similarity(Y_bi, Y_sim_bi)

print("\n--- Reconstruction Error ---")
mmd_error = model_bi.reconstruction_error(Y_bi, metric='mmd', n_samples=500)
energy_error = model_bi.reconstruction_error(Y_bi, metric='energy', n_samples=500)
print(f"MMD: {mmd_error:.6f}")
print(f"Energy Distance: {energy_error:.6f}")

visualize_comparison(Y_bi, Y_sim_bi, "Bivariate Normal: DiffusionModel")


# =============================================================================
# Example 3: 3D Mixture of Gaussians (from PDF Section 2)
# =============================================================================
print("\n" + "="*70)
print("EXAMPLE 3: 3D MIXTURE OF GAUSSIANS")
print("="*70)

np.random.seed(42)
n_samples = 800

# Component 1 (60%)
cov1 = np.array([[1.0, 0.5, 0.3],
                 [0.5, 1.5, 0.4],
                 [0.3, 0.4, 0.8]])
component1 = np.random.multivariate_normal([0, 1, -0.5], cov1, int(0.6 * n_samples))

# Component 2 (40%)
cov2 = np.array([[0.8, -0.3, 0.1],
                 [-0.3, 1.2, -0.2],
                 [0.1, -0.2, 1.0]])
component2 = np.random.multivariate_normal([2, -1, 1], cov2, int(0.4 * n_samples))

Y_3d = np.vstack([component1, component2])
np.random.shuffle(Y_3d)

print(f"\nData: {n_samples} samples from mixture of 2 MVN")
print("Original correlation matrix:")
print(np.corrcoef(Y_3d.T))

model_3d = DiffusionModel(
    timesteps=150,
    beta_start=0.0001,
    beta_end=0.02,
    schedule='cosine',
    use_pca=False
)

print("\nFitting DiffusionModel...")
model_3d.fit(Y_3d, n_steps=2000)

print("\nGenerating samples...")
Y_sim_3d = model_3d.sample(n_samples=800)

print("\n--- Hypothesis Tests ---")
results_3d = test_similarity(Y_3d, Y_sim_3d)

print("\n--- Reconstruction Error ---")
mmd_error = model_3d.reconstruction_error(Y_3d, metric='mmd', n_samples=500)
energy_error = model_3d.reconstruction_error(Y_3d, metric='energy', n_samples=500)
print(f"MMD: {mmd_error:.6f}")
print(f"Energy Distance: {energy_error:.6f}")

print("\nSimulated correlation matrix:")
print(np.corrcoef(Y_sim_3d.T))

visualize_comparison(Y_3d, Y_sim_3d, "3D Mixture of Gaussians: DiffusionModel")


# =============================================================================
# Example 4: MNIST Digits with PCA (from PDF Section 3)
# =============================================================================
print("\n" + "="*70)
print("EXAMPLE 4: MNIST DIGITS (8x8) WITH PCA")
print("="*70)

np.random.seed(42)
digits = load_digits()
Y_digits = digits.data  # Shape: (1797, 64)

print(f"\nData: {Y_digits.shape[0]} digit images of shape 8x8 (64 dimensions)")
print(f"Value range: [{Y_digits.min():.1f}, {Y_digits.max():.1f}]")

# Apply PCA as in PDF
pca = PCA(n_components=15, whiten=False)
Y_digits_pca = pca.fit_transform(Y_digits)
print(f"\nPCA applied: 64 -> 15 dimensions")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

model_digits = DiffusionModel(
    timesteps=150,
    beta_start=0.0001,
    beta_end=0.02,
    schedule='cosine',
    use_pca=True,
    pca_components=15
)

print("\nFitting DiffusionModel...")
model_digits.fit(Y_digits, n_steps=2000)

print("\nGenerating samples...")
Y_sim_digits = model_digits.sample(n_samples=44)

print("\n--- Reconstruction Error ---")
mmd_error = model_digits.reconstruction_error(Y_digits, metric='mmd', n_samples=500)
energy_error = model_digits.reconstruction_error(Y_digits, metric='energy', n_samples=500)
print(f"MMD: {mmd_error:.6f}")
print(f"Energy Distance: {energy_error:.6f}")

# Visualize digits (similar to PDF)
def visualize_digits(real_data, sim_data, n_samples=44):
    fig, axes = plt.subplots(9, 11, figsize=(11, 9), 
                            subplot_kw=dict(xticks=[], yticks=[]))
    
    # Hide middle row
    for j in range(11):
        axes[4, j].set_visible(False)
    
    # Plot real digits (top 4 rows)
    for i in range(4):
        for j in range(11):
            idx = i * 11 + j
            if idx < len(real_data):
                axes[i, j].imshow(real_data[idx].reshape(8, 8), 
                                cmap='binary', interpolation='nearest')
    
    # Plot simulated digits (bottom 4 rows)
    for i in range(4):
        for j in range(11):
            idx = i * 11 + j
            if idx < len(sim_data):
                axes[i+5, j].imshow(sim_data[idx].reshape(8, 8), 
                                  cmap='binary', interpolation='nearest')
    
    axes[0, 5].set_title("Real Digits", fontsize=12, pad=10)
    axes[5, 5].set_title("DiffusionModel Samples", fontsize=12, pad=10)
    
    plt.tight_layout()
    plt.show()

print("\n--- Visualizing Digits ---")
visualize_digits(Y_digits[:44], Y_sim_digits[:44])


print("\n" + "="*70)
print("ALL EXAMPLES COMPLETED")
print("="*70)