"""
JL-Theorem-Prover: A Rigorous Johnson-Lindenstrauss Implementation
Core Theorem: For any n points in ℝ^d and ε ∈ (0,1), ∃ linear map f:ℝ^d→ℝ^k with
              k ≥ (24 log n)/(3ε² - 2ε³) preserving pairwise distances within (1±ε)
"""
import numpy as np
from math import ceil
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import seaborn as sns

class JLTheoremProver:
    """
    A theoretically-exact JL Lemma implementation with violation diagnostics.
    
    Parameters
    ----------
    eps : float (default=0.5)
        Maximum distortion rate (1±ε) of pairwise distances. Must satisfy 0 < ε < 1.5.
    projection_type : {'gaussian', 'sparse'} (default='gaussian')
        Random projection matrix type adhering to the lemma's existence proof.
        
    Theory
    ------
    Given Lemma 2 from user input:
    k ≥ (24 log n) / (3ε² - 2ε³)
    Probability of violating (1±ε) for any pair: F ≤ 2/n²
    Union bound over all pairs: Total failure probability ≤ 1 - 1/n
    """
    def __init__(self, eps=0.5, projection_type='gaussian'):
        assert 0 < eps < 1.5, "ε must be in (0, 1.5) to satisfy 3ε² > 2ε³"
        self.eps = eps
        self.projection_type = projection_type.lower()
        self.scaler = StandardScaler()  # Critical for JL's expectation guarantees
        self.projector = None
        self.fitted_dim_ = None
        
    def _jl_theorem_dim(self, n_samples):
        """Compute minimal k per lemma: k ≥ (24 log n)/(3ε² - 2ε³)"""
        numerator = 24 * np.log(n_samples)  # Natural logarithm for lemma compliance
        denominator = 3 * self.eps**2 - 2 * self.eps**3
        return ceil(numerator / denominator)
    
    def _create_projection_matrix(self, d, k):
        """Construct random matrix per lemma's probabilistic existence proof."""
        if self.projection_type == 'gaussian':
            # Gaussian matrix with variance 1/k for E[||Ax||²] = ||x||²
            return np.random.normal(0, 1/np.sqrt(k), size=(d, k))
        elif self.projection_type == 'sparse':
            # Achlioptas' sparse construction for O(1) sparsity
            return np.sqrt(3/k) * np.random.choice(
                [0, 1, -1], size=(d, k), p=[2/3, 1/6, 1/6]
            )
        else:
            raise ValueError(f"Invalid projection type: {self.projection_type}")

    def fit_transform(self, X):
        """
        Apply JL projection with lemma-exact dimension selection.
        
        Theory
        ------
        1. Scale data to unit variance (ensures ||x_i|| ≈ O(1))
        2. Construct random matrix A ∈ ℝ^(d×k) with k from theorem
        3. Project X to ℝ^k via X_proj = X_scaled @ A
        """
        X_scaled = self.scaler.fit_transform(X)
        n_samples, d = X_scaled.shape
        
        # Calculate k using theorem's formula
        k = self._jl_theorem_dim(n_samples)
        self.fitted_dim_ = k
        
        if k >= d:
            print(f"Warning: JL dimension {k} ≥ original {d}. No projection applied.")
            return X_scaled
        
        # Build and apply projection matrix
        A = self._create_projection_matrix(d, k)
        return X_scaled.dot(A)
    
    def validate_theorem(self, X, X_proj, sample_size=500):
        """
        Empirically validate lemma's pairwise distance preservation.
        
        Theory
        ------
        For all pairs (x_i, x_j), compute ratio ||Ax_i - Ax_j||² / ||x_i - x_j||²
        Violation rate should be ≤ 2/n² per pair (validated via union bound).
        """
        # Subsample to avoid O(n²) computation
        np.random.seed(42)
        idx = np.random.choice(len(X), size=sample_size, replace=False)
        X_sample, X_proj_sample = X[idx], X_proj[idx]
        
        # Compute all pairwise squared distances
        orig_dists = pdist(X_sample, 'sqeuclidean')
        proj_dists = pdist(X_proj_sample, 'sqeuclidean')
        
        # Calculate empirical violation rate
        ratios = proj_dists / orig_dists
        violations = (ratios < (1 - self.eps)) | (ratios > (1 + self.eps))
        violation_rate = np.mean(violations)
        
        # Theoretical expected violations in subsample
        n = X.shape[0]
        sample_pairs = (sample_size * (sample_size - 1)) // 2
        theo_violations = sample_pairs * (2 / n**2)
        empirical_violations = violation_rate * sample_pairs
        
        # Plot results
        plt.figure(figsize=(10, 6))
        sns.histplot(ratios, bins=50, kde=True, stat='density')
        plt.axvline(1 - self.eps, color='r', linestyle='--', label=f'1-ε={1-self.eps:.2f}')
        plt.axvline(1 + self.eps, color='r', linestyle='--', label=f'1+ε={1+self.eps:.2f}')
        plt.title(f'JL Lemma Validation (ε={self.eps})\n'
                  f'Empirical: {empirical_violations:.1f} vs Theoretical: {theo_violations:.1f} Violations')
        plt.xlabel('Projected/Original Squared Distance Ratio')
        plt.legend()
        plt.show()
        
        return violation_rate

#==================================================================
# Example Usage with Theorem-Exact Parameters
#==================================================================
if __name__ == "__main__":
    # Generate high-dimensional data (n=1000, d=500)
    np.random.seed(42)
    X = np.random.randn(1000, 500)
    
    # Initialize JL prover with ε=0.5 (k=332 < d=500)
    jl_prover = JLTheoremProver(eps=0.5)
    
    # Apply JL projection (d=500 → k=332)
    X_proj = jl_prover.fit_transform(X)
    print(f"Original dim: {X.shape[1]} → Projected dim: {jl_prover.fitted_dim_}")
    
    # Validate theorem compliance
    violation_rate = jl_prover.validate_theorem(X, X_proj)
    
    # Theoretical vs empirical diagnostics
    n = X.shape[0]
    sample_pairs = (500 * 499) // 2  # sample_size=500
    print(f"\nTheoretical violations (subsample): {sample_pairs * 2 / n**2:.4f}")
    print(f"Empirical violation rate: {violation_rate:.4f}")
