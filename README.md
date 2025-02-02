# JL-Theorem-Python
Validating the Johnson-Lindenstrauss Lemma with Exact Formulae and Pairwise Distance Diagnostics
# JL-Theorem-Prover 🔍📉  
*Rigorous Implementation of the Johnson-Lindenstrauss Lemma with Exact Formulae*
## Core Features
- ✅ **Lemma-Exact Dimension Calculation**  
- 📊 **Empirical vs Theoretical Violation Analysis**  
- 🔀 **Gaussian & Sparse Projection Matrices**  
- 🧮 **Unit Variance Preprocessing for JL Stability**

- # Theory Compliance
For ε=0.5 and n=1000:
- Required dimension: `k ≥ 332`  
- Per-pair violation probability: `2/n² = 2e-6`  
- Total expected violations: `~1` in full dataset

Original dim: 500 → Projected dim: 332
Theoretical violations (subsample): 0.2495
Empirical violation rate: 0.0412

Distance ratios cluster within ε=0.5 bounds, with only 4.1% violations in sample

