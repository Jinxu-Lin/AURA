---
version: "1.0"
entry_mode: "assimilated"
iteration_major: 1
iteration_minor: 0
---

# Method Design: AURA — Bucketed Spectral Sensitivity for TDA Diagnosis

## Component 1: Attribution Variance Decomposition (COMPLETED)

**Purpose**: Establish that per-test-point attribution sensitivity is a real phenomenon, not an artifact of class or gradient-norm confounds.

**Method**: Two-way ANOVA with factors class (10 levels) and log(gradient_norm) (continuous, binned into quartiles).

**Response variables**:
- J10: Jaccard@10 between EK-FAC IF and K-FAC IF top-10 attributions
- tau: Kendall rank correlation between IF (EK-FAC) and RepSim rankings
- LDS: Linear Datamodeling Score (per-test-point)

**Data**: 500 CIFAR-10 test points (50 per class, stratified sampling, seed 42).

**Models**: ResNet-18 (seed 42, 95.50% test accuracy).

**TDA Methods**: EK-FAC IF, K-FAC IF, RepSim, TRAK-50.

**Gate**: Residual variance > 30% on at least one metric.

**Result**: J10 residual 77.5%, tau residual 53.4%, LDS residual 45.9%. **ALL PASS**.

---

## Component 2: Bucketed Spectral Sensitivity (BSS) (IN PROGRESS)

**Purpose**: Provide a seed-stable, per-test-point diagnostic for Hessian sensitivity.

### 2.1 Theoretical Foundation

**Random matrix theory (RMT)**: For neural network weight matrices, the eigenvalue distribution of the Hessian (or GGN) is architecture-determined. While individual eigenvector directions are unstable across training seeds (especially for non-isolated eigenvalues), the eigenvalue magnitude distribution — and in particular, the separation between outlier, edge, and bulk eigenvalues — is stable. This is the Marchenko-Pastur law for the bulk and Tracy-Widom law for edge statistics.

**Hampel's gross error sensitivity**: BSS can be viewed as a computational version of Hampel's influence function sensitivity, decomposed by spectral scale. The sensitivity of a test point's attribution to Hessian approximation error is determined by how much its gradient projects onto high-error eigenspaces.

### 2.2 BSS Computation

**Step 1**: Eigendecomposition of the GGN (top-100 eigenvalues/vectors).
- Use Kronecker-factored eigendecomposition for efficiency.
- For each layer l with Kronecker factors A_l (input) and G_l (output):
  - Eigendecompose A_l = U_A Λ_A U_A^T and G_l = U_G Λ_G U_G^T
  - GGN eigenvalues ≈ λ_a × λ_g (Kronecker product of per-layer eigenvalues)
  - GGN eigenvectors ≈ vec(U_G[:, j] U_A[:, i]^T) for corresponding (i,j) pairs
- Select top-100 eigenvalues globally across all layers

**Step 2**: Bucket assignment.
- **Outlier bucket**: Top eigenvalues with λ > 10× median (typically 3-10 eigenvalues)
- **Edge bucket**: Eigenvalues from 1× to 10× median
- **Bulk bucket**: Eigenvalues below median

**Step 3**: Per-test-point BSS computation.
For each test point z with gradient g_z:

```
BSS_bucket(z) = Σ_{λ∈bucket} (g_z^T q_λ)² · |1/λ_ekfac - 1/λ_kfac|²
```

where q_λ is the eigenvector corresponding to eigenvalue λ, and the |1/λ_ekfac - 1/λ_kfac|² term captures the Hessian approximation error at that spectral scale.

**Step 4**: Normalization.
- BSS_bucket(z) is normalized by ||g_z||² to obtain the fractional sensitivity in each bucket
- This helps disentangle gradient magnitude from spectral alignment

### 2.3 Cross-Seed Stability Hypothesis

**Claim**: BSS rankings (ordering of test points by BSS_outlier) are stable across training seeds, even though TRV rankings are not.

**Mechanism**: BSS depends on eigenvalue magnitudes (stable via RMT) and gradient-eigenspace alignments aggregated at the bucket level. Individual eigenvector directions rotate across seeds, but the aggregation within magnitude-defined buckets absorbs this rotation.

**Test**: Compute BSS for 500 test points across 5 training seeds. Gate: Spearman ρ > 0.5 for BSS_outlier rankings.

### 2.4 Known Issue: Gradient Norm Correlation

**Problem**: In the 100-point pilot, BSS_outlier correlates with gradient norm at ρ = 0.906. This is expected to some degree (larger gradients project more onto every eigenspace), but if BSS is just gradient norm in disguise, it adds no diagnostic value.

**Mitigation**: Partial BSS.
1. Regress BSS_outlier on ||g_z||² to obtain residual BSS
2. Test whether residual BSS still correlates with J10 (partial correlation > 0.1)
3. Test whether residual BSS is cross-seed stable (ρ > 0.5)
4. If partial BSS fails: consider BSS ratio (BSS_outlier / BSS_total) as an alternative that is scale-invariant

### 2.5 Computational Cost
- Eigendecomposition: ~0.5 GPU-hours per seed (Kronecker-factored, ResNet-18)
- BSS computation: ~0.1 GPU-hours per seed (500 test points × 100 eigenvalues)
- Total for 5 seeds: ~3 GPU-hours

---

## Component 3: Adaptive Method Selection (PLANNED)

**Purpose**: Leverage BSS diagnostic and cross-method disagreement to route between IF and RepSim on a per-test-point basis, improving the LDS-vs-compute Pareto frontier.

### 3.1 Routing Strategies

**Strategy A: BSS-guided routing**
- If BSS_outlier(z) > threshold → use RepSim (IF is unreliable for this point)
- If BSS_outlier(z) ≤ threshold → use IF (more informative when Hessian error is low)
- Threshold selected via LOO validation

**Strategy B: Disagreement-guided routing**
- If |tau(IF, RepSim)| > threshold for z → use RepSim
- Requires computing both methods first (higher compute cost)

**Strategy C: Class-conditional routing**
- Per-class BSS threshold (since classes have different sensitivity distributions)
- Lower overhead than per-point, but less precise

**Strategy D: Feature-based routing**
- Train a lightweight classifier on [BSS_outlier, BSS_edge, gradient_norm, class] → {IF, RepSim}
- Target: maximize per-point LDS

### 3.2 Baselines (Mandatory)

Per debate consensus:
- **IF-only**: EK-FAC IF for all test points
- **RepSim-only**: RepSim for all test points
- **TRAK-50**: TRAK with 50 models for all test points
- **W-TRAK**: Natural W-TRAK (Grosse et al.) for all test points
- **Naive ensemble**: Average IF and RepSim rankings

### 3.3 Evaluation

- Primary: Per-point LDS, mean across test set
- Secondary: Jaccard@10, Kendall tau
- Cost: GPU-hours per method/strategy
- Pareto frontier: Plot LDS vs GPU-hours for all strategies

### 3.4 Computational Cost
- LOO validation: ~4 GPU-hours (100 held-out points × multiple strategies)
- Full evaluation: ~2 GPU-hours
- Total: ~6 GPU-hours

---

## Theoretical Foundation (Cross-Component)

### Random Matrix Theory
Eigenvalue magnitude distributions of neural network Hessians are determined by architecture and dataset statistics, not training randomness. The Marchenko-Pastur law governs the bulk distribution; outlier eigenvalues follow Tracy-Widom statistics at the edge. Both are seed-stable.

### Hampel's Gross Error Sensitivity
BSS is a computational version of gross error sensitivity (Hampel 1974), decomposed by spectral scale. For each test point, BSS measures how much the IF attribution changes under perturbation of the Hessian at each spectral scale.

### Doubly Robust Estimation (Chernozhukov 2018)
Theoretical inspiration for IF+RepSim fusion. However, the doubly robust property requires error independence (H2 from original design), which is a weak assumption. We do NOT claim doubly robust properties — instead, we take a pragmatic adaptive selection approach.

### Distributional TDA (Wang et al.)
The d-TDA framework provides a unified estimand under which both IF and kernel-based methods can be viewed. This may provide theoretical justification for adaptive selection, but we use it only as conceptual motivation, not as a formal guarantee.
