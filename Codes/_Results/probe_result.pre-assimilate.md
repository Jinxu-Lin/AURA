> [ASSIMILATED: generated from iter_001/exp/results/pilots/ and phase1 results]

# Probe Results — AURA

## Phase 0 Probe (Pre-Sibyl)

**Setup**: 3 training seeds × 100 CIFAR-10 test points, last-layer Influence Functions, ResNet-18.

### Hessian Hierarchy Verification
- Condition number κ: ~1.2-1.4×10⁶ (severe ill-conditioning, confirms Hessian approximation matters)
- J@10 (EK-FAC vs EK-FAC, same approx): 1.0 (perfect agreement as expected)
- J@10 (EK-FAC vs K-FAC): 0.45-0.53 (nearly half of top-10 attributions change)
- J@10 (EK-FAC vs Diagonal): 0.21-0.28 (most attributions change)

### TRV Distribution
- TRV follows trimodal distribution: low-sensitivity cluster (~30% of points), medium (~45%), high (~25%)
- Suggests structured per-test-point variation, not random noise

### SI-TRV Correlation
- Spearman ρ(SI, TRV) ≈ 0.0 across all 3 seeds
- Interpretation: Self-Influence measures perturbation sensitivity of individual training points, NOT Hessian approximation sensitivity of test points. These are orthogonal dimensions.
- **Conclusion: A5 (SI is valid TRV proxy) FALSIFIED**

### Cross-Seed TRV Stability
- Spearman ρ(TRV_seed_i, TRV_seed_j): mean ≈ -0.006, range [-0.05, 0.04]
- TRV rankings are completely unstable across training seeds
- Root cause: TRV depends on eigenvector directions, which are seed-unstable
- **Conclusion: TRV is not a viable diagnostic. Motivates BSS approach.**

---

## Phase 1: Variance Decomposition (Sibyl System)

**Setup**: 500 CIFAR-10 test points (50 per class, stratified sampling, seed 42), ResNet-18 (seed 42, 95.50% test accuracy), full-model attributions.

**Methods**: EK-FAC IF, K-FAC IF, RepSim, TRAK-50.

### Metrics Computed
- **J10**: Jaccard@10 between EK-FAC IF and K-FAC IF top-10 attributed training points
- **tau**: Kendall rank correlation between IF (EK-FAC) and RepSim attribution rankings
- **LDS**: Linear Datamodeling Score (per-test-point)

### Two-Way ANOVA Results (class × log(gradient_norm))

| Metric | Class η² | Gradient Norm η² | Interaction η² | **Residual** | Gate (>30%) |
|--------|---------|------------------|----------------|-------------|-------------|
| J10 | 8.2% | 14.3% | ~0% | **77.5%** | **PASS** |
| tau | 18.1% | 28.5% | ~0% | **53.4%** | **PASS** |
| LDS | 22.4% | 31.7% | ~0% | **45.9%** | **PASS** |

### Interpretation
- Class and gradient norm together explain only 22.5% of J10 variance — the vast majority of per-test-point Hessian sensitivity is driven by factors beyond these two obvious confounds.
- LDS has the most explainable variance (class + gradient norm = 54.1%), but still 45.9% residual.
- **Conclusion: A1 (per-test-point sensitivity exists beyond confounds) CONFIRMED**

### Per-Class J10 Statistics
- Mean J10 ranges from 0.42 (class 3: cat) to 0.61 (class 1: automobile)
- Within-class J10 standard deviation: 0.15-0.22 (substantial within-class variation)
- Confirms that class alone does not determine sensitivity

---

## Phase 2a Pilot: BSS Computation (Sibyl System)

**Setup**: 100 CIFAR-10 test points (subset of Phase 1 500), 1 training seed (seed 42), GGN top-100 eigenvalues via Kronecker-factored eigendecomposition.

### BSS Bucket Definitions
- **Outlier**: Top 5 eigenvalues (λ > 10× median)
- **Edge**: Eigenvalues ranked 6-30 (1× to 10× median)
- **Bulk**: Eigenvalues ranked 31-100 (< 1× median)

### Results
- BSS_outlier within-class variance: 93.5% (non-degenerate — not just a class effect)
- BSS_outlier Spearman correlation with J10: -0.42 (higher BSS_outlier → lower J10 → more sensitive)
- **BSS_outlier correlation with gradient norm: ρ = 0.906** (CONCERN)
  - This is problematic: if BSS_outlier is just gradient norm in disguise, it adds no diagnostic value
  - Mitigation: Compute partial BSS (regress out gradient norm) in Phase 2a full experiment
- BSS_edge and BSS_bulk show weaker but non-trivial correlations with J10

### Cross-Seed Stability (NOT YET TESTED)
- Phase 2a full experiment requires 5 training seeds (4 still in training)
- Hypothesis: BSS rankings should be more stable than TRV because they depend on eigenvalue magnitudes (stable) rather than eigenvector directions (unstable)

---

## Phase 2b: IF-RepSim Disagreement Analysis (Sibyl System)

**Setup**: 500 test points (same as Phase 1), EK-FAC IF and RepSim attribution rankings.

### Cross-Method Correlation
- Kendall tau(IF, RepSim) per test point: mean = -0.467, std = 0.12
- Strong negative correlation: IF and RepSim produce systematically different (often inverted) rankings
- Consistent with Choe et al. finding of 0.37-0.45 correlation at concept level

### Disagreement as Diagnostic
- Binary disagreement metric: |tau(IF, RepSim)| > median → "high disagreement"
- AUROC for predicting J10_low (bottom quartile): **0.691**
- Class-stratified AUROC: **0.746** (improves when controlling for class effects)
- **Conclusion: A3 (IF-RepSim disagreement is informative) CONFIRMED**

### Interpretation
- When IF and RepSim strongly disagree on a test point's attributions, that test point is more likely to have Hessian-sensitive attributions (low J10)
- This makes intuitive sense: IF relies on Hessian inverse, RepSim does not. When they disagree, the Hessian component is likely driving IF in a different direction.
- Disagreement signal is complementary to BSS (different information source)

---

## Summary of Assumption Status

| # | Assumption | Status | Evidence |
|---|------------|--------|----------|
| A1 | Per-test-point sensitivity beyond confounds | **CONFIRMED** | J10 residual 77.5% |
| A2 | BSS cross-seed stability (ρ > 0.5) | **TESTING** | Pilot promising but ρ(BSS, grad_norm) = 0.906 concern |
| A3 | IF-RepSim disagreement informative | **CONFIRMED** | AUROC 0.691 (stratified 0.746) |
| A4 | Adaptive > uniform | **PLANNED** | Requires Phase 3 |
| A5 | SI is valid TRV proxy | **FALSIFIED** | ρ(SI, TRV) ≈ 0 |
