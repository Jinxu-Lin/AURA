---
version: "1.0"
entry_mode: "assimilated"
iteration_major: 1
iteration_minor: 0
---

# Problem Statement: Per-Test-Point Hessian Sensitivity Diagnosis for Training Data Attribution

## 1. Gap

Training Data Attribution (TDA) methods produce fundamentally different attribution rankings depending on the quality of the Hessian approximation used. Specifically, switching between EK-FAC and K-FAC approximations causes Jaccard@10 to drop from 1.0 to 0.45-0.53, meaning nearly half of the top-10 attributed training examples change. This is not a minor numerical perturbation — it represents a qualitative change in which training data are deemed most influential.

**No per-test-point diagnostic exists.** The TDA community evaluates attribution quality using global metrics (mean LDS across the entire test set), which masks dramatic per-test-point variation. Our variance decomposition (Phase 1) reveals that 77.5% of the Jaccard@10 variance, 53.4% of the Kendall tau variance, and 45.9% of the LDS variance remain after controlling for class and gradient norm — meaning the per-test-point sensitivity phenomenon is real and not explainable by known confounds.

Prior diagnostic attempts fail:
- **Self-Influence (SI)**: Proposed by Grosse et al. as a Lipschitz bound on attribution sensitivity. Our probe shows SI-TRV correlation ρ ≈ 0 — SI measures a fundamentally different dimension than Hessian sensitivity.
- **True Residual Variance (TRV)**: Direct computation of attribution variance across approximations. Cross-seed ρ ≈ -0.006 — completely unstable across training seeds, making it useless as a practical diagnostic.

## 2. Root Cause

### Why does per-test-point sensitivity vary?
Test-point gradients project differently onto the Hessian error spectrum. Points whose gradients align heavily with high-error eigenspaces (where EK-FAC and K-FAC diverge most) are sensitive; points whose gradients lie in low-error eigenspaces are robust. This is a geometric property of the interaction between the test gradient direction and the Hessian approximation error structure.

### Why have prior diagnostics failed?
1. **Global evaluation masks per-point variation**: LDS averages over all test points, treating Hessian error as a uniform problem. The per-test-point dimension is invisible.
2. **SI measures the wrong thing**: SI captures how much a training point's attribution changes under infinitesimal perturbation of the point itself — not how much it changes under Hessian approximation changes. These are orthogonal dimensions.
3. **EigenVECTOR instability**: TRV and similar diagnostics rely on eigenvector directions, which are unstable across training seeds (random matrix theory predicts eigenvector instability for non-isolated eigenvalues). However, eigenVALUE magnitude distributions are architecture-determined and seed-stable.

### Root Cause Summary
The community has no reliable per-test-point diagnostic because all prior approaches either (a) measure the wrong quantity (SI), or (b) rely on seed-unstable spectral components (eigenvector directions in TRV). The path forward requires exploiting the stable component of the spectrum: eigenvalue magnitudes.

## 3. Research Questions

- **RQ1** (CONFIRMED): Does per-test-point attribution sensitivity exist beyond class and gradient-norm confounds?
  → Yes. ANOVA residual: J10 77.5%, tau 53.4%, LDS 45.9%. All exceed the 30% gate criterion.

- **RQ2** (IN PROGRESS): Does Bucketed Spectral Sensitivity (BSS) provide a seed-stable, non-degenerate per-test-point diagnostic?
  → Pilot (100 points, 1 seed) shows BSS within-class variance 93.5%, but BSS_outlier-gradient_norm correlation ρ = 0.906 raises degeneracy concern. 5-seed stability test in progress.

- **RQ3** (PLANNED): Does BSS-guided adaptive method selection improve over uniform strategies on the LDS-vs-compute Pareto frontier?
  → Requires Phase 3 experiments.

## 4. Attack Angle

**Bucketed Spectral Sensitivity (BSS)**: Decompose each test point's attribution sensitivity by Hessian eigenvalue magnitude buckets (outlier/edge/bulk).

Formula: BSS_bucket(z_test) = Σ_{λ∈bucket} (g_test^T q_λ)² · |1/λ_ekfac - 1/λ_kfac|²

**Why BSS should work where TRV failed**: Random matrix theory guarantees that eigenvalue magnitude distributions are architecture-determined and seed-stable. By bucketing eigenvalues by magnitude rather than tracking individual eigenvector directions, BSS inherits this stability. The outlier/edge/bulk decomposition captures where in the spectrum each test point's sensitivity concentrates, providing an interpretable diagnostic.

**Known challenge**: BSS_outlier is highly correlated with gradient norm (ρ=0.906 in pilot). Partial BSS (regressing out gradient norm) is the primary mitigation strategy.

## 5. Probe Summary

### Phase 0 Probe (pre-Sibyl)
- Setup: 3 seeds × 100 test points, last-layer IF on CIFAR-10/ResNet-18
- Key findings:
  - Hessian condition number κ ~1.2-1.4×10⁶ (severe ill-conditioning)
  - J@10 degradation from 1.0 (EK-FAC vs EK-FAC) to 0.45-0.53 (EK-FAC vs K-FAC)
  - TRV follows trimodal distribution (low/medium/high sensitivity clusters)
  - SI-TRV correlation ρ ≈ 0 (orthogonal dimensions)
  - Cross-seed TRV Spearman ρ ≈ -0.006 (completely unstable)

### Phase 1 (Sibyl system)
- Setup: 500 test points (50/class stratified, seed 42), full-model, 4 methods (EK-FAC IF, K-FAC IF, RepSim, TRAK-50)
- ANOVA results:
  - J10: class η² = 8.2%, gradient_norm η² = 14.3%, **residual = 77.5%** (PASS, gate: >30%)
  - tau: class η² = 18.1%, gradient_norm η² = 28.5%, **residual = 53.4%** (PASS)
  - LDS: class η² = 22.4%, gradient_norm η² = 31.7%, **residual = 45.9%** (PASS)

### Phase 2b (disagreement analysis)
- Setup: Same 500 points
- IF-RepSim Kendall tau mean: -0.467 (strong negative correlation in rankings)
- Binary disagreement → J10 low AUROC: 0.691 (class-stratified: 0.746)
- Conclusion: Cross-method disagreement carries diagnostic signal (PASS gate)
