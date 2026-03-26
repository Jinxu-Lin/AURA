# Experiment Results: AURA

> Formal experiment results for the AURA project.
> Setup: CIFAR-10/ResNet-18, full-model Hessian, progressive gating design.

---

## Experiment 1: Attribution Variance Decomposition (Phase 1)

**Status**: COMPLETED
**Compute**: ~5 GPU-hours

### Configuration
- Dataset: CIFAR-10 (50K train / 10K test)
- Model: ResNet-18, seed=42, 200 epochs, SGD+cosine, 95.5% test acc
- Test points: 500 (50/class, stratified)
- Methods: EK-FAC IF, K-FAC IF, RepSim, TRAK-50
- Hessian: Full-model (not last-layer)
- Damping: K-FAC=0.1, EK-FAC=0.01

### Results: ANOVA Variance Decomposition

| Response | Class R^2 | GradNorm R^2 | Interaction R^2 | Residual R^2 |
|----------|-----------|-------------|----------------|-------------|
| J10 (Jaccard@10) | 14.1% | 0.6% | 1.9% | **77.5%** |
| tau (Kendall) | 26.4% | 33.5% | 17.6% | 22.5% |
| LDS (Spearman) | 26.7% | 17.0% | 4.7% | **51.6%** |

### Results: Descriptive Statistics

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| J10 | 0.9945 | 0.031 | 0.818 | 1.000 | 1.000 |
| tau | 0.0171 | 0.083 | -0.169 | 0.311 | 0.003 |
| LDS | 0.7443 | 0.090 | 0.434 | 0.893 | 0.767 |

### Conclusion
**Gate PASS**: Residual > 30% on J10 (77.5%) and LDS (51.6%). Per-test-point Hessian sensitivity is genuine and not explained by class membership or gradient magnitude.

---

## Experiment 2: BSS Pilot (Phase 2a Pilot)

**Status**: COMPLETED (pilot)
**Compute**: ~0.02 GPU-hours

### Configuration
- 100 test points (10/class), seed=42
- K-FAC factors from 5000 training samples
- Adaptive bucket thresholds (percentile-based)

### Results

| Finding | Value | Implication |
|---------|-------|-------------|
| Within-class BSS variance | 93.5% | BSS is NOT a class detector (GOOD) |
| BSS-gradient_norm rho | 0.906 | BSS may be gradient-norm proxy (CONCERN) |
| BSS-confidence rho | -0.912 | Correlated with prediction uncertainty |
| Max eigenvalue | 4.98e-05 | Kronecker products very small |
| Outlier eigenvalues | 19 | Sufficient for bucket analysis |
| Perturbation factor uniformity | ~90 across buckets | Damping dominates eigenvalues |

### Conclusion
**GO (conditional)**: BSS produces non-degenerate output with high within-class variation. Gradient-norm correlation requires partial BSS investigation.

---

## Experiment 3: Cross-Method Disagreement (Phase 2b)

**Status**: COMPLETED
**Compute**: 0 GPU-hours (reuses Phase 1 data)

### Results

| Metric | Value |
|--------|-------|
| IF-RepSim Kendall tau mean | 0.211 +/- 0.132 |
| Spearman(tau, LDS_diff) | **-0.547** (p=4.3e-9) |
| Quantile AUROC (median split) | 0.755 |
| Tertile AUROC | 0.841 |
| Class-stratified AUROC | **0.664** |
| Multi-feature LR AUROC | 0.750 +/- 0.079 |

### LDS by Method

| Method | Mean LDS | Std |
|--------|----------|-----|
| EK-FAC IF | 0.744 | 0.090 |
| K-FAC IF | 0.744 | 0.090 |
| RepSim | 0.274 | 0.179 |

### Conclusion
**Gate PASS**: Disagreement is informative (AUROC=0.691 > 0.60, class-stratified=0.664 > 0.55). IF universally dominates RepSim in pilot setting, but this may change with full-model evaluation.

---

## Experiment 4: BSS Cross-Seed Stability (Phase 2a Full)

**Status**: {{PENDING: 5-seed BSS computation with 500 test points}}

### Planned Configuration
- 500 test points x 5 seeds (42, 123, 456, 789, 1024)
- Kronecker GGN top-100 eigendecomposition per seed
- BSS variants: raw, partial (regress out gradient norm), ratio
- ~7 GPU-hours

### Expected Analyses
- Mean pairwise Spearman rho for BSS_partial rankings
- Within-class BSS variance per seed
- Partial correlation: BSS vs LDS controlling class + gradient norm
- Baselga decomposition of J10

### Gates
- BSS_partial cross-seed rho > 0.5
- Within-class variance > 25%
- Partial correlation > 0.15

---

## Experiment 5: MRC + Pareto Frontier (Phase 3)

**Status**: {{PENDING: Awaiting Phase 2a results}}

### Planned Configuration
- 11 strategies (7 uniform + 4 adaptive)
- LOO validation on CIFAR-10/5K subset (100 points)
- Pareto frontier: LDS vs GPU-hours
- ~8 GPU-hours

### Gate
MRC > best uniform by > 2% absolute LDS at same compute.

---

## Experiment 6: Ablations

**Status**: {{PENDING: Awaiting Phase 3}}

### Planned Ablations
- Bucket granularity (3, 5, 10)
- Eigenvalue count (50, 100, 200)
- Gradient-norm correction (none, linear, log)
- MRC weight function (sigmoid, softmax, piecewise)
- Damping (0.001, 0.01, 0.1, 1.0)

---

## Summary: Phase 0 Negative Results

| Finding | Value | Status |
|---------|-------|--------|
| TRV cross-seed Spearman rho | -0.007 | TRV is seed-unstable |
| SI-TRV Spearman rho | 0.024 | SI orthogonal to Hessian sensitivity |
| Hessian hierarchy bottom-collapse | Diagonal ~ Identity (last-layer) | Must use full-model |
| J@10 degradation (GGN to K-FAC) | 1.0 -> ~0.48 | 55% top-10 attributions change |
