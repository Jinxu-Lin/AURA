> [ASSIMILATED: synthesized from AURA experimental data (iter_001/exp/results/)]

# Probe Results -- AURA (Prior Empirical Evidence for FM1/FM2 Framework)

This document synthesizes AURA's completed experimental results (CIFAR-10/ResNet-18, full-model, 500 test points) as prior empirical evidence supporting the FM1/FM2 diagnostic framework. These results serve as "probe evidence" for the unified research direction.

---

## Phase 0: Probe Data Reanalysis

**Setup**: Raw probe data from 3 seeds x 100 ID test points, CIFAR-10/ResNet-18, last-layer Hessian.
**Compute**: 0 GPU-hours (pure reanalysis of cached data).

### Key Findings

**GUM Uncertainty Budget (TRV Variance Decomposition)**:
| Component | Fraction | Interpretation |
|-----------|----------|----------------|
| Seed | 1.4% | Training randomness contributes minimally |
| Class | 3.7% | Class membership explains very little |
| Residual | 94.9% | Dominated by per-point variation |

**Cross-Seed TRV Stability (CONFIRMED UNSTABLE)**:
- Mean Spearman rho: **-0.007** (essentially zero)
- TRV is NOT a test-point intrinsic property -- it is a (model-instance, test-point) joint property.
- This finding motivates the spectral approach (eigenvalue-magnitude buckets are seed-stable even when individual eigenvectors rotate).

**Correlation Matrix**: TRV is uncorrelated with SI (rho ~ 0.024), gradient norm (rho ~ -0.020), confidence (rho ~ 0.051), and entropy (rho ~ -0.032). TRV captures information orthogonal to standard point-level features.

**Hessian Hierarchy Collapse**: Only 2-3 effective levels in last-layer setting. Diagonal ~ Damped-ID ~ Identity (differences < 0.02). **Must use full-model Hessian** to preserve K-FAC/EK-FAC gap.

### Relevance to FM1/FM2 Framework

The TRV instability and Hessian hierarchy findings confirm that Hessian approximation quality is a genuine per-sample concern (not a global effect), consistent with FM1 operating at different magnitudes across samples.

---

## Phase 1: Attribution Variance Decomposition

**Setup**: 500 CIFAR-10 test points (50/class, stratified), single seed (42), full-model Hessian. Two-way ANOVA with class (10 levels) and log(gradient_norm), Type I sequential SS.
**Compute**: ~5 GPU-hours.

### Variance Decomposition Results

| Response | Class R^2 | GradNorm R^2 | Interaction R^2 | Residual R^2 |
|----------|-----------|-------------|----------------|-------------|
| J10 (EK-FAC vs K-FAC) | 0.182 | 0.009 | 0.035 | **0.775** |
| tau (IF vs RepSim) | 0.064 | 0.348 | 0.054 | **0.534** |
| LDS (per-point) | 0.121 | 0.405 | 0.015 | **0.459** |

**Gate**: Residual > 30% on at least 1 metric = **PASS** (all three pass).

### Key Observations

- **J10**: Residual dominates (77.5%) -- Hessian sensitivity is strongly per-point, not explained by class or gradient norm. This validates that FM1 (signal dilution) creates per-sample variation in attribution quality.
- **tau (IF-RepSim disagreement)**: Gradient norm explains 34.8% -- samples with larger gradients tend to have more similar IF and RepSim attributions (both methods succeed when signal is strong). Residual 53.4% confirms genuine per-point complementarity.
- **LDS**: Gradient norm explains 40.5% -- gradient geometry fundamentally determines attribution quality. This is direct evidence for FM1: in high dimensions, gradient norm is a proxy for SNR.

### Descriptive Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| J10 | 0.835 | 0.128 | 0.429 | 1.000 |
| tau (IF-RepSim) | **-0.467** | 0.114 | -0.619 | -0.048 |
| LDS | 0.297 | 0.101 | 0.089 | 0.584 |

**Critical finding**: Mean Kendall tau = -0.467 between IF and RepSim rankings. The two methods are **systematically anti-correlated** -- they do not merely disagree due to noise, but capture fundamentally different aspects of training data influence. This is the strongest single piece of evidence supporting the FM1/FM2 framework.

---

## Phase 2b: Cross-Method Disagreement Analysis

**Setup**: 500 test points, full-model, EK-FAC IF + K-FAC IF + RepSim + TRAK.
**Compute**: ~0 additional GPU-hours (reuses Phase 1 data).

### LDS Comparison

| Method | Mean LDS | Std |
|--------|----------|-----|
| EK-FAC IF | 0.297 | 0.101 |
| K-FAC IF | 0.271 | 0.106 |
| RepSim | 0.074 | 0.105 |

**IF-better**: 471/500 (94.2%). **RepSim-better**: 29/500 (5.8%).

### Disagreement Predictability

**AUROC** (predicting which method is better):
- Global AUROC (tau as predictor): **0.691**
- Quantile-stratified AUROC: **0.746**
- Class-stratified quantile AUROC: **0.746**

**Partial correlation** (tau vs LDS_diff | class + gradient_norm): **rho = 0.266, p = 1.59e-9**
This confirms the disagreement signal is NOT a class proxy -- genuine per-sample structure persists after controlling for class and gradient norm.

### Key Correlations with LDS_diff (IF advantage magnitude)

| Feature | Spearman rho | p-value |
|---------|-------------|---------|
| tau (IF-RepSim) | 0.524 | 1.25e-36 |
| log_grad_norm | 0.505 | 1.10e-33 |
| confidence | -0.436 | 1.39e-24 |
| entropy | 0.428 | 1.25e-23 |

### Gate Decision: **PASS**

- Global AUROC: 0.691 > 0.60 threshold
- Class-stratified AUROC: 0.746 > 0.55 threshold

---

## Summary: Relevance to FM1/FM2 Framework

| AURA Finding | Supports | Mechanism |
|-------------|----------|-----------|
| IF-RepSim tau = -0.467 | FM1/FM2 independence | Two methods capture different signal components (parameter-space vs representation-space) |
| J10 residual = 77.5% | FM1 per-sample nature | Hessian sensitivity varies per-sample, not just per-class |
| Gradient norm explains 40.5% of LDS | FM1 (signal dilution) | SNR depends on gradient geometry |
| Disagreement AUROC = 0.691 | Structured complementarity | IF-RepSim complementarity is predictable, not random |
| Partial rho = 0.266 after class control | Genuine per-sample effect | Not reducible to class membership |
| Cross-seed TRV rho ~ 0 | Attribution instability is real | Motivates principled method selection |

**Overall assessment**: AURA's experimental data provides strong prior evidence that parameter-space and representation-space TDA capture fundamentally different information, consistent with the FM1/FM2 diagnostic framework. The anti-correlation finding (tau = -0.467) is particularly compelling -- it cannot be explained by noise alone and directly supports the "two independent failure modes" thesis. The pending question is whether this pattern transfers to LLM scale on DATE-LM.
