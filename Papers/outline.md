# Paper Outline

**Title:** When Can You Trust Training Data Attribution? A Spectral Diagnostic Approach

**Venue target:** 9-page main body (NeurIPS / ICML format)

---

## Section-by-Section Outline

### Abstract (~0.25 pages)
- Problem: TDA methods sensitive to Hessian approximation; no per-point diagnostic exists
- Finding: 77.5% of J10 variance is per-test-point residual (ANOVA)
- Prior diagnostics fail: TRV seed-unstable, SI orthogonal
- Proposal: BSS (spectral diagnostic) + MRC (adaptive combining)
- Pilot results: BSS within-class variance 93.5%, cross-method AUROC 0.755

### 1. Introduction (~1.5 pages)
- Motivating scenario: practitioner gets different top-10 lists under EK-FAC vs K-FAC
- TDA progress and the global-metrics blind spot
- ANOVA variance decomposition result (77.5% / 51.6% residual)
- Why prior diagnostics fail (TRV, SI)
- Key insight: eigenvalue magnitudes are seed-stable (RMT)
- BSS definition and MRC combining (high-level)
- Contribution list: C1 (variance decomposition), C2 (BSS), C3 (MRC), C4 (negative results on prior diagnostics)

### 2. Related Work (~1.25 pages)
- 2.1 TDA Methods: IF lineage (TRAK, SOURCE, ASTRA, TrackStar, LoGra) + RepSim family
- 2.2 Hessian Approximation Quality: Hong et al. hierarchy, K-FAC vs EK-FAC mismatch
- 2.3 TDA Reliability and Diagnostics: SI, W-TRAK, Daunce, BIF, RIF -- positioning BSS as orthogonal
- 2.4 Spectral Analysis of Neural Networks: RMT, MP law, outlier/edge/bulk separation

### 3. Method (~2.5 pages)
- 3.1 Attribution Variance Decomposition: ANOVA model (Eq. 4), residual R^2 interpretation
- 3.2 Bucketed Spectral Sensitivity:
  - 3.2.1 Spectral decomposition of attribution error (Propositions 1)
  - 3.2.2 BSS definition with 3 buckets (Eq. 7)
  - 3.2.3 Gradient-norm correction: partial BSS (Eq. 8) and BSS_ratio (Eq. 9)
  - 3.2.4 Turnover decomposition (Baselga, Eq. 10)
- 3.3 MRC Soft Combining:
  - 3.3.1 Motivation from wireless communications
  - 3.3.2 Weight function (Eq. 13) with BSS + disagreement
  - 3.3.3 Calibration via LOO cross-validation
  - 3.3.4 Baselines (10 strategies + oracle)

### 4. Experiments (~2.75 pages)
- 4.1 Setup: CIFAR-10/ResNet-18, 4 TDA methods, 500 test points, metrics
- 4.2 Variance Decomposition: Tables 1-2, ANOVA results
- 4.3 BSS Pilot Results: Table 3, class detector check, gradient-norm correlation, perturbation factor uniformity
- 4.4 Cross-Method Disagreement: Table 4-5, quantile AUROC, per-class statistics
- 4.5 MRC Combining and Pareto Frontier: Table 6 (11 strategies), oracle gap closure
- 4.6 Ablation Studies: Table 7 (bucket granularity, eigenvalue count, grad-norm correction, weight function, damping, train subset)
- 4.7 Confound Controls: class-stratified AUROC, partial correlations, stability vs correctness

### 5. Conclusion (~0.75 pages)
- Summary of contributions and key numbers
- Limitations: scale (CIFAR-10 only), architecture specificity, BSS--gradient norm entanglement, evaluation scope, compute overhead
- Future work: scaling to larger models, downstream task evaluation, comprehensive uncertainty budget

---

## Page Budget

| Section | Pages |
|---------|:-----:|
| Abstract | 0.25 |
| 1. Introduction | 1.50 |
| 2. Related Work | 1.25 |
| 3. Method | 2.50 |
| 4. Experiments | 2.75 |
| 5. Conclusion | 0.75 |
| **Total** | **9.00** |

---

## Planned Figures

| ID | Location | Description |
|----|----------|-------------|
| Fig 1 | Sec 1 / Sec 3.2 | BSS conceptual diagram: GGN spectrum with outlier/edge/bulk buckets, test-point gradient projected onto each bucket, yielding per-bucket sensitivity scores |
| Fig 2 | Sec 4.2 | ANOVA residual visualization: scatter plot of J10 vs gradient norm colored by class, showing within-class spread dominates |
| Fig 3 | Sec 4.3 | BSS distribution: histogram of BSS_total across 100 pilot points, with outlier/edge/bulk stacked contributions |
| Fig 4 | Sec 4.4 | Disagreement--quality scatter: Kendall tau vs LDS_diff with regression line and per-class coloring |
| Fig 5 | Sec 4.5 | Pareto frontier: mean LDS vs GPU-hours for all 11 strategies, highlighting MRC position relative to oracle |

## Planned Tables

| ID | Location | Description |
|----|----------|-------------|
| Table 1 | Sec 4.2 | ANOVA variance decomposition (class, grad-norm, interaction, residual R^2) |
| Table 2 | Sec 4.2 | Descriptive statistics of J10, tau, LDS |
| Table 3 | Sec 4.3 | BSS by eigenvalue bucket (mean, std, min, max) |
| Table 4 | Sec 4.4 | Per-method LDS comparison (EK-FAC IF, K-FAC IF, RepSim) |
| Table 5 | Sec 4.4 | Per-class statistics (LDS, disagreement, class AUROC) |
| Table 6 | Sec 4.5 | Main results: 11 strategies LDS + GPU-hours |
| Table 7 | Sec 4.6 | Ablation results across 6 dimensions |

---

## Contribution Summary

1. **C1 -- Per-test-point variance decomposition.** First systematic ANOVA decomposition showing 77.5% (J10) and 51.6% (LDS) of TDA sensitivity is genuine per-test-point residual, not explained by class or gradient norm.

2. **C2 -- Bucketed Spectral Sensitivity (BSS).** A theoretically grounded, seed-stable per-point diagnostic that decomposes attribution sensitivity by GGN eigenvalue magnitude buckets. Exploits RMT stability of eigenvalue magnitudes vs instability of eigenvector directions. Pilot: 93.5% within-class variance confirms BSS is not a class proxy.

3. **C3 -- MRC soft combining.** Adaptive per-test-point weighting of IF and RepSim attributions using BSS and cross-method disagreement, inspired by Maximal Ratio Combining. Provably optimal under squared-error loss with correct variance estimates.

4. **C4 -- Negative results on prior diagnostics.** TRV is seed-unstable (cross-seed rho ~ 0) and SI is orthogonal to Hessian sensitivity (rho ~ 0), explaining why per-point reliability diagnostics have been absent despite clear need.
