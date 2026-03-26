---
version: "2.0"
created: "2026-03-17"
last_modified: "2026-03-25"
entry_mode: "design"
iteration_major: 2
iteration_minor: 0
---

# Experiment Design: AURA

## 1. Variance Decomposition (Phase 1) -- COMPLETED

### Setup
- 500 CIFAR-10 test points (50/class, stratified)
- ResNet-18, seed=42, 200 epochs, 95.5% test accuracy
- Full-model Hessian (not last-layer)
- Methods: EK-FAC IF, K-FAC IF, RepSim, TRAK-50
- Damping: K-FAC=0.1, EK-FAC=0.01

### Response Variables
- J10: Jaccard@10(EK-FAC IF, K-FAC IF)
- tau: Kendall tau(IF rankings, RepSim rankings) per test point
- LDS: Per-point Linear Datamodeling Score (EK-FAC IF vs TRAK-50)

### Analysis
Two-way ANOVA, Type I sequential SS, class entered first.

### Results

| Response | Class R^2 | GradNorm R^2 | Interaction R^2 | Residual R^2 | Gate |
|----------|-----------|-------------|----------------|-------------|------|
| J10 | 0.1409 | 0.0057 | 0.0194 | **0.7750** | PASS |
| tau | 0.2639 | 0.3349 | 0.1762 | 0.2250 | FAIL |
| LDS | 0.2668 | 0.1695 | 0.0473 | **0.5164** | PASS |

Descriptive statistics:

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| J10 | 0.9945 | 0.0310 | 0.8182 | 1.0000 | 1.0000 |
| tau | 0.0171 | 0.0825 | -0.1688 | 0.3108 | 0.0026 |
| LDS | 0.7443 | 0.0901 | 0.4342 | 0.8930 | 0.7665 |

**Gate: PASS** (residual > 30% on J10 and LDS).
**Compute**: ~5 GPU-hours.

---

## 2. BSS Cross-Seed Stability (Phase 2a) -- ~7 GPU-hours

### Setup
- 500 CIFAR-10 test points (50/class)
- 5 ResNet-18 models (seeds 42, 123, 456, 789, 1024)
- Kronecker GGN eigendecomposition: top-100 eigenvalues/eigenvectors per seed
- Adaptive bucket thresholds (percentile-based, not fixed)

### BSS Variants
1. **BSS_raw**: Standard BSS per bucket
2. **BSS_partial**: BSS residualized against ||g||^2
3. **BSS_ratio**: BSS_outlier / BSS_total

### Analyses
1. Cross-seed stability: Mean pairwise Spearman rho (10 pairs) for each BSS variant
2. Predictive power: Spearman(BSS_outlier, per-point LDS) + partial correlation controlling class and gradient norm
3. Class detector test: ANOVA BSS_outlier ~ class, within-class/total variance fraction
4. Baselga decomposition of J10: replacement vs reordering components

### Gates
| Criterion | Pass | Borderline | Fail |
|-----------|------|-----------|------|
| BSS_partial cross-seed rho | > 0.5 | 0.3-0.5 | < 0.3 |
| Within-class BSS variance | > 25% | 15-25% | < 15% |
| Partial corr (BSS vs LDS, controlling class+grad) | > 0.15 | 0.10-0.15 | < 0.10 |

### Pilot Results (1 seed, 100 points)
- Within-class variance: 93.5% (PASS H-D3)
- BSS-gradient_norm rho: 0.906 (CONCERN -- partial BSS needed)
- Perturbation factors nearly uniform (damping >> eigenvalues)
- Eigenvalue scale: max ~5e-05 (Kronecker products very small)
- Timing: ~1.2 min for 100 points (well within budget)

---

## 3. MRC + Pareto Frontier (Phase 3) -- ~8 GPU-hours

### MRC Calibration
- w(z) = sigmoid(a * BSS_partial + b * disagreement + c)
- LOO cross-validation on 300 calibration points
- Optimize (a, b, c) to maximize mean LDS

### Strategy Comparison (11 strategies)
1. Identity IF
2. K-FAC IF
3. EK-FAC IF
4. RepSim
5. TRAK-10
6. TRAK-50
7. W-TRAK
8. Naive 0.5:0.5 IF+RepSim ensemble
9. BSS-guided hard routing (IF if BSS_partial < threshold, else RepSim)
10. Disagreement-guided routing
11. MRC soft combining

### LOO Validation
- CIFAR-10/5K subset, 100 test points
- Exact leave-one-out retraining
- Validates TRAK-50 ground truth quality

### Evaluation Metrics
- Mean LDS (Spearman correlation against ground truth)
- GPU-hours per strategy
- Pareto frontier: LDS vs compute
- Class-stratified AUROC for adaptive strategies
- Oracle gap closure: (MRC - naive) / (oracle - naive)

### Gate
MRC > best uniform strategy by > 2% absolute LDS at same compute budget.

---

## 4. Confound Controls

### 4.1 Class-Stratified Analysis
All adaptive strategies must achieve within-class AUROC > 0.55. If below 0.55, routing signal is merely a class proxy.

### 4.2 Gradient-Norm Partial Correlations
For all BSS variants, report partial Spearman(BSS, LDS | class + ||g||). If partial rho < 0.10, BSS adds no information beyond gradient norm.

### 4.3 Stability vs Correctness
Compute partial Spearman(BSS, LOO_correctness | class + ||g||). Expected: < 0.1 (stable != correct).

---

## 5. Ablations

| Ablation | Variable | Values | Purpose |
|----------|----------|--------|---------|
| Bucket granularity | Number of buckets | 3, 5, 10 | Sensitivity to partitioning |
| Eigenvalue count | Top-k eigenvalues | 50, 100, 200 | How many modes matter |
| Gradient-norm correction | Regression type | None, linear, log | Best normalization |
| MRC weight function | Functional form | Sigmoid, softmax, piecewise | Robustness of combining |
| Damping | delta | 0.001, 0.01, 0.1, 1.0 | Perturbation factor sensitivity |
| Train subset size | N_train | 5K, 10K, 50K | Scalability |

---

## 6. Paper Scenarios

### Scenario A: Full Paper (all gates pass)
- C0: Variance decomposition (confirmed)
- C1: BSS diagnostic (seed-stable, informative)
- C2: MRC soft combining (Pareto-dominates uniform)
- C3: Negative results (TRV/SI failures)
- Target: NeurIPS 2026

### Scenario B: Diagnostic-Only Paper (Phase 2 pass, Phase 3 fail)
- C0 + C1 + C3
- Target: NeurIPS 2026 (poster) or TMLR

### Scenario C: Negative Results Paper (Phase 2 fail)
- C0 + C3
- Target: TMLR

---

## 7. Timeline & Compute Budget

| Phase | GPU-hours | Status |
|-------|-----------|--------|
| Phase 0: Probe reanalysis | 0 | COMPLETED |
| Phase 1: Variance decomposition | ~5 | COMPLETED |
| Phase 2a pilot: BSS pilot | ~0.5 | COMPLETED |
| Phase 2b: Disagreement analysis | 0 | COMPLETED |
| **Phase 2a full: BSS cross-seed** | **~7** | **IN PROGRESS** |
| **Phase 3: MRC + Pareto** | **~8** | **PLANNED** |
| **Ablations** | **~5** | **PLANNED** |
| **Confound controls** | **~2** | **PLANNED** |
| **Total** | **~27.5** | |

Used: ~5.5 GPU-hours. Remaining budget: ~36.5 GPU-hours. Well within 42-hour total.

---

## 8. Expected Outputs

### Tables
- Table 1: Variance decomposition (Phase 1 results)
- Table 2: BSS cross-seed stability (Spearman rho matrix)
- Table 3: Main results (LDS by strategy, with compute cost)
- Table 4: Ablation results

### Figures
- Figure 1: BSS outlier heatmap across test points, colored by class
- Figure 2: Cross-seed BSS scatter plots with rho annotations
- Figure 3: Pareto frontier (LDS vs GPU-hours)
- Figure 4: MRC weight distribution and calibration curves
- Figure 5: Baselga decomposition of attribution instability
