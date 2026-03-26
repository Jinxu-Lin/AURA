---
version: "2.0"
created: "2026-03-17"
last_modified: "2026-03-25"
entry_mode: "design"
iteration_major: 2
iteration_minor: 0
---

# Method & Experiment Design: AURA

## 1. Component 1: Attribution Variance Decomposition (COMPLETED)

### Method
Two-way ANOVA decomposing attribution sensitivity into class-conditional, gradient-norm, and residual per-test-point components. Three response variables:
- **J10**: Jaccard@10(EK-FAC IF, K-FAC IF) -- top-k overlap stability
- **tau**: Kendall tau(IF rankings, RepSim rankings) -- cross-method agreement
- **LDS**: Per-point Linear Datamodeling Score (EK-FAC IF vs TRAK-50)

Sequential sum of squares (Type I) with class entered first. 500 CIFAR-10 test points (50/class, stratified), single seed (42), full-model Hessian.

### Results (Phase 1)

| Response | Class R^2 | GradNorm R^2 | Interaction R^2 | Residual R^2 | Gate |
|----------|-----------|-------------|----------------|-------------|------|
| J10 | 0.1409 | 0.0057 | 0.0194 | **0.7750** | PASS |
| tau | 0.2639 | 0.3349 | 0.1762 | 0.2250 | FAIL |
| LDS | 0.2668 | 0.1695 | 0.0473 | **0.5164** | PASS |

**Gate PASS**: Residual > 30% on J10 (77.5%) and LDS (51.6%).

### Interpretation
J10 residual dominates at 77.5%, meaning per-test-point Hessian sensitivity is NOT explained by class membership or gradient magnitude. The phenomenon is genuine and warrants per-point diagnostics. tau residual is lower (22.5%) because IF-RepSim disagreement is more class-structured.

---

## 2. Component 2: Bucketed Spectral Sensitivity (BSS)

### 2.1 Theoretical Foundation

**Random Matrix Theory (RMT) prediction**: For neural networks, the GGN eigenvalue spectrum has:
- Outlier eigenvalues (count = number of classes) determined by class separation geometry
- Bulk eigenvalues following Marchenko-Pastur distribution
- Eigenvalue *magnitudes* are stable to O(1/sqrt(N)) across training seeds
- Eigenvalue *directions* rotate freely across seeds

This explains why TRV (which depends on eigenvector directions) has rho ~ 0 cross-seed, while BSS (using eigenvalue magnitude buckets) should be stable.

**Hampel sensitivity connection**: BSS can be viewed as a spectral decomposition of the Hampel gross error sensitivity function, partitioned by the Hessian eigenvalue spectrum.

### 2.2 BSS Definition

For test point z with gradient g_z, Hessian eigenvalues {lambda_k} with eigenvectors {v_k}, and approximate eigenvalues {tilde_lambda_k}:

```
BSS_j(z) = sum_{k in B_j} |1/(lambda_k + delta) - 1/(tilde_lambda_k + delta)|^2 * (v_k^T g_z)^2
```

where B_j partitions eigenvalues into magnitude buckets and delta is the damping term.

**Eigenvalue buckets** (adaptive, percentile-based):
- **Outlier**: Top 0.2% of eigenvalues (class-discriminative modes)
- **Edge**: Next 0.5% (transition region)
- **Bulk**: Remaining 99.3% (noise subspace)

Note: Original fixed thresholds (outlier > 100, edge > 10) were invalidated by pilot -- Kronecker eigenvalue products are extremely small (max ~5e-05). Adaptive percentile-based thresholds are required.

### 2.3 Partial BSS (Gradient-Norm Correction)

Pilot revealed BSS-gradient_norm rho = 0.906. To disentangle:

**BSS_partial**: Residuals of BSS_j regressed on ||g_z||^2:
```
BSS_partial_j(z) = BSS_j(z) - (alpha_j * ||g_z||^2 + beta_j)
```

**BSS_ratio**: BSS_outlier / BSS_total (fraction of sensitivity in outlier modes, scale-invariant).

### 2.4 Baselga Turnover Decomposition

Decompose Jaccard distance between EK-FAC and K-FAC top-k sets into:
- **Replacement component**: Points that enter/exit the top-k
- **Reordering component**: Points that stay in top-k but change rank

This separates "catastrophic" instability (different points attributed) from "mild" instability (same points, different order).

### 2.5 Cross-Seed Stability Protocol

- 5 ResNet-18 models (seeds 42, 123, 456, 789, 1024)
- 500 test points, BSS computed per seed
- Stability metric: Mean pairwise Spearman rho of BSS_outlier rankings (10 seed pairs)
- Gate: rho > 0.5

### 2.6 MRC Optimality Justification

Under Cauchy-Schwarz, the MSE-optimal combination weight for test point z is:
```
w*(z) = sigma_RepSim^2(z) / (sigma_IF^2(z) + sigma_RepSim^2(z))
```

BSS provides a spectral proxy for sigma_IF^2(z) (high BSS = high IF uncertainty). Cross-method disagreement provides a proxy for relative quality. MRC combines these signals.

---

## 3. Component 3: MRC Soft Combining

### 3.1 Weight Function

```
w(z) = sigmoid(a * BSS_partial(z) + b * disagreement(z) + c)
score(z) = w(z) * RepSim(z) + (1 - w(z)) * IF(z)
```

where disagreement(z) = |Kendall_tau(IF_rankings(z), RepSim_rankings(z))|.

When BSS_partial is high (IF unreliable) or disagreement is large, weight shifts toward RepSim.

### 3.2 Calibration

- Leave-one-out cross-validation on 300 calibration points
- Optimize (a, b, c) to maximize mean LDS against TRAK-50 ground truth
- Evaluate on held-out 200 test points

### 3.3 Baselines (11 strategies)

| # | Strategy | Type | Compute |
|---|----------|------|---------|
| 1 | Identity IF | Uniform | Low |
| 2 | K-FAC IF | Uniform | Low |
| 3 | EK-FAC IF | Uniform | Medium |
| 4 | RepSim | Uniform | Low |
| 5 | TRAK-10 | Uniform | Medium |
| 6 | TRAK-50 | Uniform | High |
| 7 | W-TRAK | Uniform | Medium |
| 8 | Naive 0.5:0.5 ensemble | Uniform | Medium |
| 9 | BSS-guided routing | Adaptive | Medium+BSS |
| 10 | Disagreement-guided routing | Adaptive | Medium |
| 11 | MRC soft combining | Adaptive | Medium+BSS |

Oracle: Per-point max(LDS_IF, LDS_RepSim).

---

## 4. Theoretical Contributions

### Proposition 1: Spectral Decomposition of Attribution Error
When H and H_tilde share eigenvectors (exact for K-FAC/EK-FAC within Kronecker structure), the attribution error decomposes exactly as:
```
||H^{-1}g - H_tilde^{-1}g||^2 = sum_k (1/lambda_k - 1/tilde_lambda_k)^2 * (v_k^T g)^2
```

### Proposition 2: Eigenvalue Bucket Stability (RMT)
Under standard assumptions (iid data, overparameterized regime), the fraction of test-gradient energy in each eigenvalue bucket converges to a data-geometric constant independent of training seed, with convergence rate O(1/sqrt(N)).

### Proposition 3: MRC Optimality
The MRC weight function w*(z) minimizes expected squared attribution error under the Cauchy-Schwarz bound, given correct estimates of per-method variance.

---

## 5. Experiment Design Summary

### Phase 2a: BSS Cross-Seed Stability (~7 GPU-hours)
- 500 test points x 5 seeds
- Kronecker GGN top-100 eigendecomposition per seed
- Compute BSS (raw + partial + ratio) per seed
- Gates: partial BSS cross-seed rho > 0.5, within-class CV > 25%, partial corr > 0.15

### Phase 2b: Disagreement Analysis (0 GPU-hours, reuses Phase 1)
- Already completed. Results: tau vs LDS_diff rho = -0.547, AUROC = 0.691, class-stratified AUROC = 0.664

### Phase 3: MRC + Pareto (~8 GPU-hours)
- LOO validation on CIFAR-10/5K subset (100 points)
- All 11 strategies evaluated
- Gate: MRC > uniform by > 2% LDS at same compute

### Phase 4: Confound Controls
- Class-stratified AUROC for all adaptive strategies (must exceed 0.55 within classes)
- Gradient-norm partial correlations for all BSS variants

### Phase 5: Ablations
- Bucket granularity (3 vs 5 vs 10 buckets)
- Eigenvalue count (top-50 vs top-100 vs top-200)
- Gradient-norm correction (none vs linear vs log)
- MRC weight function (sigmoid vs softmax vs piecewise linear)
- Damping sensitivity (delta = 0.001, 0.01, 0.1, 1.0)

---

## 6. Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1-2 | Phase 2a | BSS cross-seed stability results |
| 3 | Phase 3 | MRC calibration + Pareto frontier |
| 4 | Phase 4-5 | Confound controls + ablations |
| 5-6 | Paper | Draft, revision, submission prep |

Total remaining: ~22 GPU-hours within 42-hour budget.
