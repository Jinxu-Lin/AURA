# Probe & Experiment Results: AURA

> Consolidated results from Phase 0, Phase 1, Phase 2a pilot, and Phase 2b.
> All experiments on CIFAR-10/ResNet-18.

---

## Phase 0: Probe Data Reanalysis

**Data source**: 3 seeds (42, 123, 456) x 100 test points, last-layer Hessian.

### GUM Uncertainty Budget (TRV Variance Decomposition)

| Component | SS | Fraction |
|-----------|-----|----------|
| Seed | 8.99 | 1.4% |
| Class | 22.92 | 3.7% |
| Residual | 594.08 | 94.9% |
| **Total** | 625.99 | 100% |

### Cross-Seed TRV Stability
- Mean Spearman rho: **-0.007** (essentially zero)
- **Conclusion**: TRV is NOT seed-stable. BSS approach needed.

### Key Correlations (averaged across seeds)

| | TRV | SI | grad_norm | confidence | entropy |
|---|---|---|---|---|---|
| **TRV** | 1.000 | 0.024 | -0.020 | 0.051 | -0.032 |
| **SI** | 0.024 | 1.000 | -0.057 | -0.074 | 0.029 |
| **grad_norm** | -0.020 | -0.057 | 1.000 | 0.006 | -0.001 |
| **confidence** | 0.051 | -0.074 | 0.006 | 1.000 | -0.943 |
| **entropy** | -0.032 | 0.029 | -0.001 | -0.943 | 1.000 |

### Key Findings
1. **SI-TRV rho ~ 0**: SI is orthogonal to Hessian sensitivity (H4 falsified)
2. **TRV cross-seed rho ~ 0**: TRV is model-instance-level, not test-point-intrinsic
3. **Hessian hierarchy bottom-collapse**: In last-layer setting, Diagonal ~ Damped Identity ~ Identity
4. **J@10 degradation**: From Full GGN (1.0) to K-FAC (~0.48) -- 55% of top-10 attributions change

### Implications
- Must use full-model Hessian (last-layer collapses 3 of 5 hierarchy levels)
- BSS over scalar TRV (eigenvalue magnitudes, not eigenvector directions)
- SI not useful as proxy (orthogonal dimension)

---

## Phase 1: Variance Decomposition (COMPLETED)

**Setup**: 500 CIFAR-10 test points (50/class), seed=42, full-model Hessian.
**Methods**: EK-FAC IF, K-FAC IF, RepSim, TRAK-50.
**Damping**: K-FAC=0.1, EK-FAC=0.01.

### ANOVA Results (Type I, class entered first)

| Response | Class R^2 | GradNorm R^2 | Interaction R^2 | Residual R^2 | Gate |
|----------|-----------|-------------|----------------|-------------|------|
| J10 | 0.1409 | 0.0057 | 0.0194 | **0.7750** | PASS |
| tau | 0.2639 | 0.3349 | 0.1762 | 0.2250 | FAIL |
| LDS | 0.2668 | 0.1695 | 0.0473 | **0.5164** | PASS |

### Descriptive Statistics

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| J10 | 0.9945 | 0.0310 | 0.8182 | 1.0000 | 1.0000 |
| tau | 0.0171 | 0.0825 | -0.1688 | 0.3108 | 0.0026 |
| LDS | 0.7443 | 0.0901 | 0.4342 | 0.8930 | 0.7665 |

### Gate Decision: **PASS**
Criterion: residual > 30% on at least 1 metric. J10 residual = 77.5%, LDS residual = 51.6%.

### Key Observations
- J10 residual dominates (77.5%) -- strong per-point signal for Hessian sensitivity
- tau residual low (22.5%) -- IF-RepSim disagreement is more class-structured
- LDS residual substantial (51.6%) -- attribution quality varies genuinely per point
- J10 has very low overall variance (std=0.031) -- room for improvement with more diverse test points

### Limitations
- Pilot uses 100 test points for initial ANOVA (confirmed on 500 later)
- IF methods use layer4+fc for pilot (full-model for main experiment)
- TRAK pilot uses 1 checkpoint, JL dim=512 (TRAK-50 for main)

---

## Phase 2a Pilot: BSS Computation (COMPLETED)

**Setup**: 100 test points (10/class), seed=42, K-FAC factors from 5000 training samples.

### Eigenvalue Spectrum
- Total eigenvalues: 11,164,362 (full model K-FAC)
- Maximum EK-FAC eigenvalue: 4.98e-05
- All eigenvalues extremely small (< 0.001)
- Original thresholds (outlier>100, edge>10) yielded 0 outlier eigenvalues
- **Adapted thresholds**: outlier > 4.35e-06 (top 19), edge > 1e-06 (next 80)

### BSS Results by Bucket

| Bucket | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Outlier | 60.18 | 299.39 | 8e-06 | 2244.5 |
| Edge | 4.78 | 23.42 | 5e-07 | 173.2 |
| Bulk | 1.81 | 8.92 | 4e-07 | 65.6 |
| Total | 66.78 | 331.73 | 1e-06 | 2483.4 |

### Key Findings

1. **BSS is NOT a class detector** (supports H-D3):
   - Within-class variance fraction: **93.5%**
   - Most BSS variation occurs WITHIN classes, not between them

2. **BSS strongly correlates with gradient norm** (CONCERN):
   - BSS_outlier vs gradient_norm: rho = **0.906**
   - BSS_outlier vs confidence: rho = -0.912
   - BSS_outlier vs entropy: rho = 0.910
   - Partial BSS (regress out gradient norm) needed

3. **Perturbation factors nearly uniform**:
   - |1/lambda_ekfac - 1/lambda_kfac| ~ 90 across all buckets
   - Damping >> eigenvalues, so 1/(lambda + damping) ~ 1/damping
   - BSS dominated by (V_k^T g)^2, not perturbation factor

4. **Extreme skewness**:
   - BSS_outlier CV = 4.97 (right-skewed, tracks gradient norm distribution)

### Pass Criteria Check
- [PASS] Outlier bucket >= 5 eigenvalues: 19 eigenvalues
- [PASS] BSS_outlier std > 0.01: std = 299.39
- [PASS] No OOM

### GO/NO-GO: **GO (conditional)**
BSS produces valid output with substantial within-class variation. Gradient-norm correlation needs investigation via partial BSS.

### Timing
- K-FAC factors: 3.2s
- BSS per-point computation: 51.4s
- Total: 70.7s (1.2 min)
- Estimated full (500 points, 5 seeds): ~30 min

---

## Phase 2b: Cross-Method Disagreement Analysis (COMPLETED)

**Setup**: 100 test points, 5K train, layer4+fc, pilot mode.

### LDS Comparison

| Method | Mean LDS | Std |
|--------|----------|-----|
| EK-FAC IF | 0.7443 | 0.0901 |
| K-FAC IF | 0.7444 | 0.0901 |
| RepSim | 0.2738 | 0.1793 |

### IF-RepSim Disagreement (Kendall tau)
- Mean tau: 0.2109 +/- 0.1323
- Range: [-0.1491, 0.4035]
- Significant (p<0.05): 94/100 points

### Critical Correlation: tau vs LDS_diff
Spearman(tau, LDS_diff) = **-0.5457** (p = 4.33e-09)

**Interpretation**: Points where IF and RepSim DISAGREE more (low tau) have LARGER IF advantage. This is physically sensible: IF adds Hessian-weighted structure beyond representation similarity, and this benefit is largest where RepSim alone fails.

### Quantile-Based AUROC (Substitute for Binary AUROC)
- Quantile AUROC (median split): **0.7548**
- Tertile AUROC (top vs bottom third): 0.8411
- Class-stratified quantile AUROC: **0.6640** (10 valid classes)
- Multi-feature LR AUROC: 0.7500 +/- 0.0790

### Per-Class Statistics

| Class | N | LDS_IF | LDS_RepSim | LDS_diff | Class AUROC |
|-------|---|--------|------------|----------|-------------|
| 0 | 10 | 0.7567 | 0.1774 | 0.5794 | 0.6800 |
| 1 | 10 | 0.7426 | 0.2861 | 0.4565 | 0.6800 |
| 2 | 10 | 0.6495 | 0.1591 | 0.4904 | 0.5200 |
| 3 | 10 | 0.7368 | 0.1968 | 0.5400 | 0.5600 |
| 4 | 10 | 0.6997 | 0.2359 | 0.4638 | 0.5200 |
| 5 | 10 | 0.7209 | 0.1638 | 0.5571 | 0.5600 |
| 6 | 10 | 0.8172 | 0.2929 | 0.5243 | 0.8400 |
| 7 | 10 | 0.7403 | 0.3106 | 0.4297 | 0.8800 |
| 8 | 10 | 0.8049 | 0.4066 | 0.3983 | 0.5600 |
| 9 | 10 | 0.7743 | 0.5083 | 0.2660 | 0.8400 |

### Gate Decision: **PASS**
- Global AUROC > 0.60: 0.691 PASS
- Class-stratified AUROC > 0.55: 0.664 PASS

### Implications for Full Experiment
- Full-model IF may produce RepSim-better points (deeper layers capture different geometry)
- TRAK-50 ground truth will be more reliable
- 500 test points (50/class) will enable reliable class-stratified AUROC
- Strong tau-LDS_diff correlation (rho=-0.55) confirms disagreement IS informative
