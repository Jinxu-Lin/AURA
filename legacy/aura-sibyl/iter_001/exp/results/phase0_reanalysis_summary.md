# Phase 0: Probe Data Reanalysis — FULL MODE Summary

## Data Source
- **RAW per-point data** from 3 seeds (42, 123, 456) x 100 ID test points
- Source: `/Users/jinxulin/Research/AURA/codes/probe_experiment/outputs/attributions`
- CIFAR-10/ResNet-18, last-layer Hessian, 5 approximation levels
- Same test indices across seeds: **False**

## GUM Uncertainty Budget (TRV Variance Decomposition)

| Component | SS | Fraction |
|-----------|-----|----------|
| Seed | 3.02 | 0.4% |
| Class | 94.83 | 13.4% |
| Residual | 608.23 | 86.1% |
| **Total** | 706.08 | 100% |

## TRV Distribution

### Seed 42
| Level | Count | Fraction |
|-------|-------|----------|
| 1 | 59 | 59% |
| 2 | 21 | 21% |
| 3 | 1 | 1% |
| 5 | 19 | 19% |
Major levels (>10%): [1, 2, 5]

### Seed 123
| Level | Count | Fraction |
|-------|-------|----------|
| 1 | 38 | 38% |
| 2 | 40 | 40% |
| 3 | 3 | 3% |
| 5 | 19 | 19% |
Major levels (>10%): [1, 2, 5]

### Seed 456
| Level | Count | Fraction |
|-------|-------|----------|
| 1 | 65 | 65% |
| 2 | 11 | 11% |
| 3 | 2 | 2% |
| 5 | 22 | 22% |
Major levels (>10%): [1, 2, 5]

## Cross-Seed Stability
- 42_vs_123: Mann-Whitney p = 0.022, mean diff = -0.230
- 42_vs_456: Mann-Whitney p = 0.640, mean diff = -0.040
- 123_vs_456: Mann-Whitney p = 0.009, mean diff = 0.190

## SI-TRV Correlation
- Mean |rho| across seeds: **0.112**
- Conclusion: **no_correlation**
- Seed 42: rho = 0.043 (p = 0.672), 95% CI [-0.155, 0.239]
- Seed 123: rho = -0.180 (p = 0.073), 95% CI [-0.374, 0.014]
- Seed 456: rho = -0.114 (p = 0.259), 95% CI [-0.298, 0.099]

## Confidence Stratification
- Seed 42: high=2.12±1.53, low=1.86±1.53, p=0.127, d=0.17
- Seed 123: high=2.36±1.41, low=2.08±1.48, p=0.078, d=0.19
- Seed 456: high=1.98±1.66, low=2.08±1.63, p=0.441, d=-0.06

## Jaccard Degradation (Mean per Hessian Level)

### Seed 42 (kappa = 1224782)
| Level | Mean | Std | Median |
|-------|------|-----|--------|
| full_ggn | 1.000 | 0.000 | 1.000 |
| kfac | 0.456 | 0.162 | 0.429 |
| diagonal | 0.337 | 0.150 | 0.333 |
| damped_identity | 0.334 | 0.151 | 0.333 |
| identity | 0.334 | 0.151 | 0.333 |

### Seed 123 (kappa = 1130109)
| Level | Mean | Std | Median |
|-------|------|-----|--------|
| full_ggn | 1.000 | 0.000 | 1.000 |
| kfac | 0.532 | 0.148 | 0.538 |
| diagonal | 0.365 | 0.159 | 0.333 |
| damped_identity | 0.351 | 0.153 | 0.333 |
| identity | 0.351 | 0.153 | 0.333 |

### Seed 456 (kappa = 1357640)
| Level | Mean | Std | Median |
|-------|------|-----|--------|
| full_ggn | 1.000 | 0.000 | 1.000 |
| kfac | 0.447 | 0.177 | 0.429 |
| diagonal | 0.372 | 0.181 | 0.333 |
| damped_identity | 0.361 | 0.176 | 0.333 |
| identity | 0.361 | 0.176 | 0.333 |

## Key Implications for Phase 1
1. **Must use full-model Hessian** (last-layer collapsed 3 of 5 levels)
2. **BSS over scalar TRV** (cross-seed rho near zero for scalar TRV)
3. **Control for class first** in ANOVA (class-dominance risk from Papyan 2020)
4. **SI not useful as proxy** (orthogonal to Hessian sensitivity)
5. **Expect coarse categories** not continuous signal (per-point std ~ 0.06)

## Quality Flags
- Raw data: **YES** (not synthetic reconstruction)
- Bootstrap CIs: **YES** (n=1000)
- Statistical tests: Kruskal-Wallis, Mann-Whitney U, Spearman, KS-test
