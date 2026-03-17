# Phase 0: Probe Data Reanalysis — Summary

## Data Source
- 3 seeds (42, 123, 456) × 100 test points, CIFAR-10/ResNet-18, last-layer Hessian
- **Raw data not available** — reanalysis from aggregated statistics in probe report

## GUM Uncertainty Budget (TRV Variance Decomposition)

| Component | SS | Fraction |
|-----------|-----|----------|
| Seed | 8.99 | 1.4% |
| Class | 22.92 | 3.7% |
| Residual | 594.08 | 94.9% |
| **Total** | 625.99 | 100% |

**Interpretation**: The residual fraction (94.9%) represents
per-point variation not explained by seed or class. This gives a preliminary estimate
for Phase 1's gating criterion (residual > 30%). However, this estimate uses
synthetic class assignments and last-layer TRV — Phase 1 with full-model Hessian
may show different decomposition.

## Cross-Seed TRV Stability
- Mean Spearman rho: **-0.007** (essentially zero)
- TRV is NOT seed-stable → BSS approach uses eigenvalue-magnitude buckets instead

## Key Correlations (averaged across seeds)
| | TRV | SI | grad_norm | log_grad_norm | confidence | entropy |
|---|---|---|---|---|---|---|
| **TRV** | 1.000 | 0.024 | -0.020 | -0.020 | 0.051 | -0.032 |
| **SI** | 0.024 | 1.000 | -0.057 | -0.057 | -0.074 | 0.029 |
| **grad_norm** | -0.020 | -0.057 | 1.000 | 1.000 | 0.006 | -0.001 |
| **log_grad_norm** | -0.020 | -0.057 | 1.000 | 1.000 | 0.006 | -0.001 |
| **confidence** | 0.051 | -0.074 | 0.006 | 0.006 | 1.000 | -0.943 |
| **entropy** | -0.032 | 0.029 | -0.001 | -0.001 | -0.943 | 1.000 |

## Implications for Phase 1
1. **Must use full-model Hessian** (last-layer collapsed 3 of 5 levels)
2. **BSS over scalar TRV** (cross-seed rho ≈ 0 for scalar TRV)
3. **Control for class first** in ANOVA (class-dominance risk from Papyan 2020)
4. **SI not useful as proxy** (orthogonal to Hessian sensitivity)
5. **Expect coarse categories** not continuous signal (per-point std ≈ 0.06)
