# Phase 1 Variance Decomposition - PILOT Results

**N test points**: 100
**ANOVA type**: Type I sequential, class entered first

## Variance Decomposition (Partial R-squared)

| Response | Class R² | GradNorm R² | Interaction R² | Residual R² | Gate |
|----------|----------|-------------|----------------|-------------|------|
| J10 | 0.1409 | 0.0057 | 0.0194 | 0.8340 | PASS |
| tau | 0.2639 | 0.3349 | 0.1762 | 0.2250 | FAIL |
| LDS | 0.2668 | 0.1695 | 0.0473 | 0.5164 | PASS |

## Gate Decision: **PASS**

Criterion: residual > 30% on at least 1 metric

## Descriptive Statistics

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| J10 | 0.9945 | 0.0310 | 0.8182 | 1.0000 | 1.0000 |
| tau | 0.0171 | 0.0825 | -0.1688 | 0.3108 | 0.0026 |
| LDS | 0.7443 | 0.0901 | 0.4342 | 0.8930 | 0.7665 |

## Key Observations

- J10: Residual dominates (83.4%) - strong per-point signal
- tau: Low residual (22.5%) - class/grad_norm explain most variance
- LDS: Residual dominates (51.6%) - strong per-point signal
- J10 has very low variance (std < 0.05) - likely due to layer4+fc only setting

## Limitations

- Pilot uses 100 test points (not 500 as planned for full experiment)
- IF methods use layer4+fc only (not full model) - J10 has very low variance
- TRAK uses 1 checkpoint, JL dim=512 (pilot settings, not TRAK-50)
- Train subset is 5K (not full 50K)
- Class distribution may not be perfectly balanced (non-stratified 100 points)
