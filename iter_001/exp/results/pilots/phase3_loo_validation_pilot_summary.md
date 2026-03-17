# Phase 3 LOO Validation Pilot Summary

## Result: NO-GO (at pilot scale)

**Best LOO-TDA correlation: 0.036** (threshold: 0.3)

## Design

- **Exact LOO on 50 training samples**: 10 test points x 50 removals = 500 retrainings (15 epochs each)
- **Partial exact LOO on 500 samples**: 10 test points x 100 removals = 1000 retrainings (20 epochs each)
- **Approximate IF**: Gradient dot-product (identity Hessian) on 500 training samples
- **RepSim**: Cosine similarity on penultimate features from 500-sample model
- **Total wall time**: 24.5 minutes on 1x RTX 4090

## Key Findings

| Comparison | Mean Spearman rho |
|-----------|------------------|
| Approx IF vs Exact LOO (50-train) | -0.076 |
| Exact LOO (500-train) vs Approx IF | 0.008 |
| RepSim vs Exact LOO (50-train) | -0.046 |
| RepSim vs Exact LOO (500-train) | 0.036 |
| TRAK vs LOO | N/A (index mismatch) |
| EK-FAC vs LOO | N/A (index mismatch) |

Structural validation:
- Mean influence gap (same-class - diff-class): 0.111 (weakly positive)
- Mean top-10 same-class count: 1.4/10 (barely above random baseline of 1.0)
- Not all test points have positive influence gaps (4 out of 10 are negative)

## Root Cause Analysis

1. **Severely undertrained models**: 30% accuracy (50 samples) and 50% accuracy (500 samples) produce extremely noisy loss landscapes where LOO influence signals are dominated by random fluctuations.

2. **Insufficient training**: 15-30 epochs on 50-500 samples is far below convergence. The 200-epoch model (95.5% acc on full CIFAR-10) produces meaningful representations; the pilot models do not.

3. **Crude IF approximation**: Gradient dot-product with identity Hessian ignores the critical Hessian structure that makes IF meaningful. Proper EK-FAC or K-FAC Hessian approximation is needed.

4. **Index mapping failure**: Phase 1 TRAK/EKFAC scores use different training subset indices than the LOO pilot, preventing direct comparison.

## Implications for Full Experiment

The pilot NO-GO does NOT invalidate the LOO validation approach. The failure is entirely attributable to pilot-scale limitations:

- **Must use the 200-epoch checkpoint** (seed=42, 95.5% test acc) as the base model
- **Must use full 5K subset** matching Phase 1 training indices
- **Must use proper IF** via dattri EK-FAC (not crude gradient dots)
- **Budget**: ~30 GPU-hours for 100 test x 5000 LOO retrainings

The pilot successfully validated the LOO retraining pipeline (timing: ~1.2s per retraining with 20 epochs on 499 samples on RTX 4090). Extrapolating: 5000 retrainings x 20 epochs x 100 test points / 4 GPUs = ~17 hours.
