# Phase 2b: Cross-Method Disagreement Analysis (H-D4) — Pilot

**Date**: 2026-03-17T19:36:07.249096
**Mode**: PILOT (100 test points, 5K train, layer4+fc)

## Key Finding: IF Universally Dominates RepSim

All 100 test points have higher LDS with IF than RepSim.
This makes the original binary AUROC analysis degenerate.

## LDS Comparison

| Method | Mean LDS | Std |
|--------|----------|-----|
| EK-FAC IF | 0.7443 | 0.0901 |
| K-FAC IF | 0.7444 | 0.0901 |
| RepSim | 0.2738 | 0.1793 |

## Kendall Tau (IF-RepSim Disagreement)

- Mean: 0.2109 ± 0.1323
- Range: [-0.1491, 0.4035]
- Significant (p<0.05): 94/100

## Critical Correlation: tau vs LDS_diff

Spearman(tau, LDS_diff) = -0.5457 (p = 4.33e-09)

**Interpretation**: Points where IF and RepSim DISAGREE more (low tau) have LARGER IF advantage. This is physically sensible: IF adds Hessian-weighted structure beyond representation similarity, and this structure provides the most benefit precisely where representation similarity alone fails.

## Quantile-Based AUROC (Substitute Analysis)

- Quantile AUROC (median split): 0.7548 (best predictor: -|tau|)
- Tertile AUROC (top vs bottom third): 0.8411
- Class-stratified quantile AUROC: 0.6640 (10 valid classes)
- Multi-feature LR AUROC: 0.7500 ± 0.0790

## Per-Class Statistics

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

## Gate Decision: **PASS**


## Implications for Full Experiment

- Full-model IF may produce RepSim-better points (deeper layers capture different geometry)
- TRAK-50 ground truth will be more reliable, potentially changing IF/RepSim relative performance
- 500 test points (50/class) will enable reliable class-stratified AUROC
- The strong tau-LDS_diff correlation (rho=-0.55) suggests disagreement IS informative, motivating the full experiment
