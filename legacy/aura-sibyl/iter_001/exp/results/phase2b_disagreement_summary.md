# Phase 2b: Cross-Method Disagreement Analysis (H-D4) — FULL

**Date**: 2026-03-17T22:54:27.089270
**Mode**: FULL (500 test points, 50000 train, full-model)

## LDS Comparison

| Method | Mean LDS | Std |
|--------|----------|-----|
| EK-FAC IF | 0.2969 | 0.1009 |
| K-FAC IF | 0.2706 | 0.1062 |
| RepSim | 0.0737 | 0.1045 |

**IF-better**: 471/500 (94.2%)
**RepSim-better**: 29/500 (5.8%)

## Kendall Tau (IF-RepSim Disagreement)

- Mean: -0.4667 ± 0.1144
- Range: [-0.6187, -0.0476]
- Median: -0.5114

## Key Correlations

| Feature | Spearman rho with LDS_diff | p-value |
|---------|---------------------------|---------|
| tau | 0.5241 | 1.25e-36 |
| abs_tau | -0.5241 | 1.25e-36 |
| log_grad_norm | 0.5048 | 1.10e-33 |
| confidence | -0.4357 | 1.39e-24 |
| entropy | 0.4275 | 1.25e-23 |

**Partial correlation** (tau vs LDS_diff | class + grad_norm): rho = 0.2657, p = 1.59e-09

## AUROC Analysis

### Binary AUROC (IF-better vs RepSim-better)
- Best predictor: tau → AUROC = 0.6912
- Per predictor: {"tau": 0.6912, "-tau": 0.3088, "|tau|": 0.3088, "-|tau|": 0.6912, "log_grad_norm": 0.5967, "confidence": 0.445, "entropy": 0.5529}

### Quantile AUROC (median split on LDS_diff)
- Best predictor: tau → AUROC = 0.7775

### Class-Stratified AUROC
- Binary stratified mean: 0.6253 (4 classes)
- Quantile stratified mean: 0.7461 (10 classes)

### Per-Class Detail

| Class | N | IF-better | RepSim-better | Binary AUROC | Quantile AUROC | Mean tau |
|-------|---|-----------|---------------|-------------|----------------|---------|
| 0 | 50 | 50 | 0 | N/A | 0.8048 | -0.4649 |
| 1 | 50 | 39 | 11 | 0.7016 | 0.6864 | -0.5100 |
| 2 | 50 | 49 | 1 | N/A | 0.7040 | -0.4649 |
| 3 | 50 | 49 | 1 | N/A | 0.8112 | -0.4107 |
| 4 | 50 | 45 | 5 | 0.5689 | 0.6896 | -0.4831 |
| 5 | 50 | 50 | 0 | N/A | 0.7952 | -0.4245 |
| 6 | 50 | 46 | 4 | 0.5109 | 0.5792 | -0.5022 |
| 7 | 50 | 48 | 2 | N/A | 0.7168 | -0.4732 |
| 8 | 50 | 50 | 0 | N/A | 0.9040 | -0.4620 |
| 9 | 50 | 45 | 5 | 0.7200 | 0.7696 | -0.4720 |

### Multi-Feature Logistic Regression
- Binary: AUROC = 0.6394 ± 0.0803
  Coefficients: {"tau": 2.2368, "log_grad_norm": 0.2464, "confidence": 0.1763, "entropy": -1.0143}
- Quantile: AUROC = 0.7614 ± 0.0255
  Coefficients: {"tau": 2.3752, "log_grad_norm": 0.4964, "confidence": 0.2133, "entropy": -0.7604}

## Gate Decision: **PASS**

- Global AUROC: 0.6912 (threshold: 0.60, source: binary) → PASS
- Class-stratified AUROC: 0.7461 (threshold: 0.55, source: quantile_stratified) → PASS
- Fraction RepSim-better: 0.058 (29/500)

## Key Observations
- Full-model attribution: 471/500 IF-better, 29/500 RepSim-better
- Mean LDS: IF=0.2969, RepSim=0.0737 (diff=0.2232)
- Kendall tau(IF,RepSim) mean=-0.4667 (NEGATIVE: IF and RepSim anti-correlated in full-model)
- tau strongly correlates with LDS_diff (rho=0.5241, p=1.25e-36)
- Partial correlation (tau|class+grad_norm) remains significant: rho=0.2657 — disagreement signal is NOT just a class proxy
- Unlike pilot, full-model finds 29 RepSim-better points — routing has real value