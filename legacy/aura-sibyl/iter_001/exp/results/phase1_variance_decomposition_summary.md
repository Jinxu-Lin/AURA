# Phase 1 Variance Decomposition (FULL, 500 test points)

## Task
Two-way ANOVA on 500 CIFAR-10 test points (50/class, stratified).
Predictors: class (10 levels), log(gradient_norm). Type I sequential SS.

## Variance Decomposition Results

| Response | Class R² | GradNorm R² | Interaction R² | Residual R² |
|----------|----------|-------------|----------------|-------------|
| J10 | 0.182 | 0.009 | 0.035 | 0.775 |
| tau | 0.064 | 0.348 | 0.054 | 0.534 |
| LDS | 0.121 | 0.405 | 0.015 | 0.459 |

## Statistical Significance

| Response | Class F | Class p | GradNorm F | GradNorm p | Interact F | Interact p |
|----------|---------|---------|------------|------------|------------|------------|
| J10 | 12.49 | 7.04e-18 | 5.44 | 2.01e-02 | 2.38 | 1.20e-02 |
| tau | 6.43 | 1.20e-08 | 312.63 | 3.04e-54 | 5.41 | 4.46e-07 |
| LDS | 14.09 | 3.04e-20 | 423.01 | 7.19e-68 | 1.71 | 8.41e-02 |

## Descriptive Statistics

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| J10 | 0.8347 | 0.1283 | 0.4286 | 1.0000 | 0.8182 |
| tau | -0.4667 | 0.1145 | -0.6187 | -0.0476 | -0.5114 |
| LDS | 0.2969 | 0.1010 | 0.0888 | 0.5837 | 0.2707 |

## Gate Evaluation

**Criterion**: residual > 30% on at least 1 metric

| Metric | Residual Fraction | Pass? |
|--------|-------------------|-------|
| J10 | 0.775 | YES |
| tau | 0.534 | YES |
| LDS | 0.459 | YES |

**Overall Decision**: **PASS**

## Key Observations

- J10: Residual dominates (77.5%) - strong per-point signal
- tau: Residual dominates (53.4%) - strong per-point signal
- LDS: Moderate residual (45.9%) - meaningful per-point signal
- LDS: Grad norm explains 40.5% of variance - significant gradient effect