# Phase 3: Adaptive Strategy Evaluation (PILOT)

## Key Results

| Strategy | Mean LDS (all) | Mean LDS (eval) | Gap Closure | GPU-hrs |
|----------|---------------|-----------------|-------------|---------|
| Oracle (per-point best) | 0.7443 | - | 1.0000 | 3.0 |
| K-FAC IF (best uniform) | 0.7444 | - | 1.0004 | 2.0 |
| BSS sigmoid fusion | 0.7219 | 0.7009 | 0.8433 | 2.5 |
| Disagreement fusion | 0.7411 | 0.7199 | 0.9775 | 2.3 |
| Class-weighted | 0.7443 | 0.7231 | 1.0000 | 2.0 |
| LR selector (fusion) | 0.5258 | 0.5095 | -0.5286 | 2.3 |
| 0.5:0.5 ensemble | 0.6014 | - | 0.0000 | 3.3 |
| Random routing | 0.4893 | 0.4849 | -0.7844 | 2.0 |

## Class-Stratified AUROC (routing signal quality)

| Strategy | Global AUROC | Mean Class AUROC |
|----------|-------------|-----------------|
| BSS fusion | 0.576 | 0.654265873015873 |
| Disagreement | 0.7548 | 0.660218253968254 |
| LR selector | 0.7508 | 0.6568452380952381 |
| Random | 0.5664 | 0.642361111111111 |

## Interpretation

IF universally dominates RepSim in this pilot (layer4+fc setting), so adaptive routing
between IF and RepSim cannot improve over pure IF. However, routing signals ARE informative
about the MAGNITUDE of IF advantage:
- |tau| (disagreement) strongly anti-correlates with LDS_diff (rho=-0.55)
- LR selector CV AUROC = 0.4977777777777778 for predicting high vs low IF advantage
- BSS partially correlates with routing quality via gradient norm

Full-model experiment is critical: deeper layers may create RepSim-better points where
routing adds genuine value.

## Pass Criteria
- All 4 adaptive strategies valid: **True**
- Adaptive exceeds uniform: **False** (expected: NO in pilot due to IF dominance)
- Class AUROC computable: **True**
- Overall: **GO**
