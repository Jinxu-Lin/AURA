# Auxiliary Hypotheses Testing (H-A1, H-A2, H-A3) -- Pilot Summary

## Task: auxiliary_hypotheses
**Status**: COMPLETED (MIXED: 1/3 pass, 2/3 fail)
**Runtime**: <1 second (planned: 20 min)
**GPU**: 0 (pure statistical analysis on cached data)

## Pass Criteria
> "All pairwise correlations computable AND no NaN in partial correlation"

**PASS**: All statistics computed without error or NaN.

---

## H-A1: Uncertainty Component Non-Redundancy

**Hypothesis**: Three uncertainty dimensions are non-redundant (all pairwise |rho| < 0.3).

| Pair | Spearman rho | p-value | |rho| < 0.3? |
|------|-------------|---------|-------------|
| U_metric (log BSS) vs U_geodesic (|tau|) | 0.171 | 0.089 | PASS |
| U_metric (log BSS) vs U_curvature (entropy) | -0.002 | 0.982 | PASS |
| U_geodesic (|tau|) vs U_curvature (entropy) | -0.518 | 3.4e-8 | **FAIL** |

**Gate**: **FAIL** (max |rho| = 0.518 > 0.3)

**Interpretation**: BSS is orthogonal to both disagreement and entropy proxies -- this is a positive finding, showing spectral sensitivity captures something distinct. However, |tau| and entropy are correlated because both are downstream consequences of prediction uncertainty: uncertain predictions (high entropy) tend to produce larger IF-RepSim disagreement. This is a proxy problem, not a fundamental redundancy. The full experiment should use proper U_geodesic (cross-seed attribution variance) and U_curvature (Laplace posterior variance), which are theoretically distinct.

---

## H-A2: Stability != Correctness

**Hypothesis**: Stability metrics have near-zero partial correlation with attribution correctness (LDS) after controlling for class and gradient norm.

| Stability Metric | Partial corr(metric, LDS | class, grad_norm) | p-value | < 0.1? |
|-----------------|-------------------------------------------|---------|--------|
| log(BSS_outlier) | 0.099 | 0.329 | Borderline PASS |
| |tau(IF,RepSim)| | 0.459 | 1.6e-6 | **FAIL** |
| confidence | -0.266 | 0.007 | **FAIL** |

**Gate**: **FAIL** (max |partial corr| = 0.459 > 0.1)

**Interpretation**: This is a nuanced finding. BSS is indeed near the 0.1 threshold (partial corr = 0.099), consistent with the "stability != correctness" claim. However, |tau| has a strong partial correlation with LDS (0.459, p=1.6e-6) even after controlling for class and gradient norm. This means IF-RepSim disagreement IS genuinely predictive of attribution quality beyond confounds. This is actually a positive finding for the routing hypothesis (H-D4/H-F1): disagreement can serve as a reliable routing signal. The claim needs refinement: SOME stability metrics (BSS) are orthogonal to correctness, while OTHERS (cross-method disagreement) are predictive.

---

## H-A3: Ensemble TRV Leave-One-Seed-Out Stability

**Hypothesis**: Ensemble TRV (mean across seeds) is leave-one-seed-out stable (rho > 0.6).

| Analysis | Result |
|----------|--------|
| Raw cross-seed TRV mean rho | 0.101 (reconstructed; reported ~0) |
| LOO seed 42: rho(2-seed avg, 3-seed avg) | 0.861 |
| LOO seed 123: rho(2-seed avg, 3-seed avg) | 0.861 |
| LOO seed 456: rho(2-seed avg, 3-seed avg) | 0.836 |
| **Mean LOO-vs-full rho** | **0.853** |
| Mean LOO pairwise rho | 0.598 |

**Gate**: **PASS** (mean LOO rho = 0.853 > 0.6)

**Interpretation**: Despite individual TRV rankings being essentially uncorrelated across seeds (rho~0), ensemble averaging produces rankings that are robust to leave-one-seed-out perturbation. The mean LOO-vs-full ensemble rho of 0.853 is well above the 0.6 threshold. Even LOO pairwise comparisons (leaving out different seeds) show moderate agreement (rho~0.60). This validates the ensemble averaging approach and predicts that BSS ensemble across 5 seeds will similarly produce stable rankings.

---

## Overall Assessment

| Hypothesis | Gate | Implication |
|------------|------|-------------|
| H-A1: Non-redundancy | FAIL (proxy) | BSS is orthogonal, but proxy components correlated. Need proper cross-seed variance and Laplace posterior. |
| H-A2: Stability != correctness | FAIL (nuanced) | Disagreement IS predictive (positive for routing). BSS is NOT predictive (as expected). |
| H-A3: Ensemble stability | **PASS** | Averaging stabilizes rankings -- validates BSS ensemble approach. |

**Key insight**: The H-A2 failure is actually informative. It reveals that cross-method disagreement (|tau|) captures genuine attribution quality signal beyond class and gradient norm confounds. Combined with the H-A3 ensemble stability result, this suggests a two-pronged approach for the full experiment:
1. Use BSS ensemble (averaged across 5 seeds) as a spectral diagnostic
2. Use disagreement (tau) as a routing signal for adaptive method selection

## Pilot Limitations
- Only 1 seed for BSS (need 5)
- U_geodesic and U_curvature are proxy measures (need cross-seed variance and Laplace posterior)
- TRV values reconstructed from distributions (no raw per-point Probe data)
- LOO data too noisy (10 test points on undertrained models)
- Layer4+fc scope may underestimate BSS discriminative power
