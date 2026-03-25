# Testable Hypotheses: AURA

## Gating Hypotheses (Must pass before proceeding)

### H-G1: Per-Sample Attribution Sensitivity Exists Beyond Class and Gradient Norm

**Statement**: After controlling for class label and log(gradient norm), the residual per-test-point variance in attribution sensitivity accounts for more than 30% of total variance.

**Operationalization**: Two-way ANOVA on 500 CIFAR-10 test points (50/class). Response variables: (a) Jaccard@10(EK-FAC, K-FAC), (b) Kendall tau(IF, RepSim), (c) per-point LDS(EK-FAC IF vs TRAK-50). Sequential sum of squares with class entered first.

**Expected outcome**: Residual variance 30-50%, with class explaining 30-40% and gradient norm 10-20%.

**Pass**: Residual > 30% on at least 1 of 3 metrics.
**Borderline**: Residual 20-30% on at least 1 metric.
**Fail**: All 3 metrics show residual < 20%.

**Consequence of failure**: Kills all per-sample diagnostic proposals. Report as C1 negative result. Redirect to class-conditional method selection.

**Source**: Contrarian H-Con1, Empiricist Angle 1.

---

## Diagnostic Hypotheses (Conditional on H-G1 pass)

### H-D1: BSS Outlier-Bucket Ranking Is Seed-Stable

**Statement**: The ranking of test points by BSS_outlier (spectral sensitivity concentrated in the outlier eigenspace) has cross-seed Spearman rho > 0.5, compared to scalar TRV rho ~ 0.

**Operationalization**: Train 5 ResNet-18 on CIFAR-10 (seeds 42, 123, 456, 789, 1024). Compute GGN top-100 eigenvalues/eigenvectors via Kronecker-factored eigendecomposition. Define 3 eigenvalue buckets: outlier (lambda > 100), edge (10 < lambda < 100), bulk (lambda < 10). Compute BSS_outlier for 300 test points per seed. Report mean pairwise Spearman rho across all 10 seed pairs.

**Expected outcome**: BSS_outlier rho 0.5-0.7, driven by class-structure alignment being data-geometric rather than seed-dependent.

**Pass**: Mean pairwise rho > 0.5.
**Borderline**: Mean pairwise rho 0.3-0.5.
**Fail**: Mean pairwise rho < 0.3.

**Consequence of failure**: BSS inherits TRV's instability. Fall back to ensemble-averaged diagnostics (Interdisciplinary Angle 1) or class-conditional approach.

**Source**: Theorist Angle 1, Innovator Angle 1. Grounded in Ghorbani et al. (2019) and Papyan (2020).

### H-D2: BSS Outlier Predicts IF Reliability

**Statement**: BSS_outlier(z) correlates with per-test-point IF quality (LDS of EK-FAC IF against TRAK ground truth) with Spearman rho > 0.4. Test points with high BSS_outlier (energy concentrated in outlier eigenspace where Kronecker factorization is most accurate) have more reliable IF attributions.

**Operationalization**: Compute LDS for each of 300 test points. Compute Spearman(BSS_outlier, per-point LDS). Also compute partial correlation controlling for class and gradient norm.

**Expected outcome**: Spearman rho 0.3-0.5; partial rho (controlling for class + grad norm) > 0.2.

**Pass**: Spearman > 0.4 AND partial rho > 0.2.
**Borderline**: Spearman 0.3-0.4 OR partial rho 0.1-0.2.
**Fail**: Spearman < 0.3 OR partial rho < 0.1.

**Consequence of failure**: BSS does not predict attribution quality. Fall back to disagreement-based diagnostic or report BSS as descriptive only.

**Source**: Theorist H-Th2, cross-referencing Hong et al. finding that K-FAC eigenvalue mismatch is concentrated in outlier modes.

### H-D3: BSS Is Not Merely a Class Detector

**Statement**: Within-class BSS_outlier variance accounts for more than 25% of total BSS_outlier variance. BSS captures genuine per-sample structure beyond class membership.

**Operationalization**: ANOVA of BSS_outlier ~ class label. Report within-class variance / total variance.

**Expected outcome**: Within-class fraction 30-50%.

**Pass**: Within-class variance > 25%.
**Fail**: Within-class variance < 15%.

**Consequence of failure**: BSS reduces to a class detector. Recommend class-conditional method selection as simpler alternative.

**Source**: Contrarian's class-dominance challenge.

### H-D4: Cross-Method Disagreement Is Informative After Class Control

**Statement**: Per-test-point Kendall tau disagreement between IF and RepSim predicts which method has higher LDS, with class-stratified AUROC > 0.60.

**Operationalization**: For 500 test points, label "IF better" vs "RepSim better" based on LDS. Compute within-class AUROC of Kendall tau as predictor. Report mean across 10 classes.

**Expected outcome**: Global AUROC 0.65-0.75; class-stratified AUROC 0.55-0.65.

**Pass**: Global AUROC > 0.70 AND class-stratified AUROC > 0.60.
**Borderline**: Global AUROC 0.60-0.70 OR class-stratified AUROC 0.55-0.60.
**Fail**: Global AUROC < 0.60 OR class-stratified AUROC < 0.55.

**Consequence of failure**: Disagreement is uninformative or class-driven. Use class-conditional lookup instead.

**Source**: Pragmatist Angle 1, Empiricist Angle 2.

---

## Fusion Hypotheses (Conditional on H-D1 or H-D4 pass)

### H-F1: Adaptive Strategy Pareto-Dominates Uniform Strategies

**Statement**: At a fixed compute budget C, an adaptive strategy (cheap IF + RepSim + routing signal) achieves higher mean LDS than any uniform strategy (single method at full budget or fixed-weight ensemble). The adaptive strategy must exceed the uniform Pareto frontier by > 2% absolute LDS at the same compute budget.

**Operationalization**: Plot LDS vs GPU-hours for all uniform and adaptive strategies on 500 CIFAR-10 test points. Compute the maximum LDS gap between the adaptive frontier and the uniform frontier at each compute level.

**Expected outcome**: 2-4% LDS improvement at the K-FAC IF compute budget level.

**Pass**: Adaptive > uniform by > 2% at the adaptive strategy's own compute budget.
**Borderline**: 1-2% improvement.
**Fail**: < 1% improvement or adaptive below uniform frontier.

**Consequence of failure**: Kill Phase 2. Report Pareto frontier as constructive finding: "compute is better spent on global approximation quality." Redirect to C1 + C3 paper.

**Source**: Contrarian H-Con2, Empiricist Angle 3.

### H-F2: BSS-Guided Fusion Outperforms Naive Ensemble

**Statement**: BSS-guided adaptive weighting of IF + RepSim achieves LDS at least 1.5% absolute higher than fixed 0.5:0.5 ensemble, and closes > 30% of the gap to oracle per-point method selection.

**Operationalization**: Oracle selection = max(LDS_IF(z), LDS_RepSim(z)) per test point. BSS fusion weight: w(z) = sigmoid(-a * ||BSS(z)||_1 + b), calibrated on 300 points, evaluated on 200. Compare mean LDS of BSS fusion vs naive ensemble vs oracle.

**Expected outcome**: BSS fusion closes 30-50% of oracle gap.

**Pass**: LDS improvement > 1.5% AND gap closure > 30%.
**Borderline**: LDS improvement 0.5-1.5% OR gap closure 20-30%.
**Fail**: LDS improvement < 0.5%.

**Consequence of failure**: BSS does not translate to practical fusion improvement. Report BSS as diagnostic-only tool.

**Source**: Theorist Angle 2 (Propositions 3-4), Pragmatist Angle 3.

---

## Auxiliary Hypotheses (Informative but not gating)

### H-A1: Three Uncertainty Components Are Near-Orthogonal

**Statement**: U_metric (BSS), U_geodesic (cross-seed attribution variance), and U_curvature (Laplace posterior variance) have pairwise Spearman rho < 0.3 across test points, confirming that TRV, Daunce, and BIF measure genuinely different aspects of attribution reliability.

**Expected outcome**: rho(metric, geodesic) ~ 0.1-0.3; rho(metric, curvature) ~ 0.2-0.4; rho(geodesic, curvature) ~ 0.1-0.2.

**Source**: Theorist Angle 3, Interdisciplinary synthesis.

### H-A2: Attribution Stability Has Zero or Negative Partial Correlation with Correctness

**Statement**: After controlling for class label and gradient norm, Spearman(stability_metric, LOO_correctness) < 0.1 for any stability metric (TRV, BSS, cross-method agreement).

**Expected outcome**: Partial rho 0.05-0.15 (weak positive, not zero, but much weaker than unstratified).

**Source**: Contrarian H-Con4. Tests the "stable but wrong" concern.

### H-A3: Ensemble TRV Is More Stable Than Single-Seed TRV

**Statement**: TRV averaged across 5 training seeds (Ensemble TRV) has leave-one-seed-out Spearman rho > 0.6, compared to single-seed TRV rho ~ 0.

**Expected outcome**: Ensemble TRV rho 0.5-0.7.

**Source**: Interdisciplinary Angle 1 (Hampel's influence function), Pragmatist Angle 2.

---

## Summary Table

| ID | Hypothesis | Gate? | Success Metric | Falsification | Priority |
|---|---|---|---|---|---|
| H-G1 | Per-sample variance > 30% after class control | **YES** | Residual R2 > 0.30 | All metrics < 20% | 1 (must run first) |
| H-D1 | BSS outlier ranking seed-stable (rho > 0.5) | Phase 2 | Mean pairwise Spearman | rho < 0.3 | 2 |
| H-D2 | BSS predicts IF reliability (rho > 0.4) | Phase 2 | Spearman + partial | rho < 0.3 | 2 |
| H-D3 | BSS not class-dominated (within-class > 25%) | Phase 2 | ANOVA | Within-class < 15% | 2 |
| H-D4 | Disagreement informative after class control | Phase 2 | Class-stratified AUROC | AUROC < 0.55 | 2 |
| H-F1 | Adaptive Pareto-dominates uniform | Phase 3 | LDS gap > 2% | Gap < 1% | 3 |
| H-F2 | BSS fusion > naive ensemble | Phase 3 | LDS > 1.5%, gap > 30% | LDS < 0.5% | 3 |
| H-A1 | Three uncertainty components orthogonal | Aux | Pairwise rho < 0.3 | Any pair > 0.6 | 4 |
| H-A2 | Stability-correctness partial rho ~ 0 | Aux | Partial rho < 0.1 | Partial rho > 0.3 | 4 |
| H-A3 | Ensemble TRV stable (rho > 0.6) | Aux | LOO Spearman | rho < 0.4 | 4 |
