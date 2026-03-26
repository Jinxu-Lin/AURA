# Critique Summary

## Aggregate Scores

| Dimension | Score (1-10) | Notes |
|-----------|:------------:|-------|
| Novelty | 6.5 | BSS is genuinely new as a per-point Hessian sensitivity diagnostic; RMT connection is the key insight; MRC is more incremental |
| Soundness | 6.0 | Core math correct; RMT grounding aspirational for actual setup; MRC optimality claim overstated; perturbation factor uniformity concern |
| Experiments | 4.0 | ANOVA and pilot BSS results are solid; but C2 seed stability and C3 MRC combining are entirely PENDING (~40% complete) |
| Presentation | 7.5 | Strong opening, clear narrative, good equation motivation; intro slightly long; section numbering mismatch |
| Reproducibility | 5.5 | Good high-level detail; missing training hyperparameters and K-FAC estimation specifics; no code release mentioned |

**Weighted average: 5.7** (using P5 weights: 0.25/0.20/0.25/0.10/0.05 + significance 0.15 estimated at 6.0)

---

## Critical Issues (must fix)

1. **Cross-seed BSS stability is PENDING.** This is the central claim differentiating BSS from TRV. Without empirical evidence that BSS cross-seed rho >> 0 (while TRV rho ~ 0), the paper's core thesis is unsubstantiated. Priority: run the 5-seed experiment immediately.

2. **MRC main results are entirely PENDING.** Contribution C3 has zero empirical support. Table 6 is empty. The Pareto frontier figure does not exist. Without these, the paper has only two supported contributions (C1 ANOVA, C4 negative results), which is thin for a top venue.

3. **Perturbation factor uniformity undermines BSS.** When damping dominates eigenvalues, BSS reduces to gradient projection energy---essentially a rotated gradient norm. The partial BSS and BSS_ratio corrections are proposed but their effectiveness is PENDING. If the residual signal after correction is negligible, BSS adds nothing beyond gradient norm.

---

## Major Issues (should fix)

4. **Shared-eigenvector assumption gap.** Proposition 1 assumes K-FAC and EK-FAC share eigenvectors, but EK-FAC explicitly rotates them. Discuss the approximation quality and its impact on BSS accuracy.

5. **MRC "provably optimal" overclaim.** Proposition 3 guarantees optimality for inverse-variance weighting with known variances. The parameterized sigmoid weight (Eq. 13) with BSS as a proxy does not inherit this guarantee. Soften the claim.

6. **Section numbering mismatch.** The introduction references "Section 5 reports results. Section 6 discusses limitations" but the paper has Abstract, Sections 1-4, and Section 5 (Conclusion). Fix to match actual structure.

7. **Statistical power for BSS pilot.** 100 points with extreme skew (outlier bucket std/mean = 5x) may not reliably characterize the BSS distribution. Report confidence intervals or bootstrap standard errors.

---

## PENDING Triage

### Critical (blocks submission)
- Cross-seed BSS stability (5 seeds, 10 pairs, mean pairwise Spearman rho)
- MRC main results (Table 6: 11 strategies, LDS + GPU-hours)
- Pareto frontier figure
- Partial BSS predictive power after gradient-norm correction

### Important (strongly recommended)
- BSS_ratio results
- Full-scale 500 test points for disagreement analysis
- At least damping sensitivity ablation
- TRAK and W-TRAK LDS for complete method comparison

### Optional (can defer or move to appendix)
- Full ablation table (Table 7, all 6 dimensions)
- LOO validation of TRAK-50 ground truth
- BSS--LOO correctness partial correlation
- Class-stratified AUROC on full 500 points
