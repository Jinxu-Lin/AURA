# AURA: Spectral Sensitivity Analysis for Training Data Attribution

## Title

**Spectral Sensitivity Analysis Reveals When Training Data Attribution Fails: Bucketed Spectral Sensitivity, Variance Decomposition, and Adaptive Method Selection**

## Abstract

Training Data Attribution (TDA) methods quantify the influence of training samples on model predictions, but their results are highly sensitive to Hessian approximation choices -- the same test point can yield completely different top-k attributions depending on whether one uses K-FAC, EK-FAC, or full GGN. We investigate when and why this sensitivity arises at the per-test-point level. Our approach has three components. First, we perform a **variance decomposition** of attribution sensitivity to determine whether per-test-point variation is genuine or dominated by class membership and gradient magnitude -- a foundational question the community has not addressed. Second, we develop **Bucketed Spectral Sensitivity (BSS)**, a theoretically grounded diagnostic that decomposes per-test-point attribution error by spectral region of the Hessian, leveraging operator perturbation theory to explain *which* eigenvalue ranges cause instability for each test point. Third, we evaluate whether BSS-guided **adaptive method selection** between IF and RepSim can outperform uniform baselines on a compute-normalized Pareto frontier. We design our experimental program as a progressive gate: the variance decomposition must pass before spectral diagnostics are pursued, and spectral diagnostics must demonstrate utility before adaptive fusion is attempted. Our framework addresses the critical cross-seed instability of the original TRV metric (Spearman rho ~ 0 in pilot probes) by operating on eigenvalue-magnitude buckets rather than individual eigenvector directions, which theory predicts to be seed-stable.

## Motivation

### The Core Problem

TDA methods face a silent reliability crisis. Hong et al. (2509.23437) demonstrated that LDS strictly follows the Hessian approximation hierarchy (H >= GGN >> EK-FAC >> K-FAC), with the K-FAC-to-EK-FAC eigenvalue mismatch accounting for 41-65% of total error. Our own Probe experiment confirmed this dramatically: Jaccard@10 drops from 1.0 (Full GGN self-agreement) to ~0.48 (K-FAC) -- meaning more than half of the top-10 attributions change when switching Hessian approximation. Yet practitioners have no tool to diagnose which test points are most affected, and no principled way to mitigate the damage.

### Why Prior Attempts Failed

Our Phase 1 Probe experiment (CIFAR-10/ResNet-18, 3 seeds, 100 test points) revealed critical failures in the original TRV (TDA Robustness Value) design:

1. **Cross-seed TRV instability (CRITICAL)**: Spearman rho ~ 0 between TRV rankings across training seeds. TRV is a (model-instance, test-point) joint property, not a test-point intrinsic.
2. **SI-TRV null correlation**: Self-Influence cannot serve as a cheap TRV proxy (H4 falsified). SI captures distribution-shift sensitivity, orthogonal to Hessian-approximation sensitivity.
3. **Per-point variance insufficient**: TRV std 0.053-0.082, below the 0.15 threshold for a useful continuous routing signal.
4. **Hessian hierarchy bottom-collapse**: Diagonal ~ Damped Identity ~ Identity in last-layer setting, effectively reducing 5 tiers to 2-3.

### Why This Proposal Can Succeed

The root cause of cross-seed instability is that the original TRV depends on individual eigenvector directions, which rotate across seeds. Our key insight, grounded in random matrix theory (Ghorbani et al. 2019, Papyan 2020), is that **eigenvalue-magnitude distributions are seed-stable** even when individual eigenvectors are not. The outlier eigenvalue count equals the number of classes, and outlier eigenvalue magnitudes are determined by class separation (stable to O(1/sqrt(N)) fluctuations). By bucketing spectral energy into magnitude ranges (outlier/edge/bulk) rather than tracking individual eigenvectors, we obtain a per-test-point diagnostic that is theoretically robust to seed-induced eigenvector rotation.

However, the Contrarian perspective raises a critical prior question: does genuine per-test-point variation even exist beyond class membership and gradient magnitude? We treat this as a gating experiment.

## Research Questions

**RQ1 (Gating)**: After controlling for class label and log(gradient norm), what fraction of attribution sensitivity variance is residual per-test-point variation? If < 20%, per-sample diagnostics are fundamentally limited.

**RQ2 (Diagnostic)**: Does Bucketed Spectral Sensitivity (BSS) provide a seed-stable, per-test-point diagnostic of attribution reliability? Specifically, is cross-seed BSS ranking Spearman rho > 0.5, compared to scalar TRV rho ~ 0?

**RQ3 (Utility)**: Does BSS-guided adaptive method selection (routing between IF and RepSim) Pareto-dominate uniform strategies (best single method or fixed-weight ensemble) at equal compute budget?

## Hypotheses

See `hypotheses.md` for the complete list with falsification conditions.

## Proposed Approach

### Component 1: Attribution Variance Decomposition (Gating Experiment)

**What**: Two-way ANOVA decomposing attribution sensitivity (Jaccard@10 between EK-FAC and K-FAC, cross-method IF-RepSim disagreement, per-point LDS) into class-conditional, gradient-norm, and residual components.

**Why this must come first**: Every other angle -- spectral fingerprints, BSS, cross-method disagreement, lightweight selectors -- assumes meaningful per-test-point variation exists beyond class membership. The Probe data hints that ~20% of test points are "immune" (TRV=5), but we do not know whether this reflects genuine per-sample structure or class-conditional effects. The Contrarian correctly notes that Hessian outlier eigenspaces correspond to class-discriminative directions (Papyan 2020), so spectral diagnostics could reduce to class detectors.

**Design**: 500 CIFAR-10 test points (50/class, stratified), single seed, full-model EK-FAC IF + K-FAC IF + RepSim. ANOVA with class entered first (Type I sequential SS). Report partial R-squared for class, gradient norm, and their interaction.

**Gate criteria**: Residual variance > 30% on at least one of three response variables (J10, cross-method tau, per-point LDS) = PASS. All three < 20% = STOP all per-sample diagnostic development.

**Compute**: ~5 GPU-hours. **Critical requirement**: Must use full-model Hessian (not last-layer only), since the Probe's last-layer setting collapsed the bottom 3 hierarchy levels.

### Component 2: Bucketed Spectral Sensitivity (BSS)

**What**: For each test point z with gradient g, decompose attribution sensitivity by Hessian eigenvalue magnitude buckets:

BSS_j(z) = sum_{k in B_j} |1/lambda_k - 1/tilde_lambda_k| * (V_k^T g)^2

where B_j partitions eigenvalues by magnitude (outlier: lambda > 100, edge: 10 < lambda < 100, bulk: lambda < 10), and the perturbation factors |1/lambda_k - 1/tilde_lambda_k| capture where the K-FAC-to-EK-FAC approximation actually introduces error.

**Why BSS over alternatives**:
- **vs. scalar TRV**: BSS preserves spectral structure that scalar TRV collapses, explaining cross-seed instability.
- **vs. Innovator's Spectral Concentration Ratio (SCR)**: BSS adds perturbation-theoretic weighting |1/lambda_k - 1/tilde_lambda_k| that captures *where the approximation errors concentrate*, not just where the test gradient has energy. SCR is the cheap approximation; BSS is the theoretically motivated version.
- **vs. SI (Natural W-TRAK)**: SI collapses all spectral modes indiscriminately into a scalar. The Probe showed SI-TRV rho ~ 0 precisely because SI conflates spectral modes with opposing perturbation sensitivities.
- **vs. Daunce / BIF**: These capture model-perturbation variance and Bayesian posterior variance respectively. BSS captures Hessian-approximation-choice sensitivity -- a theoretically orthogonal dimension that practitioners directly face ("should I use EK-FAC or K-FAC?").

**Theoretical grounding**: Proposition 1 (spectral decomposition of attribution error) provides an exact decomposition when H and H_tilde share eigenvectors; Proposition 2 (eigenvalue bucket stability under RMT) predicts that BSS rankings are seed-stable because per-bucket energy depends on data geometry, not training randomness.

**Conditional on Component 1 passing**: If variance decomposition shows residual > 30%, proceed. Must also verify that BSS does not simply reduce to a class detector: measure within-class BSS_outlier variance as a fraction of total BSS_outlier variance (must be > 25%).

### Component 3: Adaptive Method Selection with Compute-Normalized Evaluation

**What**: Use BSS (or cheaper proxy features: gradient norm, confidence, entropy, cross-method disagreement) to select between IF and RepSim per test point. Evaluate on a Pareto frontier plotting LDS vs. GPU-hours.

**Why Pareto frontier, not just LDS**: The Contrarian's "routing tax" objection is legitimate. Computing BSS requires eigendecomposition (~2 GPU-hours). If that compute were instead spent on a better Hessian approximation (EK-FAC instead of K-FAC), the uniform improvement might exceed the adaptive gain. Only a compute-normalized comparison can settle this.

**Design**:
- Uniform strategies: Identity IF, K-FAC IF, EK-FAC IF, RepSim, TRAK-10, TRAK-50, naive 0.5:0.5 ensemble
- Adaptive strategies: (a) BSS-guided routing, (b) disagreement-guided routing, (c) class-conditional selection, (d) lightweight feature-based selector (logistic regression)
- Ground truth: TRAK-50 for 500 points; exact LOO on CIFAR-10/5K subset for 100 points (validation of TRAK ground truth quality)
- Class-stratified AUROC as mandatory control: if adaptive AUROC drops below 0.55 within classes, the routing signal is a class proxy

**Conditional on Components 1-2**: Only run if (a) residual per-sample variance > 20%, (b) BSS cross-seed rho > 0.4 or disagreement signal passes class-stratified test.

### Integration of Perspectives

| Perspective | Core Contribution Adopted | How Integrated |
|---|---|---|
| **Innovator** | Spectral fingerprint routing concept; multi-fidelity framing | BSS is the theoretically motivated version of SCR; fidelity hierarchy informs Pareto frontier construction |
| **Pragmatist** | Cross-method disagreement as cheapest diagnostic; engineering-first library usage | Disagreement is a candidate routing signal tested alongside BSS; TRAK/pyDVL/dattri as implementation backbone |
| **Theorist** | Operator perturbation bounds; semiparametric fusion weights; BSS derivation | BSS framework directly adopted; MSE-optimal fusion weights as theoretical anchor for adaptive weighting |
| **Contrarian** | Class-conditional null hypothesis; routing tax; stability != correctness | Variance decomposition as mandatory gating experiment; Pareto frontier evaluation; partial correlations controlling for class |
| **Interdisciplinary** | GUM uncertainty budget; Hampel's influence function; precision-weighted fusion | TRV uncertainty budget (near-zero-cost reanalysis of Probe data); precision-weighting as fusion mechanism; breakdown point as robustness certificate |
| **Empiricist** | Progressive gating design; falsification criteria; class-stratified AUROC; LOO validation | Entire experimental flow is gated; all hypotheses have pre-registered falsification conditions; LOO validation of TRAK ground truth |

### Weighting Rationale

I weighted the **Empiricist** and **Contrarian** perspectives most heavily, because they address the most fundamental risks:

1. The Empiricist's variance decomposition is the experiment that determines whether the *entire direction* of per-sample diagnostics is viable. Without it, all other proposals are building on an unverified assumption.

2. The Contrarian's challenges (class dominance, routing tax, stability != correctness) are the objections any strong reviewer will raise. Incorporating them upfront transforms AURA from "we propose X and it works" to "we rigorously investigate when X works and when it doesn't."

3. The **Theorist** provided the critical theoretical upgrade (BSS over ad hoc TRV), which directly addresses the cross-seed instability.

4. The **Pragmatist** ensured all computations use existing, battle-tested libraries rather than custom implementations.

5. The **Innovator** and **Interdisciplinary** perspectives contributed important conceptual frameworks (spectral routing, GUM uncertainty budgets) that elevate the contribution beyond an empirical study.

## Expected Contributions

**C0 (Revised)**: Bucketed Spectral Sensitivity (BSS) as a theoretically grounded, per-test-point diagnostic of attribution reliability -- replacing the unstable scalar TRV with a spectral decomposition that leverages operator perturbation theory and RMT-predicted seed stability.

**C1 (Empirical)**: First systematic variance decomposition of TDA sensitivity into class-conditional, gradient-norm, and residual per-sample components. This determines whether per-sample diagnostics can work at all, regardless of which specific diagnostic is used.

**C2 (Practical)**: Compute-normalized Pareto frontier comparing adaptive vs. uniform TDA strategies, directly addressing the "routing tax" question. Includes BSS-guided routing, disagreement-guided routing, and class-conditional selection as competing approaches.

**C3 (Negative results, if applicable)**: Honest reporting of (a) cross-seed TRV instability, (b) SI-TRV null correlation, (c) variance decomposition results (even if they show class dominance), (d) stability-correctness relationship (even if near-zero after class control). These constrain the field's expectations about per-sample TDA diagnostics.

## Experimental Plan Overview

| Phase | Experiment | GPU-hrs | Gate |
|---|---|---|---|
| 0 | Reanalyze Probe data: class-conditional TRV means, uncertainty budget | 0 | None |
| 1 | Variance decomposition (full-model, 500 points) | ~5 | Residual > 20% to proceed |
| 2a | BSS computation + cross-seed stability (5 seeds) | ~7 | BSS rho > 0.4 to proceed to Phase 3 |
| 2b | Cross-method disagreement + class-stratified AUROC | ~0 (reuses Phase 1 data) | AUROC > 0.55 within classes |
| 3 | Pareto frontier (uniform + adaptive, + LOO validation) | ~30 | Adaptive > uniform by > 2% at same budget |
| **Total (if all gates pass)** | | **~42** | |
| **Total (fast fail at Phase 1)** | | **~5** | |

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Residual per-sample variance < 20% | 40% | Direction-ending for per-sample diagnostics | Report as major negative result (class-conditional selection suffices); pivot to C1+C3 as primary contributions |
| BSS reduces to class detector | 30% | BSS has no value beyond class label | Measure within-class BSS variance; report class-conditional BSS if within-class variance is low |
| Adaptive fusion < uniform at all compute budgets | 35% | Phase 2 (RA-TDA) has no practical value | Report Pareto frontier analysis as constructive negative result; recommend compute investment in better global approximations |
| Full-model Hessian code engineering exceeds time budget | 25% | Delays experiments by 1-2 weeks | Use pyDVL's EK-FAC (supports full-model per-test-point attributions) as fallback |
| TRAK ground truth is too noisy for per-point evaluation | 20% | All LDS-based conclusions are suspect | LOO validation on CIFAR-10/5K subset (100 points); dual-metric reporting (Spearman LDS + Jaccard@k) |

## Target Venue

NeurIPS 2026 or ICML 2026. The progressive gating design means the paper can take one of several forms:
- **Best case**: BSS diagnostic + adaptive fusion outperforms uniform (full C0-C2-C3 paper)
- **Medium case**: BSS is seed-stable but adaptive fusion is marginal (C0-C1 diagnostic paper)
- **Worst case**: Per-sample variation is class-dominated (C1-C3 negative results paper, target TMLR)
