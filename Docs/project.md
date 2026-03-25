---
version: "1.0"
status: "complete"
decision: "pass"
created: "2026-03-25"
last_modified: "2026-03-25"
---

# Project: AURA — Spectral Sensitivity Analysis Reveals When Training Data Attribution Fails

## 1. Overview

### 1.1 Topic
Training Data Attribution (TDA) methods are highly sensitive to Hessian approximation choices, but no per-test-point diagnostic exists to identify unreliable attributions.

### 1.2 Initial Idea
TDA methods (Influence Functions, TRAK, RepSim) produce different attribution rankings depending on the Hessian approximation used (K-FAC vs EK-FAC vs full GGN). The same test point can receive completely different top-k attributed training examples, yet practitioners have no way to know when this happens.

We propose a three-component system: (1) Variance decomposition to confirm that per-test-point attribution variation exists beyond class and gradient-norm effects (confirmed: 77.5% residual J10 variance); (2) Bucketed Spectral Sensitivity (BSS) as a seed-stable, per-test-point diagnostic that uses eigenvalue-magnitude buckets (outlier/edge/bulk) rather than unstable eigenvector directions; (3) Adaptive method selection between IF and RepSim guided by BSS signals and cross-method disagreement.

The key theoretical insight is that while eigenvector directions are seed-unstable (explaining why prior diagnostics like TRV fail), eigenvalue magnitude distributions are architecture-determined and stable via random matrix theory. BSS exploits this stability to build a reliable diagnostic.

### 1.3 Baseline Papers

| # | Paper | Link | Relevance |
|---|-------|------|-----------|
| 1 | Hong et al. "Better Hessians" | arXiv:2509.23437 | Proves Hessian hierarchy affects LDS; motivates per-test-point analysis |
| 2 | Grosse et al. "Natural W-TRAK" | arXiv:2512.09103 | Self-Influence (SI) as Lipschitz bound, spectral condition number (κ) |
| 3 | Bae et al. "IF Failure on LLM" | arXiv:2409.19998 | IF collapses (0-7%) on LoRA, RepSim 96-100%; motivates complementarity |
| 4 | Choe et al. "Concept Influence" | arXiv:2602.14869 | IF-RepSim correlation only 0.37-0.45; quantifies method disagreement |
| 5 | Wang et al. "d-TDA" | arXiv:2506.12965 | Distributional IF framework; potential unified estimand |

### 1.4 Available Resources
- **GPU**: 4× RTX 4090 via SSH (xuchang0)
- **Timeline / DDL**: NeurIPS 2026 (submission ~May 2026)
- **Existing Assets**: Phase 0-1 probe code (Sibyl system), 500-point attribution data, pre-trained ResNet-18 (seed 42), partial eigendecomposition code
- **GPU Budget**: ~42 GPU-hours total

---

## 2. Problem & Approach

### 2.1 Baseline Analysis

#### 它们解决了什么
- Hong et al.: Established that better Hessian approximations (EK-FAC > K-FAC > diagonal) systematically improve average LDS scores, providing the Hessian hierarchy as a global quality ordering.
- Grosse et al.: Introduced Self-Influence (SI) as a Lipschitz bound on attribution perturbation sensitivity, plus spectral condition number κ as a global model-level diagnostic.
- Bae et al.: Documented IF's failure mode on large models (LoRA fine-tuned LLMs), showing RepSim as a robust fallback.
- Choe et al.: Quantified the low correlation (0.37-0.45) between IF and RepSim attributions, establishing that they measure fundamentally different aspects.
- Wang et al.: Proposed distributional TDA (d-TDA) framework unifying IF and kernel-based methods under a common estimand.

#### 它们没解决什么
- No per-test-point diagnostic: All evaluations are global averages (mean LDS across test set). A practitioner cannot identify which specific test points have unreliable attributions.
- SI as diagnostic is invalid: Our probe shows SI-TRV correlation ρ ≈ 0 — SI measures a different dimension than Hessian sensitivity.
- Cross-seed instability of existing diagnostics: TRV (True Residual Variance) has cross-seed ρ ≈ -0.006, making it useless as a stable diagnostic.
- No adaptive method selection: Methods are chosen globally (IF or RepSim for entire dataset), ignoring that different test points may benefit from different methods.

#### 为什么没解决
- **Evaluation paradigm**: LDS is a global average metric; the community optimizes for mean performance, masking per-point variation.
- **Spectral instability**: Prior attempts at spectral diagnostics relied on eigenvector directions, which are seed-unstable. No one has exploited eigenvalue magnitude stability.
- **Siloed methods**: IF and RepSim communities develop independently; the complementarity documented by Choe et al. has not been operationalized.

### 2.2 Problem Definition
- **问题一句话**: TDA methods produce different attribution rankings depending on Hessian approximation quality, but practitioners have no per-test-point diagnostic to identify which attributions are unreliable.
- **真实性论证**: Confirmed empirically — Jaccard@10 between EK-FAC and K-FAC drops to 0.45-0.53; ANOVA shows 77.5% residual variance in J10 after controlling class and gradient norm.
- **重要性论证**: Any downstream application relying on TDA (data debugging, fairness auditing, model explanation) inherits silent failures when attributions are Hessian-sensitive. Without diagnostics, practitioners cannot know when to trust results.
- **问题价值层次**: "Dimension not explored" — per-test-point Hessian sensitivity diagnosis is a new diagnostic axis.

### 2.3 Root Cause Analysis
1. **Symptom**: Different Hessian approximations produce different attribution rankings for the same test point.
2. **Intermediate cause**: Test-point gradients align differently with the Hessian error spectrum — some gradients project heavily onto high-error eigenspaces (outlier eigenvalues), making their attributions sensitive to approximation quality.
3. **Root cause**: The community evaluates TDA quality globally (average LDS), treating Hessian error as a uniform problem. Per-test-point sensitivity variation is invisible under global metrics. Prior spectral diagnostics (TRV) fail because they rely on seed-unstable eigenvector directions.

**Thought experiment**: If an oracle could perfectly identify which test points have Hessian-sensitive attributions, practitioners could (a) flag unreliable results, (b) allocate more compute (better approximation) selectively, (c) route to complementary methods. The problem disappears.

### 2.4 Proposed Approach
Three-component progressive gating design:

1. **Attribution Variance Decomposition** (COMPLETED): Two-way ANOVA decomposing J10/tau/LDS variance into class, gradient-norm, and residual components. Confirms the per-test-point phenomenon exists and is not a confound artifact.

2. **Bucketed Spectral Sensitivity (BSS)** (IN PROGRESS): For each test point, decompose attribution sensitivity by Hessian eigenvalue magnitude buckets (outlier/edge/bulk). BSS_bucket(z) = Σ_{λ∈bucket} (g_z^T q_λ)² · |1/λ_ekfac - 1/λ_kfac|². Key insight: eigenvalue magnitudes are stable across seeds (random matrix theory), so BSS rankings should be seed-stable even though TRV is not.

3. **Adaptive Method Selection** (PLANNED): Route between IF and RepSim based on BSS signals and/or cross-method disagreement. Evaluate on Pareto frontier (LDS vs compute).

Computational feasibility: GGN top-100 eigendecomposition via Kronecker factors is O(d_out² × d_in) per layer, well within 42 GPU-hour budget for ResNet-18 on CIFAR-10.

### 2.5 Core Assumptions

| # | Assumption | Type | Source | Strength | If False |
|---|------------|------|--------|----------|----------|
| A1 | Per-test-point attribution sensitivity varies beyond class/grad_norm | Empirical | Hessian hierarchy theory | **CONFIRMED** (Phase 1: 77.5% residual J10) | Diagnostic approach has no value |
| A2 | BSS outlier ranking is cross-seed stable (ρ > 0.5) | Empirical | Random matrix theory (eigenvalue distributions stable) | Medium (untested at scale) | BSS no better than original TRV |
| A3 | IF-RepSim disagreement is informative after class control | Empirical | Complementarity argument | **CONFIRMED** (Phase 2b: AUROC 0.691) | Routing has no value |
| A4 | Adaptive selection Pareto-dominates uniform | Empirical | Bayesian decision theory | Weak (untested) | Only diagnostic contribution, no fusion |
| A5 | SI is valid TRV proxy | Theoretical | Lipschitz bound | **FALSIFIED** (ρ ≈ 0 in probe) | Must compute BSS directly |

---

## 3. Validation Strategy

### 3.1 Idea Type Classification
**New diagnostic tool + new problem framing** (hybrid). The core contribution is identifying a new diagnostic dimension (per-test-point Hessian sensitivity) and proposing BSS as a theoretically grounded diagnostic. The adaptive selection is a secondary contribution leveraging the diagnostic.

Validation emphasis: (1) Does the phenomenon exist? (confirmed), (2) Is BSS a valid diagnostic? (testing), (3) Does adaptive selection improve outcomes? (planned).

### 3.2 Core Hypothesis
- **H1** (CONFIRMED): Per-test-point attribution sensitivity exists beyond class and gradient-norm confounds (residual > 30%).
- **H2** (TESTING): BSS rankings are cross-seed stable (Spearman ρ > 0.5) and non-degenerate (within-class variance > 25%, partial correlation with attribution metrics > 0.1 after gradient-norm control).
- **H3** (PLANNED): BSS-guided adaptive selection Pareto-dominates uniform strategies by > 2% absolute LDS.

### 3.3 Probe Experiment Design
- **Phase 0 Probe** (completed): 3 seeds × 100 points, last-layer IF. Confirmed Hessian impact (J@10 drops to 0.45-0.53), discovered SI-TRV null correlation, TRV cross-seed instability.
- **Phase 1** (completed): 500 points, full-model, 4 methods. ANOVA variance decomposition. All gate criteria passed.
- **Phase 2a** (in progress): BSS cross-seed stability with 5 model seeds.
- **Phase 2b** (completed): IF-RepSim disagreement analysis. AUROC 0.691.

### 3.4 Pass / Fail Criteria

| Result | Condition | Action |
|--------|-----------|--------|
| Pass | BSS cross-seed ρ > 0.5 AND adaptive > uniform by > 2% LDS | Full paper: diagnostic + fusion |
| Marginal | BSS stable but adaptive ≤ uniform | Diagnostic-only paper (Phase 1 + BSS) |
| Fail | BSS cross-seed unstable | Negative results paper; pivot to different diagnostic approach |

### 3.5 Time Budget & Resources
- Phase 2a (BSS stability): ~8 GPU-hours (5 seeds × eigendecomposition + BSS computation)
- Phase 3 (adaptive): ~6 GPU-hours (LOO validation, strategy comparison)
- Total remaining: ~14 GPU-hours (within 42-hour budget, ~28 hours used)

### 3.6 Failure Diagnosis Plan

| Failure Mode | Signature | Meaning | Action |
|--------------|-----------|---------|--------|
| BSS-gradient norm degeneracy | BSS_outlier ρ > 0.9 with gradient norm after regressing out | BSS is just gradient norm in disguise | Try partial BSS; if still degenerate, pivot to distributional diagnostic |
| BSS cross-seed instability | Spearman ρ < 0.5 across seeds | Eigenvalue bucket boundaries are seed-sensitive | Try adaptive bucket boundaries (percentile-based instead of fixed) |
| Adaptive ≤ uniform | LDS improvement < 2% | Diagnostic has value but fusion does not | Write diagnostic-only paper |
| All methods agree | J10 > 0.9 everywhere | Hessian sensitivity is negligible on this model/data | Scale to harder setting (larger model, fine-tuned LLM) |

---

## 4. Review

### 4.1 Review History

| Round | Date | Decision | Key Changes |
|-------|------|----------|-------------|
| 1 | 2026-03-16 | Go with focus | Direction refinement: decouple Phase 1-2, add W-TRAK baseline, downgrade LLM to GPT-2, H2 (error independence) highest risk |
| 2 | 2026-03-17 | Pass (probe) | Probe confirmed variance decomposition, falsified SI-TRV proxy. Pivoted from TRV to BSS approach. Gradient norm correlation flagged. |

### 4.2 Latest Assessment Summary
- **Contrarian**: BSS-gradient norm correlation (ρ=0.906) is the critical risk — if BSS is just gradient norm in disguise, the diagnostic adds nothing.
- **Comparativist**: No existing work on per-test-point Hessian sensitivity diagnosis. Gap is genuinely novel.
- **Pragmatist**: Phase 1 results are strong. BSS computation is feasible within budget. Progressive gating limits downside risk.
- **Interdisciplinary**: Random matrix theory provides solid theoretical grounding for eigenvalue stability claim, but needs empirical verification at scale.

### 4.3 Decision
- **Decision**: Pass (conditional on BSS cross-seed stability)
- **Rationale**: Phase 1 confirms the phenomenon exists (77.5% residual). BSS has sound theoretical grounding. Progressive gating limits risk.
- **Key Risks**: BSS-gradient norm correlation (ρ=0.906 in pilot); cross-seed BSS stability untested at 5-seed scale; adaptive fusion may not beat naive ensemble.
- **Unresolved Disputes**: Whether BSS adds information beyond gradient norm; whether adaptive selection provides sufficient improvement to justify the complexity.

### 4.4 Conditions for Next Module
- Priority: Resolve BSS-gradient norm degeneracy (compute partial BSS regressing out gradient norm)
- Most valuable hypothesis to test next: H2 (BSS cross-seed stability)
- Probe execution notes: Train remaining 4 ResNet-18 seeds before BSS computation; use same 500 test points for consistency

<!-- 完整辩论记录：Reviews/init/round-{N}/ -->
