---
version: "1.0"
status: "assimilated"
decision: "Go with focus"
created: "2026-03-25"
last_modified: "2026-03-25"
---

# Project: AURA — Understanding TDA Reliability Through Geometric and Spectral Analysis

> [ASSIMILATED: generated from TECA (Sibyl), TECA_old (Noesis V1), and AURA (Sibyl) project materials]

## 1. Overview

### 1.1 Topic

Understanding when and why Training Data Attribution (TDA) methods fail, through two complementary lenses: (1) geometric incommensurability between knowledge editing and attribution directions in parameter space (TECS analysis), and (2) spectral sensitivity of Hessian approximation choices at the per-test-point level (BSS analysis).

### 1.2 Initial Idea

TDA methods face a silent reliability crisis: practitioners have no way to know which test points yield trustworthy attributions. This project attacks the problem from two angles that emerged from independent investigations:

**TECA thread (completed pilot):** We proposed TECS (TDA-Editing Consistency Score) to measure cosine similarity between ROME editing update directions and aggregated TDA gradient directions at MLP layers of GPT-2-XL. The hypothesis was that if editing and attribution operate on the same knowledge subspace, their directions should align. Pilot experiments on 100 CounterFact facts produced a definitive negative result: TECS ~ 0 (Cohen's d = 0.05 vs all null baselines), confirming that editing and attribution directions occupy geometrically incommensurable subspaces. Follow-up negative path analysis showed: (a) editing subspace is ~40D distributed, attribution subspace is ~1D collapsed; (b) whitening (C^{-1}) does NOT explain the gap; (c) MEMIT shows layer-specific patterns but the same fundamental incommensurability. This negative finding is itself a significant contribution — it provides parameter-space evidence for the localization-editing disconnect (Hase et al., 2023).

**AURA thread (active):** We proposed BSS (Bucketed Spectral Sensitivity) as a per-test-point diagnostic of TDA reliability. The original TRV (TDA Robustness Value) failed due to cross-seed instability (Spearman rho ~ 0). BSS resolves this by operating on eigenvalue-magnitude buckets rather than individual eigenvectors, which random matrix theory predicts to be seed-stable. Pilot experiments (CIFAR-10/ResNet-18, 500 test points) showed: (a) genuine per-test-point variation exists beyond class effects (residual R^2 = 77.5% for J10); (b) cross-method disagreement is predictive of attribution quality (partial corr = 0.27 after controlling for class + grad_norm); (c) ensemble averaging stabilizes rankings (LOO rho = 0.85).

### 1.3 Baseline Papers

| # | Paper | Link | Relevance |
|---|-------|------|-----------|
| 1 | ROME (Meng et al. 2022) | 2202.05262 | Rank-one editing; TECS editing-side basis |
| 2 | MEMIT (Meng et al. 2022) | 2210.07229 | Multi-layer editing; negative path MEMIT analysis |
| 3 | Hase et al. 2023 — Does Localization Inform Editing? | 2301.04213 | Localization-editing disconnect; TECS contextualizes this geometrically |
| 4 | TRAK (Park et al. 2023) | ICML 2023 | Scalable TDA; attribution-side basis |
| 5 | Hong et al. 2025 — Better Hessians Matter | 2509.23437 | Hessian hierarchy; BSS theoretical foundation |
| 6 | Li et al. 2025 — Natural W-TRAK | 2512.09103 | Natural geometry; SI/kappa diagnostics |
| 7 | Li et al. 2025 — IF failure on LLM | 2409.19998 | IF unreliability; motivates need for diagnostics |
| 8 | Kowal et al. 2026 — Concept Influence | 2602.14869 | IF-RepSim low correlation (0.37-0.45) |
| 9 | Park et al. 2023 — Spearman miss-relation | 2303.12922 | Evaluation crisis; motivates independent validation |
| 10 | MDA (Li et al. 2026) | 2601.21996 | Mechanistic data attribution; subspace IF |

### 1.4 Available Resources
- **GPU**: 4x RTX 4090 (remote server, SSH MCP)
- **Timeline / DDL**: NeurIPS 2026
- **Existing Assets**:
  - TECA complete pilot code + results (100 CounterFact facts, GPT-2-XL, all negative path experiments)
  - AURA complete pilot code + results (CIFAR-10/ResNet-18, 500 test points, variance decomposition + BSS + disagreement analysis)
  - All materials consolidated in `legacy/`

---

## 2. Problem & Approach

### 2.1 Baseline Analysis

#### What they solved
- ROME/MEMIT: Locate-then-edit paradigm for factual knowledge in transformer MLPs
- TRAK/IF: Scalable training data attribution via gradient methods
- Hong et al.: Established Hessian approximation hierarchy (H >= GGN >> EK-FAC >> K-FAC)
- Natural W-TRAK: Fisher-aware geometry for attribution robustness

#### What they didn't solve
- No tool exists to diagnose per-test-point attribution reliability
- No understanding of WHY editing and attribution directions are geometrically incommensurable
- No per-test-point routing between attribution methods based on reliability signals
- Cross-seed TRV instability (Spearman rho ~ 0) invalidates naive robustness diagnostics

#### Why they didn't solve it
- TDA and knowledge editing developed in isolation; no one measured their geometric relationship
- Existing diagnostics (SI, confidence) conflate spectral modes with opposing perturbation sensitivities
- Scalar summary statistics (TRV) lose spectral structure that explains cross-seed instability

### 2.2 Problem Definition
- **One sentence**: TDA methods produce test-point attributions that are highly sensitive to Hessian approximation choices, but practitioners lack any diagnostic to identify unreliable attributions or any principled way to mitigate them.
- **Reality evidence**: Jaccard@10 drops from 1.0 (full GGN) to 0.48 (K-FAC) — more than half of top-10 attributions change with approximation choice.
- **Importance**: TDA is increasingly used for data curation, debugging, and compliance; unreliable attributions undermine all downstream applications.
- **Value tier**: "Known problem, no adequate solution" — Hong et al. established the hierarchy but provided no per-point diagnostics.

### 2.3 Root Cause Analysis
- **Symptom**: TDA rankings change dramatically with Hessian approximation
- **Intermediate cause**: Approximation errors concentrate in specific spectral regions of the Hessian, and different test points have gradient energy distributed differently across these regions
- **Root cause**: The interaction between test-point-specific gradient spectral profile and approximation-specific error profile determines per-point sensitivity — but this interaction is invisible to scalar diagnostics

Additionally, from the TECA thread:
- **Symptom**: TECS ~ 0 (editing and attribution directions don't align)
- **Root cause**: Editing operates in a ~40D distributed subspace while attribution collapses to a ~1D subspace — these are geometrically incommensurable, and this is NOT explained by ROME's C^{-1} whitening rotation

### 2.4 Proposed Approach
We propose a two-part contribution:

**(1) TECS Negative Result (completed):** Formal characterization of geometric incommensurability between knowledge editing and attribution directions. This provides the first parameter-space evidence for the localization-editing disconnect, showing that editing and attribution access fundamentally different parameter subspaces (structured, not random misalignment).

**(2) BSS Diagnostic + Adaptive Selection (active):** Bucketed Spectral Sensitivity decomposes per-test-point attribution error by Hessian eigenvalue magnitude buckets. Unlike scalar TRV (cross-seed rho ~ 0), BSS operates on eigenvalue-magnitude distributions that are seed-stable per random matrix theory. BSS-guided adaptive method selection routes between IF and RepSim per test point.

### 2.5 Core Assumptions

| # | Assumption | Type | Source | Support | If false |
|---|-----------|------|--------|---------|----------|
| A1 | Genuine per-test-point attribution sensitivity exists beyond class effects | Empirical | AURA probe | Strong (residual R^2 = 77.5% for J10) | Per-point diagnostics are fundamentally limited |
| A2 | Eigenvalue-magnitude bucket distributions are seed-stable | Theoretical | RMT (Papyan 2020) | Medium (theoretical prediction, not yet empirically verified at scale) | BSS inherits TRV's cross-seed instability |
| A3 | IF-RepSim disagreement captures genuine attribution quality signal | Empirical | AURA Phase 2b | Strong (partial corr = 0.27, p = 1.6e-9 after controlling for class + grad_norm) | Routing has no real value |
| A4 | Editing and attribution operate in incommensurable subspaces | Empirical | TECA pilot | Strong (TECS ~ 0, Cohen's d = 0.05, structured misalignment confirmed) | TECS negative result would need qualification |
| A5 | Full-model Hessian preserves K-FAC/EK-FAC gap | Empirical | AURA pilot | Medium (last-layer setting collapsed bottom 3 levels; full-model untested) | BSS diagnostic loses resolution |

---

## 3. Validation Strategy

### 3.1 Idea Type Classification
Mixed: **empirical characterization** (TECS geometric incommensurability) + **new diagnostic method** (BSS) + **adaptive routing** (method selection).
Validation weight: 40% characterization rigor, 40% diagnostic utility, 20% routing effectiveness.

### 3.2 Core Hypothesis
1. BSS provides a seed-stable per-test-point diagnostic of attribution reliability (cross-seed Spearman rho > 0.5, vs TRV rho ~ 0)
2. BSS-guided routing between IF and RepSim Pareto-dominates uniform strategies at equal compute budget

### 3.3 Probe Experiment Design
**TECA probe (COMPLETED):** GPT-2-XL, 100 CounterFact facts, ROME editing + BM25-aggregated TDA gradients, 5 null baselines. Result: TECS ~ 0, all baselines non-significant.

**AURA probe (COMPLETED):** CIFAR-10/ResNet-18, 3 seeds, 100-500 test points. Results: variance decomposition PASS, cross-method disagreement PASS, adaptive routing PASS (pilot shows IF dominates, but full-model expected to create RepSim-better points).

### 3.4 Pass / Fail Criteria

| Result | Condition | Next |
|--------|-----------|------|
| Pass | BSS cross-seed rho > 0.5 AND residual > 30% in full experiment | Full paper with both TECS + BSS |
| Marginal | BSS rho 0.3-0.5 OR residual 20-30% | Paper with TECS negative result + BSS characterization (drop routing) |
| Fail | BSS rho < 0.3 AND residual < 20% | TECS negative result paper only |

### 3.5 Time Budget & Resources
- TECA experiments: COMPLETE (0 additional GPU time needed)
- AURA full experiments: ~20 GPU-hours on 4x RTX 4090
  - Phase 1 full variance decomposition: COMPLETE
  - Phase 2 BSS + cross-seed: ~10 GPU-hours (5 seeds x 200 epochs + eigendecomposition)
  - Phase 3 adaptive selection: ~5 GPU-hours
  - Phase 4 ablations: ~5 GPU-hours

### 3.6 Failure Diagnosis Plan

| Failure Mode | Signature | Meaning | Action |
|-------------|-----------|---------|--------|
| BSS cross-seed unstable | rho < 0.3 | Eigenvalue buckets not stable enough | Focus on TECS negative result paper |
| No per-point variation | Residual < 20% in full exp | Class effects dominate | Reframe as class-conditional diagnostic |
| IF always dominates | No RepSim-better points in full model | Routing has no value | Drop routing, focus on diagnostic |
| Full-model Hessian collapse | K-FAC = EK-FAC in full model | BSS loses resolution | Use different model/dataset |

---

## 4. Review

### 4.1 Review History

| Round | Date | Decision | Key Changes |
|-------|------|----------|-------------|
| TECA_old debate | 2026-03-16 | Go with focus | 6-perspective debate; all agreed pilot cost is low |
| TECA Sibyl pilot | 2026-03-17 | NEGATIVE → Negative Path | TECS ~ 0, proceed with subspace characterization |
| AURA Sibyl pilot | 2026-03-17 | GO (gated) | Variance decomposition PASS, BSS approach validated |
| Assimilation | 2026-03-25 | Consolidate under Noesis v3 | Unified TECA + AURA into single project |

### 4.2 Latest Assessment Summary
- **TECA completed**: Definitive negative result with scientific value. Structured incommensurability confirmed.
- **AURA promising**: Strong pilot signals for BSS approach. Full experiments needed.
- **Unification opportunity**: TECA negative result + AURA BSS positive result = comprehensive story about TDA reliability.

### 4.3 Decision
- **Decision**: Go with focus
- **Rationale**: Two complementary threads with strong pilot signals. TECA provides completed negative characterization; AURA provides active diagnostic development.
- **Key Risks**: BSS cross-seed stability is theoretical prediction, not yet empirically verified at scale. Full-model Hessian computation may be prohibitively expensive.
- **Unresolved Disputes**: Whether the TECA negative result alone is sufficient for a top venue, or whether BSS results are needed to strengthen the paper.

### 4.4 Conditions for Next Module
- Formalize the unified research direction combining TECA geometric incommensurability + AURA spectral sensitivity
- Priority: verify BSS cross-seed stability with full-model Hessian (gating experiment)
- Risk: if BSS fails, fall back to TECS-only negative result paper

<!-- Complete debate records: Reviews/init/round-{N}/ (from TECA_old: legacy/teca-noesis/debate/) -->
