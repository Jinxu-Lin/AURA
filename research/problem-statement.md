---
version: "3.0"
created: "2026-03-25"
last_modified: "2026-03-25"
entry_mode: "first"
iteration_major: 3
iteration_minor: 0
---

> **v3.0**: Realigned to geometric incommensurability direction. Consistent with method-design.md v3.0 and experiment-design.md v3.0. Supersedes v1.1 (FM1/FM2) and v1.2 (transitional).

# Problem Statement

## 1. Gap Definition

### 1.1 Gap Candidate List

| # | Candidate Gap | Derivation Path | Importance | Novelty | Feasibility |
|---|--------------|----------------|-----------|---------|-------------|
| G1 | Editing and attribution directions in parameter space are geometrically incommensurable — no prior work has characterized this geometry | TECA experiments (TECS ~ 0, subspace analysis) + Hase et al. (2023) localization-editing disconnect | **High**: Resolves fundamental disconnect between knowledge localization and editing | **High**: First geometric characterization of editing-attribution parameter-space relationship | **High**: TECA pilot data exists; need cross-model validation |
| G2 | Five representation-space TDA methods outperform parameter-space methods on LLMs but lack unified evaluation | CRA_old framework + Li et al. + DDA evidence | **High**: Practitioner guidance gap | **Medium-High**: Unified framework novel; individual observations known | **Medium**: Requires DATE-LM experiments |
| G3 | Per-test-point Hessian sensitivity varies dramatically but no reliable diagnostic exists | AURA variance decomposition + BSS instability | **Medium**: Practical diagnostic tool | **Medium**: BSS may degenerate to gradient norm | **Medium**: rho=0.906 is primary risk |

### 1.2 Selected Gap (G1: Geometric Incommensurability of Knowledge Operations)

**One sentence**: Model editing (ROME/MEMIT) and training data attribution (TDA) both operate on model parameters, yet their parameter update directions show zero alignment — and this non-alignment is structured, not random — but no work has characterized the geometric relationship between these knowledge operation subspaces.

**Detailed argument**:

Knowledge editing (ROME) identifies a rank-one parameter update ΔW = (v* - Wk*)(C⁻¹k*)ᵀ at a critical layer l* to inject a new fact. Training data attribution identifies gradient directions g_z = ∇_θ L(z) to attribute model behavior to training examples. Both operate on the same parameter space, yet our TECA pilot experiments on GPT-2-XL (100 CounterFact facts) reveal:

1. **TECS ≈ 0** (Cohen's d = 0.05 vs null): The cosine similarity between editing update direction and aggregated attribution gradient is indistinguishable from chance in a ~10M-dimensional space.

2. **Structured incommensurability**: The editing subspace has effective dimensionality ~40.8 (distributed across 100 facts), while the attribution subspace collapses to ~1.2 effective dimensions (first PC explains 91% of variance). These are not randomly misaligned — they have fundamentally incompatible geometric structure.

3. **Whitening does not explain the gap**: Removing or applying ROME's C⁻¹ whitening to attribution gradients does not recover alignment (H6 rejected, d = -0.198).

4. **MEMIT shows layer-specific patterns**: Multi-layer distributed editing produces different alignment characteristics but the same fundamental incommensurability.

This finding resolves a longstanding puzzle: Hase et al. (2023) showed that knowledge localization does not predict editing success. Our geometric analysis provides the parameter-level explanation — editing and attribution access the same weights but operate in geometrically incommensurable subspaces, meaning the "knowledge" they detect is structurally different.

**Evidence type**: "Done but with fundamental flaw" (both fields assume parameter-space operations access the same knowledge structure) + "Conditions changed" (over-parameterization at LLM scale creates subspace separation).

### 1.3 Root Cause Analysis

**Root Cause Type**: Structural geometric incompatibility arising from different optimization objectives in over-parameterized models.

**Layer 1 (surface symptom)**: Knowledge localization (TDA) does not predict editing success (Hase et al. 2023).

**Layer 2 (editing structure)**: ROME's rank-one update is constrained by C⁻¹ whitening and the editing objective min||v* - Wk*||, producing a distributed ~40D subspace across facts that reflects the constrained optimization geometry, not the natural knowledge encoding.

**Layer 3 (attribution structure)**: BM25-weighted aggregated gradients collapse into a ~1D subspace dominated by surface lexical features, reflecting the gradient flow geometry of pre-training loss, not fact-specific parameter structure.

**Layer 4 (fundamental)**: In over-parameterized models (p/n ~ 10²-10³), the parameter space is large enough for editing and attribution to occupy near-orthogonal subspaces. The C⁻¹ covariance rotation (condition number ~ 10²-10³) further separates these subspaces. This is not a bug but a structural consequence of over-parameterization.

### 1.4 Gap Three-Dimensional Assessment

| Dimension | Rating | Argument |
|-----------|--------|----------|
| **Importance** | **High** | Resolves the Hase et al. (2023) localization-editing disconnect at parameter level. Provides geometric foundation for understanding why different knowledge operations access different parameter subspaces. Implications for model editing reliability, TDA method design, and knowledge representation theory. |
| **Novelty** | **High** | No prior work compares editing and attribution parameter-space geometry. TECS metric is new. Subspace characterization (effective dim, principal angles, cross-projection) applied to knowledge operations is new. The "structured incommensurability" finding is surprising and counter-intuitive. |
| **Feasibility** | **High** | TECA pilot data (GPT-2-XL, 100 facts) already demonstrates core finding. Cross-model validation (Pythia-1B, GPT-J-6B, Pythia-6.9B) requires 37-52 GPU-hours. Toy model validation is CPU-only. All code exists from TECA pilot. |

## 2. Research Questions

### 2.1 Main RQ

**Is the geometric incommensurability between model editing and training data attribution directions a universal property of transformer language models, and what geometric structure characterizes this incommensurability?**

- *Falsification*: If TECS Cohen's d > 0.5 on ANY model with N ≥ 100 facts, the incommensurability claim fails.
- *Prediction*: TECS d < 0.2 across all models; editing eff-dim 30-60, attribution eff-dim 1-5; principal angles near-random at k ≥ 20.
- *Boundary*: Addresses autoregressive transformers with ROME/MEMIT editing on factual knowledge (CounterFact). Does NOT claim about other architectures, editing methods, or knowledge types.

### 2.2 Sub-RQs

**Sub-RQ1 (Universality)**: Does TECS ≈ 0 hold across model families and scales (GPT-2-XL 1.5B, GPT-J 6B, Pythia 1B/6.9B)?
- *Falsification*: TECS d > 0.3 on any model.
- *Prediction*: d < 0.2 on all four models with consistent subspace dimensionality patterns.

**Sub-RQ2 (Subspace Characterization)**: What are the geometric properties (effective dimensionality, spectral decay, principal angles) of editing vs attribution subspaces, and is their relationship structured or random?
- *Falsification*: Principal angles statistically indistinguishable from random subspace null on all metrics.
- *Prediction*: Editing 30-60D distributed, attribution 1-5D collapsed, with structured (non-random) angular relationships at low k.

**Sub-RQ3 (Mechanism)**: Does ROME's C⁻¹ whitening explain the geometric separation?
- *Falsification*: Removing C⁻¹ yields TECS d > 0.3 (whitening is the sole cause).
- *Prediction*: C⁻¹ contributes partially (d increase 0.1-0.3 when removed) but does not fully explain incommensurability — over-parameterization is the dominant factor.

**Sub-RQ4 (Attribution Robustness)**: Does the 1D attribution collapse persist with better aggregation methods (RIF-rescaled, SVD subspace projection)?
- *Falsification*: SVD subspace (r=10) attribution yields principal angles significantly below random null.
- *Prediction*: 1D collapse is partially an artifact of BM25; RIF increases eff-dim to 3-8 but incommensurability persists.

## 3. Attack Angle

### 3.1 Selected Attack Angle: Six-Component Geometric Analysis Framework

**Core idea**: We propose TECS (TDA-Editing Consistency Score) as a scalar metric for editing-attribution alignment, then build a six-component analysis framework that progressively characterizes WHY alignment is absent: (C1) TECS confirms scalar incommensurability across models, (C2) SVD reveals asymmetric subspace dimensionality, (C3) principal angles + cross-projection quantify structural relationship, (C4) C⁻¹ ablation isolates whitening mechanism, (C5) attribution ablation tests robustness to aggregation method, (C6) toy model with known ground truth validates the geometric framework.

**Root cause match**: Components directly map to root cause layers. C1 detects Layer 1 symptom across models. C2-C3 characterize Layers 2-3 structure. C4 tests Layer 2 mechanism. C5 tests Layer 3 robustness. C6 validates Layer 4 theory in controlled setting.

**Probe evidence support**: TECA pilot (GPT-2-XL, 100 facts) provides strong prior:
- TECS d = 0.05 (indistinguishable from null): Core signal confirmed
- Editing eff-dim = 40.8, attribution eff-dim = 1.2: Asymmetry confirmed
- H6 (whitening) rejected: C⁻¹ is not the sole explanation
- H7 (structured incommensurability) confirmed: Not random misalignment
- MEMIT cross-layer d ~ 0.63: Layer-specific patterns exist

### 3.2 Limitations and Risks

1. **"Negative result" perception**: TECS ≈ 0 may be perceived as "nothing works" rather than "structured finding." **Mitigation**: Frame as geometric characterization paper, not null result. The structured incommensurability IS the finding.

2. **BM25 attribution quality**: Attribution gradients based on BM25 retrieval may be weak. **Mitigation**: RIF and SVD subspace ablations (C5) test whether better attribution changes the story.

3. **ROME specificity**: Results may be ROME-specific, not generalizable. **Mitigation**: Include MEMIT comparison; note ROME is the most widely-used factual editing method.

4. **Parameter-space direction is not function-invariant**: Cosine similarity in parameter space depends on parameterization (Codex review critique). **Mitigation**: Acknowledge explicitly; toy model (C6) provides function-invariant ground truth; subspace analysis is more robust than scalar cosine.

5. **Limited to factual knowledge**: CounterFact covers factual associations only. **Mitigation**: State boundary clearly; note this is the standard benchmark for knowledge editing.

## 4. Probe Results Integration

### 4.1 Verified Hypotheses (from TECA GPT-2-XL pilot)

| Hypothesis | Evidence | Signal Strength |
|------------|----------|----------------|
| TECS ≈ 0 (no scalar alignment) | d = 0.05, 95% CI crosses zero, Bonferroni-corrected | **Strong** |
| Structured (not random) incommensurability | Editing eff-dim = 40.8, attribution eff-dim = 1.2 | **Strong** |
| C⁻¹ whitening does NOT explain the gap | H6 rejected, d = -0.198 | **Strong** |
| MEMIT shows layer-specific patterns | Cross-layer d ~ 0.63, within-layer matches ROME | **Moderate** |

### 4.2 Unverified Hypotheses

| Hypothesis | Why Unverified | Verification Plan |
|------------|---------------|-------------------|
| Universality across models | Only GPT-2-XL tested | Exp-1: 4 models |
| Phase transition in toy model | No toy model yet | Exp-5: synthetic |
| RIF improves attribution dimensionality | Not tested | Exp-4: attribution ablation |
| Layer profile correlates with causal tracing | Only l*=17 tested | Exp-6: 48-layer sweep |

### 4.3 Unexpected Findings

1. **Attribution 1D collapse**: Expected ~10D effective dimension; found 1.2D. First PC explains 91% of variance. Suggests BM25 retrieval introduces severe bias.

2. **Asymmetric cross-projection**: G-in-D = 17.3% but D-in-G = 1.0%. Attribution partially overlaps with editing subspace, but editing subspace is essentially invisible from attribution perspective.

3. **MEMIT matched-to-own-layer alignment**: When MEMIT deltas matched to their specific layers, alignment jumps to d = 4.8-6.7, suggesting layer-specific knowledge encoding.

## 5. Metadata

- **Based on**: TECA pilot experiments (GPT-2-XL, 100 CounterFact facts, Sibyl system), TECA_old 6-perspective debate (Noesis V1), AURA variance decomposition (CIFAR-10)
- **Legacy data**: `/Users/jlin8272/Research/AURA/legacy/teca-sibyl/` (full TECA results), `/Users/jlin8272/Research/AURA/legacy/teca-noesis/` (debate materials)
- **GPU resources**: 4x RTX 4090 (xuchang0); ~37-52 GPU-hours for full experiments
- **Excluded directions** (archived in iteration-log):
  - FM1/FM2 diagnostic framework (v1.1) — different research direction; requires DATE-LM experiments not yet done
  - BSS per-test-point diagnostic — gradient norm degeneracy rho = 0.906
