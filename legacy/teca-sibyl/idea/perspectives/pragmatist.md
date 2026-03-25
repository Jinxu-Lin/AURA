# Pragmatist Perspective: TECA Research Proposals

**Agent**: sibyl-pragmatist
**Date**: 2026-03-17
**Topic**: Using model editing (ROME/MEMIT) parameter update directions as independent validation signals for TDA attribution directions; proposing the TECS (TDA-Editing Consistency Score) metric to probe knowledge geometry in parameter space.

---

## Executive Summary

The TECS idea is technically sound and computationally cheap -- both ROME's rank-1 delta and TDA gradients are deterministic, closed-form vectors in the same parameter space. The main engineering risk is **not the metric itself, but the reliability of the two signals being compared**: the pilot experiment already showed a 14% ROME success rate and gradient definition bugs. This pragmatist proposal focuses on three angles that (a) maximize reuse of existing code/data, (b) can each be validated in under 1 hour on a single 4090, and (c) produce interpretable results even if TECS shows weak signal.

**Critical constraint**: GPT-2-XL on 1x RTX 4090 (24GB). FP32 model ~6GB, leaving ~18GB for activations/gradients. Per-fact ROME edit + gradient computation: ~15-30 seconds. Budget: 200 facts total, ~1 hour per angle.

---

## Angle 1: Fix-First Minimal TECS (Improve Existing -- Highest Priority)

### Core Insight

The pilot experiment failed on engineering, not on science. Before exploring any fancy extension, the single most valuable thing is to **reproduce the basic TECS measurement with correct implementation** and see if there is any signal at all. This is not glamorous but it is the only rational first step.

### Concrete Plan

**P0-a: Zero-cost reanalysis** (~5 min)
- Load existing pilot results, filter to the 7/50 facts where ROME editing actually succeeded.
- Compute TECS only for these 7 facts. Even with N=7, a Cohen's d > 0.5 vs Null-A would be a strong hint.
- If the 7-fact TECS is indistinguishable from Null-A, the idea has a fundamental problem -- no amount of scale will fix it.

**P0-b: EasyEdit-based ROME** (~25 min)
- Replace the custom ROME implementation with EasyEdit's standard `ROMEHyperParams` + `apply_rome_to_model` for GPT-2-XL.
- EasyEdit's defaults are well-tested (CounterFact success rate >85% on GPT-2-XL in their benchmarks). This immediately addresses the 14% success rate issue.
- Key code: `from easyeditor import ROMEHyperParams, BaseEditor; editor = BaseEditor.from_hparams(hparams); metrics, edited_model, weights = editor.edit(...)`. Extract `weights` dict to get `delta_W` per layer.
- Validate on 50 CounterFact facts: target success rate >70%.

**P0-c: Gradient definition alignment** (~10 min)
- The pilot used inconsistent gradient definitions. Standardize to: `g_TDA(z_train, x_test) = nabla_theta L(x_test; theta) * nabla_theta L(z_train; theta)` for influence-function-style attribution, or simpler `g_attr = nabla_theta L(z_train; theta)` for direct gradient attribution.
- For TECS, we need the **direction** not the magnitude, so use `g_M = sum_{i in top-k} nabla_theta L(z_i; theta)` normalized to unit vector.
- Implementation: single backward pass per training sample, accumulate gradients for top-k BM25-retrieved samples, normalize.

**P0-d: Run corrected TECS on 100 facts** (~20 min)
- 100 CounterFact facts x {TECS at l*, Null-A, Null-B (l* +/- 5), Null-C}
- Primary metric: Cohen's d (TECS vs Null-A), with bootstrap 95% CI.
- Decision gate: d > 0.2 with CI not crossing 0 --> PROCEED. Otherwise --> prepare negative-result framing.

### Time Budget

| Step | Time | GPU? |
|------|------|------|
| P0-a: Reanalysis | 5 min | No |
| P0-b: EasyEdit setup + 50-fact validation | 25 min | Yes |
| P0-c: Gradient standardization | 10 min | No |
| P0-d: Full 100-fact TECS | 20 min | Yes |
| **Total** | **60 min** | |

### Success Probability: 55%

The 14% ROME success rate was almost certainly an implementation bug (EasyEdit reports >85%). Once that is fixed, there is a reasonable chance of seeing some directional alignment, because ROME's rank-1 update is literally computed from the factual association and TDA gradients also point toward that association's influence.

### Failure Modes

1. **EasyEdit incompatibility with GPT-2-XL**: EasyEdit primarily targets GPT-J/LLaMA. Mitigation: GPT-2-XL is explicitly listed in their supported models; if issues arise, use the original ROME codebase (`rome.baulab.info`) with r-ROME numerical fixes from arXiv 2403.07175.
2. **BM25 retrieval quality**: Only 6% of OpenWebText was indexed in the pilot. Mitigation: for P0, use the existing 500K index -- even partial coverage is sufficient to test the metric's behavior. Expand to 2M+ in P1.
3. **TECS is near zero even with correct implementation**: This is a valid scientific finding. Proceed to Angle 3 (negative-result framing).

---

## Angle 2: Layer-Sweep TECS Profile (Cross-Domain Transfer from Causal Tracing)

### Core Insight

Causal Tracing (Meng et al., 2022) produces a per-layer "indirect effect" profile showing which layers are critical for a fact. ROME edits at the peak layer l*. But Hase et al. (2023) showed this localization does not predict editing success. **Nobody has measured whether the DIRECTIONAL alignment between editing and attribution varies across layers** -- and this layer profile could be far more informative than either Causal Tracing or single-layer TECS alone.

The practical angle: instead of computing TECS at one layer, compute it at ALL MLP layers (GPT-2-XL has 48 layers) and plot the "TECS profile." This is cheap (delta_W and gradients can be extracted per-layer from a single forward+backward pass) and provides a rich diagnostic.

### Hypothesis

**H1**: The TECS profile will show a peak near l* (the ROME target layer) but will also reveal secondary peaks at other layers, suggesting knowledge is encoded in a distributed but structured way across layers.

**H2**: Facts where the TECS profile has a sharp single peak will have higher editing success than facts with flat profiles. This would provide a practical prediction tool: "only edit facts with peaked TECS profiles."

### Experimental Plan

| Step | Description | Time |
|------|-------------|------|
| 1 | Compute ROME delta_W at l* for 100 facts (reuse from Angle 1) | 0 min (cached) |
| 2 | For each fact, compute per-layer TDA gradient at all 48 layers | 30 min |
| 3 | Compute TECS at each layer, plot layer profile per fact | 5 min |
| 4 | Correlate TECS profile shape (peakedness, peak location) with editing success metrics | 10 min |
| 5 | Compare TECS profile vs Causal Tracing profile (Spearman correlation) | 10 min |

**Total**: ~55 min (with Angle 1 caching). **Success probability**: 45%.

### Engineering Details

- **Per-layer gradient extraction**: Use PyTorch hooks on each `transformer.h[i].mlp.c_proj.weight` to capture per-layer gradients during backward pass. Single backward pass gives all 48 layers simultaneously.
- **ROME delta_W is only at l***: For layer sweep, we need ROME's delta at multiple layers. Option A: run ROME targeting each layer (expensive). Option B (recommended): use MEMIT's multi-layer formulation which naturally distributes updates across layers, giving delta_W at each target layer.
- **Memory**: Per-layer gradient for one MLP weight matrix (1600 x 6400 for GPT-2-XL) = ~40MB FP32. 48 layers = ~2GB. Fits easily.

### Failure Modes

1. **ROME delta_W only exists at l***: Need MEMIT for multi-layer comparison. But MEMIT's update is a distributed least-squares solution, not a clean per-layer rank-1 -- the comparison geometry changes. Mitigation: use MEMIT but acknowledge the methodological difference; alternatively, run ROME at each layer independently (48 x 100 = 4800 edits, ~40 min on 4090).
2. **Layer-wise TECS profile is flat (no peaks)**: Possible if gradients lack layer specificity. Would still be a useful negative result: knowledge attribution signals are not layer-specific.

---

## Angle 3: TECS as Negative-Result Diagnostic (New Method -- Fallback)

### Core Insight

If TECS shows no signal (Angle 1 fails), the paper is not dead -- it just becomes a different paper. The key engineering insight: **a well-designed negative result about TECS reveals something important about the gap between "how knowledge was learned" (TDA) and "how knowledge can be edited" (ROME/MEMIT).**

Recent work supports this reframing:
- Hase et al. (2023, arXiv 2301.04213): Causal Tracing localization does not predict editing success.
- "The Mirage of Model Editing" (2025, arXiv 2502.11177): Editing evaluations are systematically overestimated.
- STEAM (2025, arXiv 2510.10398): Edited knowledge forms isolated residual streams, disconnected from naturally learned knowledge.

**If TECS ~ 0, it means editing and attribution operate in orthogonal parameter subspaces**, confirming STEAM's observation at the parameter level (not just representation level). This is publishable as "The Geometry of Knowledge Operations Is Incommensurable."

### Experimental Plan (adds to Angle 1)

| Step | Description | Time |
|------|-------------|------|
| 1 | Run Angle 1 (100 facts, corrected implementation) | 0 min (done) |
| 2 | If Cohen's d < 0.2: compute full pairwise cosine matrix between all 100 edit directions and all 100 attribution directions | 10 min |
| 3 | Compute principal angles between the edit subspace (span of 100 delta_W vectors) and attribution subspace (span of 100 gradient vectors) | 5 min |
| 4 | Repeat for MEMIT (multi-layer) to test if distributed editing changes the picture | 30 min |
| 5 | Visualization: subspace angle distribution, 2D UMAP of edit vs attribution directions | 10 min |

**Total**: ~55 min (conditional on Angle 1 showing no signal). **Success probability**: 70% (of producing a publishable negative result).

### Why This Angle is Practical

- Zero additional data collection needed -- same 100 facts, same gradients.
- Principal angle computation between subspaces is a well-established linear algebra technique (scipy.linalg.subspace_angles).
- Even if TECS is near zero, the *distribution* of pairwise cosines and the *principal angles* between subspaces are informative. If edit directions cluster tightly but attribution directions are diffuse (or vice versa), that tells us about the geometric structure of each operation.
- Feigenbaum et al. (2024, arXiv 2401.07526) showed that Gradient Tracing can localize editing targets differently from Causal Tracing. If TECS is low, comparing TECS computed at the Gradient-Tracing-identified layer vs the Causal-Tracing layer could reveal whether the localization mismatch extends to directional alignment.

### Failure Modes

1. **Both positive and negative results are ambiguous** (d ~ 0.15-0.25): Worst case -- not strong enough for either story. Mitigation: increase N to 200+ facts and add MEMIT comparisons to tighten confidence intervals.
2. **Reviewer dismisses negative result as "obvious"**: Mitigation: the combination of (a) principled subspace analysis, (b) comparison across ROME vs MEMIT, and (c) connection to the Hase et al. puzzle makes this non-trivial.

---

## Computational Budget Summary

| Angle | GPU Time | CPU Time | Total | Prerequisite |
|-------|----------|----------|-------|-------------|
| 1. Fix-First TECS | 45 min | 15 min | 60 min | None |
| 2. Layer-Sweep Profile | 30 min (shared) | 25 min | 55 min | Angle 1 |
| 3. Negative-Result Diagnostic | 30 min | 25 min | 55 min | Angle 1 (if fails) |
| **Combined (best case)** | **~50 min** | **~30 min** | **~80 min** | |
| **Combined (worst case, all 3)** | **~75 min** | **~40 min** | **~115 min** | |

All experiments fit within a single 4090 session. GPT-2-XL FP32 (~6GB) + peak gradient memory (~4GB per-layer) = ~10GB, leaving 14GB headroom.

---

## Recommended Priority (Pragmatist Ordering)

1. **Angle 1 (Fix-First) is mandatory**. No shortcuts. The pilot had critical implementation bugs. Until those are fixed and we see the corrected TECS values, all other analysis is premature. Specifically: start with P0-a (5 min, zero cost) -- if those 7 successful edits show TECS indistinguishable from random, the idea has a fundamental problem.

2. **If Angle 1 shows signal (d > 0.2)**: proceed to Angle 2 (Layer-Sweep). This is the highest-value extension because it produces a novel diagnostic artifact (the TECS profile) that can be compared to Causal Tracing profiles. It directly addresses the Hase et al. puzzle.

3. **If Angle 1 shows no signal (d < 0.2)**: pivot to Angle 3 (Negative-Result Diagnostic). Do NOT abandon the project. The subspace angle analysis is a clean, publishable contribution that reframes the finding constructively.

---

## Key Practical Recommendations

### On Model Choice
- **Stick with GPT-2-XL (1.5B)**. It is the right choice for this project:
  - ROME was originally validated on GPT-2-XL and GPT-J.
  - GPT-2-XL fits comfortably on 4090 with full gradient computation.
  - If reviewers complain about model size, run a validation subset (20 facts) on GPT-J-6B as a robustness check (~2 hours with gradient checkpointing).
  - Do NOT attempt GPT-J as the primary model -- gradient computation for 6B params will require aggressive checkpointing and slow everything down 4-5x.

### On ROME Implementation
- **Use EasyEdit** (https://github.com/zjunlp/EasyEdit) as the primary ROME/MEMIT implementation. It is the most battle-tested codebase (ACL 2024, 3000+ GitHub stars).
- **Extract delta_W**: After `editor.edit()`, the weight difference `edited_model.state_dict()[key] - original_model.state_dict()[key]` gives you delta_W directly.
- **Apply r-ROME fixes** (arXiv 2403.07175) if numerical instability occurs: add Tikhonov regularization to the covariance matrix inverse.

### On TDA Gradient Computation
- **Simplest viable approach**: For each fact, BM25-retrieve top-20 training documents from OpenWebText, compute per-document gradient `nabla_theta L(z_i; theta)` at the target MLP layer, aggregate as weighted sum (weights = BM25 scores), normalize to unit vector.
- **Memory optimization**: Compute gradients one document at a time, accumulate into running sum. Peak memory = model + 1 document's activations + 1 gradient tensor per layer.
- **Do NOT use TRAK or full influence functions** for the initial experiment. They are overkill for this directional comparison. Simple per-sample gradients are sufficient and far easier to debug.

### On Evaluation
- **Primary metric**: Cohen's d (TECS_real vs TECS_null) with bootstrap 95% CI. NOT p-values alone.
- **Secondary**: Spearman correlation between TECS and editing success metrics (efficacy, generalization, locality).
- **Visualization**: Histogram of TECS values overlaid with null distributions. Box plots comparing TECS across fact difficulty bins.
- **Statistical rigor**: Permutation test (1000 permutations) as a non-parametric alternative to t-test.

### On Failure Handling
- If EasyEdit fails on GPT-2-XL: fall back to the original ROME repo (rome.baulab.info) with r-ROME patches.
- If BM25 retrieval quality is too low (many facts have zero relevant documents): use a smaller, curated subset of CounterFact where the target facts are verifiably present in the OpenWebText corpus.
- If gradient computation OOMs: use gradient checkpointing (`torch.utils.checkpoint`) and reduce batch size to 1.

---

## Literature Discovered (Pragmatist-Specific)

| Paper | arXiv ID | Practical Relevance |
|-------|----------|-------------------|
| Editing Arbitrary Propositions in LLMs without Subject Labels | 2401.07526 | Gradient Tracing as alternative to Causal Tracing for ROME localization -- could be used to identify better editing layers for TECS comparison |
| Efficient Estimation of Kernel Surrogate Models for Task Attribution | 2602.03783 | Shows connection between linear surrogates and influence functions through second-order analysis; relevant if we need to upgrade from simple gradients to IF-style attribution |
| Model Unlearning via SAE Subspace Guided Projections (SSPU) | 2505.24428 | Uses SAE features to construct subspaces for targeted parameter updates; the subspace construction technique (QR decomposition) is directly applicable to our subspace angle analysis in Angle 3 |

---

## Risk Assessment (Engineering-Focused)

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| EasyEdit ROME gives <70% success rate on GPT-2-XL | 15% | High | Fall back to original ROME + r-ROME; verify CounterFact data loading is correct |
| BM25 corpus too small (6% coverage) yields garbage attribution | 40% | High | For P0, filter to facts where BM25 returns >5 relevant docs; expand corpus in P1 |
| Gradient computation OOMs on 4090 | 10% | Medium | Gradient checkpointing; compute per-layer instead of all layers simultaneously |
| TECS shows d ~ 0.1-0.2 (ambiguous zone) | 30% | Medium | Increase N to 200; add MEMIT; if still ambiguous, frame as "weak but detectable signal" with layer-sweep analysis |
| All TECS variants show d < 0.05 (no signal) | 25% | Low (if prepared) | Execute Angle 3 negative-result plan; this is a valid, publishable finding |
| Reviewer objects to GPT-2-XL as too small/old | 35% | Medium | Run 20-fact GPT-J validation; cite that ROME/MEMIT original papers used this scale |

**Overall probability of at least one publishable angle**: ~75%.

The key pragmatist insight: **the biggest risk is not that TECS fails, but that we waste time on extensions before fixing the implementation bugs from the pilot**. Fix first, measure, then decide.
