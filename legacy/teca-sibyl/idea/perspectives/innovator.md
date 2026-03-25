# Innovator Perspective: TECA Research Proposals

**Agent**: sibyl-innovator
**Date**: 2026-03-17
**Topic**: Using model editing (ROME/MEMIT) parameter update directions as independent validation signals for TDA attribution directions; proposing the TECS (TDA-Editing Consistency Score) metric to probe knowledge geometry in parameter space.

---

## Executive Summary

The TECS idea sits at an almost-virgin intersection of model editing and training data attribution. My literature search confirms **no prior work has systematically compared editing update directions with TDA gradient directions in parameter space** -- this is a genuine gap. However, the core TECS formulation (single-layer cosine similarity) is too narrow to yield a full paper. Below I propose three unconventional angles that amplify the core idea into a richer, more defensible contribution.

---

## Angle 1: Spectral TECS -- Knowledge Consistency in the Singular Value Basis

### Core Insight (Cross-Domain Transfer)

Zhou et al. (2025, arXiv 2508.16082) proved that **task vectors are approximately equal to first-epoch negative gradients** scaled by learning rate. Separately, Zhang et al. (2026, arXiv 2601.11042) showed that sequential knowledge editing collapses models by disrupting **dominant singular directions** of pretrained weight matrices. And CLM-Bench (Hu et al., 2026, arXiv 2601.17397) found that cross-lingual edit vectors are **nearly orthogonal** -- living in disjoint spectral subspaces.

**Proposal**: Instead of computing TECS as a raw cosine similarity cos(delta_W_E, g_TDA) in the full d_in x d_out weight space, **project both vectors onto the spectral basis** of the pretrained weight matrix W and measure alignment per singular value band:

```
TECS_spectral(k) = cos(U_k^T * vec(delta_W_E), U_k^T * vec(g_TDA))
```

where U_k is the subspace spanned by singular vectors with indices in band k (e.g., top-10, 10-50, 50-200, tail).

### Hypothesis

**H1**: Editing directions and TDA attribution directions will show HIGH alignment in mid-range singular bands (where factual knowledge is encoded) but LOW alignment in dominant bands (which encode general linguistic competence) and tail bands (noise). This "spectral selectivity" pattern would explain why naive TECS on full vectors might wash out the signal.

**H2**: The spectral band where alignment peaks will correlate with the band most disrupted by sequential editing collapse (Zhang et al., 2026), providing a unified geometric explanation for both editing fragility and attribution reliability.

### Why This is Novel

- SUIT (Park et al., 2025, arXiv 2509.24502) edits knowledge in subspaces but never connects to TDA.
- REVIVE (Zhang et al., 2026) preserves dominant singular subspace during editing but never measures TDA alignment.
- No prior work has decomposed the editing-attribution relationship by spectral band.

### Experimental Plan

| Step | Description | Model | Time |
|------|-------------|-------|------|
| 1 | Compute SVD of GPT-2-XL MLP weight matrices at layers 15-25 (cache once) | GPT-2-XL | 15 min |
| 2 | For 100 CounterFact facts: compute ROME delta_W and top-50 TDA gradient directions | GPT-2-XL | 30 min |
| 3 | Project both onto spectral bands [1-10], [11-50], [51-200], [201+], compute band-wise cosine | - | 5 min |
| 4 | Compare band-wise TECS against Null-A/B/C baselines | - | 10 min |

**Total**: ~60 min. **Success probability**: 55%. This is the riskiest angle but has the highest upside -- a positive result would be a genuinely new finding about knowledge geometry.

### Failure Modes

- SVD is expensive for large matrices (d_in x d_out can be 1600x6400 for GPT-2-XL MLP). Mitigation: use randomized SVD (top-200 components sufficient).
- Band-wise cosine might be noisy due to low dimensionality per band. Mitigation: aggregate across multiple facts before computing per-band statistics.
- The "mid-range band" hypothesis might be wrong -- knowledge could be distributed across all bands. This is a valid negative result: it would mean knowledge lacks clean spectral structure.

---

## Angle 2: TECS as a Causal Probe for the Localization-Editing Disconnect

### Core Insight (Reframing an Existing Puzzle)

Hase et al. (2023, arXiv 2301.04213) showed Causal Tracing localization does not predict editing success. The field has treated this as a negative result and moved on. But **nobody has asked: does the DIRECTION of the edit update (not just the LOCATION) carry information about whether the edit will succeed?**

**Proposal**: Use TECS not as a validation metric for TDA, but as a **causal probe** that predicts editing outcomes:

1. Compute TECS for each fact BEFORE executing the edit.
2. Test whether TECS predicts: (a) editing success/failure, (b) generalization score, (c) locality preservation, (d) robustness to paraphrase.
3. Compare TECS's predictive power against Causal Tracing's indirect effect (the standard localization signal).

### Hypothesis

**H3**: TECS will be a better predictor of editing success than Causal Tracing indirect effect, because TECS captures **directional alignment** in parameter space while Causal Tracing only captures **positional activation** in representation space. The Hase et al. disconnect arises because "where knowledge is activated" (Causal Tracing) and "how knowledge is geometrically encoded" (TECS) are fundamentally different questions.

**H4**: Facts where TECS is high (editing direction aligns with attribution direction) will show better locality preservation, because aligned edits are "following the natural knowledge geometry" rather than forcing an orthogonal update.

### Why This is Novel

- Hase et al. studied localization vs. editing success but never introduced a directional metric.
- No work has used TDA gradients as a predictive signal for editing quality.
- MetaKE (Liu & Wu, 2026, arXiv 2603.12677) aligns edit direction with the "feasible manifold" but discovers this manifold through meta-learning, not through TDA. TECS offers a much cheaper, closed-form alternative hypothesis about what the "feasible manifold" looks like.

### Experimental Plan

| Step | Description | Model | Time |
|------|-------------|-------|------|
| 1 | Compute TECS for 200 CounterFact facts (reuse from Angle 1) | GPT-2-XL | 0 min (cached) |
| 2 | Execute ROME edits, record success/generalization/locality/paraphrase metrics per fact | GPT-2-XL | 25 min |
| 3 | Compute Causal Tracing indirect effect for same 200 facts | GPT-2-XL | 20 min |
| 4 | Logistic regression: TECS vs CT-IE as predictors of edit success | - | 5 min |
| 5 | Spearman correlation: TECS vs each continuous editing metric | - | 5 min |

**Total**: ~55 min (with Angle 1 caching). **Success probability**: 40%. If the Hase et al. disconnect is truly about position-vs-direction, TECS should show predictive power. But the disconnect might be more fundamental (e.g., editing works by exploiting model artifacts rather than genuine knowledge structure).

### Failure Modes

- TECS might also show near-zero correlation with editing success, reproducing the Hase et al. finding in a new form. This would be a valuable negative result: it would mean the knowledge-operation disconnect extends from representation space to parameter space.
- Confound: both TECS and editing success might correlate with fact "difficulty" (frequency in pretraining data). Mitigation: control for BM25 retrieval score as a proxy for pretraining frequency.

---

## Angle 3: Multi-Operation Knowledge Fingerprinting via Direction Clustering

### Core Insight (New Method)

ROME, MEMIT, and TDA are three independent "probes" of knowledge in parameter space. Each produces a direction vector at each layer. **If knowledge has consistent geometric structure, these three probes should converge to similar directions; if not, their divergence patterns reveal heterogeneous knowledge organization.**

**Proposal**: For each fact, compute a "knowledge fingerprint" -- a set of direction vectors from multiple operations:

1. **d_ROME**: ROME rank-1 update direction at layer l
2. **d_MEMIT**: MEMIT distributed update direction at layer l (extracted from the multi-layer solution)
3. **d_TDA_top1**: Gradient direction of the single most influential training example
4. **d_TDA_agg**: Aggregated gradient direction of top-k influential examples
5. **d_KN**: Knowledge neuron activation pattern (from Integrated Gradients)

Then perform **clustering analysis** on these fingerprints across hundreds of facts to discover knowledge archetypes:

- **Type-A facts**: All 5 directions align (knowledge is "well-localized" -- a geometric sweet spot)
- **Type-B facts**: ROME/MEMIT align but TDA diverges (editing exploits a different mechanism than training attribution)
- **Type-C facts**: TDA aligns but ROME/MEMIT diverge (knowledge is trainable but not cleanly editable)
- **Type-D facts**: No alignment (knowledge is distributed/entangled)

### Hypothesis

**H5**: Knowledge archetypes will correlate with linguistic properties: simple subject-relation-object triples (e.g., "Paris is the capital of France") will be Type-A, while complex/contextual knowledge will be Type-D. This provides the first taxonomy of knowledge organization grounded in parameter-space geometry rather than linguistic categories.

**H6**: The archetype distribution will shift across model layers, with mid-layers showing more Type-A facts (consistent with Causal Tracing's observation that mid-layer MLPs are knowledge-dense).

### Connection to Recent Work

- Gupta et al. (2025, arXiv 2502.19416) showed that sequential editing shifts activations into "completely different regions in the representation space." Our fingerprinting would reveal whether these shifts correspond to transitions between knowledge archetypes.
- STEAM (Jeong et al., 2025, arXiv 2510.10398) found edited knowledge forms "isolated residual streams." Fingerprinting would test whether this isolation manifests as Type-B (editing-only alignment) in parameter space.
- Damirchi et al. (2025, arXiv 2512.22511) decomposed task vectors into shared/unique components. Our multi-operation fingerprint is the knowledge-level analog of this decomposition.

### Experimental Plan

| Step | Description | Model | Time |
|------|-------------|-------|------|
| 1 | Compute 5 direction vectors for 200 facts x 3 layers (l*-2, l*, l*+2) | GPT-2-XL | 45 min |
| 2 | Compute pairwise cosine similarity matrix (5x5) per fact | - | 5 min |
| 3 | K-means clustering on flattened similarity matrices (200 x 10-dim) | - | 2 min |
| 4 | Analyze cluster properties: fact type, linguistic complexity, layer distribution | - | 8 min |

**Total**: ~60 min. **Success probability**: 50%. The clustering might not yield clean archetypes -- knowledge organization could be a continuum rather than discrete types. But even a noisy clustering would provide the first multi-probe characterization of knowledge geometry.

### Failure Modes

- Knowledge Neuron computation via Integrated Gradients is slow. Mitigation: use gradient x activation (GxA) as a fast proxy, or drop d_KN entirely and work with 4 directions.
- MEMIT's multi-layer update makes extracting a per-layer direction non-trivial. Mitigation: use the per-layer residual from MEMIT's least-squares solution.
- 200 facts may be insufficient for stable clustering. Mitigation: use silhouette scores to validate cluster quality; if poor, report continuous similarity distributions instead.

---

## Computational Budget Summary

| Angle | GPU Time | CPU Time | Total |
|-------|----------|----------|-------|
| 1. Spectral TECS | 45 min | 15 min | 60 min |
| 2. Causal Probe | 45 min (shared) | 10 min | 55 min |
| 3. Fingerprinting | 45 min (shared) | 15 min | 60 min |
| **Combined (with caching)** | **~50 min** | **~25 min** | **~75 min** |

All experiments fit within a single 4090 GPU session. GPT-2-XL (1.5B, ~6GB FP32) leaves ample headroom for gradient computation.

---

## Recommended Priority

1. **Start with Angle 2** (Causal Probe): Lowest computational overhead if Angle 1 is also run (shared computation), and directly addresses the most famous open question in the field (Hase et al. disconnect). Even a negative result is publishable.

2. **Then Angle 1** (Spectral TECS): Provides the methodological core of the paper. If spectral band selectivity is observed, it becomes the headline result.

3. **Angle 3** (Fingerprinting): Most ambitious, best suited as a Section 5 "extended analysis" rather than the core contribution. Run only if Angles 1-2 show promising signals.

---

## Key Literature Discovered (Innovator-Specific)

These references were found through targeted cross-domain search and complement the existing literature survey:

| Paper | arXiv ID | Relevance |
|-------|----------|-----------|
| SUIT: Subspace-Aware Key-Value Mappings for Knowledge Editing | 2509.24502 | Identifies "edit-critical subspace" -- directly comparable to our spectral TECS bands |
| Spectral Characterization of Sequential Knowledge Editing Collapse (REVIVE) | 2601.11042 | Proves dominant singular directions encode general abilities; our spectral TECS tests whether mid-range bands encode factual knowledge |
| CLM-Bench: Cross-lingual Misalignment in Knowledge Editing | 2601.17397 | Shows edit vectors for different languages are orthogonal in spectral subspace -- analogous to our hypothesis that edit and TDA vectors occupy different spectral bands |
| On Task Vectors and Gradients | 2508.16082 | Proves task vectors ~ first-epoch gradients; theoretical bridge between editing deltas and TDA gradients |
| MetaKE: Meta-learning Aligned Knowledge Editing | 2603.12677 | Aligns edit direction with "feasible manifold" via meta-learning; TECS offers a cheaper, TDA-based hypothesis about this manifold |
| Decomposing Task Vectors for Refined Model Editing | 2512.22511 | Shared/unique decomposition of task vectors; our fingerprinting is the knowledge-level analog |
| Norm Growth and Stability in Localized Sequential Editing | 2502.19416 | Shows editing shifts activations to different representation regions; fingerprinting tests whether this has parameter-space correlates |
| BaFT: Basis-Level Representation Fine-Tuning for Knowledge Editing | 2503.00306 | Input-dependent subspace weighting for editing; our spectral TECS could inform which subspace bands matter |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| All TECS variants show no signal | 25% | Fatal | Pivot to negative result paper: "The Geometry of Knowledge Operations is Incommensurable" |
| ROME editing success rate too low (repeat of pilot failure) | 30% | High | Use EasyEdit standard implementation; fall back to r-ROME; filter to high-confidence edits |
| Spectral decomposition is noisy | 35% | Medium | Aggregate across facts before band-wise analysis; use confidence intervals |
| Reviewer objects that GPT-2-XL is too small/old | 40% | Medium | Run validation subset on GPT-J-6B; cite that ROME/MEMIT were designed for this scale |
| Idea already exists in unpublished work | 10% | Fatal | Literature search found no prior work; monitor arXiv weekly |

**Overall success probability for at least one publishable finding**: ~70%.
