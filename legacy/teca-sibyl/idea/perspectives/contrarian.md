# Contrarian Perspective: TECA Research Proposals

**Agent**: sibyl-contrarian
**Date**: 2026-03-17
**Topic**: Using model editing (ROME/MEMIT) parameter update directions as independent validation signals for TDA attribution directions; proposing the TECS (TDA-Editing Consistency Score) metric to probe knowledge geometry in parameter space.

---

## Executive Summary

The TECS proposal rests on at least three widely-held assumptions that, upon scrutiny, are each individually fragile and collectively may be fatal:

1. **That ROME/MEMIT editing directions reflect genuine "knowledge geometry."** Mounting evidence suggests they exploit model artifacts, not knowledge structure.
2. **That TDA gradient directions are reliable signals of knowledge provenance.** Influence functions demonstrably fail on LLMs; gradient directions are noisy and approximation-dependent.
3. **That cosine similarity in ~10M-dimensional parameter space is a meaningful metric.** High-dimensional concentration effects make this measure nearly vacuous without careful normalization.

If all three pillars are shaky, TECS is measuring the alignment between two unreliable signals using a degenerate metric. The other perspectives (innovator, pragmatist, theoretical) treat these as engineering problems to be patched. I argue they are **conceptual problems** that undermine the entire research premise. Below, I challenge each assumption with evidence and propose contrarian research directions that exploit these gaps.

---

## Challenged Assumption 1: ROME/MEMIT Editing Directions Encode "Knowledge Geometry"

### The Mainstream View

The TECS proposal assumes that ROME's rank-1 update delta_W_E represents a meaningful direction in parameter space -- one that captures how the model encodes a specific factual association. The innovator, pragmatist, and theoretical perspectives all take this for granted, treating delta_W_E as a geometric probe of knowledge.

### The Contrarian Evidence

**ROME does not edit knowledge; it edits input-output mappings via shortcut learning.** Niu et al. (2024, arXiv 2405.02421) demonstrated that the same ROME editing method can modify purely *linguistic* phenomena (not factual knowledge) with comparable success. Their conclusion is damning: "the MLP weights store complex patterns that are interpretable both syntactically and semantically, [but] these patterns do not constitute 'knowledge.'" If ROME can edit non-knowledge linguistic patterns with the same mechanism, the "direction" of delta_W_E is not a knowledge direction -- it is an *interference* direction that happens to change model outputs.

**Locate-then-edit methods suffer from shortcut learning.** Liu et al. (2025, arXiv 2506.04042) showed that ROME's optimization process systematically over-learns the subject feature while neglecting the relation feature. The editing "direction" is dominated by subject-identity shortcuts, not factual content. This means delta_W_E primarily encodes "how to recognize this subject" rather than "how this fact is geometrically stored."

**Editing directions are artifacts of the C^{-1} whitening, not knowledge structure.** The theoretical perspective's own Proposition 1 inadvertently proves this: TECS decomposes into cos(C^{-1}k*, k_i) * cos(value terms). The C^{-1} whitening is a statistical operation on the key covariance -- it rotates delta_W_E into a space defined by corpus-level statistics, not fact-level semantics. The "direction" of the edit is as much about the global key distribution as about the specific knowledge being edited. The theoretical perspective frames this as "whitening-induced incommensurability" -- I frame it as evidence that **the editing direction was never about knowledge in the first place**.

**Editing success does not require correct localization.** Hase et al. (2023, arXiv 2301.04213) showed localization and editing success are uncorrelated. Huang et al. (2025, arXiv 2502.20992) went further: individual knowledge cannot be localized at all -- what can be localized are *capabilities* (general computational patterns). If knowledge is not localized, then a rank-1 update at a single layer cannot capture its "direction." ROME works not because it follows knowledge geometry, but because it injects a sufficiently strong signal at a bottleneck layer.

**Sequential editing causes norm growth and representational collapse.** Gupta et al. (2025, arXiv 2502.19416) showed that successive edits invariably increase the Frobenius norm of updated matrices, shifting internal activations into "completely different regions in the representation space." If edits were following natural knowledge geometry, they should preserve the representational structure. The fact that they destroy it suggests the editing direction is *orthogonal* to the natural encoding manifold.

**The Mirage of Model Editing.** Evaluation benchmarks systematically overestimate editing success: 96.8% under teacher forcing drops to 38.5% in realistic evaluation (arXiv 2502.11177). If ROME's "successful" edits are largely illusory, the direction that produces them is the direction of an illusion.

### Implication for TECS

If delta_W_E does not represent knowledge geometry, then TECS = cos(delta_W_E, g_TDA) measures alignment between an *artifact direction* and an *attribution direction*. High TECS would mean TDA gradients happen to align with ROME's statistical artifacts; low TECS would mean nothing about knowledge geometry.

---

## Challenged Assumption 2: TDA Gradient Directions Are Reliable Knowledge Provenance Signals

### The Mainstream View

TECS treats the aggregated TDA gradient g_TDA = sum(nabla_W L(z_i)) as a meaningful direction that points toward "how the model learned this fact from training data." The other perspectives propose various ways to refine this signal (Fisher weighting, spectral decomposition) without questioning whether the raw signal is trustworthy.

### The Contrarian Evidence

**Influence functions demonstrably fail on LLMs.** The landmark study "Do Influence Functions Work on Large Language Models?" (arXiv 2409.19998) found a triple failure mode: (1) influence scores fail to identify actual training data for memorized facts, (2) approximation errors from iHVP estimation are overwhelming at LLM scale, and (3) representation similarity (RepSim) -- a far simpler method -- outperforms influence functions. If influence functions fail, the gradient directions they produce are noise, not signal.

**The fragility of influence functions is well-documented.** Basu et al. (2023, arXiv 2303.12922) revisited influence function reliability and found Spearman rank correlation between predicted and actual influence is near zero in many settings. The gradient direction is one component of influence estimation, and if the overall score is unreliable, the direction alone cannot be trusted.

**Layer choice for influence estimation is contested.** Vitel & Chhabra (2025, arXiv 2511.04715) showed that the conventional wisdom about which layers provide the best influence estimates is wrong: middle attention layers outperform embedding layers, contradicting prior work. More critically, they showed that the "cancellation effect" -- the theoretical basis for layer selection -- is unreliable. If we cannot even reliably choose *which layer's* gradients to use, per-layer TDA gradients are fundamentally ambiguous.

**BM25 retrieval as a proxy for training data is deeply flawed.** The TECS proposal relies on BM25-retrieved documents as training data surrogates. Chang et al. (2024, arXiv 2410.17413) showed that factual attribution (which documents contain a fact) and causal influence (which documents actually influenced the model) are fundamentally misaligned. BM25 retrieves factually relevant documents, not causally influential ones. The gradient of a BM25-retrieved document may point in a completely different direction from the gradient of the actual training document that taught the model this fact.

**Gradient directions in overparameterized models are not unique.** For an overparameterized model like GPT-2-XL, the loss landscape around the minimum is approximately flat in many directions. The gradient nabla_W L(z_i) at the trained model's parameters is nearly zero for well-fit training examples. The "direction" of a near-zero vector is dominated by numerical noise, not by meaningful knowledge structure. The proposal to "normalize to unit vector" does not fix this -- it amplifies noise into a direction.

### Implication for TECS

If g_TDA is unreliable, dominated by approximation artifacts, and computed on proxy (not actual) training data, then TECS compares an unreliable editing direction against an unreliable attribution direction. The probability of both errors aligning (or anti-aligning) in a meaningful pattern is low.

---

## Challenged Assumption 3: Cosine Similarity in ~10M-Dimensional Space Is Meaningful

### The Mainstream View

TECS = cos(vec(delta_W_E), vec(g_TDA)) is computed in R^{d_v * d_k} where d_v * d_k = 1600 * 6400 ~ 10^7 for GPT-2-XL. The other perspectives accept this metric uncritically, or propose refinements (Fisher-TECS, spectral TECS) that are still fundamentally cosine-based.

### The Contrarian Evidence

**High-dimensional concentration renders cosine similarity near-vacuous.** It is well-established that in high-dimensional spaces, the distribution of cosine similarity between random vectors concentrates tightly around zero with standard deviation ~ 1/sqrt(d). For d ~ 10^7, the noise floor of cosine similarity is ~ 1/3162. Any measured TECS above this trivially low threshold would appear "significant" even if both vectors are effectively random in the relevant subspace. The theoretical perspective's Proposition 2 acknowledges E[TECS^2] ~ 1/d_k = 1/1600, but this analysis assumes the rank-1 structure holds perfectly. In practice, both delta_W_E and g_TDA have effective rank > 1 (due to GELU nonlinearity, gradient noise, etc.), and the concentration happens in the full 10^7-dimensional space.

**Cosine similarity of learned representations can yield arbitrary results.** Steck et al. (2024, arXiv 2403.05440) showed that cosine similarity of embeddings is heavily dependent on regularization and can be "rendered even meaningless" depending on training conditions. The same concern applies to raw parameter-space gradients: their angular relationships depend on optimizer state, batch composition, and numerical precision.

**The metric does not distinguish signal from structure.** Both delta_W_E and g_TDA inherit structure from the weight matrix W they modify. If W has a dominant spectral structure (which it does -- see REVIVE, arXiv 2601.11042), both vectors will be biased toward the dominant singular directions of W. Any observed cosine similarity may reflect shared structural bias (both vectors are "colored" by the same W) rather than knowledge-level alignment. The null baseline Null-A (unrelated facts) partially controls for this, but only if unrelated facts have genuinely different structural biases -- which they may not if the dominant structure of W is fact-independent.

### Implication for TECS

The metric is either trivially easy to achieve statistical significance (due to concentration making any non-random signal look huge) or trivially easy to over-interpret (due to shared structural bias inflating cosine similarity). Either way, the raw number TECS = cos(...) has limited scientific value without extensive controls that the current proposal underspecifies.

---

## Contrarian Research Direction 1: The Anti-TECS Hypothesis -- Editing and Attribution Are Provably Orthogonal

### Proposal

Instead of hoping TECS is positive, **hypothesize that it is zero** and design experiments to prove this rigorously. The central claim: ROME editing and TDA attribution operate in fundamentally different subspaces of parameter space because they solve fundamentally different problems. Editing solves "what is the minimal perturbation that changes this output?" while attribution solves "which training data contributed to this output?" These are dual problems, not aligned ones.

### Evidence Supporting Orthogonality

1. **STEAM (arXiv 2510.10398)**: Edited knowledge forms "isolated residual streams" disconnected from naturally learned knowledge. If editing creates isolated structures, editing directions should be orthogonal to training gradient directions by construction.
2. **Hase et al. (2023)**: Localization does not predict editing success. If the "where" of knowledge is disconnected from "where editing works," the "direction" of knowledge encoding is likely disconnected from "direction of editing" too.
3. **Niu et al. (2024)**: ROME edits linguistic patterns, not just knowledge. Editing directions encode input-output mapping changes, which live in a different functional space from knowledge provenance.
4. **Gupta et al. (2025)**: Editing shifts activations into completely different representational regions. Orthogonality in representation space strongly suggests orthogonality in parameter space.

### Experimental Plan

| Step | Description | Model | Time |
|------|-------------|-------|------|
| 1 | Compute delta_W_E for 200 CounterFact facts via EasyEdit ROME | GPT-2-XL | 25 min |
| 2 | Compute g_TDA for same 200 facts (top-20 BM25 docs, aggregated gradient) | GPT-2-XL | 25 min |
| 3 | Compute principal angles between span(delta_W_E) and span(g_TDA) | - | 5 min |
| 4 | Test null hypothesis: principal angles ~ uniform on [0, pi/2] (consistent with random subspaces in R^{10M}) | - | 5 min |
| 5 | Compute TECS distribution, compare against random-vector null (not just Null-A) | - | 5 min |

**Total**: ~65 min. **Success probability**: 60%. If editing and attribution subspaces are indeed near-orthogonal, this is a clean negative result that explains the Hase et al. disconnect at the parameter-space level and warns the community against conflating "knowledge editing" with "knowledge geometry."

### Failure Modes

- Principal angles might show partial alignment (neither orthogonal nor aligned). This is the hardest outcome to interpret but would still constrain theories of knowledge geometry.
- If TECS is weakly but significantly positive, it could reflect shared structural bias (both vectors inheriting W's spectral structure) rather than knowledge-level alignment. The random-vector null (Step 5) specifically controls for this, unlike the Null-A baseline which only shuffles facts.

**Computational cost**: 65 min on 1x RTX 4090.
**Success probability**: 60%.
**Risk level**: Low -- orthogonality is well-defined and measurable.

---

## Contrarian Research Direction 2: TECS as a Confound Detector, Not a Validation Metric

### Proposal

Flip the TECS narrative entirely. Instead of using TECS to "validate TDA," use it to **detect when TDA is being confounded by the same artifacts that ROME exploits**. If TECS is high for a fact, it means the TDA gradient is pointing in the same direction as a known-artifact editing direction -- which is a *red flag* for TDA reliability, not a validation.

### Core Insight

ROME's delta_W_E encodes subject-identity shortcuts (Liu et al., 2025, arXiv 2506.04042) and statistical artifacts of the key covariance C. If g_TDA aligns with delta_W_E, it may mean the TDA method is attributing influence based on surface-level subject co-occurrence rather than genuine causal contribution. High TECS would identify facts where TDA is unreliable (confounded by the same shortcuts ROME uses), while low TECS would identify facts where TDA is capturing something genuinely different from ROME's artifact-driven editing.

### Experimental Plan

| Step | Description | Model | Time |
|------|-------------|-------|------|
| 1 | Compute TECS for 200 facts (reuse from Direction 1) | GPT-2-XL | 0 min (cached) |
| 2 | For high-TECS facts: manually inspect BM25-retrieved documents for surface-level subject overlap vs. genuine factual content | - | 20 min |
| 3 | Correlate TECS with BM25 score of top-1 document (proxy for surface overlap) | - | 5 min |
| 4 | Correlate TECS with actual causal influence (leave-one-out retraining on 10 selected facts) | GPT-2-XL | 30 min |
| 5 | Test: does high TECS predict *inflated* TDA scores (TDA overestimates influence for these facts)? | - | 5 min |

**Total**: ~60 min. **Success probability**: 50%.

### Why This Matters

If confirmed, this reframes the entire TECS narrative: instead of "TECS validates TDA," the story becomes "TECS detects TDA confounds." This is arguably a more useful contribution -- the TDA community needs confound detectors far more than it needs validation from another unreliable method (editing).

### Failure Modes

- TECS might show no correlation with BM25 surface overlap, undermining the confound hypothesis.
- Leave-one-out retraining on 10 facts may be too few for statistical power. Mitigation: use counterfactual data augmentation instead of actual retraining.

**Computational cost**: 60 min on 1x RTX 4090.
**Success probability**: 50%.
**Risk level**: Medium -- requires causal ground truth that is expensive to obtain.

---

## Contrarian Research Direction 3: The "Knowledge Has No Direction" Thesis

### Proposal

The deepest contrarian position: **knowledge in neural networks does not have a well-defined direction in parameter space.** The TECS framework, and all its extensions (spectral, Fisher, subspace), presuppose that knowledge can be characterized by a direction vector. Challenge this assumption directly.

### Evidence for Directionlessness

1. **Superposition hypothesis (Elhage et al., 2022)**: Features in neural networks are stored in superposition -- more features than dimensions. If knowledge is stored in superposition, it does not correspond to a single direction but to a pattern across many directions. The "direction" extracted by ROME or TDA is a projection of this superposition onto a rank-1 structure, losing most of the information.

2. **Distributed knowledge encoding**: Geva et al. (2023, arXiv 2304.14767) showed factual recall involves a three-step mechanism: enrichment (early MLPs encode many subject attributes), relation propagation (attention), and extraction (attention heads query the enriched subject). Knowledge recall uses *both* MLPs and attention across multiple layers and positions. A single rank-1 direction at one layer cannot capture this distributed process.

3. **Fact recall vs. heuristic use**: Saynova et al. (2024, arXiv 2410.14405) showed that LMs often answer factual queries using heuristics (e.g., Swedish-sounding name -> born in Sweden) rather than genuine fact recall. Causal tracing yields *different* patterns for heuristic-based vs. fact-recall-based answers. If the model's "knowledge direction" depends on *how* it answers (fact vs. heuristic), there is no stable knowledge direction.

4. **Multi-hop failure**: He et al. (2026, arXiv 2601.04600) showed ROME fails on multi-hop reasoning because edited knowledge at one layer cannot propagate through the multi-step recall process. This implies that single-layer directions cannot capture knowledge that requires multi-step computation.

5. **Capability > knowledge localization**: Huang et al. (2025, arXiv 2502.20992) showed that what can be localized in LLMs are *capabilities* (general computational patterns with 96.42% neuron overlap), not individual facts. If individual knowledge cannot be localized, it cannot have a direction.

### Experimental Plan

| Step | Description | Model | Time |
|------|-------------|-------|------|
| 1 | For 100 facts, compute delta_W_E at 5 different layers (l*-4, l*-2, l*, l*+2, l*+4) | GPT-2-XL | 30 min |
| 2 | For same facts, compute g_TDA at same 5 layers | GPT-2-XL | 25 min |
| 3 | Measure consistency: does the "knowledge direction" (either edit or TDA) change across layers? Compute pairwise cosine between layers for each fact | - | 5 min |
| 4 | Measure consistency: does the "knowledge direction" change across retrieval sets? Compute g_TDA with different BM25 top-k (k=5, 10, 20, 50) | GPT-2-XL | 15 min |
| 5 | Test: if knowledge has a stable direction, cross-layer and cross-retrieval consistency should be high (cosine > 0.5). If it does not, consistency ~ 0 | - | 5 min |

**Total**: ~80 min. **Success probability**: 55%.

### Predicted Outcome

I predict that the "knowledge direction" will show near-zero consistency across layers and across retrieval sets. This would demonstrate that the very premise of TECS -- that knowledge has a measurable direction in parameter space -- is flawed. The directions extracted by ROME and TDA are artifacts of the specific computational graph at a specific layer, not intrinsic properties of the knowledge.

### Failure Modes

- Knowledge might show moderate consistency (cosine 0.2-0.5) across nearby layers, suggesting partial directionality. This would weaken but not destroy the thesis.
- GPT-2-XL might be too small for superposition effects to dominate. Mitigation: if consistency is high on GPT-2-XL, test on Qwen-0.5B (smaller, more superposition) to see if it decreases.

**Computational cost**: 80 min on 1x RTX 4090.
**Success probability**: 55%.
**Risk level**: Medium-high -- a strong positive result (high consistency) would invalidate this direction entirely.

---

## Summary: Three Uncomfortable Questions TECS Cannot Answer

| # | Question | Why It Matters | What Would Settle It |
|---|----------|---------------|---------------------|
| 1 | Is ROME's delta_W_E a knowledge direction or an artifact direction? | If artifact, TECS is meaningless regardless of its value | Edit linguistic non-knowledge with ROME, measure TECS for non-knowledge edits (should be equally high if artifact-driven) |
| 2 | Is g_TDA reliable enough for directional comparison? | If unreliable, TECS inherits unreliability | Compare TECS using different TDA methods (IF, TRAK, RepSim, TracIn); if TECS varies wildly, the signal is method-dependent, not knowledge-dependent |
| 3 | Does knowledge have a stable direction in parameter space at all? | If not, the entire TECS framework is premised on a false assumption | Measure cross-layer and cross-method directional consistency |

---

## Computational Budget Summary

| Direction | GPU Time | CPU Time | Total |
|-----------|----------|----------|-------|
| 1. Anti-TECS (Orthogonality) | 50 min | 15 min | 65 min |
| 2. Confound Detector | 30 min (shared) | 30 min | 60 min |
| 3. No-Direction Thesis | 70 min | 10 min | 80 min |
| **Combined (with caching)** | **~75 min** | **~35 min** | **~110 min** |

All experiments on GPT-2-XL, single RTX 4090.

---

## Recommended Priority (Contrarian Ordering)

1. **Start with Direction 3** (No-Direction Thesis): This is the most fundamental challenge. If knowledge directions are inconsistent across layers and retrieval sets, it invalidates not just TECS but the entire class of directional knowledge probes. Run this first as a "meta-experiment" that determines whether the whole TECS enterprise is worth pursuing.

2. **Then Direction 1** (Anti-TECS Orthogonality): If Direction 3 shows some directional consistency, measure whether editing and attribution directions are aligned or orthogonal. This directly produces the TECS measurement but with the explicit hypothesis that TECS ~ 0.

3. **Direction 2** (Confound Detector) only if TECS is unexpectedly positive: If TECS > 0, the confound hypothesis provides an alternative explanation that does not require assuming knowledge geometry exists.

---

## Key Literature Discovered (Contrarian-Specific)

| Paper | arXiv ID | Contrarian Relevance |
|-------|----------|---------------------|
| What does the Knowledge Neuron Thesis Have to do with Knowledge? | 2405.02421 | ROME edits linguistic patterns, not knowledge; undermines the premise that delta_W_E is a "knowledge direction" |
| Unveiling Shortcut Learning for Locate-Then-Edit Knowledge Editing | 2506.04042 | ROME's optimization over-learns subject features, neglects relation features; editing direction is a shortcut, not knowledge geometry |
| On the Limitations of Rank-One Model Editing in Multi-hop Questions | 2601.04600 | ROME fails on multi-hop reasoning: single-layer directions cannot capture distributed knowledge |
| Capability Localization: Capabilities Can be Localized rather than Individual Knowledge | 2502.20992 | Individual knowledge cannot be localized; only general capabilities can. Fatal for the premise that knowledge has a direction |
| Do Influence Functions Work on Large Language Models? | 2409.19998 | IF triple failure on LLMs; gradient directions from IF are unreliable |
| First is Not Really Better Than Last: Layer Choice for Influence Estimation | 2511.04715 | Layer choice for TDA is contested; gradient directions are layer-dependent artifacts, not stable knowledge signals |
| Norm Growth and Stability in Localized Sequential Editing | 2502.19416 | Editing shifts activations into completely different regions; editing directions are orthogonal to natural representation structure |
| Fact Recall, Heuristics or Pure Guesswork? (PrISM) | 2410.14405 | LMs use different mechanisms (fact recall vs. heuristics) for same queries; "knowledge direction" depends on recall mechanism, not knowledge itself |
| Dissecting Recall of Factual Associations | 2304.14767 | Factual recall is a three-step distributed process across MLPs and attention; cannot be captured by a single-layer direction |
| Can Knowledge Editing Really Correct Hallucinations? | 2410.16251 | Existing benchmarks do not verify LLMs actually hallucinate before editing; editing "success" may be measuring nothing meaningful |
| Is Cosine-Similarity of Embeddings Really About Similarity? | 2403.05440 | Cosine similarity can yield arbitrary results depending on regularization; questions the fundamental metric of TECS |

---

## Risk Assessment (Contrarian-Specific)

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| TECS is actually strongly positive (d > 0.5), falsifying all three contrarian directions | 15% | Fatal for contrarian narrative | Pivot to "shared artifact" explanation; investigate whether both signals inherit W's spectral structure |
| Knowledge directions show high cross-layer consistency, undermining Direction 3 | 25% | High | Report as positive finding; narrow contrarian claim to "editing directions are artifacts" only |
| Reviewers dismiss contrarian perspective as "merely negative" | 35% | Medium | Frame as constructive: "we identify necessary preconditions for directional knowledge probes and show they are not met" |
| Orthogonality result is trivial (expected in 10M-dimensional space) | 20% | Medium | Use effective dimensionality and random subspace baselines; report principal angles relative to random, not absolute |
| The pragmatist's "fix first" approach finds strong TECS signal with corrected implementation | 30% | High | Contrarian value shifts to "explaining why TECS is positive" (shared artifacts) rather than "TECS is zero" |

**Overall probability of at least one contrarian finding surviving peer review**: ~65%.

The key contrarian insight: **the other three perspectives are optimizing a metric before asking whether the metric measures anything real.** Before computing TECS, we should first establish that (a) editing directions are about knowledge, (b) TDA directions are reliable, and (c) cosine similarity in parameter space is meaningful. None of these preconditions has been verified. The most valuable contribution of this project may not be TECS itself, but the systematic stress-testing of these preconditions -- regardless of whether TECS turns out to be positive, negative, or zero.
