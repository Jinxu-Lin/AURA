# TECA: Probing Knowledge Geometry Through the Lens of Editing-Attribution Directional Alignment in Parameter Space

## Title

**Decomposing the Geometry of Knowledge Operations: When Do Editing and Attribution Directions Align in Parameter Space?**

## Abstract

We investigate the geometric relationship between two independent probes of factual knowledge in transformer MLPs: rank-one model editing (ROME/MEMIT) and training data attribution (TDA) gradients. We propose the TDA-Editing Consistency Score (TECS), a cosine similarity metric between the editing update direction and the aggregated attribution gradient direction at the same MLP layer. Under the linear associative memory model that underlies ROME, we derive a rank-one decomposition theorem showing TECS factors into independent key-alignment and value-alignment terms, yielding closed-form predictions for the expected signal-to-noise ratio. We test TECS on GPT-2-XL across 200 CounterFact facts with five null baselines and four ablation axes, treating both positive signal (geometric consistency) and null signal (geometric incommensurability) as scientifically informative outcomes. When TECS is positive, we further test whether it predicts editing success via dose-response analysis, directly addressing the Hase et al. (2023) localization-editing disconnect. When TECS is near zero, we characterize the structured misalignment via principal angle analysis and a whitening decomposition that traces the geometric gap to ROME's covariance-inverse rotation. Either outcome advances our understanding of how knowledge is geometrically organized in parameter space and whether different knowledge operations access commensurable or incommensurable parameter subspaces.

## Motivation

Three research communities study knowledge in neural networks from different angles -- knowledge editing, training data attribution, and knowledge localization -- yet almost no work has connected them at the level of parameter-space geometry. ROME/MEMIT produce deterministic, closed-form update vectors (delta_W) that modify factual associations. TDA methods produce gradient vectors (g_TDA) that trace how training data shaped the same weight matrices. Both vectors live in exactly the same parameter space (d_v x d_k per MLP layer), making directional comparison not only possible but natural.

The core scientific question is deceptively simple: **do these two fundamentally different knowledge operations point in the same direction in parameter space?** If yes, it means knowledge has a consistent geometric structure that is discoverable by independent methods -- a strong claim about the nature of knowledge encoding. If no, it means editing and attribution operate in geometrically incommensurable subspaces, explaining the persistent localization-editing disconnect (Hase et al., 2023) and the "isolated residual streams" phenomenon (STEAM, Jeong et al., 2025) at the parameter level.

This question has never been asked systematically. The editing and TDA literatures have developed in isolation, each with its own evaluation paradigms. TECS bridges them by providing the first metric that simultaneously probes both.

## Research Questions

**RQ1 (Existence):** Is TECS at the ROME editing layer l* significantly different from chance, as measured against five null baselines (random-fact, wrong-layer, failed-edit, shuffled-gradient, random-direction)?

**RQ2 (Decomposition):** Does the theoretical rank-one decomposition (TECS = key-alignment x value-alignment) hold empirically? Which component carries more signal?

**RQ3 (Predictive power):** Does TECS predict editing success metrics (efficacy, generalization, locality) after controlling for fact difficulty and relation type? Does it succeed where Causal Tracing localization failed (Hase et al., 2023)?

**RQ4 (Layer profile):** How does TECS vary across layers? Does the layer profile reveal distributed or localized knowledge geometry?

**RQ5 (Incommensurability characterization):** If TECS is near zero, is the misalignment structured (editing and attribution directions occupy distinct but organized subspaces) or random? Does ROME's C^{-1} whitening explain the geometric gap?

## Core Methodology

### TECS Definition

For a factual association (subject s, relation r, object o) edited at layer l* of an MLP with weight matrix W in R^{d_v x d_k}:

```
TECS = cos(vec(delta_W_E), vec(g_M))
```

where delta_W_E is ROME's rank-one update at l*, and g_M = sum_{i in top-k} w_i * nabla_W L(z_i; theta) / ||...|| is the BM25-weighted, normalized aggregation of per-sample training gradients at l*.

### Theoretical Foundation (from Theoretical perspective)

Under the linear associative memory model (W*k = v), TECS admits a rank-one decomposition:

```
TECS(z_i) = sign(alpha) * cos(C^{-1} k*, k_i) * cos(v* - Wk*, d_v_i)
```

This decomposition:
- Predicts the null distribution: E[TECS_random^2] ~ 1/d_k (Proposition 2)
- Predicts the signal amplification for relevant training data: SNR ~ rho_k * rho_v * sqrt(d_k) (Proposition 3)
- For GPT-2-XL (d_k=1600), even weak correlations (rho_k * rho_v > 0.08) should produce Cohen's d > 0.3 with N=100 facts

### Experimental Protocol (from Empiricist perspective)

**Phase 1 -- ROME Validation** (15 min GPU): EasyEdit ROME on 200 CounterFact facts. Gate: efficacy > 75%.

**Phase 2 -- TDA Gradient Validation** (20 min GPU): Per-document gradients for top-20 BM25-retrieved documents. Sanity checks: gradient norm correlation with fact presence; within-vs-between cluster cosine similarity. Gate: within-cluster > between-cluster at p < 0.01.

**Phase 3 -- TECS Measurement** (15 min GPU): TECS for 200 facts against five null baselines (Null-A through Null-E). Primary metric: Cohen's d with bootstrap 95% CI. Decision gate: d > 0.2 proceeds to positive path; d < 0.2 proceeds to negative path.

**Phase 4 -- Ablation** (10 min GPU): Sensitivity to top-k cutoff, weighting scheme, loss definition, gradient scope. TECS is trustworthy only if robust across ablation axes (d variation < 20%).

**Phase 5 -- Theory Test** (5 min CPU): Decomposed TECS vs full TECS correlation; empirical null distribution vs theoretical 1/sqrt(d_k) prediction.

### Positive Path (if d > 0.2)

**Dose-Response Analysis** (from Empiricist): Spearman correlation between TECS and editing metrics, controlling for fact difficulty and relation type. Logistic regression for predictive utility with LOOCV.

**Layer-Sweep Profile** (from Pragmatist): TECS at all 48 layers; compare profile shape with Causal Tracing indirect effect profile. Test whether peaked profiles predict editing success.

**Spectral TECS** (from Innovator): Project both vectors onto spectral bands of W (top-10, 10-50, 50-200, tail singular values); test whether alignment peaks in mid-range bands where factual knowledge is hypothesized to reside.

### Negative Path (if d < 0.2)

**Subspace Geometry** (from Theoretical): Principal angle analysis between editing subspace S_E and attribution subspace S_A. Compare against random subspace baseline.

**Whitening Decomposition** (from Theoretical + Empiricist): Compare TECS_whitened (standard, with C^{-1}) vs TECS_unwhitened (raw, without C^{-1}). If unwhitened >> whitened, ROME's statistical whitening is the primary source of geometric incommensurability.

**MEMIT Comparison** (from Empiricist): Does the multi-layer distributed editing of MEMIT show different alignment patterns with TDA?

**Structured vs Random Misalignment** (from Empiricist + Contrarian): Intra-subspace clustering, cross-subspace projection residuals, principal angle distribution characterization.

## Hypotheses

**H1 (Core signal):** TECS at l* is significantly higher than Null-A (random fact) with Cohen's d > 0.3 for N=200 facts on GPT-2-XL.

**H2 (Decomposition validity):** The rank-one decomposition product cos(key-alignment) * cos(value-alignment) correlates with full TECS at Spearman rho > 0.7.

**H3 (Dose-response):** TECS shows significant positive correlation with ROME editing efficacy (Spearman rho > 0.2, p < 0.01) after controlling for covariates.

**H4 (Layer specificity):** TECS at l* is significantly higher than at l* +/- 5, confirming layer-specific alignment.

**H5 (Spectral selectivity):** TECS alignment peaks in mid-range singular value bands (indices 10-200) rather than dominant bands (top-10) or tail bands (200+).

**H6 (Whitening gap):** If TECS ~ 0, then TECS_unwhitened > TECS_whitened with d > 0.3, attributing the misalignment to ROME's C^{-1} rotation rather than fundamental knowledge geometry.

**H7 (Structured incommensurability):** If TECS ~ 0, the minimum principal angle between S_E and S_A is significantly smaller than the random subspace baseline, confirming structured (not random) misalignment.

## Expected Contributions

1. **First systematic comparison of editing and attribution directions in parameter space.** No prior work has measured the geometric relationship between ROME/MEMIT update vectors and TDA gradient vectors. This fills Research Gap 1 identified in the literature survey.

2. **Rank-one decomposition theorem for TECS.** The theoretical result that TECS factors into key-alignment and value-alignment components provides both interpretability and closed-form signal predictions, grounding the metric in the linear associative memory framework.

3. **Dual-outcome experimental design.** The paper is designed to produce a publishable contribution regardless of whether TECS is positive or near-zero. Positive TECS reveals geometric consistency; near-zero TECS with structured incommensurability characterization reveals the geometric basis for the localization-editing disconnect.

4. **Rigorous experimental methodology.** Five null baselines, four ablation axes, pre-registered decision gates, and dose-response analysis set a new standard for empirical rigor in knowledge geometry studies.

5. **Connection to the Hase et al. puzzle.** Whether TECS predicts editing success or not, the study provides new evidence on the relationship between knowledge localization, editing, and attribution -- the central open question in the field.

## Design Principles

### Addressing Contrarian Concerns

The contrarian perspective raised three fundamental challenges that this proposal explicitly addresses rather than ignores:

1. **"ROME directions may be artifacts, not knowledge geometry."** We do not assume TECS measures "knowledge geometry" -- we test it. The whitening decomposition (H6) specifically tests whether TECS reflects knowledge structure or ROME's statistical artifacts. The dose-response analysis (H3) tests whether TECS has functional relevance beyond geometric similarity.

2. **"TDA gradients are unreliable for LLMs."** Phase 2 includes explicit gradient sanity checks before computing TECS. If gradients lack fact-specificity, we report this as a finding about TDA limitations rather than proceeding with unreliable measurements.

3. **"Cosine similarity in 10M-dimensional space is near-vacuous."** We include Null-E (random direction baseline) specifically to calibrate against dimensional concentration. The theoretical framework provides closed-form predictions for the noise floor.

### Empiricist's Rigor Requirements

- Cohen's d with bootstrap CI as primary metric (not p-values alone)
- Bonferroni correction for multiple comparisons
- Pre-registered analysis plan before Phase 3
- All intermediate tensors saved for reproducibility
- EasyEdit exact commit hash reported

### Pragmatist's Engineering Priorities

- Fix implementation bugs before measuring (EasyEdit replaces custom ROME)
- GPT-2-XL as primary model (ROME's native scale)
- 1-hour per experiment constraint respected
- Each phase has explicit gate criteria

## Computational Budget

| Component | GPU Time | CPU Time | Total |
|-----------|----------|----------|-------|
| Phase 1-5 (Core TECS) | 50 min | 15 min | 65 min |
| Positive path (Dose-Response + Layer-Sweep + Spectral) | 30 min | 25 min | 55 min |
| Negative path (Subspace + Whitening + MEMIT) | 20 min | 20 min | 40 min |
| **Total (worst case, both paths)** | **~75 min** | **~40 min** | **~115 min** |
| **Typical path** | **~55 min** | **~25 min** | **~80 min** |

All experiments on GPT-2-XL (1.5B, ~6GB FP32) on a single RTX 4090 (24GB). Peak VRAM: ~10GB with per-layer gradient computation.

## Perspective Weighting and Synthesis Rationale

This proposal is not a compromise but a structured synthesis. Here is how each perspective contributed and why:

**Pragmatist (highest weight on execution):** The "fix first, measure second" principle is the foundation. Phases 1-2 are directly from the pragmatist. The gate-based decision flow ensures we never build on unreliable measurements.

**Empiricist (highest weight on methodology):** The five-null-baseline design, ablation axes, dose-response analysis, and statistical rigor requirements are all from the empiricist. This is what distinguishes a convincing paper from a speculative one.

**Theoretical (highest weight on interpretability):** The rank-one decomposition theorem transforms TECS from an opaque cosine similarity into a decomposable, predictable metric. The subspace geometry framework provides rigorous tools for the negative-result path.

**Contrarian (highest weight on intellectual honesty):** Every major concern is addressed by design, not dismissed. The whitening decomposition, gradient sanity checks, and dimensional concentration baseline are direct responses to contrarian challenges. The dual-outcome design means we cannot "fail to find what we wanted" -- both outcomes are scientifically valued.

**Innovator (selective integration):** Spectral TECS is included as an extension on the positive path because it generates the novel prediction (mid-range band selectivity) that no other perspective produces. The Multi-Operation Fingerprinting idea (Angle 3) is deferred as too ambitious for the first round.

**Interdisciplinary (framing, not core methodology):** The CLS (Complementary Learning Systems) analogy -- ROME as "hippocampal" (fast, sparse) encoding vs TDA as "neocortical" (slow, distributed) encoding -- provides a compelling narrative framework. The consolidation-TECS correlation (P4) is included as a secondary analysis. The spin-glass and immunological angles are noted as future directions but not central to the first paper.

## Risk Assessment

| Scenario | Probability | Paper Outcome |
|----------|------------|---------------|
| Strong TECS signal (d > 0.5) + dose-response | 25% | Strong positive: "TECS validates editing-attribution alignment" |
| Moderate TECS (0.2 < d < 0.5), no dose-response | 20% | Moderate positive: "Detectable but non-predictive geometric alignment" |
| TECS ~ 0 with structured misalignment | 30% | Strong negative: "Structured incommensurability of knowledge operations" |
| TECS ~ 0 with whitening explaining the gap | 15% | Mechanistic: "ROME's whitening rotates away from natural knowledge geometry" |
| Dead end (random misalignment, no structure) | 10% | Pivot required |

**Overall probability of at least one publishable outcome: ~85%.**
