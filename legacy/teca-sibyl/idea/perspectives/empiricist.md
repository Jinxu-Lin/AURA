# Empiricist Perspective: TECA Research Proposals

**Agent**: sibyl-empiricist
**Date**: 2026-03-17
**Topic**: Using model editing (ROME/MEMIT) parameter update directions as independent validation signals for TDA attribution directions; proposing the TECS (TDA-Editing Consistency Score) metric to probe knowledge geometry in parameter space.

---

## Executive Summary

The TECS idea proposes measuring cosine similarity between two well-defined vectors in parameter space: ROME's rank-1 update direction and TDA's aggregated gradient direction. From an experimentalist's standpoint, **the biggest threat to this project is not whether the metric is interesting, but whether the measurements are trustworthy**. The pilot experiment already exposed severe confounders: 14% ROME success rate (likely a bug, not a finding), inconsistent gradient definitions, and minimal corpus coverage. Before any theoretical framing matters, we need clean measurements with proper controls.

This empiricist proposal designs three experiment-driven research angles, each organized around **what can actually be measured, what confounders lurk, and what result would falsify the hypothesis**. I prioritize ablation-first thinking: every positive claim about TECS must survive at least three null baselines and two ablation axes.

**Hardware constraint**: GPT-2-XL (1.5B, ~6GB FP32) on 1x RTX 4090 (24GB). This leaves ~18GB for activations and gradients -- sufficient for per-layer gradient computation (one MLP weight matrix at 1600 x 6400 = ~40MB FP32).

---

## Angle 1: Controlled TECS Measurement with Comprehensive Null Baselines

### Core Insight (Improve Existing -- Highest Priority)

The fundamental experimental question is: **does TECS at the ROME editing layer l* differ from chance?** This is a measurement problem, not a theory problem. The pilot failed to answer it because of implementation defects. Before measuring TECS, we must first establish that both signals being compared (ROME delta_W and TDA gradients) are individually valid.

### Confounders Identified

I identify **seven** specific confounders that could produce spurious TECS signal or mask a real one:

| # | Confounder | Mechanism | Severity |
|---|-----------|-----------|----------|
| C1 | Low ROME success rate | Only 7/50 edits succeeded; failed edits produce meaningless delta_W directions | **Critical** |
| C2 | Gradient definition mismatch | Pilot used inconsistent loss definitions between TDA gradient and ROME objective | **Critical** |
| C3 | BM25 corpus coverage | Only 6% of OpenWebText indexed; most facts may have zero relevant training documents | **High** |
| C4 | Layer selection confound | ROME always edits layer l* selected by Causal Tracing; if l* is wrong, TECS at l* is meaningless | **High** |
| C5 | Dimensional dominance | In d=10^7 dimensional space, cosine similarity concentrates around 0; even small biases (e.g., shared batch normalization statistics) can dominate | **Medium** |
| C6 | Gradient aggregation method | Choice of top-k, weighting scheme, and normalization all affect the TDA gradient direction | **Medium** |
| C7 | CounterFact selection bias | CounterFact facts are specifically chosen to be editable; results may not generalize to arbitrary factual knowledge | **Low** |

### Experimental Design: Five-Phase Protocol

**Phase 1: Validate ROME signal quality** (~15 min GPU)

Before measuring TECS, establish that ROME edits are actually working:
- Use EasyEdit's standard ROME implementation for GPT-2-XL with default hyperparameters.
- Edit 200 CounterFact facts (stratified sample: 100 common relations, 100 rare relations).
- Measure efficacy (target probability before/after), generalization (paraphrase accuracy), and locality (neighborhood accuracy).
- **Gate criterion**: ROME efficacy > 75% on this sample. If not, debug EasyEdit configuration before proceeding. The original ROME paper reports ~99% efficacy on CounterFact for GPT-2-XL; anything below 80% indicates a configuration problem, not a scientific finding.
- **Record**: For each fact, save the full delta_W tensor, the efficacy score, and the target layer l*.

**Phase 2: Validate TDA gradient signal quality** (~20 min GPU)

Before comparing TDA gradients with ROME, establish that the gradients are meaningful:
- For 200 facts, BM25-retrieve top-20 documents from OpenWebText (use existing 500K index).
- Compute per-document gradient g_i = nabla_{W_{l*}} L(z_i; theta) at the ROME target layer, where L is the standard next-token prediction loss on the document.
- Sanity check 1: **Gradient norm correlation**. For documents that explicitly contain the target fact (substring match), gradient norms at l* should be larger than for topically related but fact-absent documents. If not, the gradient computation is likely wrong.
- Sanity check 2: **Gradient direction stability**. For the same fact, compute pairwise cosine similarity between gradients of its top-5 retrieved documents. These should show higher mutual similarity than gradients from documents for different facts (within-cluster vs between-cluster). Use a permutation test (500 permutations) to assess significance.
- **Aggregate**: g_M = sum_{i in top-k} w_i * g_i / ||sum_{i in top-k} w_i * g_i||, where w_i = BM25(fact_i, doc_i).
- **Gate criterion**: Within-cluster cosine similarity > between-cluster cosine similarity at p < 0.01. If not, the TDA gradients lack fact-specific directionality and TECS cannot work regardless of the metric definition.

**Phase 3: Measure TECS with five null baselines** (~15 min GPU)

Compute TECS for 200 facts and compare against comprehensive null distributions:

| Baseline | Construction | What It Controls For |
|----------|-------------|---------------------|
| **Null-A** (random edit) | TECS between fact_i's TDA gradient and fact_j's ROME delta_W (j != i, random) | Fact-specificity of the alignment |
| **Null-B** (wrong layer) | TECS at layers l* +/- 5 using the same fact's TDA gradient and ROME delta_W projected to those layers | Layer-specificity of the alignment |
| **Null-C** (failed edit) | TECS for facts where ROME efficacy < 50% | Whether successful editing is necessary for alignment |
| **Null-D** (shuffled gradient) | TECS using randomly permuted dimensions of the TDA gradient vector | Whether alignment is in the gradient direction or just gradient magnitude structure |
| **Null-E** (random direction) | TECS between ROME delta_W and a random unit vector in the same space | Dimensional concentration baseline for d=10^7 |

**Primary metric**: Cohen's d (TECS_real vs each null) with bootstrap 95% CI (10,000 resamples).
**Secondary metric**: Area under the ROC curve (AUROC) for classifying real vs null TECS values.
**Tertiary**: Permutation test (1000 permutations) as non-parametric alternative.

**Phase 4: Ablation on gradient computation choices** (~10 min GPU)

The TDA gradient direction depends on several arbitrary choices. Test sensitivity to each:

| Ablation Axis | Variants | Expected Impact |
|---------------|----------|----------------|
| Top-k cutoff | k = 5, 10, 20, 50 | Higher k dilutes signal with irrelevant documents |
| Weighting scheme | Uniform, BM25 score, 1/rank | BM25 score should outperform uniform if retrieval quality matters |
| Loss definition | Document loss (full sequence), Fact-specific loss (only tokens after subject) | Fact-specific loss should give sharper gradient direction |
| Gradient scope | Full MLP weight matrix, Only c_proj, Only c_fc | ROME edits c_proj specifically; c_fc is a placebo |

Report TECS under each variant. If TECS is robust to these choices (Cohen's d varies < 20%), the metric is measuring a genuine signal. If TECS is fragile (d varies > 50%), the metric is an artifact of implementation choices.

**Phase 5: Effect size calibration** (~5 min CPU)

Even if TECS is statistically significant, is it scientifically meaningful?
- Compute the **explained variance** of TECS: what fraction of the total variance in cos(delta_W, g_TDA) is explained by fact identity vs noise?
- **Benchmarks**: Compare TECS effect size against known effect sizes in the knowledge editing literature:
  - Causal Tracing indirect effect: Cohen's d ~ 2-5 for critical layers vs non-critical layers (Meng et al., 2022)
  - Editing efficacy vs random baseline: d ~ 10+ (trivially large)
  - Hase et al. (2023) localization-editing correlation: r ~ 0 (null result)
- If TECS d < 0.2: the signal is too weak to be scientifically interesting regardless of p-value. Proceed to Angle 3.
- If TECS 0.2 < d < 0.5: moderate signal, potentially interesting with larger N and additional controls.
- If TECS d > 0.5: strong signal, proceed to Angle 2 for deeper analysis.

### Hypothesis and Falsification Criteria

**H1**: TECS at the ROME editing layer l* is significantly higher than all five null baselines (Cohen's d > 0.3, bootstrap 95% CI excluding 0).

**Falsification**: If TECS_real is not significantly higher than Null-A AND Null-B, the hypothesis is rejected. Specifically:
- TECS ~ Null-A means the alignment is not fact-specific (fatal for the TECS concept).
- TECS ~ Null-B means the alignment is not layer-specific (undermines the claim about knowledge localization geometry).
- TECS >> Null-A but TECS ~ Null-B means alignment is fact-specific but not layer-specific -- an interesting finding that suggests distributed knowledge encoding.
- TECS >> Null-A AND TECS >> Null-B means both fact-specific and layer-specific alignment -- the strongest positive result.

### Time Budget

| Phase | Time | GPU? |
|-------|------|------|
| Phase 1: ROME validation | 15 min | Yes |
| Phase 2: TDA gradient validation | 20 min | Yes |
| Phase 3: TECS + 5 null baselines | 15 min | Yes |
| Phase 4: Ablation (4 axes x 4 variants) | 10 min | Yes |
| Phase 5: Effect size calibration | 5 min | No |
| **Total** | **65 min** | |

### Success Probability: 50%

The controlled measurement could go either way. ROME's rank-1 update and TDA gradients both relate to factual knowledge in the same MLP weight space, so there is theoretical reason for alignment. But the high dimensionality (d ~ 10^7) means cosine similarity concentrates near 0, and a real signal must overcome this concentration. The pilot's 14% ROME success rate was almost certainly an implementation bug (EasyEdit reports >85% on CounterFact for GPT-2-XL), so we cannot use the pilot to predict the outcome.

### Failure Modes and Mitigations

1. **EasyEdit ROME gives < 75% efficacy on GPT-2-XL** (prob: 15%). Mitigation: verify CounterFact data loading format matches EasyEdit expectations; try the original ROME codebase with r-ROME patches (arXiv 2403.07175).
2. **TDA gradient sanity checks fail** (prob: 25%). This is the most likely failure point. Mitigation: (a) increase BM25 corpus to 2M+ documents, (b) filter to facts where at least 3 retrieved documents contain the target entity as substring, (c) try representation-similarity-based attribution (RepSim from arXiv 2409.19998) as alternative to gradient-based TDA.
3. **TECS is significant against Null-A/D/E but not Null-B** (prob: 30%). This means the alignment is real but not layer-specific. Reframe as: "knowledge geometry is consistent across layers, not localized" -- still publishable, contradicts localization hypothesis.
4. **All effect sizes < 0.2** (prob: 25%). Pivot to Angle 3 (negative result framing). This is a valid outcome.

---

## Angle 2: Dose-Response Analysis -- TECS vs Editing Success Metrics

### Core Insight (New Method)

If TECS measures genuine geometric alignment between editing and attribution directions, it should show a **dose-response relationship** with editing outcomes. Facts where TECS is high should be easier to edit (higher efficacy, better generalization, better locality preservation) and better attributed (higher TDA relevance score). This dose-response pattern is the strongest possible evidence for TECS's validity, because it cannot be explained by dimensional artifacts or implementation choices.

### Why Dose-Response is the Gold Standard

In experimental science, the dose-response relationship is what separates correlation from mechanism. Showing that "TECS is nonzero" is a weak claim. Showing that "higher TECS predicts better editing and better attribution, in a monotonic relationship with the right functional form" is a strong claim. This is analogous to the difference between "drug has nonzero effect" and "drug effect increases with dose."

The key confounders to control:

| Confounder | How It Could Produce Spurious Dose-Response | Control |
|-----------|---------------------------------------------|---------|
| Fact difficulty | Easy facts may have both high TECS and high editing success, not because TECS predicts editing, but because both are easy for simple facts | Stratify by fact difficulty (measured by pre-edit model perplexity on the target) |
| Relation type | Some relation types (e.g., "capital of") are both well-represented in training data (high TDA) and well-localized (high editing success) | Include relation type as a covariate in regression |
| Subject frequency | High-frequency subjects have better BM25 retrieval AND better ROME editing | Control for subject frequency in OpenWebText (measured by BM25 retrieval score) |

### Experimental Design

**Step 1: Compute TECS and editing metrics for 200 facts** (reuse from Angle 1, 0 min).

For each fact, record:
- TECS value (from Phase 3)
- ROME efficacy (probability of target after editing)
- ROME generalization (paraphrase accuracy, 5 paraphrases per fact from CounterFact)
- ROME locality (neighborhood accuracy, 5 neighbors per fact from CounterFact)
- Pre-edit perplexity of target (fact difficulty proxy)
- Relation type (from CounterFact metadata)
- BM25 retrieval score (subject frequency proxy)

**Step 2: Dose-response analysis** (~10 min CPU)

- **Primary**: Spearman rank correlation between TECS and each editing metric, with 95% CI from bootstrap.
- **Controlled**: Partial Spearman correlation, controlling for fact difficulty and relation type.
- **Visualization**: Scatter plots with LOWESS smoothing for each TECS-vs-metric pair. Look for monotonicity and saturation effects.
- **Threshold analysis**: Bin facts into TECS quartiles. Plot editing success rate per quartile. If there is a dose-response, Q4 (highest TECS) should show monotonically higher editing success than Q1 (lowest TECS).

**Step 3: Predictive utility test** (~5 min CPU)

Can TECS predict which facts will be successfully edited before editing?
- Train a logistic regression: P(editing success) ~ TECS + covariates (difficulty, relation, frequency).
- Evaluate with leave-one-out cross-validation.
- **Compare against**: (a) TECS alone, (b) covariates alone, (c) TECS + covariates.
- If TECS adds significant predictive power over covariates alone (likelihood ratio test p < 0.05), it has genuine practical utility.
- Report AUROC for each model.

**Step 4: Reverse dose-response -- TECS vs TDA quality** (~15 min GPU)

TECS should also predict TDA quality. For each fact:
- Compute a "TDA relevance score": fraction of top-10 retrieved documents that contain the target entity as substring (proxy for ground-truth relevance).
- Correlation between TECS and TDA relevance score, controlling for BM25 quality.
- If TECS is high when both editing and attribution are good, it validates the interpretation that TECS measures alignment between two independently valid knowledge probes.

### Hypothesis and Falsification Criteria

**H2**: TECS shows a significant positive dose-response relationship with editing efficacy (Spearman rho > 0.2, p < 0.01) after controlling for fact difficulty and relation type.

**H3**: TECS adds significant predictive power for editing success beyond covariates alone (likelihood ratio test p < 0.05, AUROC improvement > 0.05).

**Falsification**:
- If the Spearman correlation is positive but disappears when controlling for covariates, the dose-response is spurious (explained by shared confounders).
- If TECS predicts editing success but not TDA quality (or vice versa), the metric measures something specific to one operation, not a shared knowledge geometry.
- If the dose-response is non-monotonic (inverted-U or U-shaped), the interpretation of TECS as "alignment" is wrong.

### Time Budget

| Step | Time | GPU? |
|------|------|------|
| Step 1: Reuse Angle 1 data | 0 min | No |
| Step 2: Dose-response analysis | 10 min | No |
| Step 3: Predictive utility test | 5 min | No |
| Step 4: Reverse dose-response | 15 min | Yes |
| **Total** | **30 min** | |

### Success Probability: 40%

Dose-response is a high bar. Even if TECS is nonzero on average (Angle 1 succeeds), the per-fact variance may be too high for a clean dose-response. The main risk is that editing success is dominated by factors orthogonal to TECS (e.g., syntactic structure of the CounterFact prompt, tokenization artifacts). Hase et al. (2023, arXiv 2301.04213) already showed that causal tracing localization does not predict editing success; TECS may suffer the same fate. However, TECS measures directional alignment (a richer signal than scalar localization), so there is reason to believe it could succeed where localization failed.

---

## Angle 3: Controlled Negative-Result Protocol -- Geometric Incommensurability

### Core Insight (New Method -- Contingent on Angle 1 Failure)

If TECS shows no signal (Cohen's d < 0.2 in Angle 1), the scientific question shifts from "how aligned are editing and attribution?" to "how misaligned are they, and why?" A well-designed negative result must go beyond "TECS is near zero" to provide **mechanistic explanation** for the misalignment. This requires additional measurements that characterize the geometry of the misalignment.

The key experimental insight: near-zero mean TECS does not mean the editing and attribution directions are random with respect to each other. They could be systematically misaligned in a structured way (e.g., editing directions cluster in one parameter subspace, attribution directions cluster in a different subspace). Distinguishing "random misalignment" from "structured misalignment" is the core measurement challenge of a negative result.

### Experimental Design

**Step 1: Subspace construction** (~5 min CPU)

- Collect 200 ROME delta_W vectors (from Angle 1, already vectorized as d_v * d_k = 1600 * 6400 = 10.24M dimensions).
- Collect 200 TDA gradient vectors (same dimensionality).
- Compute SVD of each matrix (200 x 10.24M). Use randomized SVD (top-100 components) for efficiency.
- Define: S_E = top-r principal subspace of editing directions; S_A = top-r principal subspace of attribution directions; where r is chosen by the elbow criterion on singular value decay.

**Step 2: Principal angle analysis** (~5 min CPU)

- Compute principal angles between S_E and S_A using scipy.linalg.subspace_angles.
- **Key diagnostic**: If the minimum principal angle theta_min is close to 0, the subspaces overlap significantly (structured alignment). If theta_min is close to 90 degrees, the subspaces are nearly orthogonal (structured misalignment). If the principal angle distribution matches a random subspace baseline, the misalignment is unstructured.
- **Random baseline**: Generate two random r-dimensional subspaces in R^{10.24M} and compute their principal angles. Repeat 100 times for confidence interval. The expected minimum principal angle for random subspaces scales as ~ arcsin(sqrt(r/d)), which for r=20, d=10^7 is ~ 0.08 degrees. So any theta_min above this is informative.

**Step 3: Structured misalignment characterization** (~10 min CPU)

Three measurements to characterize the nature of the misalignment:

(a) **Intra-subspace clustering**: Within S_E, do editing directions cluster by relation type? Within S_A, do attribution directions cluster by relation type? If editing directions cluster but attribution directions do not (or vice versa), the two operations impose different organizational structures on knowledge.
- Measure: Silhouette score of relation-type clusters within each subspace.

(b) **Cross-subspace projection**: Project each editing direction onto S_A and measure the residual. If residuals are uniformly large, editing directions contain no information about the attribution subspace. If residuals are large for some facts but small for others, there is a fact-dependent alignment pattern.
- Measure: Distribution of ||proj_{S_A}(delta_W_i)|| / ||delta_W_i|| across 200 facts.

(c) **Whitening decomposition**: ROME's update uses C^{-1} (key covariance whitening). Compute TECS with and without whitening:
- TECS_whitened: Standard TECS (using ROME's actual delta_W with C^{-1}).
- TECS_unwhitened: TECS using delta_W_raw = (v* - Wk*) k*^T (removing the C^{-1} factor).
- If TECS_unwhitened >> TECS_whitened, the whitening operation is the primary source of misalignment. This is a concrete, mechanistic explanation: ROME deliberately rotates the editing direction away from the natural gradient direction to achieve better functional editing, at the cost of geometric consistency with training data attribution.
- Connect to STEAM (arXiv 2510.10398): if whitening-induced misalignment explains why edited knowledge forms "isolated residual streams."

**Step 4: MEMIT comparison** (~20 min GPU)

MEMIT distributes edits across multiple layers. Does multi-layer editing show different geometric relationships with TDA?
- Compute MEMIT delta_W for the same 200 facts (using EasyEdit).
- MEMIT produces delta_W at multiple layers; compute TECS at each layer.
- Compare MEMIT TECS profile vs ROME TECS (single layer).
- **Prediction**: If the misalignment is due to ROME's C^{-1} whitening, MEMIT (which uses a different optimization objective) may show different alignment patterns.

### Hypothesis and Falsification Criteria

**H4**: The editing-attribution misalignment is structured, not random. Specifically: the minimum principal angle between S_E and S_A is significantly smaller than the random subspace baseline (theta_min < theta_random at p < 0.01), but significantly larger than 0 (confirming partial but incomplete overlap).

**H5**: The whitening decomposition explains a significant fraction of the misalignment. Specifically: TECS_unwhitened > TECS_whitened with Cohen's d > 0.3.

**Falsification**:
- If principal angles match the random subspace baseline exactly, the misalignment is genuinely random and TECS has no scientific content (worst case).
- If TECS_unwhitened ~ TECS_whitened, the whitening is not the explanation and we need to look elsewhere.
- If MEMIT shows the same misalignment pattern as ROME (despite different optimization), the misalignment is a property of the knowledge geometry itself, not the editing algorithm.

### Time Budget

| Step | Time | GPU? |
|------|------|------|
| Step 1: Subspace construction | 5 min | No |
| Step 2: Principal angle analysis | 5 min | No |
| Step 3: Structured misalignment | 10 min | No |
| Step 4: MEMIT comparison | 20 min | Yes |
| **Total** | **40 min** | |

### Success Probability: 65% (of producing a publishable negative result)

Negative results are publishable if they are (a) rigorously measured, (b) provide mechanistic explanation, and (c) connect to known open problems. This design achieves all three: rigorous principal angle analysis with random baselines, whitening decomposition as mechanistic explanation, and connection to the Hase et al. (2023) localization-editing disconnect and STEAM's isolated residual stream finding.

---

## Overall Experimental Matrix

| Angle | Prerequisite | Decision Gate | If Pass | If Fail |
|-------|-------------|---------------|---------|---------|
| 1. Controlled TECS | None | Cohen's d > 0.2 for TECS vs Null-A | Proceed to Angle 2 | Proceed to Angle 3 |
| 2. Dose-Response | Angle 1 passes | Spearman rho > 0.2 after controlling for covariates | Strong positive paper | Weak positive: TECS is nonzero but not predictive |
| 3. Negative Result | Angle 1 fails | Structured misalignment (theta_min < random) | Publishable negative paper | Dead end (random misalignment) |

**Best case** (Angles 1+2 both pass, ~135 min total): "TECS: A Validated Metric for Measuring Knowledge Geometry Consistency Between Model Editing and Training Data Attribution."

**Middle case** (Angle 1 passes, Angle 2 fails, ~95 min total): "TECS Reveals Fact-Specific Parameter-Space Alignment Between Editing and Attribution, But Cannot Predict Editing Success."

**Worst acceptable case** (Angle 1 fails, Angle 3 shows structured misalignment, ~105 min total): "The Geometry of Knowledge Operations Is Structured But Incommensurable: Why Editing and Attribution Directions Diverge in Parameter Space."

**Dead end** (Angle 1 fails AND Angle 3 shows random misalignment): Pivot entirely. Probability estimate: ~10%.

---

## Computational Budget Summary

| Component | GPU Time | CPU Time | Total |
|-----------|----------|----------|-------|
| Angle 1 (Controlled TECS) | 50 min | 15 min | 65 min |
| Angle 2 (Dose-Response) | 15 min | 15 min | 30 min |
| Angle 3 (Negative Result) | 20 min | 20 min | 40 min |
| **Combined (worst case, all 3)** | **~65 min** | **~35 min** | **~100 min** |
| **Likely path (Angle 1 + one of 2/3)** | **~55 min** | **~25 min** | **~80 min** |

All experiments fit comfortably on a single RTX 4090 session. Peak VRAM: GPT-2-XL FP32 (~6GB) + per-layer gradient (~40MB) + activations (~4GB) = ~10GB, leaving 14GB headroom.

---

## Key Experimental Recommendations

### On Statistical Rigor

- **Never report p-values without effect sizes.** Cohen's d with bootstrap CI is the primary metric. p-values are secondary.
- **Always include power analysis.** Before running the 200-fact experiment, compute the minimum detectable effect size at 80% power with N=200. For a two-sample t-test: d_min ~ 0.28. If we need to detect smaller effects, increase N to 500 (feasible: ~2.5 hours total GPU time).
- **Multiple comparisons correction.** With 5 null baselines and 4 ablation axes, we have ~20 statistical tests. Apply Bonferroni correction (alpha = 0.05/20 = 0.0025) for the primary claims.
- **Pre-register the analysis plan.** Before running Phase 3, commit the exact statistical tests and decision thresholds to a timestamped file in the experiment directory. This prevents post-hoc rationalization of ambiguous results.

### On Confounder Control

- **The most dangerous confounder is BM25 retrieval quality.** If the retrieved documents are topically related but do not actually contain the target fact, the TDA gradient will point in a "topic direction" rather than a "fact direction." This would inflate TECS for common topics and deflate it for rare topics, creating a confound with fact difficulty.
  - **Mitigation**: For each fact, compute a "retrieval precision" score: fraction of top-10 documents that contain the target subject AND object as substrings. Report TECS separately for high-precision (>50%) and low-precision (<20%) retrieval sets. If TECS is only significant in the high-precision group, the signal is driven by retrieval quality, not knowledge geometry.

### On Reproducibility

- **Fix random seeds everywhere.** BM25 retrieval is deterministic given the index. ROME editing is deterministic given the model weights (no stochastic optimization). The only source of randomness is the permutation tests and bootstrap -- fix seeds for these.
- **Save all intermediate artifacts.** For each fact: ROME delta_W tensor, TDA gradient tensor, all editing metrics, all retrieval scores. This enables post-hoc analysis without re-running GPU-intensive computations.
- **Use EasyEdit's exact commit hash** for ROME/MEMIT implementation. Report it in the paper.

### On Model Choice

- **GPT-2-XL is the right choice for a methods paper.** ROME was originally validated on GPT-2-XL. Using the same model avoids confounds from implementation differences. If reviewers request larger models, run a 20-fact validation on GPT-J-6B with gradient checkpointing (~2 hours).
- **Do NOT use instruction-tuned models.** Instruction tuning changes the loss landscape and may affect both ROME editing behavior and TDA gradients. Use the base pretrained model only.

### On Reporting Negative Results

- **A negative result is only publishable if the measurement is trustworthy.** This requires:
  1. Demonstrated that ROME edits work correctly (Phase 1 gate passed).
  2. Demonstrated that TDA gradients carry fact-specific information (Phase 2 gate passed).
  3. Showed that TECS's null value is not due to implementation artifacts (Phase 4 ablation consistent).
- **Frame negative results constructively.** "Editing and attribution operate in geometrically incommensurable parameter subspaces" is more publishable than "our metric didn't work." The subspace analysis in Angle 3 provides this constructive framing.

---

## Literature Discovered (Empiricist-Specific)

These references were found through targeted searches on evaluation methodology, experimental confounders, and statistical best practices for knowledge editing and TDA:

| Paper | arXiv ID / Venue | Empirical Relevance |
|-------|-----------------|---------------------|
| The Mirage of Model Editing: Revisiting Evaluation in the Wild | 2502.11177 (ACL 2025) | Demonstrates that teacher-forcing inflates editing success from 38.5% to 96.8%; our evaluation must use autoregressive decoding only. Critical confounder to avoid. |
| Scalable Influence and Fact Tracing for LLM Pretraining | 2410.17413 | Finds misalignment between factual attribution and causal influence at scale; BM25 outperforms IF for explicit fact retrieval. Directly relevant to our BM25 retrieval quality confounder (C3). |
| How Well Can Knowledge Edit Methods Edit Perplexing Knowledge? | 2406.17253 | Shows negative correlation between "perplexingness" (conflict with learned hierarchies) and editing effectiveness. Fact difficulty is a critical covariate in our dose-response analysis (Angle 2). |
| STEAM: Semantic-Level Knowledge Editing | 2510.10398 | Edited knowledge forms isolated residual streams disconnected from pre-existing knowledge. Supports our H5 (whitening-induced misalignment) and Angle 3 negative-result framing. |
| Does Localization Inform Editing? | 2301.04213 (NeurIPS 2023) | Causal Tracing localization does not predict editing success (r ~ 0). Sets the precedent for our Angle 2: TECS may or may not succeed where localization failed, but we must test this explicitly. |
| Rebuilding ROME: Resolving Model Collapse (r-ROME) | 2403.07175 | Fixes numerical instability via Tikhonov regularization on C^{-1}. If ROME gives anomalous delta_W directions due to ill-conditioned C, our Angle 3 whitening analysis will detect it. |
| Benchmarking and Rethinking Knowledge Editing for LLMs | 2505.18690 | Current parameter-based editing methods perform poorly under realistic autoregressive conditions; context-based SCR outperforms across settings. Confirms that Angle 1 Phase 1 validation is essential. |
| Enhancing TDA for LLMs with Fitting Error Consideration (DDA) | 2410.01285 | Addresses fitting errors in influence functions via debias+denoise; AUC 91.64%. Relevant if our TDA gradient sanity checks (Phase 2) fail and we need a more robust attribution method. |
| CLM-Bench: Cross-lingual Misalignment | 2601.17397 | Edit vectors for different languages are nearly orthogonal, residing in disjoint subspaces. Validates our Angle 3 approach: subspace orthogonality analysis is a proven diagnostic technique for knowledge geometry. |

---

## Risk Assessment (Experiment-Focused)

| Risk | Probability | Impact | Mitigation | Detection |
|------|------------|--------|------------|-----------|
| ROME efficacy < 75% (implementation bug) | 15% | **Critical** -- invalidates all downstream measurements | Try original ROME + r-ROME; verify CounterFact loading | Phase 1 gate check |
| TDA gradients lack fact-specificity | 25% | **Critical** -- TECS cannot work without meaningful gradients | Expand corpus; filter high-precision retrieval; try RepSim | Phase 2 sanity checks |
| TECS shows d ~ 0.1-0.2 (ambiguous zone) | 30% | **High** -- cannot claim positive or negative result | Increase N to 500; tighten CI; report as "weak signal" with caveats | Phase 5 calibration |
| Dose-response confounded by fact difficulty | 35% | **Medium** -- Angle 2 inconclusive | Partial correlation controlling for difficulty; stratified analysis | Step 2 covariate analysis |
| BM25 retrieval quality too low for 500K corpus | 40% | **Medium** -- weak TDA signal | Expand to 2M+ docs; filter facts with >3 relevant docs | Retrieval precision metric |
| Subspace analysis trivially near 90 degrees | 20% | **Low** -- expected if TECS is near zero | Compare against random baseline; effective alignment dimension metric | Step 2 of Angle 3 |
| GPT-2-XL too small for reviewers | 35% | **Low** -- addressable in revision | 20-fact GPT-J validation; cite ROME's original use of GPT-2-XL | Reviewer response |

**Overall probability of publishable result**: ~75%.
- Positive paper (Angles 1+2): ~35%
- Negative paper (Angle 1 fails + Angle 3): ~40%
- Dead end (all angles fail): ~10%
- Ambiguous (inconclusive): ~15%

---

## Recommended Execution Order

1. **Phase 1 (ROME validation) is non-negotiable.** Do not compute TECS until you have confirmed >75% editing success. The pilot's 14% rate indicates a critical bug that must be resolved first.

2. **Phase 2 (TDA gradient validation) before Phase 3.** If gradients lack fact-specificity, TECS will be noise regardless. Fixing the gradient computation is more important than measuring TECS.

3. **Phases 3-5 depend on Phases 1-2 passing.** If either gate fails, stop and debug before proceeding.

4. **Angle 2 or Angle 3, not both initially.** After Angle 1, choose the appropriate path based on the decision gate. Only run the other angle if you have time remaining in the 1-hour budget.

5. **Pre-register before Phase 3.** Write down the exact statistical tests, decision thresholds, and interpretation rules before seeing the TECS values. This is the single most important step for credibility.

The empiricist's core principle: **measure first, theorize second, and never trust a measurement without proper controls.**
