# TECA Experiment Methodology

## Overview

We measure the TDA-Editing Consistency Score (TECS) — the cosine similarity between ROME's rank-one editing update and the aggregated TDA gradient — at MLP layers of GPT-2-XL. The study follows a gated, dual-outcome design: positive TECS triggers dose-response and spectral analysis; null TECS triggers subspace incommensurability characterization.

## Model and Dataset

- **Model:** GPT-2-XL (1.5B parameters, 48 layers, d_k = d_v = 1600)
- **Dataset:** CounterFact (Meng et al., 2022) — 200 facts for full experiments, 100 for pilots
- **Editing framework:** EasyEdit (pinned commit hash) for ROME/MEMIT
- **TDA retrieval:** BM25 over WikiText/Wikipedia to retrieve top-20 training documents per fact
- **Hardware:** Single NVIDIA RTX 4090 (24GB VRAM). GPT-2-XL in FP32 requires ~6GB; peak VRAM ~10GB with per-layer gradient computation.

## Phase 1: ROME Editing Validation (Setup)

**Objective:** Confirm EasyEdit ROME produces valid edits on GPT-2-XL.

**Protocol:**
1. Install EasyEdit, load GPT-2-XL
2. Apply ROME editing to N facts from CounterFact at layer l* = 17 (ROME's default for GPT-2-XL)
3. Measure editing metrics: efficacy (post-edit P(new_object) > P(old_object)), generalization (paraphrase success rate), locality (unrelated fact preservation)

**Gate:** Efficacy > 75%. If fails, debug EasyEdit configuration before proceeding.

**Outputs:** `exp/results/phase1_rome_validation.json` — per-fact editing metrics and delta_W vectors saved as tensors.

## Phase 2: TDA Gradient Validation

**Objective:** Confirm per-document training gradients at MLP layers carry fact-specific signal.

**Protocol:**
1. For each fact, BM25-retrieve top-20 documents from a Wikipedia subset
2. Compute per-document gradients ∇_W L(z_i; θ) at layer l* for each retrieved document
3. Sanity checks:
   - Gradient norm correlation with fact-presence score in document
   - Within-cluster (same-fact docs) vs between-cluster (different-fact docs) cosine similarity
4. Aggregate: g_M = Σ w_i · ∇_W L(z_i; θ) / ||...|| with BM25 weights

**Gate:** Within-cluster cosine > between-cluster cosine at p < 0.01 (permutation test, 1000 permutations).

**Outputs:** `exp/results/phase2_tda_validation.json` — gradient quality metrics; aggregated g_M tensors saved per fact.

## Phase 3: Core TECS Measurement

**Objective:** Measure TECS against five null baselines to determine whether editing-attribution alignment exists.

**TECS definition:** cos(vec(Δ_W_E), vec(g_M)) at layer l*

**Null baselines:**
- **Null-A (random fact):** TECS between fact_i's editing direction and fact_j's TDA gradient (j ≠ i)
- **Null-B (wrong layer):** TECS at layers l* ± 5 instead of l*
- **Null-C (failed edit):** TECS for facts where ROME editing failed (efficacy < 50%)
- **Null-D (shuffled gradient):** TECS with randomly permuted gradient vector components
- **Null-E (random direction):** TECS between editing direction and a random unit vector in R^{d_v × d_k}

**Primary metric:** Cohen's d (TECS_real vs Null-A) with 10,000 bootstrap 95% CI.

**Decision gate:**
- d > 0.2 → Positive path (Phase 4-5 + dose-response + layer sweep + spectral)
- d ≤ 0.2 → Negative path (Phase 4-5 + subspace geometry + whitening + MEMIT)

**Outputs:** `exp/results/phase3_tecs_core.json` — per-fact TECS values, null distribution statistics, Cohen's d with CI.

## Phase 4: Ablation Study

**Objective:** Verify TECS robustness across four ablation axes.

**Axes:**
1. **Top-k cutoff:** k ∈ {5, 10, 20, 50} documents in TDA aggregation
2. **Weighting scheme:** BM25 weights vs uniform weights vs TF-IDF weights
3. **Loss definition:** Cross-entropy on object token vs full-sequence loss vs margin loss
4. **Gradient scope:** Layer l* only vs layers [l*-2, l*+2] aggregated

**Robustness criterion:** TECS Cohen's d variation < 20% across ablation settings.

**Outputs:** `exp/results/phase4_ablation.json` — TECS statistics per ablation configuration.

## Phase 5: Theoretical Decomposition Test

**Objective:** Validate the rank-one decomposition theorem.

**Protocol:**
1. Compute decomposed TECS: key_align = cos(C^{-1} k*, k_i), value_align = cos(v* - Wk*, d_v_i)
2. Compare decomposed product (key_align × value_align) with full TECS via Spearman correlation
3. Compare empirical null distribution variance with theoretical prediction 1/d_k
4. Measure which component (key or value alignment) carries more signal

**Outputs:** `exp/results/phase5_decomposition.json` — correlation, component contributions, null distribution comparison.

## Positive Path Extensions

### Dose-Response Analysis (H3)
- Spearman correlation: TECS vs editing efficacy, controlling for pre-edit perplexity and relation type
- Logistic regression: P(editing success) ~ TECS + covariates, LOOCV AUROC
- Compare predictive power with Causal Tracing localization score

### Layer-Sweep Profile (H4, H9)
- TECS computed at all 48 layers for a subset of 50 facts
- Compare layer profile shape with Causal Tracing indirect effect profile
- Test whether peaked TECS profile predicts editing success

### Spectral TECS (H5)
- SVD of W at layer l*; project both Δ_W_E and g_M onto spectral bands: top-10, 10-50, 50-200, 200+ singular values
- Test whether alignment peaks in mid-range bands (indices 10-200)

## Negative Path Extensions

### Subspace Geometry (H7)
- Principal angle analysis: scipy.linalg.subspace_angles between S_E = span(Δ_W_1,...,Δ_W_N) and S_A = span(g_1,...,g_N)
- Compare against 100 random subspace samples of same dimension
- Characterize: structured (θ_min < random baseline at p < 0.01) vs random misalignment

### Whitening Decomposition (H6)
- Compute TECS_unwhitened: use raw k* instead of C^{-1} k* in the editing direction
- Compare TECS_whitened vs TECS_unwhitened via Cohen's d
- If unwhitened >> whitened: ROME's covariance-inverse rotation is primary source of geometric gap

### MEMIT Comparison
- Apply MEMIT (multi-layer editing) to same 200 facts
- Measure alignment between MEMIT update and TDA gradients across edited layers
- Test whether distributed editing shows different alignment patterns

## Baselines

| Baseline | Description | Purpose |
|----------|-------------|---------|
| Null-A | Random-fact TECS | Fact-specificity control |
| Null-B | Wrong-layer TECS (l* ± 5) | Layer-specificity control |
| Null-C | Failed-edit TECS | Edit-quality control |
| Null-D | Shuffled-gradient TECS | Gradient-structure control |
| Null-E | Random-direction TECS | Dimensional-concentration control |

## Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| Cohen's d (bootstrap) | Effect size: real TECS vs null | d > 0.2 for positive path |
| Spearman rho | Decomposition validity | rho > 0.7 (H2) |
| Partial Spearman rho | Dose-response | rho > 0.2 (H3) |
| LOOCV AUROC | Predictive utility | ΔAUROC > 0.05 (H3) |
| Principal angles | Subspace geometry | θ_min < random at p < 0.01 (H7) |

## Expected Visualizations

- **Table 1:** Main TECS results — mean TECS, Cohen's d, 95% CI for each null baseline comparison
- **Table 2:** Ablation robustness — TECS statistics across 4 ablation axes × settings
- **Figure 1:** Architecture/pipeline diagram — ROME editing direction vs TDA gradient direction in parameter space
- **Figure 2:** TECS distribution — histogram of real TECS vs Null-A overlay, with effect size annotation
- **Figure 3:** Layer profile — TECS across all 48 layers with Causal Tracing overlay (if positive path)
- **Figure 4:** Decomposition scatter — full TECS vs decomposed product, colored by key/value dominance
- **Figure 5 (positive):** Dose-response — TECS vs editing efficacy scatter with regression line
- **Figure 5 (negative):** Subspace geometry — principal angle distribution vs random baseline
- **Figure 6 (positive):** Spectral TECS — band-wise alignment bar chart
- **Figure 6 (negative):** Whitening decomposition — TECS_whitened vs TECS_unwhitened distributions

## Reproducibility

- All random seeds fixed at 42
- EasyEdit pinned to specific commit hash (recorded in experiment logs)
- All intermediate tensors (Δ_W, g_M, decomposed components) saved to `exp/tensors/`
- Bonferroni correction applied for all multi-comparison statistical tests
- Analysis notebooks saved to `exp/analysis/`

## Computational Budget

| Task Group | GPU Time | CPU Time | GPU Count |
|-----------|----------|----------|-----------|
| Phase 1 (ROME validation) | 15 min | 5 min | 1 |
| Phase 2 (TDA gradients) | 20 min | 5 min | 1 |
| Phase 3 (TECS measurement) | 15 min | 10 min | 1 |
| Phase 4 (Ablation) | 10 min | 5 min | 1 |
| Phase 5 (Decomposition) | 0 min | 10 min | 0 (CPU) |
| Positive extensions | 30 min | 25 min | 1 |
| Negative extensions | 20 min | 20 min | 1 |
| **Total (worst case)** | **~75 min** | **~55 min** | — |
