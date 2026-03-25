---
version: "3.0"
created: "2026-03-25"
last_modified: "2026-03-25"
entry_mode: "first"
iteration_major: 3
iteration_minor: 0
---

> **v3.0 design**: Complete rewrite for geometric incommensurability direction. Six experiments (Exp-0 through Exp-6) corresponding to six method components (C1-C6).

# Experiment Design

## 1. 探針 → 完整実験衔接

**Probe**: TECA pilot on GPT-2-XL (100 CounterFact facts, ROME editing, BM25-weighted TDA gradients, 5 null baselines).

**Extensions**:

| Dimension | Probe | Full | Justification |
|-----------|-------|------|---------------|
| Models | GPT-2-XL | + GPT-J-6B, Pythia-1B, Pythia-6.9B | Universality (Sub-RQ1) |
| Facts | 100 | 200-300 | Statistical power + diversity |
| Editing | ROME only | + MEMIT (full covariance) | Editing method generality |
| Attribution | BM25-weighted | + Raw mean, RIF, SVD subspace | Attribution robustness (C5) |
| Whitening | Not ablated | Whitened vs unwhitened | Mechanism (Sub-RQ3) |
| Layers | L17 only | Full 48-layer sweep | Layer profile (Sub-RQ4) |
| Theory | None | Toy model | Ground-truth validation |

## 2. Pilot 快速验証（Experiment 0）

### 2.1 验証目標

Validate two assumptions before investing full budget:

1. **Cross-model**: TECS ~ 0 on Pythia-1B (cheapest additional model)
2. **C^{-1} effect**: TECS_unwhitened differs from TECS_whitened on existing GPT-2-XL data

### 2.2 実験方案

**Exp-0a**: TECS on Pythia-1B with 50 CounterFact facts. ROME at Pythia editing layer. BM25 TDA gradients. Null-A comparison only.

**Exp-0b**: On GPT-2-XL, compute delta_W_unwhitened = C * delta_W for 100 facts (C from precomputed ROME stats). Compare TECS_unwhitened vs TECS_whitened.

### 2.3 Pass/Adjust/Fail

| Exp-0a (Pythia-1B) | Exp-0b (Whitening) | Action |
|----|---|---|
| TECS d < 0.3 (pass) | d(unwhitened vs whitened) > 0.2 (detectable) | Full proceed |
| 0.3 < d < 0.5 (marginal) | Any | Investigate; proceed cautiously |
| d > 0.5 (alignment) | Any | GPT-2-XL may be anomalous; re-evaluate contribution scope |

### 2.4 時間予算

~4 GPU-hours total (< 10% of budget).

## 3. 核心実験

### Exp-1: Cross-Model TECS Replication (Sub-RQ1)

**Goal**: Test universality of TECS ~ 0.

**Models**:

| Model | Params | d_k | d_v | Editing layer | GPU |
|-------|--------|-----|-----|---------------|-----|
| GPT-2-XL | 1.5B | 1600 | 1600 | L17 | 4090 (existing) |
| GPT-J-6B | 6B | 4096 | 4096 | L6 (causal tracing) | A6000 |
| Pythia-1B | 1B | 2048 | 2048 | TBD (causal trace first) | 4090 |
| Pythia-6.9B | 6.9B | 4096 | 4096 | TBD | A6000 |

**Protocol per model**: 200 CounterFact facts. ROME editing. BM25-weighted TDA gradients (top-20 documents). 5 null baselines. Cohen's d primary metric. Bonferroni correction.

**Metrics**: TECS mean, std, Cohen's d vs Null-A, 95% bootstrap CI (10000 resamples).

**Compute**: ~25-35 GPU-hours across models.

**Expected**: TECS d < 0.2 on all models. Subspace asymmetry qualitatively universal.

**If contradicted**: TECS d > 0.3 on larger models → model-size-dependent transition → investigate what architectural features change.

### Exp-2: Cross-Model Subspace Characterization (Sub-RQ2)

**Goal**: Quantify subspace properties across models; test scaling predictions.

**Protocol**: From Exp-1 data, per model compute:
- Editing subspace (D matrix): SVD → effective dim, spectral decay, 90% variance cutoff
- Attribution subspace (G matrix): SVD → effective dim, spectral decay, 90% variance cutoff
- Principal angles at k = 1, 5, 10, 20, 50
- Cross-projection fractions (D-in-G, G-in-D) at k = 10, 20
- Random subspace null (1000 trials per k)
- p-values for structured vs random

**Metrics**: Effective dimensionality (spectral entropy), spectral decay rate (power law fit), principal angle statistics, Grassmann distance, cross-projection asymmetry ratio.

**Compute**: CPU post-processing. Negligible.

**Expected**:
- Editing eff-dim ~ 30-60 (sub-linear scaling with d_k)
- Attribution eff-dim ~ 1-5 (model-invariant)
- Principal angles match random null at k >= 20
- Cross-projection asymmetry (G-in-D >> D-in-G) universal

**Scaling analysis**: Plot editing eff-dim and attribution eff-dim vs model hidden dimension. Fit power law: eff-dim ~ d_k^alpha. Predict alpha_editing ~ 0.3-0.5, alpha_attribution ~ 0.

### Exp-3: C^{-1} Whitening Ablation (Sub-RQ3)

**Goal**: Isolate ROME's covariance-inverse contribution to incommensurability.

**Protocol**: On GPT-2-XL (primary) + one additional model:

1. Compute delta_W_unwhitened = C * delta_W (recover pre-whitening direction) for all facts
2. Compute TECS_unwhitened
3. SVD of D_unwhitened → subspace properties
4. Principal angles between D_unwhitened and G
5. Cross-projection of unwhitened subspaces
6. Compare all metrics between whitened and unwhitened

**Statistical test**: Paired t-test on TECS per fact. Cohen's d for the whitened-vs-unwhitened difference.

**Compute**: ~2-3 GPU-hours.

**Three outcome scenarios**:

| Scenario | TECS_unwhitened d vs Null-A | Interpretation |
|----------|---------------------------|----------------|
| C^{-1} is main cause | d > 0.3 | ROME's whitening drives incommensurability |
| C^{-1} contributes | 0.1 < d < 0.3 | Partial contribution; fundamental separation remains |
| C^{-1} irrelevant | d < 0.1 | Incommensurability is fundamental to editing-attribution dichotomy |

All scenarios are publishable.

### Exp-4: Attribution Method Ablation (C5)

**Goal**: Test whether attribution ~1D collapse is aggregation artifact.

**Protocol**: On GPT-2-XL, 100 facts:

| Method | g_M | Expected eff-dim |
|--------|-----|-------------------|
| BM25-weighted (baseline) | sum w_i g_i / norm | ~1 (observed) |
| Raw mean | mean(g_i) / norm | ~1 if loss-dominated |
| RIF-rescaled | sum (w_i/(1-h_i)) g_i / norm | ~1-3 |
| SVD subspace (r=5) | top-5 left singular vectors of G | 5 (construction) |
| SVD subspace (r=10) | top-10 left singular vectors of G | 10 (construction) |

For each: SVD → eff-dim → principal angles with D → TECS (where applicable).

**Key test**: SVD subspace retains multi-dimensional attribution representation. If principal angles remain near-random even with r=10, incommensurability is not caused by 1D collapse.

**Compute**: ~3-5 GPU-hours (RIF dominates).

**Expected**: Eff-dim remains ~1 for scalar aggregations. SVD subspace at r=5 shows marginal angle improvement but still near-random at k >= 10.

### Exp-5: Toy Model Validation (C6)

**Goal**: Validate theoretical framework with known ground truth.

**Setup**: Linear associative memory W = sum v_i k_i^T:
- d in {100, 200, 500, 1000}, n in {10, 20, 50, 100}
- d/n (over-parameterization) varies from 2 to 100
- k_i, v_i ~ N(0, I_d), independent
- 10 random seeds per (d, n) configuration

**Editing**: ROME-style: delta_W = (C + lambda*I)^{-1} k* (v_new - Wk*)^T, C = (1/n) sum k_i k_i^T.

**Attribution**: g_i = (Wk_i - v_i) k_i^T. g_M = mean(g_i) / norm.

**Measurements**: TECS, subspace analysis, principal angles, C^{-1} ablation, ground-truth angle (delta_W vs true knowledge direction v_target k_target^T).

**Key predictions**:
1. TECS ~ 0 for d/n > 10
2. TECS increases as d/n → 1
3. C^{-1} effect larger in toy model
4. Phase boundary at d/n ~ 5-15

**Output**: 2D heatmap of Cohen's d vs (d, n). Phase boundary curve.

**Compute**: CPU only, < 1 hour.

### Exp-6: Layer Profile Analysis (Sub-RQ4)

**Goal**: Map editing-attribution geometry across transformer layers.

**Protocol**: GPT-2-XL, 100 facts:
- ROME editing at l*=17 (standard)
- TDA gradients at all 48 MLP layers
- TECS(l) = cos(vec(delta_W at 17), vec(g_M at l)) for each l
- Attribution subspace SVD at each layer

**Additional**: Compare TECS layer profile with causal tracing indirect effect profile (from Meng et al. 2022).

**Compute**: ~3-5 GPU-hours (48x gradient computation).

**Expected**: TECS varies but remains < 0.3 at all layers. Possible weak peak near l*=17. Attribution eff-dim may vary across layers.

## 4. Baseline 选択与論証

Analysis paper — "baselines" are comparison conditions:

| Comparison | Purpose |
|------------|---------|
| Random subspace null (1000 trials) | Gold standard for "is this structured?" |
| TECS Null-A through Null-E | Established null framework from TECA |
| MEMIT vs ROME | Editing method generality |
| Whitened vs unwhitened | Mechanism ablation |
| Multiple attribution methods | Attribution robustness |
| Cross-model | Universality |
| Toy model analytical predictions | Theory validation |

## 5. 指標定義

### Primary

| Metric | Formula | RQ |
|--------|---------|-----|
| Cohen's d (TECS vs Null-A) | (mean_real - mean_null) / pooled_sd | Main RQ |
| Effective dimensionality | exp(-sum p_i log p_i) | Sub-RQ2 |
| Min principal angle | theta_1 from SVD of U_D^T U_G | Sub-RQ2 |
| Grassmann distance | sqrt(sum theta_i^2) | Sub-RQ2 |
| TECS_unwhitened - TECS_whitened | Paired difference | Sub-RQ3 |
| Cross-projection fraction | tr(P^T M^T M P) / tr(M^T M) | Sub-RQ2 |

### Auxiliary

- TECS per-layer profile
- Spectral decay rate (power law exponent)
- Attribution eff-dim per aggregation method
- Toy model TECS vs d/n (phase boundary)

### Statistical framework

- All: Cohen's d + 10000 bootstrap 95% CIs
- Multiple comparisons: Bonferroni across 5 null baselines
- Cross-model: Per-model results reported independently
- Effect size: |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large

## 6. 応用価値

1. **Cross-paradigm caution**: Do NOT use editing success as evidence for attribution quality
2. **Editing method design**: If C^{-1} drives incommensurability, future editing methods could align with attribution geometry
3. **Attribution improvement**: Multi-dimensional methods may capture richer information than 1D aggregation
4. **Knowledge geometry theory**: Constrains theories of knowledge storage in transformers

## 7. 効率験証

Not applicable (analysis framework, not computational method).

## 8. 科学発見（可選）

If findings hold:
1. New axis for localization debate: editing and attribution subspaces provide complementary geometric evidence
2. Does editing method choice (ROME vs PMET vs MEND) change subspace geometry?
3. Can attribution be improved by projecting into editing subspace? (G captures 17.3% of D)

## 9. データセットと計算規划

### Total budget

| Experiment | GPU-hours | Priority |
|------------|-----------|----------|
| Exp-0 (Pilot) | 4 | P0 (gate) |
| Exp-1 (Cross-model) | 25-35 | P1 (core) |
| Exp-2 (Subspace) | ~0 (CPU) | P1 |
| Exp-3 (C^{-1}) | 2-3 | P1 |
| Exp-4 (Attribution) | 3-5 | P2 |
| Exp-5 (Toy model) | ~0 (CPU) | P1 |
| Exp-6 (Layer profile) | 3-5 | P2 |
| **Total** | **~37-52** | Within 80h |

### Timeline

| Week | Activity |
|------|----------|
| W1 | Exp-0 (pilot) + Exp-5 (toy model, CPU parallel) |
| W2-3 | Exp-1 (cross-model, parallel GPUs) + Exp-3 (whitening) |
| W4 | Exp-2 (CPU post-processing) + Exp-4 (attribution) + Exp-6 (layers) |
| W5-6 | Analysis + writing |
| W7-8 | Revision + submission |

### Data

- **CounterFact**: 21,919 facts (Meng et al. 2022). Use 200-300, sampled for relation-type diversity.
- **Training data**: Pile subset (ROME training corpus). BM25 top-20 per fact.
- **Models**: All on HuggingFace. No custom training.

## 10. 予想結果与失败预案

### Expected results

| Finding | Expected | Confidence |
|---------|----------|------------|
| TECS d vs Null-A (all models) | < 0.2 | High |
| Editing eff-dim | 30-60 | Medium |
| Attribution eff-dim | 1-5 | Medium-High |
| Principal angles vs random (k=20) | p > 0.5 | High |
| TECS_unwhitened d | 0.1-0.5 (higher than whitened) | Medium |
| Toy model phase transition | d/n ~ 5-15 | Low |

### Failure responses

| Failure | Severity | Response |
|---------|----------|----------|
| Pythia-1B TECS d > 0.5 | Critical | Single-model case study with deeper mechanism analysis |
| C^{-1} no effect (d < 0.1) | Adjust | Narrative: fundamental, not ROME artifact (strengthens theory) |
| SVD subspace changes angles | Adjust | Narrative: aggregation destroys dimensionality |
| Toy model fails | Adjust | Weaken theory, strengthen empirical claims |
| Attribution eff-dim > 10 on large models | Adjust | 1D collapse is scale-dependent; investigate |

## 11. NeurIPS 2026 Paper Scenarios

### Scenario A: Full contribution (all findings hold across models)

Universal incommensurability + subspace asymmetry + mechanism (C^{-1}) + theory (toy model). 8-page NeurIPS main conference paper.

**Title**: "The Geometry of Knowledge Operations: Editing and Attribution Directions are Incommensurable in Transformer Parameter Space"

### Scenario B: Partial universality

GPT-2-XL incommensurability + partial replication on some models + mechanism + toy model. Still NeurIPS-worthy if 3/4 models confirm.

### Scenario C: Single-model with deep mechanism

Only GPT-2-XL but with strong C^{-1} explanation + toy model validation + rich layer profile. Position as "detailed case study revealing a fundamental geometric property."

### Scenario D: Pivot required (Exp-0a fails)

If Pythia shows strong alignment, fundamental rethinking needed. Possible pivot to "model-dependent knowledge geometry" with GPT-2-XL as a case where incommensurability arises due to specific architectural features.

## 12. Back-References

- Problem statement: `research/problem-statement.md` (v1.2) — Gap G1, Main RQ, Sub-RQ1-4
- Method design: `research/method-design.md` (v3.0) — Components C1-C6
- Probe results: `legacy/teca-sibyl/results/` — TECA experimental data
- AURA probe: `Codes/_Results/probe_result.md` — Supplementary CIFAR-10 data
- Iteration log: `iteration-log.md` — Direction pivot history
