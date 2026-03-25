---
version: "1.2"
created: "2026-03-25"
last_modified: "2026-03-25"
entry_mode: "fr_revise"
iteration_major: 1
iteration_minor: 2
---

> **v1.2 fr_revise**: Major direction pivot. Abandoned both Direction A (BSS diagnostic) and Direction B (FM1/FM2 DATE-LM). New direction: Geometric Incommensurability of Knowledge Operations — leveraging TECA's definitive negative result (TECS ~ 0) as the core contribution. The negative result resolves Hase et al. (2023) localization-editing disconnect at the parameter level. Formalize review round-1 issues addressed: Direction A/B inconsistency resolved (chose neither — pivoted to TECA geometric analysis); RIF/BIF engaged as ablation targets; DATE-LM probe no longer needed (evidence comes from TECA, not benchmark performance).

# Problem Statement

## 1. Gap 定义

### 1.1 Gap 候选列表

| # | 候选 Gap | 推导路径 | 重要性 | 新颖性 | 可解性 |
|---|----------|---------|--------|--------|--------|
| G1 | Editing and attribution directions in parameter space are geometrically incommensurable, but no one has characterized WHY or what this reveals about knowledge organization | TECA experiments (TECS ~ 0, subspace analysis) + Hase et al. localization-editing disconnect + ROME/MEMIT literature | **高**: Resolves a fundamental tension across editing, TDA, and mechanistic interp communities | **高**: First systematic geometric characterization of editing-attribution relationship in parameter space | **中-高**: TECA data already collected; need theoretical framework + extended experiments |
| G2 | Five representation-space TDA methods outperform parameter-space methods but lack unified evaluation and theoretical explanation (FM1/FM2 framework) | CRA_old framework + AURA variance decomposition + Li et al. + DDA evidence | **中-高**: Practitioner guidance gap | **中**: Unified framework novel but individual observations known | **中**: Requires DATE-LM experiments with uncertain RepSim performance; review flagged 3 structural issues |
| G3 | Per-test-point Hessian sensitivity varies dramatically (77.5% residual variance) but no reliable diagnostic exists (BSS direction) | AURA Phase 0-2b variance decomposition + TRV instability + SI orthogonality | **中**: Practical diagnostic tool | **中**: BSS novel but may degenerate to gradient norm | **中**: BSS-gradient_norm correlation (rho=0.906) is primary risk; CIFAR-10 only |

### 1.2 选定 Gap（G1: Geometric Incommensurability of Knowledge Operations）

**一句话**: Knowledge editing (ROME/MEMIT) and training data attribution (TDA) operate on the same weight matrices but in geometrically incommensurable parameter subspaces — the editing subspace is ~40-dimensional and distributed while the attribution subspace is ~1-dimensional and collapsed — yet no prior work has characterized this geometric disconnect or explained what it reveals about how knowledge is organized in transformer parameters.

**详细论证**:

Three research communities study knowledge in neural networks from different angles — knowledge editing, training data attribution (TDA), and mechanistic interpretability — yet almost no work has connected them at the level of **parameter-space geometry**. ROME/MEMIT produce deterministic rank-1 update vectors delta_W that modify factual associations. TDA methods produce gradient vectors g_M that trace how training data shaped the same weight matrices. Both live in exactly the same parameter space (d_v x d_k per MLP layer), making directional comparison natural and meaningful.

The core finding from TECA experiments (GPT-2-XL, 100 CounterFact facts, 5 null baselines):

1. **TECS ≈ 0**: The cosine similarity between editing and attribution directions is indistinguishable from chance (Cohen's d = 0.05 vs Null-A; Bonferroni-corrected: all 5 comparisons non-significant). This is a definitive negative result with tight confidence intervals (95% CI crosses zero: [-0.00117, 0.00146]).

2. **Dramatic subspace asymmetry**: The editing subspace has effective dimensionality ~40.8 with variance distributed across ~44 components (90% explained by top-44), while the attribution subspace collapses to effective dimensionality ~1.2 with 94.8% variance in the top-10 directions. This structural asymmetry — one operation distributes across parameter space while the other concentrates — is itself a novel empirical finding.

3. **Random-level misalignment confirmed**: Principal angle analysis (k=10,20,50) shows the editing and attribution subspaces are approximately as orthogonal as random subspaces of the same dimension (p=0.084 at k=10; p>=0.989 at k>=20). Minimum principal angle at k=10 is 63.7° (random baseline: 66.8° +/- 2.1°).

4. **MEMIT shows layer-specific patterns**: Multi-layer distributed editing (MEMIT, 30 facts) achieves strong matched-layer alignment (d=6.7 at L13, d=7.4 at L14) but cross-layer alignment comparable to ROME (d~0.63). Distributing edits across layers does NOT fundamentally change the editing-attribution geometric relationship.

5. **Cross-projection asymmetry**: D (editing) variance captured by G (attribution) subspace: only 1.0% (k=10). G variance captured by D subspace: 17.3% (k=10). The attribution subspace has a small foot in the editing space, but not vice versa.

This finding resolves the persistent **localization-editing disconnect** (Hase et al., 2023): causal tracing localizes knowledge to specific MLP layers, but editing success doesn't correlate with localization strength. Our geometric analysis provides a parameter-level explanation: editing and attribution access the SAME layers but DIFFERENT subspaces within those layers. The disconnect is not about which layers matter, but about which directions within those layers each operation uses.

**Evidence type**: "做了但有根本缺陷" — The knowledge editing and TDA communities both implicitly assume their operations access commensurable parameter subspaces. This assumption is wrong, and the failure to question it has left the localization-editing disconnect unexplained for 3+ years.

### 1.3 Root Cause 分析

**Root Cause Type**: 被忽视的维度 (parameter-space geometry as a lens connecting editing and attribution).

**Layer 1 (surface symptom)**: Causal tracing localizes knowledge to specific MLP layers, but editing success doesn't correlate with localization strength (Hase et al., 2023). TDA methods and editing methods produce different rankings/behaviors even when targeting the same factual knowledge.

**Layer 2 (intermediate)**: Editing and attribution are designed with fundamentally different objectives:
- ROME solves a constrained least-squares problem: min ||W_new k* - v*|| subject to W_new k_j = W k_j for j != target, with the key step being covariance-inverse whitening (C^{-1}). This optimization is designed for **precise factual insertion** with minimal collateral damage.
- TDA gradient aggregation computes g_M = weighted sum of per-sample loss gradients, capturing how training data shaped the current weight configuration. This is designed for **influence tracing** via first-order Taylor approximation.

These different optimization objectives select fundamentally different directions in the same parameter space.

**Layer 3 (structural root cause)**: The parameter space of an MLP layer (R^{d_v x d_k}, typically ~10^7 dimensions) is vastly over-parameterized relative to the number of facts stored (~10^4-10^5 facts in practical LLMs). Different knowledge operations can access the SAME information through DIFFERENT parameter-space routes — analogous to different basis sets representing the same function space. Specifically:
- ROME's C^{-1} rotation projects onto statistically decorrelated directions, spreading delta_W across ~40 dimensions (effective dim = 40.8)
- TDA gradient aggregation concentrates on the dominant loss gradient direction, collapsing to ~1 dimension (effective dim = 1.2)
- The 40D-vs-1D asymmetry reflects a fundamental difference in how the two operations navigate the over-parameterized landscape

**Oracle validation**: If we had a perfect theory of knowledge geometry in transformers (knowing exactly which parameter directions encode which facts), would the editing-attribution disconnect disappear? Yes — we could verify whether both operations access the same "knowledge directions," regardless of their algorithmic paths. The absence of such a theory is precisely why the geometric characterization we propose is needed.

### 1.4 Gap 三维评价

| 维度 | 评价 | 论证 |
|------|------|------|
| **重要性** | **高** | Connects three active communities (editing: ROME/MEMIT/PMET; TDA: IF/TRAK/RepSim; mechanistic interp: causal tracing). Resolves a 3-year-old open question (Hase et al. 2023 disconnect). The "negative result IS the contribution" framing has strong precedent at top venues. Understanding knowledge geometry is foundational for both safe model editing and reliable attribution. |
| **新颖性** | **高** | No prior work has systematically measured the geometric relationship between editing and attribution directions in parameter space. The subspace asymmetry (40D vs 1D) and random-level misalignment are completely novel empirical observations. The theoretical framework connecting C^{-1} whitening to incommensurability is new. |
| **可解性** | **中-高** | Core TECA experiments already completed (GPT-2-XL, 100 facts). Need: (a) extended experiments on additional models for robustness, (b) theoretical framework formalization, (c) whitening ablation (H6), (d) toy model validation. GPU budget: 4x RTX 4090 + 4x A6000, timeline ~8 weeks to NeurIPS 2026. Estimated additional compute: 40-80 GPU-hours, well within budget. |

## 2. 研究问题

### 2.1 Main RQ

**To what extent are knowledge editing directions and training data attribution directions geometrically commensurable in transformer parameter space, and what does their incommensurability reveal about the structure of knowledge encoding?**

- *Falsification*: If TECS shows significant positive alignment (Cohen's d > 0.3) on a majority of tested models/fact sets, the incommensurability thesis is wrong, and we pivot to characterizing positive alignment patterns.
- *Prediction*: Based on TECA pilot, we predict TECS ~ 0 across models, with structured subspace asymmetry (editing distributed ~40D, attribution collapsed ~1D) as a general property of autoregressive transformers with rank-1 editing.
- *Boundary*: Addresses parameter-space geometry at the MLP layer level. Does NOT address attention layers, residual stream geometry, or behavioral-level alignment.
- *Independent value*: Even if the theoretical explanation is incomplete, the empirical characterization of subspace geometry constrains future theories of knowledge storage and provides concrete guidance for hybrid editing-attribution systems.

### 2.2 Sub-RQs

**Sub-RQ1 (Universality)**: Does the TECS ~ 0 finding generalize beyond GPT-2-XL to other autoregressive transformers (GPT-J-6B, Pythia-1B, Pythia-6.9B)?
- *Falsification*: TECS shows d > 0.3 on >= 2/4 tested models.
- *Prediction*: TECS ~ 0 is a general property, not model-specific. Subspace dimensionality profiles may scale with model size but the qualitative asymmetry persists.

**Sub-RQ2 (Subspace Characterization)**: What are the quantitative properties of the editing and attribution subspaces — effective dimensionality, spectral decay rate, principal angle distribution, cross-projection fractions — and how do they vary with model size?
- *Falsification*: Subspace properties are indistinguishable from random projections (no structured asymmetry detected across models).
- *Prediction*: Editing subspace is ~40D distributed (spectral decay approximately uniform within effective rank), attribution subspace is ~1D collapsed (sharp spectral cutoff), and effective dimensionalities scale sub-linearly with model hidden dimension.

**Sub-RQ3 (Mechanism — C^{-1} Whitening)**: Does ROME's covariance-inverse rotation (C^{-1}) explain the geometric incommensurability, or is it a more fundamental property of how knowledge operations access parameter space?
- *Falsification*: TECS_unwhitened (raw, without C^{-1}) >> TECS_whitened (standard) with d > 0.5 → C^{-1} is the primary cause, and the incommensurability is an artifact of ROME's algorithm rather than fundamental geometry.
- *Prediction*: C^{-1} contributes to the incommensurability (TECS_unwhitened > TECS_whitened) but does not fully explain it. The 40D-vs-1D dimensionality asymmetry persists even without whitening, reflecting a more fundamental separation between how constrained optimization (editing) and first-order approximation (attribution) navigate over-parameterized spaces.

**Sub-RQ4 (Layer Profile)**: How does the editing-attribution geometric relationship vary across transformer layers, and does the profile correlate with causal tracing indirect effect?
- *Falsification*: TECS is uniform across layers (no layer-specific geometry).
- *Prediction*: MEMIT data shows layer-specific variation in matched-layer alignment (d ranges from 4.8 to 7.4 across L13-L16). The layer profile may correlate with causal tracing peaks, providing a geometric explanation for why causal tracing identifies specific layers.

## 3. 攻击角度

### 3.1 候选攻击角度（简表）

| 角度 | 核心 idea | Root cause 匹配度 | 可行性 |
|------|-----------|------------------|--------|
| **A: Geometric Incommensurability Analysis** | Characterize editing-attribution parameter geometry via subspace analysis, principal angles, spectral decomposition, C^{-1} ablation. Frame the negative result as contribution. | **高**: Directly measures the root cause (different operations → different subspaces due to over-parameterization) | **高**: Core data exists; theory formalizable; 40-80 GPU-hours additional |
| B: FM1/FM2 Diagnostic Framework | Signal-processing framing of parameter-space TDA failures; 2x2 ablation on DATE-LM | **中**: Addresses TDA failure modes but not editing-attribution disconnect | **低**: 155 GPU-hours needed; RepSim performance uncertain; 3 structural issues from review |
| C: BSS Per-Test-Point Diagnostic | Bucketed spectral sensitivity for Hessian sensitivity prediction | **低**: Different problem entirely (per-point diagnosis vs knowledge geometry) | **中**: BSS-gradient norm correlation (rho=0.906) primary risk; CIFAR-10 only |

### 3.2 选定攻击角度 (A: Geometric Incommensurability Analysis)

**核心 idea**: We propose a systematic geometric analysis framework that characterizes the relationship between knowledge editing and training data attribution in transformer parameter space. Using principal angle analysis between editing and attribution subspaces, spectral decomposition revealing their dimensionality asymmetry, ablation of ROME's C^{-1} whitening transform, and cross-model validation, we reveal that these two knowledge operations are geometrically incommensurable. We formalize this finding by connecting it to the over-parameterization of MLP weight matrices and the distinct optimization objectives of editing vs attribution, providing the first parameter-level explanation for the Hase et al. (2023) localization-editing disconnect.

**与 root cause 因果匹配论证**: The root cause is that different knowledge operations select different parameter-space directions due to distinct optimization objectives in a vastly over-parameterized space. The attack angle directly measures this: (1) principal angle analysis quantifies the geometric separation, (2) spectral decomposition reveals the dimensionality mechanism (40D vs 1D), (3) C^{-1} ablation isolates the algorithmic contribution of ROME's whitening, (4) cross-model validation tests whether the phenomenon is general or architecture-specific. Each analysis component directly addresses one aspect of the root cause chain.

**探针结果支持程度**: TECA pilot experiments provide strong support:
- TECS ~ 0 (d = 0.05) confirms incommensurability at the scalar level
- Subspace analysis (40D vs 1D effective dimensionality) reveals the structural mechanism
- MEMIT comparison (30 facts) shows the pattern persists across editing methods
- 5 null baselines + Bonferroni correction ensure statistical rigor
- Cross-projection asymmetry (1.0% vs 17.3%) provides additional geometric detail
- All Phase 1-3 data collected and validated on GPT-2-XL

### 3.3 局限性与风险

1. **GPT-2-XL only (so far)**: Current data from a single model. If incommensurability is model-specific, the contribution is weakened. **Mitigation**: Extend to GPT-J-6B and Pythia-1B/6.9B (Sub-RQ1). Estimated cost: ~20-30 GPU-hours on A6000, well within budget.

2. **ROME-specific editing**: ROME's rank-1 update with C^{-1} is one editing approach. PMET, MEND, GRACE use different update structures. **Mitigation**: MEMIT already provides multi-method evidence. PMET (which doesn't use C^{-1}) would be the strongest ablation — include if time permits.

3. **BM25-weighted gradient aggregation for TDA**: The attribution direction g_M depends on top-k BM25 retrieval and weighting scheme. **Mitigation**: Ablation of aggregation method (raw mean, IF-weighted, subspace projection via top-r SVD of gradient matrix). Partially completed in TECA; extend systematically.

4. **"Parameter space direction is not function-invariant" (Codex review critique)**: Gradient directions depend on parameterization; reparameterization changes directions without changing the function. **Mitigation**: Valid concern for claims about "intrinsic knowledge geometry." We frame results as properties of the STANDARD parameterization used by both editing and attribution methods — since practitioners operate in this parameterization, the geometric findings are practically relevant even without function-invariance claims.

5. **RIF/BIF effect on attribution subspace**: Rescaled IF (2506.06656) and Bayesian IF improve parameter-space attribution quality. If RIF produces attribution directions that better align with editing directions, the "fundamental incommensurability" claim weakens — the gap may be partially attributable to classical IF's approximation errors rather than fundamental geometry. **Mitigation**: Include RIF-based attribution directions as an ablation. If RIF changes subspace geometry, this is itself a finding about how Hessian corrections reshape parameter-space knowledge access. Either outcome is informative.

6. **Theoretical depth risk**: The gap between "we observe incommensurability" and "we explain WHY it exists" is significant. Without a compelling theoretical framework, reviewers may view the paper as "just an observation." **Mitigation**: (a) C^{-1} whitening ablation provides a mechanistic pathway, (b) over-parameterization theory connects to established literature, (c) toy model with KNOWN knowledge locations provides ground-truth validation of the framework. The toy model is the strongest theoretical anchor: if we can show that in a controlled setting where we KNOW the knowledge directions, both editing and attribution access them differently, the geometric framework is validated.

7. **"So what?" critique — practical implications unclear**: Knowing that editing and attribution subspaces are orthogonal is interesting, but reviewers will ask what to DO with this finding. **Mitigation**: Frame practical implications clearly: (a) caution against using editing success as evidence for attribution quality (or vice versa), (b) design of future attribution methods that account for the geometric structure, (c) the subspace characterization itself enables new hybrid methods that project into the joint editing-attribution space.

## 4. 探針結果整合

### 4.1 已验证假設

| 假设 | 证据 | 信号强度 |
|------|------|---------|
| H1 (TECS ~ 0): Editing and attribution directions do NOT align at the editing layer | TECS mean = 0.000157, Cohen's d = 0.05 vs Null-A, all 5 null comparisons non-significant (Bonferroni) | **强** (definitive negative) |
| ROME editing technically reliable | 100/100 efficacy, mean P(new) = 0.978 | **强** |
| TDA gradients technically valid | 100/100 valid gradients, mean norm = 0.19, angular variance = 0.048 (moderate coherence) | **强** |
| Subspace asymmetry (editing ~40D, attribution ~1D) | Effective dim: editing 40.8, attribution 1.2; editing 90% at top-44, attribution 94.8% at top-10 | **强** |
| Random-level misalignment (not structured) | Principal angles at k=10,20,50: p >= 0.084; min angle 63.7° vs random 66.8° +/- 2.1° | **强** (k>=20 clearly random; k=10 marginal) |
| MEMIT preserves incommensurability | Cross-layer d ~ 0.63 (comparable to ROME); matched-layer high but reflects structural correlation | **中** (30 facts; simplified MEMIT) |
| Cross-projection asymmetry | D-in-G: 1.0%; G-in-D: 17.3% (k=10) | **中** |

### 4.2 未验证假設

| 假设 | 未验证原因 | 后续验证建议 |
|------|-----------|-------------|
| Universality across models (Sub-RQ1) | Only GPT-2-XL tested | Run on GPT-J-6B, Pythia-1B, Pythia-6.9B (~20-30 GPU-hours) |
| C^{-1} whitening as primary mechanism (Sub-RQ3, H6) | Whitening ablation not completed | Compute TECS_unwhitened (remove C^{-1} from ROME delta_W) and compare subspace properties |
| Layer-wise profile correlates with causal tracing (Sub-RQ4) | Only L17 (ROME) and L13-17 (MEMIT) tested | Full 48-layer TECS sweep on GPT-2-XL |
| Model-size scaling of subspace dimensionalities | Single model only | Compare effective dims across Pythia-1B vs 6.9B |
| RIF effect on attribution subspace | RIF not implemented in TECA | Compute RIF-based attribution directions and repeat subspace analysis |
| Toy model validation | Not implemented | Construct synthetic associative memory with known storage locations |

### 4.3 意外发现

1. **Extreme subspace asymmetry (40D vs 1D)**: We expected both subspaces to be low-dimensional but similar in rank. The 40:1 ratio was unexpected and suggests that ROME's constrained optimization explores a much richer parameter manifold than loss-gradient-based attribution. This asymmetry is a standalone contribution.

2. **MEMIT matched-layer alignment is very high (d = 6.7-7.4) but misleading**: High matched-layer TECS reflects shared loss function structure (delta_W at layer l and g_M at the same layer l both derive from the same loss), not genuine knowledge-geometric alignment. The informative comparison (cross-layer TECS) remains near-random. This is an important methodological caveat for any future work comparing editing and attribution.

3. **Attribution subspace collapses to ~1 dimension**: Top-1 principal component explains ~80% of attribution variance (effective dim = 1.2). This means all 100 facts' TDA gradients essentially point in the SAME direction — suggesting that BM25-weighted gradient aggregation is dominated by a single loss landscape mode rather than fact-specific information.

4. **Cross-projection is asymmetric (G captures 17.3% of D; D captures only 1.0% of G)**: The attribution direction has a small but non-negligible foot in the editing subspace, but the editing subspace has almost zero presence in the attribution subspace. This suggests a hierarchical relationship: editing accesses a richer space that partially overlaps with attribution, but attribution is too narrow to cover editing's manifold.

### 4.4 探針局限性

1. **Single model (GPT-2-XL)**: Most critical. All findings may be architecture- or scale-specific.
2. **100 CounterFact facts**: Sufficient for statistical power but limited in factual diversity (primarily entity-relation-object triples).
3. **BM25 retrieval for training data**: Top-k retrieval may miss important training samples, biasing the attribution subspace toward lexically similar documents.
4. **Simplified MEMIT (identity covariance)**: GPU memory constraints required identity instead of actual covariance matrices, potentially affecting MEMIT's true subspace.
5. **No whitening ablation completed**: H6 (C^{-1} as mechanism) is the most theory-relevant hypothesis but remains untested.
6. **CPU-only subspace analysis in 199-dimensional joint subspace**: Preserves all relevant geometry (both subspaces have <= 100 vectors) but limits resolution for finer-grained analysis.

## 5. 元数据

- **基于 Startup 产出版本**: project.md v1.0 (assimilated from CRA_old + AURA + CRA)
- **探针结果来源**: legacy/teca-sibyl/results/ (negative_subspace_results.json, negative_memit_results.json); Codes/_Results/probe_result.md (AURA CIFAR-10 data as supplementary context)
- **GPU 资源约束**: 4x RTX 4090 (xuchang0) + 4x A6000 (jinxulin); ~40-80 GPU-hours for extended experiments (well within budget)
- **Formalize Review Round 1 addressal**:
  - Direction A/B inconsistency → resolved by pivoting to neither: chose TECA geometric incommensurability direction (G1), which leverages existing experimental data and avoids both DATE-LM uncertainty (Direction B) and BSS degeneracy risk (Direction A)
  - DATE-LM probe → no longer needed: our evidence comes from TECA parameter-space geometry (GPT-2-XL + CounterFact), not from DATE-LM benchmark performance
  - RIF/BIF literature gap → engaged in Section 3.3 risk #5: RIF-based attribution as ablation target; either outcome (alignment changes or doesn't) is informative
- **Key excluded directions**:
  - FM1/FM2 2x2 ablation on DATE-LM (155 GPU-hours needed; RepSim LDS = 0.074 in AURA; 3 structural review issues)
  - BSS per-test-point diagnostic (different problem scope; gradient norm degeneracy rho = 0.906; CIFAR-10 only)
  - TECS as TDA validation metric (TECS ~ 0 means editing is NOT a validation channel — but this negative result IS the contribution)
- **Iteration log update**: v1.2 pivot from FM1/FM2 and BSS to geometric incommensurability. Excluded: FM1/FM2 DATE-LM direction (infeasible compute + uncertain RepSim), BSS direction (gradient norm degeneracy risk). New direction leverages completed TECA data.
