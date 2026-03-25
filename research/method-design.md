---
version: "3.0"
created: "2026-03-25"
last_modified: "2026-03-25"
entry_mode: "first"
iteration_major: 3
iteration_minor: 0
---

> **v3.0 design**: Complete rewrite for geometric incommensurability direction. Replaces BSS diagnostic method-design (v2.0). Six-component geometric analysis framework for characterizing editing-attribution parameter-space relationship.

# Method Design

## 1. Probe Signal Summary

From TECA experiments (GPT-2-XL, 100 CounterFact facts):

**Core negative result**: TECS ~ 0 (Cohen's d = 0.05 vs Null-A). Editing and attribution directions do not align.

**Subspace asymmetry**: Editing effective dim = 40.8 (90% at top-44); attribution effective dim = 1.2 (94.8% in top-10). Principal angles near 90 deg, indistinguishable from random at k>=20.

**MEMIT** (30 facts, simplified): Cross-layer d ~ 0.63. Matched-layer d = 4.8-7.4 (shared loss artifact).

**Cross-projection**: G-in-D = 17.3%, D-in-G = 1.0% (k=10).

**Design constraints**: C1 extend beyond GPT-2-XL; C2 ablate C^{-1}; C3 test attribution aggregation; C4 address function-invariance; C5 budget 40-80 GPU-hours.

## 2. Compute Budget

| Component | GPU-hours | GPU type |
|-----------|----------|----------|
| Cross-model TECS (GPT-J-6B) | 15-20 | A6000 |
| Cross-model TECS (Pythia-1B, 6.9B) | 10-15 | A6000/4090 |
| C^{-1} whitening ablation | 2-3 | 4090 |
| RIF attribution directions | 3-5 | 4090 |
| Full 48-layer sweep | 3-5 | 4090 |
| Aggregation method ablation | 2-3 | 4090 |
| Toy model experiment | 1-2 | CPU/4090 |
| Extended fact sets (200+) | 5-8 | 4090 |
| **Total** | **~41-61** | Within budget |

## 3. Attack Angle to Component Mapping

| Sub-goal | Component | Root Cause Layer | Validation |
|----------|-----------|-----------------|------------|
| Quantify scalar incommensurability | C1: TECS metric | L1: Surface measurement | Exp-1 |
| Characterize subspace geometry | C2: SVD + Principal Angles | L2-L3: Dimensionality asymmetry | Exp-2 |
| Identify mechanism | C3: C^{-1} Whitening Ablation | L3: Algorithmic vs fundamental | Exp-3 |
| Test attribution robustness | C4: Attribution Method Ablation | Constraint C3: aggregation artifact? | Exp-4 |
| Validate framework | C5: Toy Model Ground Truth | Oracle validation | Exp-5 |
| Map layer-wise geometry | C6: Layer Profile Analysis | Sub-RQ4: Depth variation | Exp-6 |

## 4. Framework Overview

This is an **analysis framework**, not a model. Six components characterize editing-attribution parameter-space geometry:

```
Input: Model M, Fact set F, Editing method E, Attribution method A
  -> C1: TECS (scalar alignment per fact + null baselines)
  -> C2: Subspace SVD (effective dim, spectral decay)
  -> C3: Principal Angles + Cross-Projection (structured vs random)
  -> C4: C^{-1} Ablation (whitened vs unwhitened)
  -> C5: Attribution Ablation (BM25, raw, RIF, SVD subspace)
  -> C6: Toy Model (ground-truth validation)
```

## 5. Component Details

### C1: TECS Metric

**Function**: Scalar alignment between editing and attribution directions per fact.

**I/O**: delta_W (d_v x d_k), g_M (d_v x d_k) -> TECS = cos(vec(delta_W), vec(g_M)).

**Causal argument**: Root cause is different operations selecting different directions. TECS is the natural scale-invariant directional metric. No simpler alternative captures directional (vs magnitude) alignment.

**Statistical framework**: 5 null baselines (Null-A: random fact; B: cross-layer; C: shuffled gradient; D: random direction; E: test gradient). Primary: Cohen's d with 10000 bootstrap. Bonferroni for 5 comparisons.

**Complexity**: O(d_v * d_k) per fact. Trivial.

**Validation**: Exp-1. Expected: d < 0.2 across models.

### C2: Subspace Construction and Spectral Analysis

**Function**: Construct editing/attribution subspaces, characterize spectral properties.

**I/O**: D (n x p), G (n x p) -> SVD -> effective dimensionality = exp(spectral entropy), variance profiles, decay rates.

**Causal argument**: TECS gives per-fact scalars. To understand WHY TECS ~ 0, we need subspace geometry. If editing uses ~40D and attribution ~1D, alignment is near-impossible in 10^7-D space -- this IS the explanation. Spectral entropy captures distributional information that rank or top-k variance cannot.

**Complexity**: Reduce to n-dimensional joint subspace first (n << p). O(n^3). Trivial.

**Validation**: Exp-2. Expected: editing eff-dim 30-60, attribution 1-5.

### C3: Principal Angle and Cross-Projection

**Function**: Principal angles between subspaces + cross-projection variance fractions.

**I/O**: Top-k singular vectors of D, G -> k angles, Grassmann distance, p-values vs 1000 random trials; cross-projection fractions.

**Causal argument**: TECS ~ 0 could be structured near-orthogonality (organized but non-overlapping) or random orthogonality. Principal angles disambiguate via null hypothesis testing. Cross-projection reveals partial overlaps invisible to angles (TECA: G-in-D = 17.3% vs D-in-G = 1.0% shows hierarchical structure).

**Complexity**: O(k^3 + 1000*k^3). Negligible.

**Validation**: Exp-2. Expected: random-level at k >= 20; cross-projection asymmetry universal.

### C4: C^{-1} Whitening Ablation

**Function**: Remove ROME's covariance-inverse rotation, repeat all analyses.

**I/O**: Standard delta_W + C -> delta_W_unwhitened = C * delta_W -> TECS_unwhitened, subspace of D_unwhitened, angles vs G.

**Causal argument**: ROME's C^{-1} decorrelates key space, emphasizing low-variance directions. TDA gradients are in natural basis. If C has high condition number (Marchenko-Pastur predicts this), the rotation is extreme. This is the single most theory-informative ablation: distinguishes "ROME artifact" from "fundamental geometry."

**Implementation**: delta_W_raw = k* (v_new - Wk*)^T (outer product without covariance inversion). Or multiply delta_W by C.

**Complexity**: O(d_k^2) per fact. ~2-3 GPU-hours total.

**Validation**: Exp-3. Three scenarios all publishable:
- TECS_unwhitened d > 0.3: C^{-1} is main cause
- 0.1 < d < 0.3: C^{-1} contributes but doesn't explain
- d < 0.1: incommensurability is fundamental

### C5: Attribution Method Ablation

**Function**: Test whether ~1D attribution collapse is aggregation artifact.

**Methods**: BM25-weighted (baseline), raw mean, RIF-rescaled (2506.06656), SVD subspace (top-5, top-10).

**Causal argument**: Attribution collapses to ~1D. Could be genuine (all facts' gradients dominated by same loss direction) or BM25 artifact. SVD subspace is key: retains multi-dimensional representation. If principal angles remain near-random even at r=10, incommensurability transcends the 1D collapse.

**RIF integration**: Addresses formalize review RIF/BIF concern. RIF rescaling may change gradient weighting and thus attribution direction.

**Complexity**: ~3-5 GPU-hours (RIF dominates).

**Validation**: Exp-4. Expected: eff-dim ~1 for scalar aggregations; SVD subspace marginal improvement.

### C6: Toy Model Validation

**Function**: Validate framework with known ground truth.

**Setup**: Linear associative memory W = sum v_i k_i^T. d in {100,500,1000}, n in {10,50,100}. ROME-style editing with C^{-1}. Gradient-based attribution.

**Causal argument**: Theory claims over-parameterization + C^{-1} -> incommensurability. Toy model tests this with complete ground truth. Correct predictions validate theory; failures reveal which assumptions break.

**Key prediction**: TECS ~ 0 for d/n > 10; increases as d/n -> 1. Phase boundary at d/n ~ 5-15.

**Complexity**: CPU only, < 1 hour.

**Validation**: Exp-5. 2D heatmap of d vs (d,n).

## 6. Causal Chain

**Gap**: Editing and attribution access same weights but different subspaces. No characterization exists.

**Root Cause**: (1) Different optimization objectives -> different directions. (2) Over-parameterization -> multiple equivalent routes -> no convergence constraint.

**Method**: C1 detects incommensurability. C2 reveals structural mechanism (40D vs 1D). C3 tests structured vs random. C4 isolates ROME's contribution. C5 tests aggregation artifacts. C6 validates theory.

**Why complete**: Each component addresses a distinct aspect. Remove any one and a gap remains in the explanation.

## 7. Theoretical Analysis

### 7.1 TECS Rank-1 Decomposition

Under linear associative memory: TECS = sign(alpha) * cos(C^{-1}k*, k_i) * cos(v*-Wk*, d_v_i).

Null distribution: E[TECS^2] ~ 1/d_k. For GPT-2-XL (d_k=1600): E[|TECS|] ~ 0.025.
SNR ~ rho_k * rho_v * sqrt(d_k). TECA result (d=0.05) implies rho_k * rho_v < 0.013.

### 7.2 Over-parameterization

p = d_v * d_k ~ 10^7, n ~ 10^2-10^3. Over-parameterization ratio ~ 10^2-10^3.

Editing eff-dim predicted ~ min(n, d_k) with ROME spreading. Observed 40.8 in range.
Attribution collapse to ~1 predicted when loss residuals are approximately parallel.

### 7.3 C^{-1} Condition Number

Marchenko-Pastur: condition number of C ~ (1+sqrt(d_k/N))^2 / (1-sqrt(d_k/N))^2 ~ 10^2-10^3 for GPT-2-XL. C^{-1} amplifies low-variance directions by this factor, creating extreme rotation away from natural gradient directions.

## 8. Method Positioning

**Inherits**: TECS metric and null framework from TECA. Standard SVD-based subspace analysis.

**Changes**: From TECA's scalar question to full geometric characterization. Added cross-model universality, C^{-1} ablation, RIF attribution ablation, toy model.

**Novel relative to**: Hase et al. (behavioral disconnect -> we give parameter-level explanation); ROME/MEMIT (editing effectiveness -> we study editing's geometric footprint); TDA literature (attribution quality -> we study attribution's geometric footprint).

## 9. Probe Code Reuse

**Directly usable** (`Codes/experiments/teca/`): pilot_rome.py, pilot_tecs_core.py, pilot_tda_gradient*.py, negative_subspace_geometry.py, negative_memit_experiment.py, precompute_memit_stats.py.

**Needs adaptation**: negative_whitening.py -> full C^{-1} ablation.

**Create new**: Cross-model pipeline, RIF attribution, toy model, 48-layer sweep, aggregation ablation.
