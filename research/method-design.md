---
version: "3.0"
created: "2026-03-25"
last_modified: "2026-03-25"
entry_mode: "first"
iteration_major: 3
iteration_minor: 0
---

> **v3.0**: Geometric incommensurability analysis framework. Six components characterizing editing-attribution parameter-space relationship.

# Method Design

## 1. Probe Signal Summary

TECA (GPT-2-XL, 100 CounterFact facts): TECS ~ 0 (d=0.05). Editing eff-dim=40.8, attribution eff-dim=1.2. Principal angles near-random. MEMIT cross-layer d~0.63. G-in-D=17.3%, D-in-G=1.0%.

Constraints: extend beyond GPT-2-XL; ablate C^{-1}; test attribution aggregation; 40-80 GPU-hours.

## 2. Compute Budget

Total ~41-61 GPU-hours: cross-model 25-35h, C^{-1} ablation 2-3h, RIF 3-5h, layer sweep 3-5h, aggregation 2-3h, toy model ~0h (CPU), extended facts 5-8h.

## 3. Component Mapping

| Sub-goal | Component | Validation |
|----------|-----------|------------|
| Scalar incommensurability | C1: TECS | Exp-1: cross-model |
| Subspace geometry | C2: SVD + Principal Angles | Exp-2: characterization |
| Mechanism | C3: C^{-1} Ablation | Exp-3: whitened vs unwhitened |
| Attribution robustness | C4: Attribution Ablation | Exp-4: raw, RIF, SVD |
| Framework validation | C5: Toy Model | Exp-5: synthetic memory |
| Layer geometry | C6: Layer Profile | Exp-6: 48-layer sweep |

## 4. Framework Overview

Analysis framework (not model/algorithm). Six components: C1 TECS (scalar alignment + null baselines) -> C2 SVD (effective dim, spectral decay) -> C3 Principal Angles + Cross-Projection (structured vs random) -> C4 C^{-1} Ablation (mechanism) -> C5 Attribution Ablation (robustness) -> C6 Toy Model (ground truth).

## 5. Component Details

### C1: TECS Metric
cos(vec(delta_W), vec(g_M)) per fact. 5 null baselines, Cohen's d, Bonferroni. O(d_v*d_k). Expected: d < 0.2 across models.

### C2: SVD + Spectral Analysis
D, G matrices -> SVD -> effective dim (spectral entropy), variance profiles. Reduce to n-dim joint subspace. Expected: editing 30-60D, attribution 1-5D.

### C3: Principal Angles + Cross-Projection
k principal angles between span(D_k), span(G_k). 1000 random trials for null. Cross-projection fractions. Expected: random-level at k>=20; G-in-D >> D-in-G.

### C4: C^{-1} Whitening Ablation
delta_W_unwhitened = C * delta_W or k*(v_new-Wk*)^T. Repeat C1-C3. Most theory-informative ablation. Three publishable scenarios: C^{-1} main cause (d_unwhitened > 0.3), partial (0.1-0.3), irrelevant (<0.1).

### C5: Attribution Method Ablation
BM25 (baseline), raw mean, RIF-rescaled (2506.06656), SVD subspace (r=5,10). Tests 1D collapse artifact. SVD subspace is key: if angles remain near-random at r=10, incommensurability transcends collapse.

### C6: Toy Model
Linear associative memory W = sum v_i k_i^T. Vary d/n from 2-100. ROME-style editing + gradient attribution. Ground truth known. Predicted phase transition at d/n ~ 5-15.

## 6. Causal Chain

Gap: editing and attribution access same weights, different subspaces. Root cause: different optimization objectives + over-parameterization. Method: C1 detects, C2 characterizes, C3 tests structure, C4 isolates mechanism, C5 tests robustness, C6 validates theory. Each component necessary; none sufficient alone.

## 7. Theory

TECS rank-1 decomposition: TECS = sign(alpha)*cos(C^{-1}k*,k_i)*cos(v*-Wk*,d_v_i). Null: E[TECS^2] ~ 1/d_k. Over-parameterization: p/n ~ 10^2-10^3 allows orthogonal subspaces. C^{-1} condition number ~ 10^2-10^3 creates extreme rotation.

## 8. Positioning

Novel: no prior work compares editing-attribution parameter-space geometry. Resolves Hase et al. (2023) localization-editing disconnect at parameter level. Inherits TECS from TECA pilot; adds cross-model, C^{-1} ablation, RIF, toy model.

## 9. Code Reuse

Reuse: Codes/experiments/teca/ (pilot_rome.py, pilot_tecs_core.py, pilot_tda_gradient*.py, negative_subspace_geometry.py). Create new: cross-model pipeline, RIF attribution, toy model, 48-layer sweep.
