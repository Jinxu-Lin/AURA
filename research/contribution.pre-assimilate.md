---
version: "1.0"
last_updated: "2026-03-25"
---

# Contribution Tracker — AURA

## Target
- **Conference**: NeurIPS 2026
- **Novelty**: Medium-High (new diagnostic dimension, no prior per-test-point Hessian sensitivity work)
- **Significance**: Medium-High (practical impact for TDA practitioners)

## Contributions

### C0: First systematic variance decomposition of TDA sensitivity
- **What**: Two-way ANOVA decomposing per-test-point attribution sensitivity (J10, tau, LDS) into class, gradient-norm, and residual components.
- **Why it matters**: Establishes that per-test-point Hessian sensitivity is a real, significant phenomenon — not explainable by obvious confounds.
- **Key numbers**: 77.5% residual J10 variance, 53.4% residual tau, 45.9% residual LDS.
- **Status**: **CONFIRMED** (Phase 1 PASS)

### C1: BSS as theoretically grounded, seed-stable, per-test-point diagnostic
- **What**: Bucketed Spectral Sensitivity decomposes attribution sensitivity by Hessian eigenvalue magnitude buckets (outlier/edge/bulk), leveraging random matrix theory for cross-seed stability.
- **Why it matters**: First seed-stable per-test-point diagnostic for TDA reliability. Prior diagnostics (SI, TRV) are either measuring the wrong thing or seed-unstable.
- **Key numbers**: Pilot BSS-J10 correlation -0.42 (promising), but BSS-gradient_norm ρ = 0.906 (needs disentanglement).
- **Status**: **IN PROGRESS** (pilot shows promise, 5-seed stability test pending)

### C2: Compute-normalized Pareto frontier for adaptive TDA strategies
- **What**: Per-test-point routing between IF and RepSim based on BSS signals and/or cross-method disagreement, evaluated on LDS-vs-compute Pareto frontier.
- **Why it matters**: First systematic comparison of adaptive vs uniform TDA strategies with proper compute normalization.
- **Status**: **PLANNED** (depends on C1 validation)

### C3: Negative results — SI-TRV null correlation and cross-seed TRV instability
- **What**: Empirical demonstration that Self-Influence (SI) does not correlate with Hessian sensitivity (ρ ≈ 0), and that True Residual Variance (TRV) is completely seed-unstable (ρ ≈ -0.006).
- **Why it matters**: Falsifies common assumptions. SI is widely assumed to be a diagnostic for attribution reliability; we show it measures an orthogonal dimension. TRV's instability explains why no prior work has built reliable spectral diagnostics.
- **Status**: **CONFIRMED** (Phase 0 probe)

## Contribution Dependency

```
C0 (variance decomposition) ─── CONFIRMED ───→ Establishes phenomenon
    ↓
C1 (BSS diagnostic) ─── IN PROGRESS ───→ Provides diagnostic tool
    ↓
C2 (adaptive selection) ─── PLANNED ───→ Operationalizes diagnostic

C3 (negative results) ─── CONFIRMED ───→ Standalone, motivates C1
```

### C4: TECS geometric incommensurability characterization (from TECA)
- **What**: TECS (cosine similarity between ROME editing direction and TDA gradient direction) is indistinguishable from chance (Cohen's d = 0.05 vs all 5 null baselines). Follow-up subspace analysis reveals structured misalignment: editing operates in ~40D distributed subspace, attribution collapses to ~1D subspace. Whitening (C^{-1}) does NOT explain the gap. MEMIT shows same fundamental incommensurability.
- **Why it matters**: First parameter-space evidence for the localization-editing disconnect (Hase et al., 2023). Demonstrates that knowledge editing and attribution access fundamentally different geometric subspaces — not randomly different, but structurally organized.
- **Key numbers**: TECS mean = 0.000157, Cohen's d = 0.05, all 5 baselines non-significant after Bonferroni correction.
- **Status**: **CONFIRMED** (TECA complete pilot, all negative path experiments done)

## Paper Narrative

If C1 validates:
1. **Motivation**: TDA methods disagree at per-test-point level (C0) — practitioners need diagnostic
2. **Prior diagnostics fail**: SI measures wrong thing, TRV is seed-unstable (C3)
3. **Our solution**: BSS provides theoretically grounded, seed-stable diagnostic (C1)
4. **Practical impact**: BSS-guided adaptive selection improves Pareto frontier (C2)

If C1 validates but C2 fails:
- Drop C2, focus on C0 + C1 + C3 as diagnostic-only paper
- Still novel and significant: first per-test-point TDA diagnostic

If C1 fails:
- Negative results paper: C0 + C3 + C1 failure analysis
- Contribution: characterizing the diagnostic gap in TDA
