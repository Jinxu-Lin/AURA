> [ASSIMILATED: generated from iter_001/exp/results/]

# Experiment Results — AURA

## Phase 0: Hessian Hierarchy Verification (Pre-Sibyl Probe)

**Setup**: 3 seeds × 100 CIFAR-10 test points, last-layer IF, ResNet-18.

### Hessian Condition Numbers
| Seed | κ (EK-FAC) | κ (K-FAC) | κ Ratio |
|------|-----------|-----------|---------|
| 42 | 1.21×10⁶ | 3.45×10⁶ | 2.85 |
| 123 | 1.34×10⁶ | 3.72×10⁶ | 2.78 |
| 456 | 1.18×10⁶ | 3.51×10⁶ | 2.97 |

### Attribution Agreement (J@10)
| Comparison | Mean J@10 | Std | Interpretation |
|-----------|-----------|-----|----------------|
| EK-FAC vs EK-FAC (same) | 1.000 | 0.000 | Deterministic (sanity check) |
| EK-FAC vs K-FAC | 0.487 | 0.183 | ~50% of top-10 change |
| EK-FAC vs Diagonal | 0.243 | 0.156 | ~75% of top-10 change |
| K-FAC vs Diagonal | 0.312 | 0.167 | ~70% of top-10 change |

### SI-TRV Correlation
| Seed | Spearman ρ(SI, TRV) | p-value |
|------|---------------------|---------|
| 42 | 0.012 | 0.907 |
| 123 | -0.034 | 0.738 |
| 456 | 0.008 | 0.936 |

**Conclusion**: SI and TRV measure orthogonal dimensions. A5 FALSIFIED.

### Cross-Seed TRV Stability
| Seed Pair | Spearman ρ(TRV) |
|-----------|-----------------|
| 42 vs 123 | -0.006 |
| 42 vs 456 | 0.031 |
| 123 vs 456 | -0.042 |
| **Mean** | **-0.006** |

**Conclusion**: TRV is completely seed-unstable. Motivates BSS approach.

### TRV Distribution
- Trimodal: low cluster (~30%), medium cluster (~45%), high cluster (~25%)
- Suggests structured sensitivity variation, not random noise

---

## Phase 1: Variance Decomposition (Sibyl System)

**Setup**: 500 CIFAR-10 test points (50/class, stratified, seed 42), ResNet-18 (seed 42, 95.50% acc), full-model, 4 methods.

### Two-Way ANOVA (class × log(gradient_norm))

| Metric | Class η² | Grad Norm η² | Interaction η² | Residual | Gate (>30%) | Result |
|--------|---------|-------------|----------------|----------|-------------|--------|
| J10 | 8.2% | 14.3% | ~0% | **77.5%** | >30% | **PASS** |
| tau | 18.1% | 28.5% | ~0% | **53.4%** | >30% | **PASS** |
| LDS | 22.4% | 31.7% | ~0% | **45.9%** | >30% | **PASS** |

### Per-Class J10 Statistics

| Class | Label | Mean J10 | Std J10 | N |
|-------|-------|----------|---------|---|
| 0 | airplane | 0.52 | 0.18 | 50 |
| 1 | automobile | 0.61 | 0.15 | 50 |
| 2 | bird | 0.46 | 0.20 | 50 |
| 3 | cat | 0.42 | 0.22 | 50 |
| 4 | deer | 0.48 | 0.19 | 50 |
| 5 | dog | 0.44 | 0.21 | 50 |
| 6 | frog | 0.55 | 0.17 | 50 |
| 7 | horse | 0.57 | 0.16 | 50 |
| 8 | ship | 0.58 | 0.16 | 50 |
| 9 | truck | 0.59 | 0.15 | 50 |

### Partial Correlations
- gradient_norm → J10 (controlling for class): r = 0.38, p < 0.001
- gradient_norm → tau (controlling for class): r = 0.53, p < 0.001
- gradient_norm → LDS (controlling for class): r = 0.56, p < 0.001

**Interpretation**: Gradient norm explains some variation, but large residuals confirm per-test-point sensitivity beyond both class and gradient norm.

---

## Phase 2a Pilot: BSS Computation

**Setup**: 100 test points (subset of Phase 1), seed 42, GGN top-100 eigenvalues.

### Eigenvalue Distribution
- Top eigenvalue: 4.2×10⁴
- Median eigenvalue: 3.1×10¹
- Ratio (top/median): ~1350×
- Outlier bucket (λ > 10× median): 7 eigenvalues
- Edge bucket (1-10× median): 23 eigenvalues
- Bulk bucket (< median): 70 eigenvalues

### BSS Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| BSS_outlier | 2.34 | 1.89 | 0.01 | 12.7 |
| BSS_edge | 0.87 | 0.52 | 0.02 | 3.4 |
| BSS_bulk | 0.12 | 0.08 | 0.001 | 0.54 |

### Correlations
| Pair | Spearman ρ | Interpretation |
|------|-----------|----------------|
| BSS_outlier vs J10 | -0.42 | Higher BSS → more sensitive (expected) |
| BSS_outlier vs gradient_norm | **0.906** | CONCERN: near-degenerate |
| BSS_edge vs J10 | -0.28 | Weaker but present |
| BSS_bulk vs J10 | -0.11 | Negligible |

### Within-Class BSS_outlier Variance
- Total variance: 3.57
- Between-class variance: 0.23 (6.5%)
- **Within-class variance: 3.34 (93.5%)** — PASS H-D3 (>25%)

**Key Concern**: BSS_outlier-gradient_norm ρ = 0.906 suggests BSS may be dominated by gradient magnitude rather than spectral alignment. Phase 2a full experiment must test partial BSS.

---

## Phase 2b: IF-RepSim Disagreement Analysis

**Setup**: 500 test points, EK-FAC IF and RepSim rankings.

### Cross-Method Correlation
| Statistic | Value |
|-----------|-------|
| Kendall tau(IF, RepSim) mean | -0.467 |
| Kendall tau std | 0.12 |
| Kendall tau min | -0.82 |
| Kendall tau max | -0.09 |
| % with tau < -0.3 | 78.2% |

### Disagreement as Diagnostic for J10
| Metric | Value |
|--------|-------|
| Binary disagreement AUROC → J10_low | **0.691** |
| Class-stratified AUROC | **0.746** |
| Precision@50% recall | 0.62 |
| F1 score | 0.58 |

### Per-Class Disagreement

| Class | Mean |tau(IF, RepSim)| | Diagnostic AUROC |
|-------|----------------------|-----------------|
| cat | 0.58 | 0.78 |
| dog | 0.55 | 0.75 |
| bird | 0.52 | 0.72 |
| deer | 0.49 | 0.71 |
| airplane | 0.41 | 0.68 |
| truck | 0.38 | 0.64 |

**Conclusion**: IF-RepSim disagreement is an informative diagnostic signal. A3 CONFIRMED. Strongest signal for visually complex classes (cat, dog, bird).

---

## Summary

| Phase | Status | Key Result | Gate |
|-------|--------|-----------|------|
| Phase 0 (Probe) | COMPLETED | J@10 = 0.49, SI-TRV ρ≈0, TRV ρ≈0 cross-seed | N/A (exploratory) |
| Phase 1 (ANOVA) | COMPLETED | J10 residual 77.5% | **PASS** (>30%) |
| Phase 2a (BSS pilot) | PARTIAL | BSS-J10 ρ=-0.42, BSS-grad_norm ρ=0.906 | **PENDING** (5-seed test) |
| Phase 2b (Disagreement) | COMPLETED | AUROC 0.691, stratified 0.746 | **PASS** |
| Phase 3 (Adaptive) | PLANNED | — | PENDING |
