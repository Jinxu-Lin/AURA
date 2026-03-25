---
version: "1.0"
entry_mode: "assimilated"
iteration_major: 1
iteration_minor: 0
---

# Experiment Design: AURA

## §1: Variance Decomposition Experiment (COMPLETED)

**Corresponds to**: method-design.md §Component 1

### Setup
- **Data**: CIFAR-10, 500 test points (50 per class, stratified sampling, seed 42)
- **Model**: ResNet-18 (seed 42, 95.50% test accuracy, standard CIFAR-10 training)
- **Methods**: EK-FAC IF, K-FAC IF, RepSim, TRAK-50
- **Attribution scope**: Full-model (all layers), top-5000 training points per test point

### Metrics
- **J10**: Jaccard@10 between EK-FAC IF and K-FAC IF top-10 attributions
- **tau**: Kendall rank correlation between IF (EK-FAC) and RepSim attribution rankings (top-500)
- **LDS**: Linear Datamodeling Score (per-test-point), using leave-one-out retraining ground truth

### Analysis
- Two-way ANOVA: class (10 levels) × log(gradient_norm) (continuous, binned into quartiles)
- Partial correlations: gradient_norm → J10, controlling for class
- Within-class variance decomposition

### Gate Criterion
- Residual variance > 30% on at least one metric

### Results

| Metric | Class η² | Gradient Norm η² | Interaction η² | Residual | Gate |
|--------|---------|------------------|----------------|----------|------|
| J10 | 8.2% | 14.3% | ~0% | **77.5%** | **PASS** |
| tau | 18.1% | 28.5% | ~0% | **53.4%** | **PASS** |
| LDS | 22.4% | 31.7% | ~0% | **45.9%** | **PASS** |

**Conclusion**: Per-test-point attribution sensitivity is a real phenomenon beyond class and gradient-norm confounds.

---

## §2: BSS Cross-Seed Stability (IN PROGRESS)

**Corresponds to**: method-design.md §Component 2

### Setup
- **Data**: Same 500 test points as §1
- **Models**: 5 ResNet-18 trained with seeds {42, 123, 456, 789, 1024}
  - Seed 42: trained (95.50% test acc)
  - Seeds 123, 456, 789, 1024: training in progress
- **Eigendecomposition**: GGN top-100 eigenvalues per seed via Kronecker-factored eigendecomposition
- **BSS computation**: Three buckets per test point per seed
  - Outlier: λ > 10× median eigenvalue
  - Edge: 1× to 10× median
  - Bulk: < 1× median

### Primary Analysis
- Cross-seed BSS_outlier ranking: Spearman ρ for all (5 choose 2) = 10 seed pairs
- Cross-seed BSS_edge and BSS_bulk rankings: Same analysis
- Comparison: Cross-seed BSS ρ vs cross-seed TRV ρ (expected: BSS >> TRV)

### Gate Criteria
- **H-D1**: Cross-seed BSS_outlier Spearman ρ > 0.5 (mean across 10 pairs)
- **H-D2**: Partial correlation of BSS_outlier with J10, controlling for gradient norm, > 0.1
- **H-D3**: BSS_outlier within-class variance > 25% (non-degenerate)

### Gradient Norm Disentanglement (Critical Ablation)
- Compute partial BSS: regress BSS_outlier on ||g_z||², take residuals
- Test partial BSS cross-seed stability (ρ > 0.5)
- Test partial BSS correlation with J10 (> 0.1)
- Alternative: BSS ratio = BSS_outlier / BSS_total (scale-invariant)

### Pilot Results (100 points, 1 seed)
- BSS_outlier within-class variance: 93.5% (PASS H-D3)
- BSS_outlier-gradient_norm ρ: 0.906 (CONCERN — motivates disentanglement)
- BSS_outlier-J10 correlation: -0.42 (promising — higher BSS → more sensitive)

### Compute Budget
- Eigendecomposition: ~0.5 GPU-hours × 5 seeds = 2.5 GPU-hours
- BSS computation: ~0.1 GPU-hours × 5 seeds = 0.5 GPU-hours
- Model training (4 remaining seeds): ~4 GPU-hours
- **Total: ~7 GPU-hours**

---

## §3: Adaptive Selection & Pareto Frontier (PLANNED)

**Corresponds to**: method-design.md §Component 3

### Setup
- **Data**: CIFAR-10, 5000 training subset, 100 test points for LOO validation
- **Attribution methods**: EK-FAC IF, RepSim
- **BSS**: Pre-computed from §2

### Strategies Under Test

| Strategy | Description | Compute Cost |
|----------|-------------|-------------|
| IF-only | EK-FAC IF for all points | 1× (baseline) |
| RepSim-only | RepSim for all points | 0.3× |
| TRAK-50 | TRAK with 50 models | 5× |
| W-TRAK | Natural W-TRAK (Grosse et al.) | ~2× |
| Naive ensemble | Average IF + RepSim rankings | 1.3× |
| BSS-guided | Route IF/RepSim by BSS_outlier threshold | 0.7-1.3× |
| Disagreement-guided | Route by |tau(IF, RepSim)| | 1.3× (need both) |
| Class-conditional | Per-class BSS threshold | 0.7-1.3× |
| Feature-based | Lightweight classifier on [BSS, grad_norm, class] → {IF, RepSim} | 0.7-1.3× |

### Evaluation Protocol
1. **LOO validation** (100 held-out points): For each strategy, compute per-point LDS using leave-one-out ground truth
2. **Threshold selection**: BSS/disagreement thresholds selected by LOO cross-validation (5-fold)
3. **Pareto frontier**: Plot mean LDS vs GPU-hours for all strategies
4. **Statistical test**: Paired t-test on per-point LDS (strategy vs IF-only baseline)

### Gate Criterion
- **H-F1**: At least one adaptive strategy achieves > 2% absolute LDS improvement over best uniform strategy at the same compute budget

### Compute Budget
- LOO ground truth: ~2 GPU-hours (100 points × 100 retrained models, cached from Phase 1)
- Strategy evaluation: ~2 GPU-hours (attributions already computed)
- W-TRAK baseline: ~2 GPU-hours
- **Total: ~6 GPU-hours**

---

## §4: Confound Controls

### Training Variance Control
- 5-seed stability check for all metrics (not just BSS)
- Gate: J10, tau, LDS rankings have Spearman ρ > 0.6 across seeds
- If metric rankings are seed-unstable, per-point analysis is meaningless

### OOD Confound Control
- Include 20 OOD test points (from CIFAR-100, similar classes)
- Measure TRV and BSS on OOD points
- Gate: Correlation(TRV, OOD_flag) < 0.8 (ensure TRV is not just measuring OOD-ness)

### Compute Budget Normalization
- All strategy comparisons compute-normalized (LDS per GPU-hour)
- Report absolute LDS AND cost-adjusted LDS

### Metric Gaming
- LDS is a linear metric — supplement with non-linear metrics:
  - Jaccard@10 (top-k overlap)
  - Kendall tau (full ranking)
  - NDCG@10 (graded relevance)

---

## §5: Ablation Structure

### BSS Bucket Granularity
| Configuration | Buckets | Rationale |
|--------------|---------|-----------|
| 2-bucket | Outlier / Non-outlier | Simplest possible |
| 3-bucket (default) | Outlier / Edge / Bulk | Matches spectral theory |
| 5-bucket | Outlier / Upper-edge / Lower-edge / Upper-bulk / Lower-bulk | Finer decomposition |

### Eigenvalue Count
| Configuration | Top-K | Rationale |
|--------------|-------|-----------|
| Top-50 | 50 | Fewer eigenvalues, faster, may miss edge effects |
| Top-100 (default) | 100 | Balance of coverage and cost |
| Top-200 | 200 | More coverage, captures bulk better |

### Damping Sensitivity
- K-FAC damping values: {1e-3, 1e-4, 1e-5, 1e-6}
- Measure how BSS changes with damping (sensitive → less reliable diagnostic)

### Gradient Norm Disentanglement
| Configuration | Method | Expected Outcome |
|--------------|--------|------------------|
| Raw BSS | No correction | High ρ with grad_norm (~0.9) |
| Partial BSS | Regress out ||g_z||² | Lower ρ, test if still diagnostic |
| BSS ratio | BSS_outlier / BSS_total | Scale-invariant, should decorrelate |
| Normalized BSS | BSS / ||g_z||² | Direct normalization |

---

## Timeline

| Phase | Status | GPU-hours | Calendar |
|-------|--------|-----------|----------|
| Phase 1 (Variance Decomposition) | COMPLETED | ~8 | Done |
| Phase 2a (BSS Stability) | IN PROGRESS | ~7 | 1 week (waiting on model training) |
| Phase 2b (Disagreement) | COMPLETED | ~2 | Done |
| Phase 3 (Adaptive Selection) | PLANNED | ~6 | 1 week after Phase 2a |
| Ablations | PLANNED | ~5 | Parallel with Phase 3 |
| **Total** | | **~28** | ~2-3 weeks remaining |

GPU budget: 42 hours total, ~20 hours used, ~22 remaining. Comfortable margin.
