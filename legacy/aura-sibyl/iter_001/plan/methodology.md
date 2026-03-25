# AURA: Methodology

## Overview

AURA investigates the reliability of Training Data Attribution (TDA) methods through spectral sensitivity analysis. The experimental program follows a **progressive gating** design: each phase must pass pre-registered criteria before subsequent phases proceed, ensuring compute is not wasted on unsupported directions.

**Three components**:
1. Attribution Variance Decomposition (gating)
2. Bucketed Spectral Sensitivity (BSS) diagnostic
3. Adaptive method selection with compute-normalized evaluation

## Model and Dataset

- **Dataset**: CIFAR-10 (standard 50K train / 10K test split)
- **Model**: ResNet-18 (11.2M parameters, fits on a single GPU)
- **Training**: Standard SGD, lr=0.1, momentum=0.9, weight_decay=5e-4, cosine annealing, 200 epochs
- **Seeds**: Primary seed 42; multi-seed experiments use {42, 123, 456, 789, 1024}
- **Subsets**: CIFAR-10/5K (5000 training samples, stratified) for LOO validation

Rationale: CIFAR-10/ResNet-18 is the standard testbed in TDA literature (Hong et al., Bae et al., Park et al.). Full-model Hessian computation is feasible (unlike ImageNet-scale), enabling ground-truth spectral analysis.

## TDA Methods and Libraries

| Method | Library | Configuration |
|--------|---------|---------------|
| K-FAC IF | pyDVL / dattri | Kronecker-factored Fisher, damping=0.1 |
| EK-FAC IF | pyDVL / dattri | Eigenvalue-corrected K-FAC, full-model |
| GGN IF | custom (torch.autograd) | Full Gauss-Newton, last 2 layers for tractability |
| RepSim | custom | Cosine similarity on penultimate-layer representations |
| TRAK-10 | trak library | 10 model checkpoints, JL dimension=4096 |
| TRAK-50 | trak library | 50 model checkpoints, JL dimension=4096 |

**Ground truth hierarchy**: LOO (exact, on 5K subset) > TRAK-50 > TRAK-10. TRAK-50 serves as primary ground truth for the 500-point evaluation set; LOO on 100 points validates TRAK quality.

## Phase 0: Probe Data Reanalysis

**Goal**: Extract maximum signal from existing Probe experiment data (3 seeds x 100 test points, CIFAR-10/ResNet-18, last-layer setting).

**Analyses**:
- Class-conditional TRV means and within-class variance
- GUM-style uncertainty budget: partition TRV variance into seed, class, and residual
- Correlation matrix: TRV vs SI vs gradient norm vs confidence vs entropy

**Compute**: 0 GPU-hours (pure reanalysis of cached data).

## Phase 1: Attribution Variance Decomposition (H-G1)

**Goal**: Determine whether genuine per-test-point variation exists beyond class membership and gradient magnitude.

**Design**:
- 500 CIFAR-10 test points (50 per class, stratified random)
- Single model (seed=42), **full-model** Hessian (not last-layer)
- Compute: EK-FAC IF, K-FAC IF, RepSim, TRAK-50 attributions for all 500 points
- Response variables:
  - J10: Jaccard@10(EK-FAC IF, K-FAC IF)
  - tau: Kendall tau(IF rankings, RepSim rankings) per test point
  - LDS: Per-point Linear Datamodeling Score (EK-FAC IF vs TRAK-50)
- Analysis: Two-way ANOVA with class (10 levels) and log(gradient_norm) as predictors. Type I sequential SS with class entered first.
- Report: partial R-squared for class, gradient_norm, interaction, and residual

**Gate**: Residual > 30% on at least 1 metric = PASS. All < 20% = STOP.

**Compute**: ~5 GPU-hours.

**Critical**: Must use full-model Hessian. The Probe used last-layer only, which collapsed the bottom 3 hierarchy levels (Diagonal ~ Damped Identity ~ Identity). Full-model preserves the full K-FAC/EK-FAC gap.

## Phase 2a: BSS Computation and Cross-Seed Stability (H-D1, H-D2, H-D3)

**Conditional on Phase 1 passing.**

**BSS Definition**:
For test point z with gradient g, eigenvalue buckets B_j:
```
BSS_j(z) = sum_{k in B_j} |1/lambda_k - 1/tilde_lambda_k| * (V_k^T g)^2
```

Buckets:
- Outlier: lambda > 100 (top ~10 eigenvalues, class-discriminative)
- Edge: 10 < lambda < 100 (transition region)
- Bulk: lambda < 10 (noise subspace)

**Design**:
- 5 ResNet-18 models (seeds 42, 123, 456, 789, 1024)
- GGN top-100 eigenvalues/eigenvectors via Kronecker-factored eigendecomposition per seed
- 300 test points (30/class), BSS computed per seed
- Cross-seed stability: mean pairwise Spearman rho of BSS_outlier rankings (10 pairs)
- Predictive power: Spearman(BSS_outlier, per-point LDS) + partial correlation controlling for class and grad norm
- Class detector test: ANOVA of BSS_outlier ~ class, report within-class / total variance

**Gate**: BSS cross-seed rho > 0.4, partial correlation > 0.1, within-class variance > 25%.

**Compute**: ~7 GPU-hours (5 model trainings already available from multi-seed, eigendecomposition ~0.5h each).

## Phase 2b: Cross-Method Disagreement Analysis (H-D4)

**Reuses Phase 1 data** (no additional GPU compute).

**Design**:
- 500 test points with IF and RepSim attributions from Phase 1
- Label each point: "IF better" vs "RepSim better" by LDS comparison
- Compute Kendall tau(IF rankings, RepSim rankings) per point
- Class-stratified AUROC: mean AUROC across 10 classes

**Gate**: Global AUROC > 0.60 AND class-stratified AUROC > 0.55.

## Phase 3: Pareto Frontier and Adaptive Selection (H-F1, H-F2)

**Conditional on Phase 2a or 2b passing.**

**Uniform strategies** (each at known compute cost):
- Identity IF, K-FAC IF, EK-FAC IF, RepSim
- TRAK-10, TRAK-50
- Naive 0.5:0.5 IF+RepSim ensemble

**Adaptive strategies**:
- (a) BSS-guided routing: route to IF when BSS_outlier is high, RepSim otherwise
- (b) Disagreement-guided routing: route based on IF-RepSim Kendall tau
- (c) Class-conditional selection: lookup table by class
- (d) Feature-based selector: logistic regression on (grad_norm, confidence, entropy, disagreement)

**BSS fusion**:
```
w(z) = sigmoid(-a * ||BSS(z)||_1 + b)
score(z) = w(z) * IF(z) + (1 - w(z)) * RepSim(z)
```
Calibration: 300 points (train), 200 points (test).

**LOO Validation**:
- CIFAR-10/5K subset, 100 test points
- Exact leave-one-out retraining
- Validates TRAK-50 ground truth quality

**Evaluation**:
- Pareto frontier: mean LDS vs GPU-hours for all strategies
- Class-stratified AUROC as mandatory control (adaptive AUROC must exceed 0.55 within classes)
- Oracle gap closure: (BSS_fusion - naive_ensemble) / (oracle - naive_ensemble)

**Gate**: Adaptive > uniform by > 2% LDS at same compute budget.

**Compute**: ~30 GPU-hours (dominated by LOO retraining on 5K subset).

## Baselines

| Baseline | Purpose |
|----------|---------|
| K-FAC IF (uniform) | Standard cheap TDA method |
| EK-FAC IF (uniform) | Gold-standard Kronecker IF |
| RepSim (uniform) | Non-gradient baseline |
| TRAK-50 (uniform) | Ensemble-based TDA |
| 0.5:0.5 ensemble | Naive fusion baseline |
| Class-conditional selector | Tests whether class label alone suffices |
| Random routing | Sanity check that routing signal > random |

## Metrics

| Metric | Description | Used In |
|--------|-------------|---------|
| Jaccard@10 | Overlap of top-10 attributed training points | Phase 1 (J10) |
| Kendall tau | Rank correlation between two attribution methods | Phase 1, 2b |
| LDS (Spearman) | Linear Datamodeling Score against ground truth | All phases |
| Partial R-squared | Variance explained after controlling for covariates | Phase 1 |
| Spearman rho | Cross-seed ranking stability | Phase 2a |
| AUROC | Predictive discrimination of routing signal | Phase 2b, 3 |
| GPU-hours | Compute cost for Pareto analysis | Phase 3 |

## Software Requirements

```
torch>=2.1
torchvision
trak>=0.3
pydvl>=0.9
dattri>=0.1
scipy
statsmodels  # ANOVA
scikit-learn
matplotlib
seaborn
```

## Expected Visualizations

- **Table 1**: Variance decomposition results (class R2, grad_norm R2, residual R2) across 3 response variables
- **Figure 1**: BSS outlier heatmap across test points, colored by class (shows within-class variation)
- **Figure 2**: Cross-seed BSS stability scatter plots (seed_i vs seed_j rankings) with Spearman rho annotations
- **Figure 3**: Pareto frontier plot (LDS vs GPU-hours) showing uniform and adaptive strategies
- **Figure 4**: Ablation bar chart (BSS routing vs disagreement routing vs class-conditional vs random)
- **Table 2**: Main results: LDS by method, with compute budget and class-stratified AUROC
- **Figure 5**: BSS fusion weight distribution and per-class calibration curves
- Architecture diagram: BSS computation pipeline (Hessian -> eigendecomposition -> bucketing -> per-point sensitivity)
