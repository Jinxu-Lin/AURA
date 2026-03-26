# AURA Code Layout

## Project Context

AURA investigates per-test-point Hessian sensitivity in Training Data Attribution (TDA).
Primary target: **Scenario B (diagnostic-only paper)** -- BSS as gradient-eigensubspace projection diagnostic.
MRC (C3) is exploratory/appendix only, gated by RepSim-wins pre-gate.

## Code Locations

### 1. Sibyl Legacy Code (READ-ONLY REFERENCE)

```
iter_001/exp/code/
├── phase0_reanalysis.py              <- Phase 0: TRV/SI reanalysis (COMPLETED)
├── phase1_train_resnet.py            <- ResNet-18 training (COMPLETED)
├── phase1_attribution_pilot*.py      <- IF/RepSim/TRAK attribution (v4 = final, COMPLETED)
├── phase1_variance_decomposition.py  <- Phase 1 ANOVA (COMPLETED)
├── phase1_verify_hessian*.py         <- Hessian verification (v4 = final, COMPLETED)
├── phase1_compile_results.py         <- Results compilation (COMPLETED)
├── phase2a_bss_compute_pilot.py      <- BSS pilot computation (COMPLETED)
├── phase1_setup_env.sh               <- Environment setup
└── phase3_loo_validation_pilot.py    <- LOO validation pilot (reference only)
```

Sibyl code paths use `/home/jinxulin/sibyl_system/projects/AURA/`. DO NOT modify.

### 2. Noesis v3 Experiment Code (WORKING CODEBASE)

```
Codes/experiments/
├── phase0_reanalysis_full.py                <- Phase 0 reanalysis (extended, COMPLETED)
├── phase1_attribution_gpu1.py               <- Phase 1 attributions GPU 1 (COMPLETED)
├── phase1_attribution_gpu2.py               <- Phase 1 attributions GPU 2 (COMPLETED)
├── phase1_combine_results.py                <- Phase 1 results merger (COMPLETED)
├── phase1_compute_ekfac_if.py               <- EK-FAC IF computation (COMPLETED)
├── phase1_retrain_with_checkpoints.py       <- Multi-seed training (COMPLETED)
├── phase1_trak_only.py                      <- TRAK computation (COMPLETED)
├── phase1_variance_decomposition_full.py    <- Phase 1 ANOVA 500pts (COMPLETED)
├── phase2a_bss_analysis.py                  <- Phase 2a BSS analysis (pilot, COMPLETED)
├── phase2b_disagreement.py                  <- Phase 2b disagreement (COMPLETED)
├── phase2b_disagreement_v2.py               <- Phase 2b v2 (COMPLETED)
├── phase2b_disagreement_full.py             <- Phase 2b full 500pts (COMPLETED)
├── phase3_uniform_baselines.py              <- Phase 3 uniform baselines (COMPLETED)
├── phase3_adaptive_strategies.py            <- Phase 3 adaptive strategies (COMPLETED)
├── auxiliary_hypotheses_pilot.py             <- Auxiliary hypothesis tests (COMPLETED)
└── teca/                                    <- LEGACY (previous direction, DO NOT USE)
```

### 3. New Experiment Scripts (TO BE WRITTEN)

```
Codes/experiments/
├── phase2a_bss_crossseed.py           <- NEW: BSS 5-seed stability (CRITICAL GATE)
├── phase2a_anova_crossseed.py         <- NEW: Per-seed ANOVA cross-validation
├── phase2a_randomized_bucket.py       <- NEW: Randomized-bucket mechanism control
├── phase2a_damping_ablation.py        <- NEW: Early damping ablation
├── phase2a_principal_angles.py        <- NEW: Outlier subspace stability
├── phase3_repsim_wins_pregate.py      <- NEW: RepSim-wins pre-gate
├── phase3_mrc_calibration.py          <- NEW: MRC soft combining (conditional)
├── phase4_confound_controls.py        <- NEW: Confound controls
├── phase5_ablations.py                <- NEW: Ablation studies
└── compile_all_results.py             <- NEW: Final results compilation
```

### 4. Core Utilities (Shared, Reusable)

```
Codes/core/
├── bss.py        <- BSS computation: eigendecomposition, bucket partitioning,
│                    partial BSS, BSS_ratio, randomized-bucket variant
├── anova.py      <- ANOVA utilities: Type I SS, R^2 decomposition, bootstrap CI
├── metrics.py    <- J10, tau, LDS, AUROC, Baselga decomposition, ICC(2,1)
├── data.py       <- CIFAR-10 loading, stratified test point sampling, seed management
├── kfac.py       <- K-FAC/EK-FAC factor computation and eigendecomposition
├── training.py   <- ResNet-18 training with checkpointing (multi-seed)
└── utils.py      <- Partial correlation, progress reporting, reproducibility setup
```

### 5. Configuration & Scripts

```
Codes/configs/
├── phase2a.yaml       <- Phase 2a: seeds, test points, eigenvalue count, damping
├── phase3.yaml        <- Phase 3: MRC calibration parameters, strategy list
├── ablations.yaml     <- Phase 5: ablation variable grid
└── base.yaml          <- Shared: paths, GPU assignment, random seeds

Codes/scripts/
├── run_phase2a.sh     <- Launch Phase 2a (all sub-experiments)
├── run_phase3.sh      <- Launch Phase 3 (conditional on Phase 2a gate)
├── run_ablations.sh   <- Launch ablations
└── plot_results.py    <- Generate all paper figures
```

## Data Locations

| Type | Path | Git |
|------|------|-----|
| Sibyl results (read-only) | `iter_001/exp/results/` | tracked |
| Sibyl checkpoints | `iter_001/exp/checkpoints/` | gitignored |
| Generated data (weights, eigenvalues, BSS arrays) | `Codes/_Data/` | gitignored |
| Experiment results (markdown summaries) | `Codes/_Results/` | tracked |
| CIFAR-10 | `~/Resources/Datasets/` or auto-download | N/A |
| Trained models | `Codes/_Data/models/` | gitignored |

## Key Dependencies

- torch >= 2.1, torchvision
- trak >= 0.3 (TRAK computation)
- pydvl >= 0.9 (EK-FAC IF, K-FAC IF via dattri)
- dattri >= 0.1 (alternative IF implementations)
- scipy, statsmodels (ANOVA, partial correlations)
- scikit-learn (logistic regression, AUROC)
- matplotlib, seaborn (visualization)
- pyyaml (config loading)

## Conventions

- All experiments: CIFAR-10 / ResNet-18
- Seeds: {42, 123, 456, 789, 1024}
- Damping: K-FAC=0.1, EK-FAC=0.01 (default; ablation varies)
- Ground truth: TRAK-50 (LOO dropped per resource constraints)
- Test points: 500 (50/class, stratified), same indices across all experiments
- Every script: deterministic seeding (torch + numpy + random + CUDA)
- Results: markdown to `_Results/`, raw arrays to `_Data/` (gitignored)
- Every code change: commit + push for multi-machine sync
- GPU assignment: specify via CUDA_VISIBLE_DEVICES at top of each script

## Critical Design Review Binding Conditions

These conditions from design_review round-1 are NON-NEGOTIABLE:

1. **Scenario B (diagnostic-only) is the primary paper target.** MRC is exploratory only.
2. **BSS narrative reframed under damping dominance.** BSS measures gradient-eigensubspace projection, NOT Hessian sensitivity. All code comments and result descriptions must use this framing.
3. **Randomized-bucket mechanism control** in Phase 2a.
4. **Per-seed ANOVA cross-validation** in Phase 2a.
5. **RepSim-wins pre-gate** (>15% points where RepSim LDS > IF LDS) before any Phase 3 MRC work.
6. **LDS is the primary metric.** J10 is secondary (ceiling effects: mean=0.9945, median=1.0).
