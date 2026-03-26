# AURA Code Layout

## Overview

AURA has two code locations:
1. **Legacy Sibyl code**: `iter_001/exp/code/` -- original experiment implementations (read-only reference)
2. **v3 structure**: `Codes/` -- new Noesis-framework code

## v3 Code Structure

```
Codes/
├── CLAUDE.md          <- This file
├── core/              <- Reusable core components
│   ├── bss.py         <- BSS computation (eigendecomposition, bucket partitioning, partial BSS)
│   ├── anova.py       <- Variance decomposition utilities
│   ├── metrics.py     <- J10, tau, LDS, AUROC computation
│   └── data.py        <- CIFAR-10 data loading, test point sampling
├── experiments/       <- Experiment-specific wrappers
│   ├── phase1_variance_decomposition/
│   ├── phase2a_bss_stability/
│   ├── phase2b_disagreement/
│   └── phase3_mrc_pareto/
├── probe/             <- Probe experiment code (Phase 0)
├── scripts/           <- Utility scripts (training, evaluation, plotting)
├── configs/           <- Experiment configuration files
├── _Data/             <- Generated data (gitignored)
└── _Results/          <- Experiment result summaries (committed)
    ├── probe_result.md
    └── experiment_result.md
```

## Legacy Sibyl Code

```
iter_001/exp/code/
├── train_resnet.py        <- ResNet-18 training on CIFAR-10
├── compute_attributions.py <- IF/RepSim/TRAK attribution computation
├── variance_decomposition.py <- Phase 1 ANOVA
├── bss_pilot.py           <- Phase 2a BSS pilot
├── disagreement_analysis.py <- Phase 2b disagreement
└── utils/                 <- Shared utilities
```

## Key Dependencies

- torch >= 2.1, torchvision
- trak >= 0.3 (TRAK computation)
- pydvl >= 0.9 (EK-FAC IF, K-FAC IF)
- dattri >= 0.1 (alternative IF implementations)
- scipy, statsmodels (ANOVA)
- scikit-learn (logistic regression, AUROC)
- matplotlib, seaborn (visualization)

## Conventions

- All experiments use CIFAR-10/ResNet-18
- Seeds: {42, 123, 456, 789, 1024}
- External data: ~/Resources/Datasets/ (CIFAR-10 auto-downloads)
- Pretrained models: ~/Resources/Models/ or Codes/_Data/
- Results written to Codes/_Results/ as markdown
- Every code change: commit + push for multi-machine sync
