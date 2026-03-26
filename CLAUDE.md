# CLAUDE.md — AURA

## Project Overview

**AURA** (Adaptive Unified Robust Attribution) investigates per-test-point Hessian sensitivity in Training Data Attribution (TDA) using Bucketed Spectral Sensitivity (BSS) on CIFAR-10/ResNet-18.

- **Topic**: Per-test-point Hessian sensitivity diagnosis via BSS
- **Problem**: TDA methods give different attribution rankings depending on the Hessian approximation (K-FAC vs EK-FAC vs GGN), but no per-test-point diagnostic exists. Prior diagnostics (TRV) are cross-seed unstable (rho ~ 0); SI is orthogonal (rho ~ 0).
- **Approach**: (1) Variance decomposition confirming per-point variation exists (77.5% residual J10); (2) BSS diagnostic using eigenvalue-magnitude buckets (stable via RMT) instead of eigenvector directions; (3) MRC soft combining guided by BSS + disagreement signals.
- **Target**: NeurIPS 2026
- **Researcher**: Jinxu Lin

## Resources

- **GPU**: 4x RTX 4090 via SSH (xuchang0)
- **GPU Budget**: ~42 GPU-hours total (~20 used, ~22 remaining)
- **Noesis path**: ~/Research/Noesis

## Current State

- **Active Module**: Research (implement phase)
- **Pipeline Status**: `pipeline-status.json` -> research module
- **Research Status**: `Docs/research-module-status.json` -> implement phase
- **Paper Status**: `Papers/paper-status.json` -> P3 (critique)

### Completed
- Phase 0: Probe reanalysis (TRV cross-seed rho ~ 0, SI orthogonal, J@10 degrades 1.0 -> 0.45)
- Phase 1: Variance decomposition (J10 residual 77.5%, tau residual 22.5%, LDS residual 51.6%) -- ALL GATES PASS
- Phase 2a pilot: BSS computation (within-class CV=93.5%, BSS-gradient_norm rho=0.906)
- Phase 2b: IF-RepSim disagreement (Kendall tau=-0.467, AUROC=0.691, class-stratified=0.746)

### In Progress
- Phase 2a full: BSS cross-seed stability (5 seeds, 500 points)
- Partial BSS (regress out gradient norm)

### Planned
- Phase 3: MRC soft combining + Pareto frontier

## Directory Structure

```
AURA/
├── CLAUDE.md                    <- This file
├── research/                    <- v3 research documents
│   ├── problem-statement.md     <- Gap + RQs + attack angle
│   ├── method-design.md         <- C1/C2/C3 method + experiment design
│   ├── experiment-design.md     <- Detailed experiment plan
│   └── contribution.md          <- Contribution tracker
├── Codes/                       <- v3 code structure
│   ├── CLAUDE.md                <- Code layout guide
│   ├── core/                    <- Reusable core components
│   ├── experiments/             <- Experiment-specific wrappers
│   ├── probe/                   <- Probe experiment code
│   ├── scripts/                 <- Utility scripts
│   └── _Results/                <- Experiment results
│       ├── probe_result.md      <- Probe + Phase 0-1-2a-2b results
│       └── experiment_result.md <- Formal experiment results
├── iter_001/                    <- Legacy Sibyl system code + results
│   ├── idea/                    <- Proposals, hypotheses, problem statements
│   ├── plan/                    <- Methodology
│   └── exp/                     <- Code + results
│       ├── code/                <- Sibyl-era experiment code
│       └── results/             <- Phase 0/1/2a/2b result summaries
├── Docs/
│   ├── project.md               <- Project overview (init module output)
│   ├── init-module-status.json
│   └── research-module-status.json
├── Papers/                      <- Paper pipeline
│   ├── outline.md
│   ├── sections/                <- Individual paper sections
│   ├── paper.md                 <- Merged paper
│   ├── critique/                <- P3 critique
│   ├── review.md                <- P5 review
│   └── paper-status.json
├── Reviews/                     <- Debate records
├── pipeline-status.json
├── iteration-log.md
└── pipeline-evolution-log.md
```

## Key Constraints

- All experiments run on CIFAR-10/ResNet-18 (tractable for full-model Hessian analysis)
- Progressive gating: Phase 1 must pass before Phase 2, Phase 2 before Phase 3
- BSS uses eigenvalue-magnitude buckets (outlier/edge/bulk), NOT eigenvector directions
- Must disentangle BSS from gradient norm (partial BSS after regression)
- W-TRAK is mandatory baseline
- Report all negative results honestly (TRV cross-seed failure, SI orthogonality)

## Code Conventions

- Legacy Sibyl code in `iter_001/exp/code/` -- read-only reference
- New v3 code in `Codes/` following deep/shallow decoupling: `core/` for reusable, `experiments/` for wrappers
- External data: `~/Resources/Datasets/` (CIFAR-10), `~/Resources/Models/` (pretrained)
- Generated data: `Codes/_Data/`
- All code changes: commit + push to GitHub for multi-machine sync
