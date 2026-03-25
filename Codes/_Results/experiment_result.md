> [ASSIMILATED: placeholder, DATE-LM experiments pending]

# Experiment Results -- AURA (TDA Diagnostic Framework)

## Status: DATE-LM Experiments Pending

The core DATE-LM experiments (2x2 ablation on Pythia-1B across 3 tasks) have not yet been executed.

## Preliminary Results (CIFAR-10/ResNet-18)

AURA's completed experiments on CIFAR-10/ResNet-18 provide preliminary small-scale evidence. See `Codes/_Results/probe_result.md` for full details.

Key numbers:
- IF-RepSim Kendall tau: -0.467 (anti-correlated)
- J10 (EK-FAC vs K-FAC): mean 0.835, residual variance 77.5%
- Per-point LDS: IF mean 0.297, RepSim mean 0.074
- Disagreement AUROC: 0.691

## Pending Experiments

1. **DATE-LM 2x2 ablation** (primary): {TRAK, IF} x {standard, contrastive} vs {RepSim, RepT} x {standard, contrastive} on Pythia-1B, 3 tasks, 5 seeds.
2. **CIFAR-10 contrastive variants** (supplementary): Contrastive-RepSim and Contrastive-RepT on existing AURA infrastructure.
3. **LoRA vs full FT comparison** on DATE-LM (FM1 artifact test).

## Timeline

Estimated 3-4 weeks from experiment start. See `research/experiment-design.md` for detailed compute budget.
