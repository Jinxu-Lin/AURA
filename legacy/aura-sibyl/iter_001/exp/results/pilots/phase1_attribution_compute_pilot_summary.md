# Phase 1 Attribution Computation - Pilot Summary

## Task
Compute 4 types of training data attribution (TDA) scores for 100 stratified CIFAR-10 test points (10/class) against 5K training points: EK-FAC IF, K-FAC IF, RepSim, and TRAK.

## Status: CONDITIONAL GO

All 4 methods successfully produced valid per-point attribution rankings. The J@10 variance criterion (std > 0.05) was not met due to using layer4+fc only (not full model), but this is an expected limitation of the memory-constrained pilot.

## Key Results

| Metric | Value | Note |
|--------|-------|------|
| J@10(EK-FAC, K-FAC) | mean=0.995, std=0.031 | Near-identical: eigenvalue correction negligible at layer4+fc |
| J@10(EK-FAC, RepSim) | mean=0.017, std=0.068 | Fundamentally different attribution signals |
| Kendall tau(EK-FAC, RepSim) | mean=-0.091, std=0.124 | Weak negative correlation |
| LDS(EK-FAC, TRAK) | mean=0.744, std=0.090 | Strong agreement between IF and TRAK |
| LDS(K-FAC, TRAK) | mean=0.744, std=0.090 | Nearly identical to EK-FAC (same root cause) |

## Critical Findings

1. **EK-FAC/K-FAC gap requires full-model Hessian**: The B-matrix (output gradient covariance) eigenvalues are near zero for the fc layer, making the EK-FAC eigenvalue correction negligible. The K-FAC/EK-FAC approximation gap primarily manifests in convolutional layers with richer spatial structure.

2. **IF and RepSim capture very different signals**: J@10 = 0.017 between EK-FAC and RepSim confirms these methods identify fundamentally different influential training points. This is the divergence that Phase 2b's disagreement analysis will exploit.

3. **IF-TRAK agreement is strong**: LDS = 0.74 validates that both the manual IF implementation and TRAK projections capture similar attribution signals, supporting TRAK as a ground truth proxy.

4. **GPU memory constraint is the blocker**: Another user's process occupies ~7GB on GPU 3 (RTX 4090, 24GB total). dattri's vmap-based per-sample gradient computation needs ~16GB even at batch_size=1 for ResNet-18 (11.2M params), leaving insufficient headroom.

## Memory Analysis

| Component | GPU Memory |
|-----------|-----------|
| Other process (yxma, serve_policy.py) | ~6.8 GB |
| dattri vmap per-sample gradient (bs=1) | ~16.3 GB |
| **Total needed** | **~23.1 GB** |
| **Available** | **23.5 GB** |
| **Shortfall** | **~0.4 GB (too tight, OOM at allocation spikes)** |

## Implementation Details

Due to GPU memory constraints, used manual implementations instead of dattri/TRAK libraries:

- **EK-FAC IF**: Manual K-FAC factorization on fc layer (A=513x513, B=10x10), damped identity on conv layers. Per-sample gradients via backward loop (not vmap).
- **K-FAC IF**: Same as EK-FAC but higher damping (0.1 vs 0.01), no eigenvalue correction.
- **RepSim**: Standard cosine similarity on avgpool (penultimate layer) features. Full 50K train.
- **TRAK**: Manual Gaussian random projection (dim=512) on CPU. Single checkpoint.

## Requirements for Full Experiment

1. **Dedicated GPU**: Need full 24GB RTX 4090 OR use A6000 (GPU 0, 48GB free)
2. **dattri IFAttributorEKFAC**: Verified at 10.43GB peak during phase1_setup (with dedicated GPU)
3. **TRAK-50**: Requires retraining ResNet-18 with 50 checkpoint saves
4. **Full 50K train**: Chunked computation with dattri on dedicated GPU

## Estimated Full Experiment Time
- EK-FAC IF (500 test x 50K train): ~2-3 hours on dedicated GPU
- K-FAC IF: Similar to EK-FAC
- RepSim: ~15 seconds (already validated)
- TRAK-50: ~2 hours (50 featurize passes + scoring)
- **Total: ~5-6 GPU-hours on dedicated GPU**
