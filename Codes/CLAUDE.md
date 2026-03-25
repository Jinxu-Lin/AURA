# AURA Code — Geometric Incommensurability of Knowledge Operations

## Project Summary

Characterize the geometric relationship between knowledge editing (ROME/MEMIT) and training data attribution (TDA) directions in transformer parameter space. Core finding: TECS ~ 0 (editing and attribution are geometrically incommensurable). Six-component analysis framework.

## Direction (v3.0, NeurIPS 2026)

The project has pivoted from FM1/FM2 diagnostic framework and BSS diagnostic to **geometric incommensurability analysis**. The core contribution is the NEGATIVE result: editing and attribution directions do not align in parameter space, with dramatic subspace asymmetry (editing ~40D, attribution ~1D).

## Key Experimental Code

### TECA Pipeline (reuse from `experiments/teca/`)
- `pilot_rome.py` — ROME editing, delta_W extraction for GPT-2-XL
- `pilot_rome_validation.py` — ROME validation (efficacy checks)
- `pilot_tda_gradient_v2.py` — TDA gradient computation with BM25 retrieval
- `pilot_tecs_core.py` — TECS computation + 5 null baselines + Cohen's d
- `negative_subspace_geometry.py` — SVD, principal angles, cross-projection, null distribution
- `negative_memit_experiment.py` — MEMIT comparison
- `negative_whitening.py` — Whitening analysis (extend for full C^{-1} ablation)
- `precompute_memit_stats.py` — Covariance matrix computation

### Legacy Code (reference only)
- `../iter_001/exp/code/` — AURA Sibyl system (CIFAR-10 experiments)
- `../legacy/teca-sibyl/` — TECA Sibyl system (GPT-2-XL experiments, results in results/)

### New Code Needed
- `experiments/teca/cross_model_tecs.py` — Adapt pipeline for GPT-J, Pythia
- `experiments/teca/attribution_ablation.py` — Raw mean, RIF, SVD subspace aggregation
- `experiments/teca/layer_sweep.py` — 48-layer TECS sweep
- `core/rif_rescaling.py` — RIF leverage score rescaling (per 2506.06656)
- `experiments/toy_model/associative_memory.py` — Synthetic linear associative memory

## Data Paths

- **Models**: HuggingFace (GPT-2-XL, GPT-J-6B, Pythia-1B, Pythia-6.9B)
- **CounterFact**: Download or from existing ROME data
- **Generated data**: `_Data/` (gitignored)
- **Results**: `_Results/` (git tracked, markdown format)
- **TECA results**: `../legacy/teca-sibyl/results/` (JSON, reference)

## GPU Resources

- **xuchang0**: 4x RTX 4090 (24GB) — GPT-2-XL, Pythia-1B experiments
- **jinxulin**: 4x A6000 (48GB) — GPT-J-6B, Pythia-6.9B experiments
- **Total budget**: 40-80 GPU-hours additional

## Reproducibility

- Seeds: `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`, `torch.cuda.manual_seed_all(42)`
- CUDA deterministic: `torch.backends.cudnn.deterministic = True`
- Environment: `pip freeze > requirements.txt` before first experiment
- Record: git commit hash + GPU model + CUDA version per experiment

## Experiment Execution Order

See `experiment-todo.md` for full checklist:
1. Phase 0: Pilot (Pythia-1B TECS + whitening ablation) — GATE
2. Phase 1: Cross-model TECS + subspace + C^{-1} + toy model
3. Phase 2: Attribution ablation + layer profile
4. Phase 3: Analysis + writing

## Debug Guide

- **ROME OOM**: Reduce batch size; use gradient checkpointing; for 6.9B use A6000
- **TECS all zero**: Check delta_W and g_M norms; verify ROME actually edited the fact
- **Principal angles all 90**: Expected if TECS ~ 0 at high k; check at k=1 for any signal
- **SVD numerical issues**: Use `torch.linalg.svd` with full_matrices=False; reduce to joint subspace first

## Version Sync

After each experiment or code change:
```bash
cd /Users/jlin8272/Research/AURA
git add -A
git commit -m "implement: <description>"
git push  # if remote configured
```
