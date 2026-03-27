# Pilot Verification Result

> Generated: 2026-03-27
> Config: configs/pilot.yaml
> Environment: CPU (no GPU), Python 3.13.12, PyTorch

---

## Phase 0: Sanity Checks

### 0a. Overfit Check
- **Config**: 32 CIFAR-10 samples, Adam lr=0.001, 50 epochs
- **Result**: Loss 14.38 -> 0.024, accuracy 18.8% -> 100%
- **Status**: **PASS** -- ResNet-18 can perfectly memorize small subsets

### 0b. Gradient Check
- **Config**: 1 forward-backward pass, 16 CIFAR-10 samples
- **Result**: All 62 parameters have valid, non-zero, non-NaN gradients
- **Status**: **PASS**

### 0c. Shape Check (Full Pipeline)
- Model forward: (4, 3, 32, 32) -> (4, 10)
- K-FAC factors: A_cov (513, 513), B_cov (10, 10)
- Top-50 eigendecomposition: eigenvalues (50,), range [1.73e-04, 5.27e-01]
- Per-sample gradients: (10, 5130) for fc layer (512*10 weight + 10 bias)
- Gradient projections: (10, 50)
- BSS: total (10,), outlier (10,), edge (0,), bulk (49,)
- BSS partial (gradient-norm corrected): (10,)
- BSS ratio: (10,)
- **Status**: **PASS** -- all pipeline stages produce correct shapes

### 0d. Unit Tests
- 31/31 tests pass (test_anova, test_bss, test_data, test_integration, test_kfac, test_metrics, test_training, test_utils)
- Integration test covers full pipeline: model -> K-FAC -> BSS -> metrics -> ANOVA
- **Status**: **PASS**

---

## Phase 1: Pilot Training

- **Config**: pilot.yaml (5 epochs, seed 42, full CIFAR-10)
- **Result**: 72.87% test accuracy after 5 epochs (17.1 min on CPU)
- **Note**: Low accuracy expected -- full 200 epochs needed for 94-96%. Pilot only validates training infrastructure.
- **Status**: **PASS** -- training loop functional, checkpoint saved

---

## Phase 2: Pilot BSS Pipeline (dry-run)

### Phase 2a: BSS Cross-Seed Stability
- **Config**: 1 seed (42), 20 test points, top-50 eigenvalues
- **K-FAC factors**: computed successfully on 5K training subset
- **Eigenvalue range**: [9.43e-06, 4.14e-04] (consistent with expected Kronecker product scale)
- **BSS values**:
  - bss_outlier: [4.11e-08, 2.88e+02], no NaN
  - bss_partial: [-9.90e+01, 1.80e+02], no NaN
  - bss_ratio: [2.94e-05, 9.76e-01], no NaN
  - bss_total: [1.13e-04, 3.02e+02], no NaN
- **Bucket partition**: 1 outlier, 0 edge, 49 bulk (50 eigenvalues)
- **Cross-seed analysis**: skipped (needs >= 2 seeds, pilot has 1)
- **Status**: **PASS** -- per-seed pipeline runs without error

### Phase 2a Augmented: Controls & Diagnostics
- Randomized-bucket control: loaded cached eigenvalues successfully
- **Status**: **PASS** -- pipeline functional

### Phase 2b: Disagreement Analysis
- Placeholder data (20 points), pipeline execution validated
- **Status**: **PASS** -- pipeline functional

### Phase 3 Pre-Gate: RepSim-Wins Check
- Placeholder data (20 points), pipeline execution validated
- **Status**: **PASS** -- pipeline functional

### Phase 4: Confound Controls
- Class-stratified AUROC computed per seed
- Partial correlations computed
- **Status**: **PASS** -- pipeline functional

---

## Pilot Summary

| Check | Status | Notes |
|-------|--------|-------|
| Overfit | PASS | Loss -> 0, 100% accuracy on 32 samples |
| Gradient flow | PASS | All 62 params have valid gradients |
| Shape consistency | PASS | Full pipeline shapes verified |
| Unit tests | PASS | 31/31 pass |
| Training infra | PASS | Checkpoint saved, 72.87% in 5 epochs |
| BSS pipeline (per-seed) | PASS | K-FAC -> eigen -> projections -> BSS chain works |
| Evaluate phases | PASS | All phases (2a, 2a-aug, 2b, 3-pregate, confound) run |

**Overall Pilot Verdict: PASS**

All code infrastructure is verified functional on CPU. Ready for GPU execution of full experiments.

### Needs GPU Environment
- Full 200-epoch training (seeds 789, 1024)
- Phase 2a with 5 seeds x 500 test points
- Full Kronecker eigendecomposition (top-100)
- Ablation experiments

---

## Re-Validation (2026-03-27, post-scaffold)

### Integration Tests
- `pytest tests/test_integration.py -v`: **3/3 passed** (2.63s)
  - test_full_forward_backward: PASS
  - test_all_components_independently: PASS
  - test_memory_footprint: PASS

### Evaluate Pipeline Re-Run
- `evaluate.py --config configs/pilot.yaml --phase phase2a --dry-run`: **PASS**
  - K-FAC factors computed, eigenvalue range [9.47e-06, 4.18e-04]
  - BSS (raw, partial, ratio) computed for 20 test points
  - Single-seed limitation correctly handled (cross-seed skipped)
- `evaluate.py --config configs/pilot.yaml --phase confound --dry-run`: **PASS**
  - Class-stratified AUROC: 1.0000 (trivially perfect on 20 points, expected)
  - Partial correlations computed without error

### Baseline Config Verification
- `baseline_uniform.yaml --phase phase2a --dry-run`: **PASS**
  - Seed 42 processed, seed 123 correctly skipped (no checkpoint)
  - Pipeline runs with strategy-comparison config structure
- `baseline_gradnorm.yaml --phase phase2a --dry-run`: **PASS**
  - Gradient norm diagnostic baseline config created and verified
  - Config loads correctly with base.yaml inheritance

---

### Known Limitations (CPU pilot)
- Training only 5 epochs (72.87%) vs expected 94-96% at 200 epochs
- Only 1 seed (cross-seed stability not testable)
- Only 20 test points (not 500)
- Only top-50 eigenvalues (not top-100)
