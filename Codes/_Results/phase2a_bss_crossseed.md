# AURA Experiment Results: phase2a

Generated: 2026-03-27T17:15:49.056573
Dry-run: True

## Phase 2a: BSS Cross-Seed Stability

- Seeds: [42, 123]
- Test points: 20
- Damping: 0.01
- Top-k eigenvalues: 50

### Eigenvalue Spectrum Summary

| Seed | Max | Min | Mean | Std |
|------|-----|-----|------|-----|
| 42 | 4.12e-01 | 5.51e-05 | 2.70e-02 | 6.95e-02 |
| 123 | 4.08e-01 | 5.56e-05 | 2.68e-02 | 6.90e-02 |

### Cross-Seed Stability (Mean Pairwise Spearman Rho)

| BSS Variant | Mean Rho | Std |
|-------------|----------|-----|
| bss_partial | 0.9985 | 0.0000 |
| bss_ratio | 1.0000 | 0.0000 |
| bss_raw | 1.0000 | 0.0000 |

### ICC(2,1) Reliability

- bss_partial: 0.9999
- bss_ratio: 1.0000

### Within-Class Variance: 0.3099

### Gate Evaluation

| Gate | Value | Threshold | Result |
|------|-------|-----------|--------|
| bss_partial_rho | 0.9985 | 0.5 | **PASS** |
| within_class_variance | 0.3099 | 0.25 | **PASS** |
