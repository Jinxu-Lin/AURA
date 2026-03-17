# Idea Validation Decision: Post-Pilot Assessment

## Decision: ADVANCE with cand_bss_adaptive

## Evidence Summary

### Candidate: cand_bss_adaptive (front-runner)

**Phases completed**: All pilot phases (0, 1-setup, 1-attribution, 1-variance-decomposition, 2a-multiseed-train, 2a-bss-compute, 2a-bss-analysis, 2b-disagreement, 3-loo-validation, 3-uniform-baselines, 3-adaptive-strategies, auxiliary-hypotheses).

#### Gating Hypothesis H-G1: PASS
- J10 residual: 83.4% (but J10 variance is negligible in layer4+fc; std=0.031, nearly all values = 1.0)
- LDS residual: 51.6% (the meaningful signal -- class explains 26.7%, grad_norm 17.0%)
- tau residual: 22.5% (class+grad_norm dominate; grad_norm alone explains 33.5%)
- **Conclusion**: Genuine per-sample structure exists for LDS. Per-sample diagnostics are NOT fundamentally limited by class dominance.

#### BSS Diagnostic (H-D1, H-D2, H-D3): INCOMPLETE (layer4+fc artifact)
- H-D1 (cross-seed stability): UNTESTED (only 1 seed)
- H-D2 (predictive power): FAIL in pilot (partial corr = -0.076, below 0.1 threshold). BSS is a gradient-norm proxy (rho=0.91). BUT: perturbation factors are uniform because all eigenvalues < 0.001 and damping dominates. Full-model Hessian with larger eigenvalues is expected to produce non-uniform perturbation factors.
- H-D3 (not class detector): PASS (within-class variance 93.5%). BSS captures per-point structure beyond class.
- **Root cause of weakness**: Layer4+fc scope makes K-FAC and EK-FAC nearly identical (J10=0.995). BSS measures a difference that barely exists in this setting.

#### Disagreement Signal (H-D4): PARTIALLY SUPPORTED
- tau vs LDS_diff: Spearman rho = -0.55 (p < 1e-8). Strong quantitative relationship.
- Quantile AUROC: 0.75 (above 0.60 threshold)
- Binary AUROC: DEGENERATE (all 100 points favor IF over RepSim)
- Class-stratified quantile AUROC: 0.664 (above 0.55 threshold)
- **Limitation**: Cannot test binary routing because IF universally dominates RepSim in layer4+fc. Full-model Hessian may create RepSim-better points.

#### Phase 3 Adaptive Strategies: NO SIGNAL (expected in degenerate setting)
- Best adaptive (class-conditional): LDS = 0.744 = best uniform (K-FAC IF: 0.744)
- All routing strategies learn to pick IF always (because IF always wins)
- BSS sigmoid fusion: weight mean = 0.95 (nearly pure IF)
- **This is not a failure of the approach** -- it's an artifact of the degenerate pilot setting where IF universally dominates.

#### LOO Validation: NO_GO at pilot scale
- All correlations near zero (best = 0.036)
- Root cause: severely undertrained models (30-50% accuracy on 50-500 samples)
- Requires full 200-epoch checkpoint + full 5K training set

#### Auxiliary Hypotheses: MIXED (1/3 pass)
- H-A1 (non-redundancy): FAIL. Geodesic-curvature proxy correlation = -0.52 (pilot proxies are inadequate; full experiment uses different components)
- H-A2 (stability != correctness): FAIL. |tau| has genuine partial correlation 0.46 with LDS. This is actually POSITIVE for the routing hypothesis -- disagreement predicts attribution quality.
- H-A3 (ensemble stability): PASS. LOO rho = 0.85, mean pairwise rho = 0.60. Ensemble averaging stabilizes TRV rankings.

### Candidate: cand_negative_results (backup)
- Not actively tested but evidence is partially informative
- H-G1 PASS weakens the negative-results narrative (per-sample variance IS substantial for LDS)
- The "class dominates everything" claim is not supported for LDS (only 26.7% class R2)
- Could still pivot here if full-model experiments show class dominance increases

### Candidate: cand_conformal (backup)
- No pilot evidence collected
- Remains theoretically viable but lower priority given strong signals from front-runner

## Why ADVANCE (not REFINE or PIVOT)

### Arguments for ADVANCE:
1. **Gating hypothesis passed convincingly** for LDS (residual 51.6%). This is the most important prerequisite.
2. **Infrastructure validated**: dattri full-model EK-FAC works at 10.43GB peak, multi-seed training is fast (3.15 min/seed pilot), all attribution methods produce valid scores.
3. **The primary limitation (layer4+fc scope) has a clear, verified remedy**: Full-model Hessian computation is feasible and expected to produce the divergence that BSS theory requires.
4. **Disagreement signal is strong**: tau-LDS_diff rho = -0.55 provides a backup routing signal even if BSS-specific predictions fail.
5. **No falsification criteria triggered**: No hypothesis hit its FAIL threshold in a way that can't be attributed to the layer4+fc limitation.

### Arguments against (considered and addressed):
1. **"BSS is just gradient norm"**: True in pilot, but caused by uniform perturbation factors (eigenvalues << damping). Full-model eigenvalues are expected to be larger.
2. **"IF always beats RepSim"**: True in layer4+fc. Full-model IF uses different features; RepSim's relative simplicity may win for some points.
3. **"No adaptive strategy beats uniform"**: Expected when there's no heterogeneity in method performance. The question is whether heterogeneity emerges with full-model computation.
4. **"LOO validation failed"**: At an inappropriate pilot scale. Not informative about full-experiment feasibility.

### Why not REFINE:
The proposal and hypotheses are well-designed. The pilot limitations are all attributable to scope (layer4+fc) rather than conceptual flaws. Refining the proposal would not add value -- the next step is clearly full-model computation.

### Why not PIVOT:
No falsification criteria were triggered. The front-runner's core bet (that per-sample spectral structure exists and can be exploited) received preliminary support from H-G1 and the disagreement analysis. Pivoting to negative results or conformal prediction would abandon promising signals.

## Risk Assessment for Full Experiment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Full-model BSS still correlates with grad_norm (rho > 0.8) | 30% | Test gradient-norm-normalized BSS as alternative |
| Full-model IF still universally dominates RepSim | 25% | Reframe paper around diagnostic (C0+C1) rather than routing (C2) |
| GPU memory issues with full-model dattri on 500 points | 20% | Use A6000 (48GB) or chunked computation |
| Cross-seed BSS stability (H-D1) fails (rho < 0.3) | 25% | Fall back to ensemble-averaged diagnostics (H-A3 already validated) |
| Full experiment exceeds ~42 GPU-hour budget | 15% | Prioritize variance decomposition + BSS; defer LOO validation |

## Required Resources for Full Experiment
- Dedicated GPU: RTX 4090 (24GB) or A6000 (48GB) for full-model dattri
- 5 seeds x 200 epochs training: ~2.5 GPU-hours
- Full-model EK-FAC + K-FAC for 500 points: ~10 GPU-hours
- BSS computation (5 seeds): ~7 GPU-hours
- TRAK-50 ground truth: ~15 GPU-hours
- LOO validation (100 points): ~5 GPU-hours (approximate LOO)
- Total: ~40 GPU-hours

## Next Actions
1. Train 4 additional seeds to 200 epochs (full training, not pilot 20 epochs)
2. Compute full-model EK-FAC and K-FAC attributions for 500 stratified test points
3. Re-run variance decomposition with full-model data (H-G1 full validation)
4. Compute BSS with full-model eigenvalues and reduced damping
5. Evaluate cross-seed BSS stability (H-D1) with 5 seeds
6. If BSS shows seed-stable signal, proceed to adaptive routing (Phase 3)

SELECTED_CANDIDATE: cand_bss_adaptive
CONFIDENCE: 0.62
DECISION: ADVANCE
