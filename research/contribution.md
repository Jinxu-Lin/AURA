---
version: "2.0"
created: "2026-03-16"
last_modified: "2026-03-25"
---

# Contribution Tracker: AURA

## Contribution List

| # | Contribution | Type | Status | Dependencies |
|---|-------------|------|--------|-------------|
| C0 | **Attribution Variance Decomposition**: First systematic ANOVA decomposing TDA sensitivity into class, gradient-norm, and residual per-test-point components. Residual J10 = 77.5%, proving per-point diagnostics can work. | Empirical finding | **CONFIRMED** (Phase 1) | None |
| C1 | **Bucketed Spectral Sensitivity (BSS)**: Theoretically grounded per-test-point diagnostic using eigenvalue-magnitude buckets (outlier/edge/bulk) instead of seed-unstable eigenvector directions. Grounded in RMT and operator perturbation theory. | Diagnostic tool | **TESTING** (Phase 2a) | C0 pass |
| C2 | **MRC Soft Combining**: BSS-guided adaptive method selection between IF and RepSim with Cauchy-Schwarz optimal weights. Includes Pareto frontier evaluation against 11 strategies. | Method innovation | **PLANNED** (Phase 3) | C1 pass |
| C3 | **Negative Results (SI/TRV)**: (a) Cross-seed TRV instability (rho ~ 0), (b) SI-TRV null correlation (rho ~ 0), (c) honest reporting of variance decomposition including tau failure. Constrains community expectations about per-sample TDA diagnostics. | Negative results | **CONFIRMED** (Phase 0) | None |

## Evolution History

### Startup (2026-03-16)
- C0 was "TRV diagnostic" (Jaccard@k stability)
- C2 was "SI-TRV theory-empirical bridge"
- C3 was "RA-TDA adaptive fusion"

### Post-Probe Pivot (2026-03-17)
- TRV cross-seed rho ~ 0 --> pivot from TRV to BSS (eigenvalue buckets)
- SI-TRV rho ~ 0 --> SI abandoned as proxy, becomes negative result
- RA-TDA paused --> redesigned as MRC soft combining

### Post-Phase 1 (2026-03-18)
- ANOVA confirms per-point variation (77.5% residual J10) --> C0 becomes variance decomposition
- tau residual only 22.5% --> honest reporting, not a universal finding
- BSS pilot shows within-class CV = 93.5% (not a class detector) but gradient-norm correlation concern

### Current (v2.0, 2026-03-25)
- C0 confirmed, C1 testing, C2 planned, C3 confirmed
- BSS gradient-norm correlation (rho=0.906) being addressed via partial BSS

## Publication Value Assessment

| Dimension | Rating | Evidence |
|-----------|--------|----------|
| Novelty | **Medium-High** | No prior work on per-test-point Hessian sensitivity diagnosis; BSS is new; orthogonal to Daunce/BIF |
| Significance | **Medium-High** | Serves all TDA method users; variance decomposition is foundational |
| Conference fit | **High** | NeurIPS 2026 accepts meta-methodology + evaluation contributions |

## Contribution Ceiling by Scenario

| Scenario | Contributions | Ceiling |
|----------|--------------|---------|
| All gates pass | C0 + C1 + C2 + C3 | Strong NeurIPS paper |
| BSS stable, fusion marginal | C0 + C1 + C3 | NeurIPS poster / TMLR |
| BSS unstable | C0 + C3 | TMLR (negative results) |

## Metadata
- **Target**: NeurIPS 2026
- **Last updated**: 2026-03-25
- **Current phase**: Research module, implement (Phase 2a full experiment)
