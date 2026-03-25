## [Methodologist] 方法论者视角

### 评估协议完整性

**Variance Decomposition (Completed)**:
- Data split: Clean — 500 test points from 10K test set, no overlap with training
- ANOVA design: Type I sequential SS with class entered first — appropriate given the Papyan 2020 class-structure concern
- Multiple comparisons: Three response variables (J10, tau, LDS) — Bonferroni correction applied
- **Concern**: Only 1 seed (seed 42) for full-model evaluation. Cross-seed validation not done for the variance decomposition itself. The 77.5% residual may be seed-specific.

**BSS Cross-Seed Stability (Planned)**:
- 5 seeds is sufficient for pairwise Spearman rho estimation (10 pairs)
- Gate criterion (rho > 0.5) is appropriate — well above chance, below the 0.85 that would suggest trivial correlation
- **Concern**: The 500 test points may not be enough to detect differences in cross-seed stability between BSS_outlier, BSS_edge, and BSS_bulk. Should report per-bucket rho separately.

**Adaptive Selection (Planned)**:
- LOO validation design is appropriate for compute-constrained setting
- Pareto frontier evaluation (LDS vs GPU-hours) is the right comparison framework
- **Concern**: The 2% absolute LDS improvement threshold (H-F1) may be too generous or too strict depending on the variance. Should specify as "statistically significant improvement" (paired t-test p < 0.05) rather than a fixed percentage.

### Baseline 公平性与时效性

**For BSS evaluation**:
- Gradient norm is the correct "simple baseline" — BSS must beat it
- Cross-method disagreement is the correct "strong baseline" — already achieves AUROC 0.69-0.75
- Missing: **SI (Self-Influence)** — should be included as a baseline even though pilot showed rho ~ 0 with TRV. Different question: does SI predict J10?
- Missing: **Confidence/entropy baselines** — trivially available, should be reported

**For adaptive selection**:
- IF-only and RepSim-only are correct baselines
- Naive ensemble (average rankings) is correct
- Missing: **Oracle routing** (per-test-point best method) — should be reported as upper bound
- Missing: **W-TRAK** as a uniform alternative that may dominate both IF and RepSim

### Ablation 完整性

The experiment design includes 4 BSS ablations (bucket granularity, eigenvalue count, damping sensitivity, gradient norm disentanglement). This is thorough.

**Missing ablations**:
1. **Layer selection**: BSS computed across all layers vs last-layer-only vs per-layer BSS — to verify that full-model is necessary
2. **Training data subset size**: BSS with 5K vs 50K training points — to test if BSS requires full training set access
3. **Test set selection**: BSS robustness to test set composition — is BSS stable across different 500-point samples?

### 评估协议 Overall

**Pass with minor additions**:
- Add SI, confidence, entropy as baselines for BSS evaluation
- Add oracle routing as upper bound for adaptive selection
- Specify statistical test for H-F1 instead of fixed percentage
- Report per-bucket cross-seed rho separately
- Add layer selection ablation
