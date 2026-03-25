## [Empiricist] 实验科学性视角

### RQ 覆盖度

**RQ1 (Variance decomposition)**: Fully covered by Phase 1 experiment. ANOVA design is sound, gate criteria are pre-registered. COMPLETED with PASS.

**RQ2 (BSS diagnostic)**: Well-covered by Phase 2 experiment. Cross-seed stability test (5 seeds × 500 test points) is the right design. Gate criteria (rho > 0.5) is pre-registered.

**RQ3 (Adaptive selection)**: Covered by Phase 3 experiment. LOO validation + Pareto frontier evaluation. Conditional on RQ2 pass.

**TECA RQ (Geometric incommensurability)**: Fully covered by completed TECA pilot. 5 null baselines, Bonferroni correction, negative path analysis. COMPLETED.

No orphan experiments. No overclaimed results. Coverage is strong.

### 探针 → 完整实验衔接

**Scale gap assessment**: The main experiments are at the SAME scale as the probe (CIFAR-10/ResNet-18). This is both a strength and weakness:
- Strength: No scale-up uncertainty. Full-model experiments are natural extensions of the probe.
- Weakness: Results may not generalize to LLM scale. The paper must be honest about this limitation.

**Probe-to-full differences**:
1. Probe: 1 seed → Full: 5 seeds (addresses the key weakness)
2. Probe: last-layer only → Full: full-model (addresses hierarchy collapse)
3. Probe: 100 test points → Full: 500 test points (more statistical power)

All differences are well-identified and have clear scaling strategies.

### Baseline 公平性

The experiment design includes fair baselines for both BSS evaluation and adaptive selection. BSS is compared against:
- Free baselines: gradient norm, class label, confidence, entropy
- Moderate-cost baselines: cross-method disagreement (requires both IF and RepSim)
- No-diagnostic baseline: uniform method selection

This is a good comparison set. The key question is whether BSS provides value BEYOND the free baselines.

### Statistical Rigor

**Strengths**:
- Pre-registered gate criteria for each phase
- Bonferroni correction for multiple comparisons
- Bootstrap confidence intervals for TECA results
- Paired comparisons for method selection (same test points)

**Concerns**:
1. The 5-seed design gives only 10 pairwise correlations — limited statistical power for testing whether BSS is MORE stable than TRV. Consider also reporting: (a) mean absolute difference in BSS rankings across seeds, (b) intra-class correlation coefficient (ICC) for BSS across seeds.
2. No power analysis for detecting the rho > 0.5 threshold. With 500 test points, what is the minimum detectable rho? (Should be very small, so this is likely fine.)
3. The partial BSS analysis (regressing out gradient norm) should use BOTH linear and rank-based regression to check robustness.

### Reproducibility

**Strong**: All code exists, hardware is specified (4x RTX 4090), model architecture and training details are documented, random seeds are specified.

**Weak**: No explicit environment specification (conda env, package versions). Should document the conda environment from the Sibyl setup.

### Overall Assessment

The experimental design is thorough, well-gated, and appropriately conservative. The main limitation is CIFAR-10/ResNet-18 scale, but for the BSS diagnostic story this is acceptable — the paper's contribution is the analytical framework, not SOTA results. The TECA experiments are at a different scale (GPT-2-XL) and provide complementary evidence.

**Score**: Pass. Minor additions recommended (ICC for cross-seed stability, rank-based partial BSS, environment specification).
