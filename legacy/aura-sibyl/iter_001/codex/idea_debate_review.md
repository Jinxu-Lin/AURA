# Codex 独立评审 - idea_debate

**评审时间**: 2026-03-17
**模型**: Codex (GPT-5)

## 评审意见

### 1. Overlooked Risks

- **Selection bias from progressive gating**: If Phase 2/3 are only run after Phase 1 passes, you create a "winner's curse" pipeline. Reported downstream gains can be inflated by design.
- **Diagnostic circularity**: BSS, IF quality, and routing features are all curvature/gradient-derived. If that shared signal is wrong, all three components can fail together while still appearing internally consistent.
- **Bucket instability risk**: Eigenvalue bucket boundaries (outlier/edge/bulk) may be unstable across seeds and checkpoints; this can make BSS variance look like signal when it is discretization noise.
- **Robustness/security blind spot**: No explicit stress test for adversarially manipulated training points or outliers, despite attribution methods being attackable in compensation-like settings.
- **Generalization risk**: The plan appears too CIFAR-10/ResNet-centric. A method that "works" there may fail on ViT, NLP fine-tuning, or diffusion-style regimes.

### 2. Assumption Vulnerabilities

- **A1: Residual variance exists and is large (>30%)**. If false, both BSS and adaptive routing lose purpose.
- **A2: Cross-seed spectral structure is alignable**. If eigenspaces rotate/mix, bucketed comparisons can be meaningless even with similar spectra.
- **A3: First-order perturbation remains informative in non-convex deep nets**. This is a known fragility point for influence methods.
- **A4: Stability implies reliability**. High cross-seed rho can still be consistently wrong w.r.t. true retraining effects.
- **A5: Routing signal is method-differential**. If BSS correlates equally with both IF and RepSim error, selector gains vanish.

### 3. Methodological Gaps

- **No ground-truth faithfulness anchor**: You need at least a small exact/near-exact retraining benchmark (leave-one/group-out) to validate that BSS predicts real attribution error, not just surrogate stability.
- **Potential ANOVA misspecification**: Class + gradient norm additive decomposition ignores interactions and nonlinear effects; residual % can be misleading.
- **Insufficient statistical plan**: Seven hypotheses with hard thresholds need power analysis, CIs, multiplicity control, and uncertainty on rho/AUROC.
- **Possible evaluation leakage**: If routing is tuned and evaluated on same test points/seeds, reported Pareto gains are optimistic.
- **Metric narrowness**: LDS-only wins are weak if top-k overlap, rank calibration, and subgroup harms are not also tracked.

### 4. Missing Alternatives / Baselines

- Strong recent baselines seem missing: **MAGIC (2025)**, **Rescaled IF (2025)**, **parameter-weighted TDA (ICLR 2026)**, **Concept Influence (2026)**.
- Classical but important controls: **TracIn**, **Representer Points**, **DataInf**.
- Add trivial-but-strong sanity baselines: random routing, always-IF, always-RepSim, and oracle routing upper bound.
- Add robustness baseline: adversarial/outlier stress tests from recent attribution attack literature.

### 5. Novelty Assessment

- **Current novelty: moderate (not high)**. The main novelty is the *integration* (gating + BSS + routing), not clearly a fundamentally new attribution estimator.
- **BSS could be publishable** if you show it predicts true retraining error across architectures/datasets and not just seed agreement.
- **Top-venue bar (NeurIPS/ICML)**: borderline today. Likely needs stronger cross-domain evidence, tighter causal validation, and clearer theory-to-practice guarantees.

### 6. Concrete Improvements (Actionable)

1. Add a **gold-standard faithfulness suite**: small-scale exact retraining for leave-one/group-out; evaluate BSS as a calibrated predictor of true attribution error.
2. Replace/supplement ANOVA with a **hierarchical mixed-effects model** (class, seed, interaction terms), and pre-register multiplicity correction across H-G1...H-F2.
3. Use **nested evaluation for routing**: train selector on one split of test points/seeds, evaluate on held-out split; report regret vs oracle.
4. Expand to at least **2 datasets + 2 model families** (e.g., CIFAR-100/Tiny-ImageNet and ResNet/ViT) before claiming general mechanism.
5. Include **2025-2026 baselines** (MAGIC, RIF, parameter-weighted TDA, Concept Influence) and adversarial stress tests; otherwise claims risk being outdated.

### Sources Referenced

- https://arxiv.org/abs/2509.23437
- https://arxiv.org/abs/2303.14186
- https://arxiv.org/abs/2310.00902
- https://arxiv.org/abs/2006.14651
- https://arxiv.org/abs/1905.13289
- https://arxiv.org/abs/2002.08484
- https://arxiv.org/abs/2504.16430
- https://arxiv.org/abs/2506.06656
- https://arxiv.org/abs/2506.05647
- https://arxiv.org/abs/2602.14869

## 评分

6/10

**理由**: Strong problem choice and good fail-fast discipline via progressive gating. However, key claims rest on fragile assumptions (residual variance existence, spectral bucket stability, first-order perturbation validity), proxy-heavy validation without ground-truth anchoring, and limited external validity (CIFAR-10/ResNet-only). The statistical plan needs power analysis and multiplicity correction for 7 hypotheses. With stronger ground-truth validation, broader baselines (including 2025-2026 methods), and tighter statistics, this could move into strong-conference territory.
