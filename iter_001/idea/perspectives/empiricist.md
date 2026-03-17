# Empiricist Perspective: AURA — Beyond Point Estimates

**Agent**: sibyl-empiricist
**Date**: 2026-03-17
**Topic**: TDA Robustness Value (TRV) and Sensitivity-Aware Training Data Attribution

---

## Executive Summary

The Probe results provide a clear empirical verdict: the original TRV definition (Jaccard@k across Hessian approximation tiers) has **three empirical failures** — cross-seed instability (Spearman rho ~ 0), insufficient per-point variance (std 0.053-0.082 vs. 0.15 threshold), and null SI-TRV correlation. As an experimentalist, I refuse to treat these failures as "problems to engineer around." Instead, I ask: **what experimental designs would produce unambiguous, reproducible evidence for or against per-test-point attribution sensitivity?** Every proposal must specify exact falsification criteria, confounders to control, and the minimum effect size that would justify the computational overhead.

My central thesis: **the most valuable experiment AURA can run right now is not another rescue attempt for TRV, but a controlled decomposition of attribution variance** that cleanly separates class-conditional effects, gradient-norm effects, and genuine per-sample residuals. This decomposition determines whether *any* per-sample diagnostic (TRV, BSS, spectral fingerprint, cross-method disagreement) can work — and saves months of effort if the answer is no.

---

## Angle 1: Attribution Variance Decomposition — The Experiment That Must Come First

### Motivation

Every other perspective (Innovator, Pragmatist, Theorist, Interdisciplinary) proposes a new operationalization of per-test-point sensitivity. But none of them tests the **foundational assumption** that per-test-point variation exists beyond class membership and gradient magnitude. The Contrarian correctly raised this concern (H-Con1), but proposed a large-scale negative-result study (~80 GPU-hours). I propose a **surgical 3-hour experiment** that answers the same question.

### Core Hypothesis

**H-Emp1**: After controlling for class label and gradient norm (log-scale), the residual per-test-point variance in attribution sensitivity (measured by any of: Jaccard@10 between EK-FAC and K-FAC, cross-method TRAK-IF disagreement, or RepSim-IF disagreement) accounts for **more than 30% of total variance**. If residual variance is < 20%, per-sample diagnostics add negligible information beyond class-conditional method selection.

### Why 30% Is The Right Threshold

- At 30% residual variance, a per-sample diagnostic (even a perfect one) can explain ~0.55 correlation with the full sensitivity signal (sqrt(0.30) ~ 0.55). This is the minimum useful regime for routing.
- Below 20%, a class-conditional lookup table (10 entries for CIFAR-10) captures > 80% of the information. No per-sample diagnostic can justify its compute cost.
- The Probe data hints at the answer: TRV distributions differ across seeds (Level 1 ranges from 38% to 65%), suggesting model-instance noise, but within-seed TRV has a three-modal structure that could be class-driven or genuinely per-sample.

### Experimental Design

**Population**: 500 test points from CIFAR-10 (50 per class, stratified random sampling from correctly classified test set).

**Treatments**: Compute attributions under 3 methods on the same model (seed 42):
1. EK-FAC IF (via Hong et al. code or pyDVL)
2. K-FAC IF (same codebase, different approximation)
3. RepSim (penultimate layer cosine similarity, ~50 lines)

**Response variables** (per test point):
- J10: Jaccard@10 between EK-FAC and K-FAC attributions (Hessian sensitivity)
- D_IF_Rep: Kendall tau distance between IF (EK-FAC) and RepSim rankings (cross-method disagreement)
- LDS_IF: per-test-point LDS of EK-FAC IF against TRAK-50 pseudo ground truth

**Covariates** (per test point):
- Class label (categorical, 10 levels)
- log(gradient norm) at the trained model
- Prediction confidence (softmax probability of true class)
- Prediction entropy
- Loss value

**Analysis**:
1. **Two-way ANOVA**: J10 ~ class + log(||grad||) + class:log(||grad||) + residual
2. **Partial R-squared**: Compute R^2 for class alone, log(||grad||) alone, and the full model. Residual variance = 1 - R^2_full.
3. **Hierarchical decomposition**: Use sequential sum of squares (Type I) with class entered first, then gradient norm, to get unambiguous variance attribution.
4. **Robustness check**: Repeat the ANOVA on D_IF_Rep and LDS_IF to confirm the decomposition is not metric-specific.

**Controls**:
- **Stratified sampling**: Exactly 50 points per class eliminates class imbalance artifacts.
- **Single seed**: Deliberately single-seed to avoid conflating cross-seed variance with per-point variance. The question is "within a single model, is per-point sensitivity predictable beyond class?" Cross-seed stability is a separate question.
- **Multiple response variables**: If all three (J10, D_IF_Rep, LDS_IF) show < 20% residual, the conclusion is robust across metrics.

| Step | Description | Time | Compute |
|------|-------------|------|---------|
| 1 | Train 1 ResNet-18 on CIFAR-10 (seed 42) | 30min | 1 GPU |
| 2 | Compute EK-FAC IF for 500 test points (Hong et al. code) | 1.5h | 1 GPU |
| 3 | Compute K-FAC IF for same 500 test points | 1h | 1 GPU |
| 4 | Compute RepSim for 500 test points | 10min | 1 GPU |
| 5 | Compute TRAK-50 pseudo ground truth for 500 test points | 45min | 2 GPUs |
| 6 | Extract covariates (grad norm, confidence, entropy, loss) | 10min | 1 GPU |
| 7 | ANOVA + partial R-squared analysis | 15min | CPU |
| **Total** | | **~4h** | **~5 GPU-hours** |

**Pilot (15 min)**: Use the existing Probe data (100 ID test points, 3 seeds). Compute class-conditional TRV means for each seed. If between-class TRV variance > 60% of total TRV variance in all 3 seeds, the residual is likely too small for per-sample diagnostics. This pilot uses zero additional compute.

### Success Criteria

| Metric | Pass | Borderline | Fail |
|--------|------|------------|------|
| Residual variance (J10) | > 30% | 20-30% | < 20% |
| Residual variance (D_IF_Rep) | > 30% | 20-30% | < 20% |
| Residual variance (LDS_IF) | > 30% | 20-30% | < 20% |
| Class alone R^2 | < 50% | 50-70% | > 70% |
| At least 1 metric passes | Required | — | All fail = stop |

### Falsification Protocol

**If all three metrics show residual < 20%**: Report as a major negative finding. This falsifies not just TRV but *all* per-sample diagnostic proposals (BSS, spectral fingerprint, disagreement, lightweight selector). Recommend class-conditional method selection as a simpler, sufficient alternative. This is a publishable finding at TMLR or NeurIPS (negative results are undervalued but highly citable — cf. Basu et al. 2020, 600+ citations).

**If residual is 20-30%**: Borderline regime. Per-sample diagnostics may add marginal value. Proceed cautiously with Angles 2-3, but frame any improvement as "incremental beyond class-conditional baseline" rather than "transformative diagnostic."

**If residual > 30%**: Genuine per-sample structure exists. Proceed to Angles 2-3 to determine which diagnostic best captures it.

---

## Angle 2: Controlled Disagreement Experiment — Testing Whether Cross-Method Disagreement Is Informative

### Motivation

The Pragmatist proposes cross-method disagreement (TRAK-IF-RepSim) as a per-test-point diagnostic. This is the simplest and cheapest proposal. But the Pragmatist's own success criterion (Spearman(disagreement, LDS gap) > 0.3) is weak — a correlation of 0.3 explains only 9% of variance, which may not justify the overhead of computing multiple attribution methods.

I propose a more rigorous test with tighter controls and a higher bar.

### Core Hypothesis

**H-Emp2**: Per-test-point cross-method disagreement (Kendall tau between IF and RepSim attributions) predicts **which method is more accurate** for that test point (measured by LDS against TRAK ground truth), with AUROC > 0.70 — not just 0.65 as the Pragmatist suggests. The higher threshold is needed because at AUROC 0.65, the expected LDS improvement from oracle routing is < 2% absolute (given both methods are reasonable), which is within noise.

**H-Emp3**: The predictive power of disagreement (H-Emp2) remains significant (AUROC > 0.60) **after controlling for class label**. If AUROC drops below 0.55 after class stratification, the disagreement signal is merely a proxy for class-conditional method preference, not a genuine per-sample diagnostic.

### Why The Class-Conditional Control Is Non-Negotiable

Consider: IF and RepSim may systematically disagree on certain classes (e.g., classes with fine-grained features vs. classes with holistic textures). If so, per-sample disagreement simply reflects class membership, and a lookup table mapping class -> preferred method is strictly simpler, cheaper, and more reliable. The Pragmatist's proposal does not include this control. The Contrarian correctly anticipated this risk (H-Con1) but proposed a large-scale investigation. My design embeds the control directly into the primary experiment.

### Experimental Design

**Pre-requisite**: Angle 1 must show residual variance > 20% (otherwise stop here — class-conditional selection suffices).

**Data**: Same 500 test points and attributions from Angle 1.

**Additional computation**:
- Label each test point: "IF better" (LDS_IF > LDS_RepSim) vs. "RepSim better" (LDS_RepSim > LDS_IF)
- Compute Kendall tau(IF ranking, RepSim ranking) per test point = disagreement signal
- Extract features: disagreement, |disagreement|, confidence, gradient norm, entropy, loss

**Analysis**:
1. **Global AUROC**: Logistic regression predicting method superiority from disagreement alone. Must exceed 0.70.
2. **Class-stratified AUROC**: Within each class, compute AUROC. Report mean and min across classes. Mean must exceed 0.60.
3. **Incremental AUROC**: Compare (a) class-only model (10-level categorical) vs. (b) class + disagreement model. The AUROC improvement from adding disagreement must be > 0.05 (from likelihood ratio test).
4. **Effect size of adaptive selection**: Compute average LDS of oracle-selected method vs. always-IF vs. always-RepSim vs. class-conditional selection vs. disagreement-guided selection. The disagreement-guided selector must improve over class-conditional by > 1.5% absolute LDS.

| Step | Description | Time | Compute |
|------|-------------|------|---------|
| 1 | Per-test-point LDS for RepSim (from Angle 1 TRAK ground truth) | 15min | CPU |
| 2 | Per-test-point Kendall tau(IF, RepSim) | 10min | CPU |
| 3 | Logistic regression + AUROC analysis | 10min | CPU |
| 4 | Class-stratified AUROC | 10min | CPU |
| 5 | Adaptive selection LDS comparison | 10min | CPU |
| **Total** | **Marginal over Angle 1** | **~1h** | **0 GPU-hours** |

### Success Criteria

| Metric | Pass | Borderline | Fail |
|--------|------|------------|------|
| Global AUROC (disagreement -> method choice) | > 0.70 | 0.60-0.70 | < 0.60 |
| Mean class-stratified AUROC | > 0.60 | 0.55-0.60 | < 0.55 |
| Incremental AUROC over class-only | > 0.05 | 0.02-0.05 | < 0.02 |
| Adaptive LDS improvement over class-conditional | > 1.5% absolute | 0.5-1.5% | < 0.5% |

### Falsification Protocol

**If global AUROC < 0.60**: Cross-method disagreement is uninformative for method selection. This falsifies the Pragmatist's Angle 1 and the Innovator's spectral routing (which uses a more expensive version of the same signal). Report as negative result.

**If global AUROC > 0.70 but class-stratified AUROC < 0.55**: The disagreement signal is entirely class-driven. Recommend class-conditional method selection. This is a constructive negative result: "you don't need a per-sample diagnostic, just measure which method is best per class on a calibration set."

**If both pass**: Genuine per-sample routing signal exists. Proceed to Angle 3 for the full adaptive framework.

---

## Angle 3: Pareto Frontier Analysis — Does Adaptive Attribution Beat Uniform Investment?

### Motivation

Even if per-sample disagreement is informative (Angle 2 passes), it may not be *useful* in practice. The Contrarian (H-Con2) correctly noted that the compute spent on diagnostic signals could instead be spent on better global methods. This is the "routing tax" problem. I propose a direct Pareto frontier comparison.

### Core Hypothesis

**H-Emp4**: At a fixed compute budget of C GPU-hours, an adaptive strategy (compute cheap IF + RepSim + disagreement diagnostic, then route) achieves higher mean LDS than any uniform strategy (compute the best single method with full budget C). The adaptive strategy must Pareto-dominate the uniform frontier by > 2% absolute LDS at the same compute budget.

**H-Emp5**: The adaptive LDS advantage persists on at least 2 of 3 test datasets (CIFAR-10, CIFAR-100, one text classification dataset) to rule out benchmark overfitting.

### Why This Is The Deal-Breaker Experiment

If H-Emp4 fails, the entire AURA adaptive framework has no practical value — you are always better off investing compute in a better Hessian approximation (EK-FAC instead of K-FAC) or a larger ensemble (TRAK-100 instead of TRAK-50). This is the experiment that every reviewer will ask about, and it is better to know the answer before writing the paper.

### Experimental Design

**Uniform strategies (points on the Pareto frontier)**:
- Identity IF: ~0.1 GPU-hours, lowest quality
- K-FAC IF: ~0.5 GPU-hours
- EK-FAC IF: ~1.5 GPU-hours
- RepSim: ~0.2 GPU-hours
- TRAK-10: ~0.3 GPU-hours
- TRAK-50: ~1.5 GPU-hours
- Naive ensemble (IF + RepSim, fixed 0.5/0.5): ~1.7 GPU-hours (IF cost + RepSim cost)
- Naive ensemble (IF + TRAK-10, fixed 0.5/0.5): ~0.8 GPU-hours

**Adaptive strategies**:
- Disagreement-guided (Angle 2): RepSim + K-FAC IF + routing = ~0.8 GPU-hours
- Class-conditional selection: RepSim + K-FAC IF + class lookup = ~0.7 GPU-hours (slightly cheaper, no per-point computation)
- Disagreement-guided with EK-FAC: RepSim + EK-FAC IF + routing = ~1.8 GPU-hours

**Evaluation**: Mean LDS across 500 test points (against TRAK-50 ground truth for CIFAR-10, against LOO for CIFAR-10/5K-subset).

**Confounders to control**:
1. **Ground truth quality**: TRAK-50 is a noisy proxy. For 100 test points on CIFAR-10 with 5K training subset, also compute exact LOO (feasible: 5000 retraining runs x 30s = ~42 GPU-hours, parallelizable on 4 GPUs in ~10h). Compare LDS rankings under both ground truth sources.
2. **Metric sensitivity**: Report both Spearman LDS and Jaccard@10 agreement with ground truth. If conclusions differ, the evaluation metric is confounded (as warned by Park et al. 2303.12922).
3. **Compute measurement**: All compute budgets measured in actual wall-clock GPU-hours on RTX 4090, not theoretical FLOPs. This ensures fair comparison.

| Step | Description | Time | Compute |
|------|-------------|------|---------|
| 1 | Compute all uniform strategy attributions for 500 test points | 4h | 4 GPUs |
| 2 | Compute adaptive strategy attributions + routing | 2h | 2 GPUs |
| 3 | LOO ground truth for 100 test points on CIFAR-10/5K | 10h | 4 GPUs |
| 4 | LDS computation under both ground truths | 30min | CPU |
| 5 | Pareto frontier plotting + statistical comparison | 30min | CPU |
| **Total** | | **~17h wall** | **~30 GPU-hours** |

**Pilot (1 hour)**: Compute Identity IF, K-FAC IF, RepSim, TRAK-10 for 100 test points on CIFAR-10/5K (cheaper training set). Plot LDS vs. compute for uniform methods. If EK-FAC IF already dominates all other methods at its compute budget (no method at lower compute is within 5% LDS), adaptive selection has very limited headroom.

### Success Criteria

| Metric | Pass | Borderline | Fail |
|--------|------|------------|------|
| Adaptive LDS > best uniform at same budget | > 2% absolute | 1-2% | < 1% |
| Adaptive dominates on >=2/3 datasets | Yes | 1/3 | 0/3 |
| LOO-based LDS confirms TRAK-based rankings | Spearman > 0.7 | 0.5-0.7 | < 0.5 |
| Adaptive > naive ensemble (fixed 0.5/0.5) | > 1.5% | 0.5-1.5% | < 0.5% |

### Falsification Protocol

**If adaptive < uniform at all compute budgets**: Kill Phase 2 entirely. Report as a strong negative result: "compute is better spent on global approximation quality than per-sample routing." This directly answers the Contrarian's strongest objection and is itself a publishable finding.

**If adaptive > uniform but < naive ensemble**: Per-sample routing adds no value over simply combining methods with fixed weights. Kill the routing/diagnostic component but retain the ensemble contribution (which is trivial and not novel).

**If adaptive > naive ensemble by > 1.5%**: RA-TDA has genuine incremental value. Proceed to full paper writing with this as the primary experimental evidence.

---

## Cross-Cutting Experimental Concerns

### Confounder Inventory

Every experiment proposed by other perspectives has uncontrolled confounders. I catalog the most serious ones:

| Confounder | Affected Proposals | Control Required |
|------------|-------------------|------------------|
| **Class membership** | All per-sample diagnostics (TRV, BSS, spectral fingerprint, disagreement) | Class-stratified analysis (Angles 1-2) |
| **Gradient norm** | SI, BSS, spectral fingerprint | Partial correlation controlling for ||grad|| |
| **Training seed variance** | TRV, any single-seed diagnostic | Multi-seed + ensemble averaging (Probe already showed rho ~ 0) |
| **Ground truth noise** | All LDS-based evaluations | LOO ground truth on small subset (Angle 3) |
| **Evaluation metric choice** | All LDS-based comparisons | Report both Spearman LDS and Jaccard@k (Angle 3) |
| **Hessian approximation code quality** | All IF-based methods | Use 2+ independent implementations (pyDVL + Hong et al.) |
| **Last-layer vs. full-model** | All Probe-based conclusions | Full-model Hessian in Angle 1 (essential) |
| **Dataset specificity** | All single-dataset results | Multi-dataset validation in Angle 3 |

### The Full-Model Hessian Gap

The Probe used **last-layer only** Hessian computation (512->10, 5130 params). This is a severe limitation:

1. Last-layer Hessian is nearly diagonal (hence Diagonal ~ Damped Identity ~ Identity in the Probe), collapsing 3 of 5 hierarchy levels.
2. Hong et al. (2509.23437) showed the EK-FAC -> K-FAC eigenvalue mismatch (the largest single error source, 41-65%) operates on the **full Kronecker structure across all layers**, not just the last layer.
3. Any experiment claiming to measure "Hessian approximation sensitivity" on last-layer only is measuring an attenuated signal.

**Recommendation**: Angle 1 must use full-model Hessian approximation. If Hong et al.'s code supports per-test-point attribution extraction (their API may only expose aggregate LDS), invest the engineering time to add this interface. If not feasible, use pyDVL's EK-FAC implementation (which computes per-test-point attributions by default) or dattri's IF module.

**Cost implication**: Full-model EK-FAC IF is ~3-5x more expensive than last-layer IF for ResNet-18 (from ~1.5h to ~5-8h for 500 test points). Budget accordingly.

### The TRAK Ground Truth Problem

Multiple proposals (Pragmatist Angle 1, Angle 3, my Angle 2-3) use TRAK scores as "ground truth" for evaluating other methods. But TRAK itself is an approximation:

1. **TRAK-50 vs. Datamodel**: Park et al. (2303.12922) showed TRAK-50 has Spearman correlation ~0.75 with true Datamodel scores on CIFAR-10. This means ~44% of TRAK-50's variance is noise.
2. **Per-test-point TRAK quality varies**: Some test points have high TRAK signal-to-noise, others low. Using TRAK as ground truth introduces systematic bias toward test points where TRAK happens to be accurate.
3. **TRAK evaluates TRAK-like methods favorably**: TRAK and IF share the same gradient-based paradigm. Using TRAK as ground truth may systematically favor IF over RepSim.

**Mitigation in my design**: Angle 3 includes exact LOO ground truth on a small subset (100 test points, CIFAR-10/5K). This provides an unbiased validation of whether TRAK-based LDS rankings agree with true LOO-based rankings. If Spearman < 0.5, all TRAK-based evaluation is suspect.

### Hyperparameter Sensitivity in TDA Evaluation

Recent work by Wang et al. (2505.24261, "Taming Hyperparameter Sensitivity in Data Attribution") demonstrated that TDA evaluation results are highly sensitive to hyperparameter choices (damping coefficient, number of iterations, projection dimension). Any experiment must report sensitivity to these hyperparameters, not just the best configuration.

**Control**: For each IF computation, sweep damping coefficient over {0.01, 0.1, 1.0, 10.0} and report the range of resulting LDS. If the range exceeds 5% absolute, the "optimal" method depends more on hyperparameter tuning than on the method itself — undermining any adaptive selection claim.

---

## Recommended Execution Order

### Phase 0: Zero-Compute Pilot (30 minutes)

Re-analyze existing Probe data (100 test points, 3 seeds):
1. Compute class-conditional TRV means per seed.
2. Estimate between-class TRV variance / total TRV variance.
3. If > 60% in all 3 seeds: **strong evidence for class dominance** (supports fast-failing Angle 1).
4. If < 40% in any seed: residual per-sample structure may exist, proceed with Angle 1.

### Phase 1: Variance Decomposition (Angle 1, ~5 GPU-hours)

This is the **gating experiment**. Run it first. If it shows residual < 20%, stop all per-sample diagnostic development and redirect to:
- Class-conditional method selection (trivial to implement)
- Negative-result paper on structural limitations of per-sample TDA diagnostics

### Phase 2: Conditional on Phase 1 Pass (Angle 2, ~0 additional GPU-hours)

Uses same data. Run the controlled disagreement experiment with class-stratified AUROC. If class-stratified AUROC < 0.55, fall back to class-conditional selection.

### Phase 3: Conditional on Phase 2 Pass (Angle 3, ~30 GPU-hours)

The expensive Pareto frontier experiment. Only run if Phases 1-2 confirm that per-sample adaptation has genuine headroom. Includes LOO ground truth validation.

### Total Budget

| Scenario | GPU-Hours | Wall Time |
|----------|-----------|-----------|
| Phase 0 only (fast fail) | 0 | 30 min |
| Phase 0 + Phase 1 | 5 | 5h |
| Phase 0 + Phase 1 + Phase 2 | 5 | 6h |
| Phase 0 + Phase 1 + Phase 2 + Phase 3 | 35 | 20h |

This is a progressive investment strategy: most of the compute is committed only after the foundational assumptions are validated.

---

## Integration with Other Perspectives

### What I Endorse

- **Pragmatist Angle 1 (Disagreement Diagnostic)**: Sound idea but needs the class-conditional control I add in Angle 2. Without it, any positive result is suspect.
- **Contrarian's negative-result thesis**: Intellectually honest and potentially high-impact. My Angle 1 provides the surgical version of their proposed investigation.
- **Theorist's BSS**: Elegant theoretical framework, but must survive Angle 1's variance decomposition. If class label + gradient norm explain > 80% of BSS variance, the theoretical beauty is empirically empty.

### What I Challenge

- **Innovator Angle 1 (Spectral Fingerprint Routing)**: The hypothesis that "eigenvalue-magnitude bucket energy is stable across seeds" (H-Inn1) is theoretically grounded (Papyan 2020) but needs to pass the class-conditional null. If spectral fingerprint reduces to class membership (as the Contrarian predicts), the routing signal adds nothing beyond knowing the class label. My Angle 1 directly tests this.

- **Pragmatist Angle 3 (Lightweight Selector)**: A logistic regression using (confidence, loss, gradient norm, entropy, margin) to predict method quality. The risk: these features are all class-correlated. A logistic regression trained on CIFAR-10 will learn "class X prefers IF" through these features as proxies for class membership. **Required control**: Train the selector on 8 classes, evaluate on the held-out 2 classes. If AUROC drops below 0.55 on unseen classes, the selector is memorizing class-method associations, not learning generalizable per-sample signals.

- **Interdisciplinary Angle 1 (Hampel's GES)**: The analogy to robust statistics is mathematically precise but computationally vacuous. Computing the "ensemble TRV" (Hampel's asymptotic influence function analogue) requires M=10 models, which is 10x the cost of a single TRV. The theoretical guarantee is real but the practical value is unclear — my Angle 3 will determine whether the compute investment is justified.

- **Theorist Angle 2 (Semiparametric Fusion)**: Requires IF and RepSim to share a common estimand. As the Theorist acknowledges, they do not — IF estimates counterfactual loss change, RepSim estimates representation similarity. The semiparametric framework cannot be applied without first establishing what is being estimated. This is a pre-requisite that no amount of mathematical sophistication can bypass.

### Mapping to AURA Contributions

| Contribution | Angle 1 Impact | Angle 2 Impact | Angle 3 Impact |
|-------------|----------------|----------------|----------------|
| C0 (TRV diagnostic) | Determines if per-sample TRV adds value beyond class label | Tests disagreement as alternative C0 | Determines if C0 has practical value |
| C1 (Empirical characterization) | Provides the ANOVA decomposition as a core empirical finding | Characterizes method disagreement structure | Characterizes compute-quality tradeoffs |
| C3 (RA-TDA fusion) | Gating: if residual < 20%, C3 is dead | Tests one specific fusion strategy | Determines if C3 beats uniform baselines |
| C4 ("Stable != Correct") | Directly tests stability-correctness link via LDS stratification | Tests disagreement-correctness link | Tests whether stability-based routing improves actual quality |

---

## Key Open-Source Resources

**Libraries to use (all validated for CIFAR-10/ResNet-18)**:
- [TRAIS-Lab/dattri](https://github.com/trais-lab/dattri): Unified TDA benchmark, IF/TRAK/TracIn (NeurIPS 2024). Use for standardized evaluation.
- [MadryLab/trak](https://github.com/MadryLab/trak): TRAK scores with custom CUDA kernel. Use for ground truth computation.
- [aai-institute/pyDVL](https://github.com/aai-institute/pydvl): IF with EK-FAC, K-FAC, CG, Arnoldi, Nystrom. Use for multi-approximation IF.
- Probe code: `~/Research/AURA/codes/probe_experiment/` (already validated, reuse training pipeline + attribution extraction).

**Critical reference papers**:
- Hong et al. (2509.23437): Hessian hierarchy evidence. Full-model chain code needed.
- Park et al. (2303.12922): TRAK + LDS evaluation. Ground truth methodology.
- Kowal et al. (2602.14869): IF-RepSim correlation 0.37-0.45. Baseline for disagreement.
- Wang et al. (2505.24261): Hyperparameter sensitivity in TDA. Evaluation methodology.
- DATE-LM (2507.09424): Multi-task TDA benchmark. External validation source.
- Basu et al. (2020): IF fragility. Context for negative results.
- Papyan (2020): Hessian class-structure dominance. Supports class-conditional null hypothesis.

---

## Literature Search Log

1. **Web: "training data attribution evaluation methodology ablation study benchmark pitfalls 2025 2026"** — Found DATE-LM (2507.09424) as unified LLM TDA benchmark; AblationBench for automated ablation planning; user-centered TDA critique (2409.16978). Confirmed: no prior work systematically decomposing TDA sensitivity variance into class vs. per-sample components. Gap validated for Angle 1.

2. **Web: "influence function Hessian approximation sensitivity per-sample stability evaluation 2025 2026"** — Found Hong et al. (2509.23437), ASTRA (2507.14740), BIF (2509.26544), Hessian-free IF (2405.17490), GRASS (2505.18976). All improve Hessian approximation quality globally, none provide per-sample sensitivity decomposition. Confirmed: global improvements dominate the literature, per-sample analysis is the gap.

3. **Web: "cross-method disagreement diagnostic uncertainty quantification data attribution 2025 2026"** — No results specific to TDA cross-method disagreement. Ensemble disagreement is used in general ML uncertainty quantification but not applied to TDA method selection. Gap confirmed for Angle 2.

4. **Web: "dattri benchmark TDA evaluation leaderboard TRAK influence function RepSim comparison results 2025 2026"** — Found dattri (NeurIPS 2024) benchmark results: TRAK-50 dominates non-linear models, IF methods infeasible at scale. LoRIF (2601.21929) for scalable IF. Confirmed TRAK as practical ground truth source. Note: dattri does not include RepSim in its benchmark — RepSim evaluation must be done independently.

5. **Web: "training data attribution reproducibility confounders seed variance evaluation pitfalls retraining ground truth 2025"** — Found Wang et al. (2505.24261) on hyperparameter sensitivity in TDA evaluation. Ground truth computation requires 1000 model retraining for GPT-2. Confirmed: evaluation methodology is itself a confounder that must be controlled. Essential citation for Angle 3's LOO validation.

---

## Summary of Falsification Criteria

| Hypothesis | Experiment | Falsification Condition | Consequence if Falsified |
|------------|-----------|------------------------|-------------------------|
| H-Emp1: Residual per-sample variance > 30% | Angle 1 ANOVA | Residual < 20% for all 3 metrics | **Kill all per-sample diagnostics** |
| H-Emp2: Disagreement AUROC > 0.70 | Angle 2 logistic regression | AUROC < 0.60 | Kill disagreement-based routing |
| H-Emp3: Class-stratified AUROC > 0.60 | Angle 2 stratified analysis | Class-stratified AUROC < 0.55 | Disagreement is class proxy; use class lookup |
| H-Emp4: Adaptive > uniform at same budget | Angle 3 Pareto frontier | Adaptive < uniform at all budgets | **Kill Phase 2 (RA-TDA)** |
| H-Emp5: Multi-dataset generalization | Angle 3 multi-dataset | Fails on 2/3 datasets | Overfitting to CIFAR-10 artifacts |

**The empiricist's commitment**: every positive claim must survive these falsification tests. Any result that does not is reported as a negative finding, not swept under the rug. The most valuable experiments are the ones that can produce unambiguous failures.
