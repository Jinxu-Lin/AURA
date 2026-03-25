# Contrarian Perspective: AURA — Beyond Point Estimates

**Agent**: sibyl-contrarian
**Date**: 2026-03-17
**Topic**: TDA Robustness Value (TRV) and Sensitivity-Aware Training Data Attribution

---

## Executive Summary

The AURA project has already suffered two critical empirical defeats in its own Probe: cross-seed TRV Spearman rho approximately 0, and SI-TRV correlation null. Rather than treating these as "findings to work around" (as the Innovator, Pragmatist, and Theorist perspectives do), I argue these failures are **diagnostic of deeper structural problems** that the proposed rescue angles inherit. Specifically, I challenge three widely-held assumptions in this project and in the TDA community more broadly:

1. **"Per-test-point attribution sensitivity varies meaningfully"** — The Probe's own data contradicts this; the rescue angles (spectral fingerprints, BSS, disagreement diagnostics) are increasingly elaborate attempts to extract signal from what may be noise.
2. **"Adaptive method fusion can outperform uniform baselines"** — The overhead of computing any diagnostic signal may not justify the marginal gains over simply running the best single method or a naive ensemble.
3. **"Attribution stability is a useful proxy for attribution correctness"** — Even the theoretically elegant BSS/SSP decomposition cannot bridge the fundamental gap between "consistent across approximations" and "actually correct."

My contrarian thesis: **the most impactful contribution AURA can make is a rigorous negative result paper** that demonstrates the structural reasons why per-test-point TDA diagnostics are harder than they appear, rather than yet another method paper with marginal improvements.

---

## Challenged Assumption 1: "Per-Test-Point Attribution Sensitivity Varies Meaningfully"

### The Mainstream Claim

The entire AURA project rests on the premise that different test points exhibit meaningfully different sensitivity to Hessian approximation choices. The Probe was designed to verify this. All three other perspectives (Innovator, Pragmatist, Theorist) accept this premise and propose alternative operationalizations (spectral fingerprints, cross-method disagreement, BSS) to salvage it.

### The Counter-Evidence

**The Probe killed this assumption and nobody is admitting it.**

- **Cross-seed TRV Spearman rho approximately 0**: This is not merely "TRV is model-instance-level." It means the *ranking* of test points by sensitivity is essentially random across training seeds. If the sensitivity were a genuine per-test-point property (determined by data geometry, as the root cause analysis claims), it would be stable across seeds. It is not.

- **TRV standard deviation 0.053-0.082**: Below the Fail threshold of 0.10 defined in the problem statement's own Pass criteria. The distribution is concentrated, not spread across multiple tiers.

- **Hessian eigenspectrum class-structure dominance**: Papyan (2020) and Ghorbani et al. (2019) showed that the top Hessian eigenvalues correspond to class-discriminative directions. This means the spectral structure is primarily determined by **class membership**, not individual sample properties. The Innovator's spectral fingerprint and the Theorist's BSS will likely reduce to class-conditional diagnostics — adding no information beyond knowing which class a test point belongs to. Papyan et al. (AAAI 2021, "A Deeper Look at the Hessian Eigenspectrum") explicitly showed that "the number of outliers in the Hessian eigenspectrum is approximately the number of classes" and that outlier eigenspaces are consistent across layers, suggesting a global class-level rather than sample-level structure.

- **K-FAC eigenvalue mismatch is architecture-level**: The Probe's contribution tracker (C0 update) notes that "Kronecker decomposition comes from network architecture itself (per-layer independence assumption)." The mismatch pattern between EK-FAC and K-FAC eigenvalues is determined by the Kronecker factorization structure, which is architecture-specific — not test-point-specific. All test points' gradients pass through the **same** Kronecker-factored approximation, so the amplification/compression pattern is globally shared.

### What If The Opposite Is True?

**Hypothesis (Contrarian)**: Per-test-point variation in attribution sensitivity is dominated by class membership and gradient norm, with negligible residual variation after controlling for these two factors. Specifically:

- **H-Con1**: After controlling for class label and gradient norm, the within-class TRV (or BSS, or cross-method disagreement) variance is < 20% of the total variance. The "per-test-point" signal is actually a "per-class" signal contaminated by gradient magnitude scaling.

This would explain the Probe failure: cross-seed instability arises because different seeds produce slightly different class-conditional gradient distributions, causing the *within-class ranking* to shuffle (low signal-to-noise), while the *between-class structure* is preserved but too coarse (only C = 10 groups for CIFAR-10) to serve as a useful diagnostic.

### Evidence From Literature

- **Basu et al. (2020, "Influence Functions in Deep Learning Are Fragile")**: Showed that influence function accuracy depends strongly on network architecture, depth, width, and regularization — all **global** properties. The paper found shallow networks produce reasonable estimates while deeper networks are "often erroneous," with no per-sample structure to the failure.

- **Hammoudeh & Lowd (2023, "Revisiting the Fragility of Influence Functions")**: Revisited Basu et al. and argued that the instability comes from **validation procedures** rather than the methods themselves. If the evaluation is noisy, any per-sample diagnostic built on it will inherit that noise.

- **DATE-LM benchmark (2507.09424)**: Recent comprehensive benchmarking shows no single TDA method dominates across tasks. Method performance depends on task type, model architecture, and evaluation design — again, global factors rather than per-sample properties.

### Research Direction Exploiting This Gap

**Negative Result Paper: "The Myth of Per-Sample Attribution Sensitivity"**

Design a controlled experiment that decomposes TDA sensitivity variance into:
1. Class-conditional component (between-class variance of sensitivity)
2. Gradient-norm component (correlation of sensitivity with ||grad||)
3. Residual per-sample component (after controlling for 1 and 2)

If the residual is small (< 20% of total variance), this is a publishable negative result that saves the community from pursuing per-sample diagnostics. Use ANOVA decomposition on multiple datasets (CIFAR-10, CIFAR-100, Tiny-ImageNet) with multiple TDA methods (IF, TRAK, RepSim).

**Compute cost**: ~6 GPU-hours (reuses existing attribution computations from any other angle).
**Success probability for the negative result**: 55%. The Probe data already trends this way.

---

## Challenged Assumption 2: "Adaptive Fusion Outperforms Uniform Baselines"

### The Mainstream Claim

Multiple perspectives propose some form of adaptive method selection or fusion: the Innovator's spectral routing, the Pragmatist's lightweight selector, the Theorist's semiparametric fusion. All assume that per-test-point adaptation yields meaningful gains over uniform strategies.

### The Counter-Evidence

**The "routing tax" problem**: Computing any diagnostic signal (BSS, spectral fingerprint, cross-method disagreement, or even cheap features like confidence/entropy) requires additional computation. This "routing tax" must be amortized over the marginal gain from adaptive selection. For AURA, this tax is severe:

- **BSS/Spectral Fingerprint**: Requires eigendecomposition of the GGN (even approximate Lanczos costs ~2 GPU-hours for ResNet-18). This is comparable to the cost of simply computing attributions with a better Hessian approximation (EK-FAC instead of K-FAC), which directly improves all test points uniformly.

- **Cross-method disagreement**: Requires computing attributions with 2-3 methods. But if you've already computed TRAK + IF + RepSim for the diagnostic, you can simply **ensemble all three** with fixed weights. The diagnostic step is redundant — the information it uses (multiple method outputs) already contains the improvement.

- **Self-consistency trap**: The pre-Sibyl debate synthesis correctly identified this: "the cost of computing TRV may be comparable to the cost of just using the best Hessian approximation." No rescue angle has resolved this fundamental problem.

**Empirical evidence against adaptive gains**:

- **Kowal et al. (2602.14869)**: IF-RepSim correlation is 0.37-0.45. This low correlation means the methods *do* capture different signals. But crucially, low correlation between *values* does not imply that a routing signal can reliably predict *which is more accurate* for a given point. The correlation between method disagreement and actual quality difference is a second-order effect that may be undetectable.

- **LLM routing literature analogy**: Recent work on adaptive LLM routing (BEST-Route, Route-To-Reason, 2025) shows that routing gains exist but are **task-dependent and often marginal**. In TDA, where the "task" (compute attributions for a test point) is much more uniform than diverse NLP tasks, the gains from routing are expected to be even smaller.

- **The "Daunce baseline" threat**: Daunce (ICML 2025) already provides per-sample attribution uncertainty without any Hessian approximation chain — it works by perturbing model parameters directly. If uncertainty quantification is the goal, Daunce is simpler, cheaper, and already published. AURA's adaptive fusion must beat not just naive ensemble but also Daunce's uncertainty-weighted approach.

### What If The Opposite Is True?

**Hypothesis (Contrarian)**: For standard benchmarks (CIFAR-10/ResNet-18, CIFAR-100, GPT-2), the best uniform strategy (either the single best method or a fixed-weight ensemble) matches or exceeds any adaptive strategy within the same compute budget. Specifically:

- **H-Con2**: Given a fixed compute budget C, allocating C entirely to the best single TDA method (e.g., EK-FAC IF with full compute budget) produces higher average LDS than splitting C between (a) a cheaper TDA method + (b) a diagnostic computation + (c) adaptive routing/fusion.

- **H-Con3**: A naive fixed-weight ensemble of IF + RepSim (weights 0.5/0.5) achieves LDS within 2% of any oracle-weighted adaptive fusion on standard benchmarks. The Pragmatist's own success criterion for Angle 3 is AUROC > 0.65 for method selection — even if achieved, this translates to < 3% LDS improvement given that both methods are already reasonable.

### Evidence From Literature

- **Natural W-TRAK (2512.09103)**: Achieved 68.7% certified ranking pairs by simply changing the metric space (from Euclidean to natural Wasserstein) — a **uniform** transformation applied to all test points. No per-sample adaptation needed. This suggests that **global improvements** (better metrics, better approximations) dominate per-sample adaptation.

- **ASTRA (2507.14740, NeurIPS 2025)**: Improved IF by using EK-FAC preconditioner on Neumann series iterations — again, a **uniform** improvement. Applied the same number of iterations to all test points. Won NeurIPS acceptance without any per-sample adaptation.

- **Efficient Ensembles (Deng et al., 2405.17293)**: Showed that simple ensemble strategies (Dropout, LoRA) improve TDA quality uniformly, reducing training cost by 80%. The improvement is **method-level**, not sample-level.

### Research Direction Exploiting This Gap

**"Compute-Optimal TDA: Why Uniform Beats Adaptive"**

Systematic Pareto-frontier analysis of TDA quality vs. compute budget:
1. Plot LDS vs. GPU-hours for: (a) single methods at various fidelity levels (Identity IF → K-FAC IF → EK-FAC IF → GGN IF), (b) fixed-weight ensembles (IF + RepSim, IF + TRAK), (c) adaptive fusion with diagnostic overhead included in budget.
2. Hypothesis: the Pareto frontier is dominated by uniform methods at every budget level.
3. If confirmed, this directly argues against the entire AURA adaptive fusion narrative.

**Compute cost**: ~8 GPU-hours.
**Success probability**: 50%. The single-method Pareto frontier is likely strong because better Hessian approximations have been shown to uniformly improve LDS (Hong et al., 2509.23437).

---

## Challenged Assumption 3: "Attribution Stability Is a Useful Proxy for Correctness"

### The Mainstream Claim

AURA's core narrative is that TRV (or any stability measure) serves as a diagnostic for attribution *quality*. The implicit logic: if attributions are stable across Hessian approximations, they are more likely to be correct. The Theorist frames this as BSS predicting LDS, the Pragmatist as disagreement predicting quality gaps, the Innovator as spectral fingerprint routing to the correct method.

### The Counter-Evidence

**The "stable but wrong" problem is not a corner case — it is the default.**

- **First-order linearity bias**: ALL influence function variants (regardless of Hessian approximation) share the same fundamental limitation: they linearize the loss landscape around theta*. For deep networks, this linearization is always wrong to some degree. Points where the linearization is most wrong may actually have the **most stable** IF attributions, because the gradient direction is simple (dominated by a single class-discriminative eigenvector), while points where the linearization happens to work (complex gradient structure in the bulk eigenspace) may appear **unstable** because multiple Hessian approximations interact with the complex gradient differently.

- **The contribution tracker (C4) acknowledges this**: "Quantitatively demonstrate TRV-high but attribution-incorrect cases." This is not an edge case to document — it is potentially the **majority case**. If BSS_outlier is high (test gradient concentrated in outlier eigenspace), the IF linearization is dominated by class-discriminative directions where all Hessian approximations agree (because outlier eigenvalues are well-approximated) — yielding high stability. But this class-discriminative dominance means the attribution simply reflects class membership, not actual training data influence. The attribution is stable and wrong.

- **Park et al. (2303.12922) Spearman miss-relation**: LDS (Spearman correlation) itself can assign high scores to attributions that miss the true influential set. If the evaluation metric is flawed, optimizing a diagnostic that predicts LDS is doubly flawed — you are optimizing a proxy of a proxy.

- **Attribution vs. Influence gap (Li et al., 2410.17413, G4)**: TrackStar demonstrated that "attribution is not influence" — high-attribution training points are not necessarily the ones whose removal changes model behavior. Stability of attribution rankings says nothing about whether those rankings capture actual influence.

### What If The Opposite Is True?

**Hypothesis (Contrarian)**: Attribution stability (measured by TRV, BSS, cross-method agreement, or any other metric) has **zero or negative** correlation with attribution correctness (measured by LOO ground truth or counterfactual retraining), after controlling for trivially predictive features (class match, gradient norm).

- **H-Con4**: Spearman(stability_metric, LOO_correctness | class, ||grad||) < 0.1 for any stability metric proposed by the other perspectives. The apparent correlation (if any) in unstratified analysis is entirely explained by class-conditional effects.

- **H-Con5**: In the high-stability subgroup, the fraction of "correctly attributed" test points (top-10 overlap with LOO ground truth > 30%) is **not significantly higher** than in the low-stability subgroup (one-sided Fisher exact test p > 0.05).

### Evidence From Literature

- **Bae et al. (2024)**: Showed that discrepancies between IF estimates and exact LOO retraining arise from non-convexity, initialization variance, and optimization dynamics. These factors affect all test points' attributions but do not create stable per-sample variation in error magnitude — the error is fundamentally about global landscape properties.

- **Distributional TDA (Mlodozeniec et al., 2506.12965)**: Revealed that single-run LOO correlation is "approximately 0%" — not because methods fail but because single-run evaluation has insufficient signal-to-noise ratio. If ground truth itself requires distributional treatment (averaging over training runs), then any per-sample diagnostic computed on a single model instance is fundamentally limited.

- **"Stable != Correct" in other domains**: In numerical analysis, condition number (stability) and accuracy are related but not equivalent. A well-conditioned problem can still produce wrong answers if the algorithm is systematically biased. In TDA, the "algorithm" (IF linearization) is systematically biased for all deep networks — stability of this bias across Hessian approximations is not informative about correctness.

### Research Direction Exploiting This Gap

**"When Stability Deceives: The Anti-Correlation Between Attribution Consistency and Correctness"**

1. Compute exact LOO ground truth for a small subset (100-200 test points on CIFAR-10 with 5K training set, feasible with ~83 GPU-hours on 4 GPUs).
2. Compute multiple stability metrics: TRV, BSS, cross-method disagreement, spectral fingerprint, prediction confidence.
3. Compute partial correlations controlling for class label and gradient norm.
4. Test H-Con4 and H-Con5 rigorously.

If stability-correctness correlation is indeed near zero (or negative), this is a high-impact finding that challenges the entire "reliability diagnosis" paradigm — including Daunce and BIF, not just AURA.

**Compute cost**: ~100 GPU-hours for LOO ground truth (the bottleneck), ~5 GPU-hours for stability metrics.
**Success probability**: 40%. This requires significant compute investment but the Probe results suggest the direction is promising.

---

## The Strongest Contrarian Proposal: A Rigorous Negative Results Paper

### Why This Is The Most Valuable Contribution

The TDA field is currently in an expansion phase: new methods (ASTRA, Natural W-TRAK, Daunce, BIF, AirRep) are published at every top venue, each claiming improvements on specific benchmarks. What the field **lacks** is rigorous negative results that constrain the space of what is achievable.

A paper titled **"Structural Limitations of Per-Test-Point TDA Diagnostics"** would:

1. **Save community effort**: If per-sample diagnostics are fundamentally limited by class-conditional structure, dozens of research groups pursuing per-sample reliability metrics can redirect effort toward more productive directions (better global methods, distributional approaches, task-specific evaluation).

2. **Be highly citable**: Negative result papers in ML have high impact when well-executed. "Influence Functions in Deep Learning Are Fragile" (Basu et al., 2020) has 600+ citations precisely because it constrained the field's expectations.

3. **Be honest about the Probe results**: The Probe showed TRV is unstable and SI is uninformative. Other perspectives are engineering workarounds; a negative results paper treats the Probe as the finding, not the obstacle.

4. **Complement rather than compete with AURA's other angles**: Even if some rescue angles succeed (e.g., BSS achieves cross-seed stability), the negative results paper provides context: "here's what doesn't work and why, and here's the narrow conditions under which something does work."

### Paper Structure

1. **Introduction**: TDA methods produce unreliable attributions depending on implementation choices. Recent work proposes per-test-point diagnostics. We systematically test whether such diagnostics can work.

2. **Theoretical Analysis**: Decompose attribution sensitivity into class-conditional, gradient-norm, and residual components. Show that under Hessian class-structure dominance (Papyan 2020), the residual is bounded by O(1/sqrt(C)) where C is the number of classes.

3. **Empirical Investigation**:
   - Experiment 1: ANOVA decomposition of TRV/BSS/disagreement across 3 datasets, 3 model architectures, 5 seeds. Measure class-conditional vs. residual variance.
   - Experiment 2: Compute-optimal Pareto frontier showing uniform methods dominate adaptive ones.
   - Experiment 3: Stability-correctness partial correlation after controlling for class and gradient norm.

4. **Constructive Conclusion**: Per-test-point diagnostics are limited, but *class-conditional* diagnostics are valid and cheap. Recommend: (a) report class-conditional LDS in all TDA papers; (b) use class-conditional method selection rather than per-sample routing; (c) invest compute in better global approximations rather than per-sample diagnostics.

### Experimental Plan

| Step | Description | Time | Compute |
|------|-------------|------|---------|
| 1 | Train models: ResNet-18/CIFAR-10 (5 seeds), ResNet-50/CIFAR-100 (3 seeds), ViT-Small/Tiny-ImageNet (3 seeds) | 8h | 4 GPUs |
| 2 | Compute EK-FAC IF + K-FAC IF + RepSim for 500 test points per dataset | 12h | 4 GPUs |
| 3 | Compute TRV, BSS, cross-method disagreement, spectral fingerprint per test point | 2h | 1 GPU |
| 4 | ANOVA decomposition: class vs. gradient-norm vs. residual | 30min | CPU |
| 5 | Compute TRAK ground truth for Pareto frontier analysis | 4h | 2 GPUs |
| 6 | Pareto frontier: LDS vs. compute for uniform vs. adaptive strategies | 1h | CPU |
| 7 | LOO ground truth on CIFAR-10/5K-subset for stability-correctness analysis | 20h | 4 GPUs |
| 8 | Partial correlation analysis controlling for class and gradient norm | 30min | CPU |
| **Total** | | **~48h wall** | **~80 GPU-hours** |

### Success Criteria

- Class-conditional + gradient-norm components explain > 80% of TRV/BSS/disagreement variance (H-Con1)
- Uniform methods dominate Pareto frontier at every compute budget level (H-Con2)
- Fixed-weight ensemble within 2% LDS of any adaptive strategy (H-Con3)
- Stability-correctness partial correlation < 0.1 after class/gradient-norm control (H-Con4)

### Failure Modes (When Contrarian Is Wrong)

1. **Residual per-sample variance is large (> 30%)**: This would mean the Probe failure was due to last-layer Hessian limitation or small sample size, and full-model BSS/spectral fingerprints do capture genuine per-sample structure. Probability: 30%.
2. **Adaptive fusion achieves > 5% LDS improvement over uniform**: This would justify the routing overhead, especially if the diagnostic is cheap (disagreement-based). Probability: 25%.
3. **Stability-correctness correlation is significant (> 0.3)**: This would validate the entire AURA diagnostic narrative. Probability: 20%.

If any of these failure modes materialize, the paper pivots to: "Structural limitations exist but are partially surmountable under conditions X, Y, Z" — still a valuable contribution that maps the boundary.

---

## Comparative Assessment of All Perspectives

| Criterion | Innovator | Pragmatist | Theorist | **Contrarian** |
|-----------|-----------|------------|----------|----------------|
| **Intellectual honesty about Probe failure** | Low (reframes failure as feature) | Medium (sidesteps TRV) | Medium (explains why TRV failed) | **High (takes failure seriously)** |
| **Risk of "solution in search of problem"** | High (spectral routing may add no information beyond class label) | Medium (disagreement may be uninformative) | High (theoretical framework may be elegant but empirically empty) | **Low (tests whether the problem exists at all)** |
| **Novelty if succeeds** | High | Medium | Very High | **High (negative results are undervalued)** |
| **Value if fails** | Low (yet another method that doesn't work) | Medium (descriptive findings remain) | Low (theory without empirical support) | **High (partial negative results still constrain the field)** |
| **Addresses the "compute TRV vs. use better approximation" trap** | No | No | No | **Yes (Pareto frontier analysis)** |
| **Compute cost** | 4-8 GPU-hours (pilot) | 4-5 GPU-hours (pilot) | 5-8 GPU-hours (pilot) | **~80 GPU-hours (full), ~10 GPU-hours (pilot)** |

---

## Recommended Strategy

**If AURA must produce a positive-result paper**: Pursue the Pragmatist's Angle 1 (disagreement diagnostic) as the lowest-risk option, but **include the class-conditional control experiments from H-Con1** as ablations. This way, even if the diagnostic "works," the paper honestly reports how much of the signal is class-level vs. genuinely per-sample. This is the difference between a "we propose X and it works" paper and a "we deeply understand when and why X works" paper — the latter is more publishable at top venues.

**If AURA can afford intellectual ambition**: Write the negative results paper. It will be harder to publish (negative results face editorial bias) but more impactful and more honest. Target TMLR (which explicitly welcomes negative results) or NeurIPS Datasets & Benchmarks track.

**What I would NOT recommend**: Pursuing the Theorist's Angle 3 (information-geometric decomposition) or the Innovator's Angle 2 (conformal attribution sets). Both are theoretically beautiful but require the per-sample variation assumption to hold strongly — the very assumption the Probe challenged. Building elaborate theoretical frameworks on a shaky empirical foundation is the fastest path to a rejected paper.

---

## Integration With AURA Contribution Structure

The contrarian perspective reshapes the contribution structure:

- **C0 (TRV diagnostic)**: TRV as originally defined is dead (Probe killed it). The rescue attempts (BSS, spectral fingerprint, disagreement) should be tested against the class-conditional null hypothesis before being accepted as replacements. C0 becomes: "under what conditions does per-sample diagnostic add information beyond class label?"

- **C1 (Empirical characterization)**: Strengthened, not weakened, by the contrarian perspective. The empirical characterization now includes the ANOVA decomposition and the class-conditional dominance finding — this is **more** informative than simply reporting TRV distributions.

- **C2 (SI-TRV bridge)**: Dead. The Probe killed it. The Theorist's explanation (SI collapses spectral structure) is a rationalization — the real explanation is simpler: there is insufficient per-sample signal to bridge.

- **C3 (RA-TDA fusion)**: Must survive the Pareto frontier test. If adaptive fusion cannot beat uniform methods at equal compute budget, C3 has no value regardless of how principled the fusion weights are.

- **C4 ("Stable != Correct")**: This is actually the **strongest** AURA contribution from a contrarian viewpoint. Documenting the decoupling (or anti-correlation) between stability and correctness is a genuine finding with implications for the entire field. Elevate C4 from supporting evidence to potential primary contribution.

---

## Literature Search Log

1. **Web: "influence functions are fragile" Basu 2020 follow-up 2024 2025**: Found Basu et al. (2020) original paper on IF fragility in deep learning; Hammoudeh & Lowd (2023) revisiting and attributing fragility to validation procedures; Bae et al. (2024) showing IF-LOO discrepancies from non-convexity, initialization variance, and optimization dynamics. Confirms that IF instability is global (architecture/training-level), not per-sample.

2. **Web: "data attribution methods benchmark no clear winner 2025 2026"**: Found DATE-LM (2507.09424) benchmark showing no single TDA method dominates across tasks; dattri library (NeurIPS 2024) providing unified but inconclusive comparisons. Supports H-Con2: method selection depends on global factors, not per-sample properties.

3. **Web: "Hessian eigenspectrum class structure dominates neural network"**: Found Papyan (2020), Papyan et al. (AAAI 2021) showing outlier eigenvalues correspond to class-discriminative directions with number of outliers approximately equal to number of classes; Ghorbani et al. (2019) establishing bulk+outlier spectral structure. Directly supports H-Con1: per-sample spectral variation is dominated by class membership.

4. **Web: "influence function unreliable Hessian approximation criticism 2024 2025"**: Found recent work (2409.17357, 2405.17490, 2508.07297) improving Hessian approximations globally rather than per-sample; BIF (2509.26544) providing posterior variance as alternative uncertainty estimate. Confirms trend: improvements are method-level, not sample-level.

5. **Web: "adaptive method selection routing overhead uniform baseline 2025"**: Found LLM routing literature (BEST-Route, Route-To-Reason) showing routing gains are task-dependent and often marginal; no evidence of per-sample TDA routing outperforming uniform strategies. Gap confirmed for H-Con2/H-Con3.

6. **Web: "training data attribution stability seed variance model-level 2025"**: Found Natural W-TRAK (2512.09103) achieving stability through global metric change (not per-sample); Daunce (2505.23223) providing uncertainty through model perturbation ensemble; efficient ensembles (2405.17293) improving quality uniformly. All improvements are global, supporting the contrarian thesis.

---

## Key References

**Supporting the contrarian position**:
- Basu et al. (2020), "Influence Functions in Deep Learning Are Fragile" — IF instability is architecture-level
- Hammoudeh & Lowd (2023), "Revisiting the Fragility of Influence Functions" — instability from validation procedures, not per-sample properties
- Papyan (2020), "Traces of Class/Cross-Class Structure Pervade Deep Learning Spectra" — Hessian structure dominated by class, not sample
- Papyan et al. (AAAI 2021), "A Deeper Look at the Hessian Eigenspectrum" — outlier count = class count, homogeneous across layers
- Ghorbani et al. (2019), "An Investigation into Neural Net Optimization via Hessian Eigenvalue Density" — bulk+outlier structure is global
- Park et al. (2303.12922) — LDS miss-relation weakens any stability-correctness bridge
- Li et al. (2410.17413) — attribution != influence, stability of attribution is not stability of influence
- DATE-LM (2507.09424) — no single TDA method dominates; method quality depends on task-level factors
- Mlodozeniec et al. (2506.12965) — single-run LOO correlation approximately 0%, fundamental signal-to-noise limitation

**Existing AURA evidence supporting contrarian position**:
- Probe results: cross-seed TRV rho approximately 0, SI-TRV rho approximately 0
- Probe: TRV std 0.053-0.082, below the pre-registered Fail threshold of 0.10
- Debate synthesis: 4/6 perspectives flagged H2 (IF-RepSim error independence) as weakest assumption
- Debate synthesis: Contrarian's own "W-TRAK + naive ensemble may suffice" identified as deal-breaker baseline
