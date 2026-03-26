# External AI Review — RS (Strategic Review, Round 2) — AURA

**Reviewer**: External AI (Codex / Independent)
**Date**: 2026-03-16
**Artifact reviewed**: `research/problem-statement.md` v1.1 (rs_revise) + `research/contribution.md`
**Score**: 5 / 10

---

## Overall Impression

AURA tackles a real and underappreciated structural problem in TDA: that aggregate evaluation metrics obscure per-sample reliability variation. The problem framing is technically sophisticated and the literature coverage is unusually thorough for a pre-probe stage. However, the project's strategic viability currently hinges almost entirely on a single empirical bet — whether TRV variance across test points is large enough to matter — and the paper-level contribution story has a circularity problem: the most compelling claim (RA-TDA adaptive fusion) depends on a chain of three sequential empirical pass conditions, each of which may not hold.

---

## The Blind Spot Report

**1. The "diagnosis without ground truth" problem is unaddressed.**
TRV measures agreement across Hessian approximations, not agreement with the true influence. A high-TRV test point means "all cheap approximations agree" — but if the ground truth is the exact Hessian (H), and EK-FAC/K-FAC both diverge from H in the same direction for a given test point, TRV will be high while the attribution is wrong. The paper acknowledges the "stable != correct" concern but treats it as a secondary empirical validation task (step 8 in the probe). In fact, this is a foundational epistemological issue: TRV is a proxy for agreement among approximations, not a proxy for correctness. Without establishing that the TRV-high set reliably overlaps with the "attribution close to exact-H" set, the diagnostic value of TRV is unclear. The probe's LDS comparison (TRV-high vs TRV-low) is necessary but not sufficient — LDS itself is a known noisy proxy (the document acknowledges this, §1.2 point 3). There is a risk of building a diagnostic on top of a diagnostic, both of which are approximate.

**2. The adaptive fusion framing assumes IF and RepSim are interchangeable in the low-TRV regime, but their estimands differ fundamentally.**
The document correctly notes that "IF captures loss sensitivity, RepSim captures semantic similarity" and that they are "complementary rather than competing." But the proposed fusion strategy implicitly assumes that in low-TRV regions (where IF is unreliable due to Hessian approximation), RepSim provides a useful substitute. This is not obvious: RepSim answers a different question entirely. Substituting a representation-similarity score for a counterfactual LOO score in the low-TRV regime does not produce a better estimate of counterfactual influence — it produces a *different* answer to a different question. The document acknowledges this by retreating from "Doubly Robust estimation" to "diagnostic-guided adaptive ensemble," but the practical claim ("RA-TDA outperforms naive 0.5:0.5 ensemble on LDS") is now harder to motivate theoretically. Why should the ensemble weighted by a Hessian-approximation-reliability signal improve LDS scores if the two methods do not share an estimand?

**3. The choice of CIFAR-10/ResNet-18 as the probe setting may systematically bias the conclusion toward a positive result.**
ResNet-18 on CIFAR-10 is a relatively well-conditioned system where Hessian approximations are known to work better than on large language models or ViTs. The probe validates the concept on a small-scale setting, and the path to showing that TRV is useful in the settings practitioners actually care about (LoRA fine-tuning of LLMs, where IF already fails badly per §1.2) is not laid out. If TRV variance is measurable on CIFAR-10/ResNet-18 but vanishingly small on LLM fine-tuning, the diagnostic tool is irrelevant where it would matter most.

**4. SI as a "cheap proxy" for TRV has a structural conflict the document underweights.**
SI(z) = phi(z)^T Q^{-1} phi(z) is computed using a specific Hessian approximation Q (EK-FAC, per §3.2 step 6). TRV measures *cross-approximation stability*. The theoretical connection via Theorem 5.4 of Natural W-TRAK bounds approximation sensitivity in a Wasserstein metric — but TRV is defined as Jaccard@k on top-k sets, which is a discrete rank-based quantity. The gap between a continuous Lipschitz bound in Wasserstein space and a discrete Jaccard similarity in rank space is large, and there is no intermediate theorem bridging them. If the correlation Spearman(SI, TRV) is moderate (0.4-0.6) rather than strong (>0.7), the "SI as cheap proxy" framing is empirically motivated but theoretically weak — and the simpler alternative (running all approximations directly) dominates.

**5. The differentiation from Daunce/BIF rests on an "orthogonality" claim that is asserted but not demonstrated.**
The document argues that TRV (method-selection uncertainty), Daunce (training randomness), and BIF (Bayesian parameter uncertainty) are "theoretically orthogonal." The argument is plausible but not derived — it is stated as a conceptual claim without a formal proof or an empirical falsification condition. A skeptical reviewer will ask: if you ran all three methods on the same dataset, what would the correlation between their uncertainty estimates be? If Daunce variance and TRV are strongly correlated in practice (both driven by "how sensitive this test point's gradient is to the model"), the orthogonality claim collapses and AURA becomes an incremental variant of Daunce with a different computational mechanism.

---

## Strengths

- **Literature grounding is exceptional.** The document synthesizes 15+ recent papers (including 2025/2026 work) with precise attribution of which claims come from which papers. The Hessian hierarchy from Hong et al., the SI Lipschitz bound from Natural W-TRAK, and the IF/RepSim complementarity from Kowal et al. are all correctly cited and integrated.
- **The problem is real and practically motivated.** Users of TRAK/SOURCE/EK-FAC-IF genuinely face the question "should I trust this attribution for this specific test point?" No current tool answers it. The framing of TDA reliability as a per-sample diagnostic rather than a global method ranking is a legitimate and underexplored perspective.
- **The probe design is methodologically careful.** The 3-seed stability check, the partial correlation controlling for gradient norm, and the TRV-high vs TRV-low LDS comparison are all appropriate controls. The explicit fail conditions and the articulation of what different failure modes mean scientifically are above average for this field.
- **The two-phase structure with conditional commitment is strategically sound.** Gating Phase 2 (RA-TDA fusion) on empirical validation of Phase 1 (TRV variance) is the right approach given the risk profile.
- **The "stable != correct" concern is self-identified.** This is the hardest conceptual challenge for the project, and the team has recognized it early.

---

## Weaknesses / Concerns

1. **The paper's minimum viable contribution (Phase 1 only: C0+C1+C2+C4) may be too weak for NeurIPS/ICML oral/spotlight.** A new diagnostic metric plus an empirical characterization of its distribution on a single model/dataset, without a downstream application that demonstrates improved performance, is a poster at best. The contribution.md acknowledges this ("Solid NeurIPS/ICML poster"), but the timeline and effort investment implied by the probe suggests the team is aiming higher. There is a mismatch between the intellectual ambition of the framing and the expected empirical footprint.

2. **The "evaluation crisis" framing is leveraged rhetorically but not resolved.** §1.2 correctly notes that LDS has miss-relation problems, distributional TDA suggests single-run LDS is noisy, and attribution != influence. AURA then proposes to validate TRV using LDS (step 8 in the probe, §3.3 pass criteria: "Cohen's d > 0.5 in LDS difference"). This is circular: if LDS is an unreliable metric, it cannot serve as the ground truth against which TRV is validated.

3. **The choice of EK-FAC vs K-FAC as the primary Hessian pair is a double-edged sword.** Hong et al. identify this as the largest error source (41-65% of total approximation error), which makes it the best candidate for large TRV variance. But it also means the project is anchored to a very specific and narrow comparison within the Hessian hierarchy. A reviewer may ask why TRV should be defined with respect to this pair specifically rather than H vs K-FAC or GGN vs EK-FAC.

4. **No discussion of TRV stability across tasks/domains.** If TRV is a per-test-point property of the model and data, its distribution may vary dramatically across (model, dataset, task) combinations. A diagnostic tool that works on CIFAR-10/ResNet-18 but shows no variance on BERT fine-tuning or GPT-2 is not useful at the target venue's level of generality.

5. **The fusion mechanism in Phase 2 (RA-TDA) lacks a concrete algorithm sketch.** §2.2 describes the intuition ("low TRV -> increase RepSim weight") but does not specify: (a) how weights are computed from TRV/SI values, (b) whether the fusion is over attribution scores or ranked lists, (c) the normalization scheme. Without this, the comparison against "naive 0.5:0.5 ensemble" is underspecified.

---

## The "Simpler Alternative" Challenge

**The simplest alternative**: Report uncertainty via *ensembling multiple Hessian approximations* directly — compute attributions under EK-FAC, K-FAC, Block-GGN, and take the intersection of top-k lists as the "reliable" attribution set, and the union-minus-intersection as the "uncertain" region. This requires no new theory and no proxy (SI). It is directly interpretable. It does not improve attribution quality, but it provides the diagnostic signal AURA claims TRV provides, more transparently and with less theoretical scaffolding.

**Why AURA might still win**: The ensemble approach's cost is O(M * N_test * N_train) for every query, with no cheap proxy. If SI reliably predicts which test points have high ensemble disagreement, then computing SI alone (cheap) is more practical than running all M approximations. AURA's value proposition is therefore *specifically* the SI-TRV connection, not TRV itself. If Spearman(SI, TRV) is weak, the simpler alternative dominates.

**Challenge to the authors**: Can you make the argument that SI is a cheap proxy for cross-approximation agreement *before* running the probe? If the theoretical derivation is solid, state it as a theorem. If it is only empirically motivated, acknowledge that the simpler alternative (direct ensemble) is the baseline you must beat in computation efficiency times predictive accuracy.

---

## Specific Recommendations

1. **Resolve the epistemological circularity in TRV validation.** If LDS is unreliable, find a more trustworthy ground truth. Consider using small datasets where exact LOO retraining is feasible as a calibration set, validating TRV against exact LOO influence rather than approximate LDS. This would dramatically strengthen the claim.

2. **Formalize or remove the "theoretical orthogonality" claim with Daunce/BIF.** Either derive a formal condition under which TRV and Daunce variance would be uncorrelated, or commit to running all three methods on the same test set and reporting the empirical correlation matrix as part of the paper. The "three dimensions are orthogonal" claim is currently unfalsifiable and will invite skepticism.

3. **Specify RA-TDA's algorithm before the probe.** The adaptive fusion mechanism should be fully specified (weight computation, normalization, aggregation over ranked lists) before running experiments. An underspecified method cannot be meaningfully compared to a baseline.

4. **Add a large-pretrained-model sanity check to the probe.** Even a small-scale version (e.g., GPT-2 on a subset of Pile) would dramatically increase the probe's evidential value and relevance to the target venue. CIFAR-10/ResNet-18 results alone will be treated as preliminary.

5. **Add the trivial baseline comparison to the probe.** Compute Spearman(prediction_confidence, TRV) alongside Spearman(SI, TRV). If softmax entropy explains as much TRV variance as SI, the SI-TRV theoretical connection is undermined. This takes ~10 lines of code and completely changes the interpretation of results.

6. **Reframe the contribution hierarchy.** Consider whether the paper's core contribution could be framed as: "We show that a simple, interpretable measure of cross-approximation consistency (TRV) predicts attribution quality, and that this signal can be cheaply approximated via SI." This is a cleaner, more falsifiable claim that does not require the fusion mechanism to work.

---

## Score: 5 / 10

Calibrated to NeurIPS/ICML standards (7+ = likely accept, 5-6 = weak reject / borderline).

**Rationale**: The problem identification and literature synthesis are genuinely strong (would score 7-8 on framing alone). The score is pulled down by: (a) the unresolved epistemological circularity in using LDS to validate a metric proposed to address LDS's limitations; (b) the SI-TRV theoretical gap between a Wasserstein Lipschitz bound and a Jaccard rank stability measure; (c) the Phase 2 contribution being underspecified; (d) the single-dataset probe scope; (e) the undemonstrated orthogonality claim vs. Daunce/BIF. The project has a viable path to 7+ if the probe shows strong TRV variance, the SI-TRV correlation is robust (>0.6), the paper includes at least one result on a pretrained model setting, and the LDS circularity is addressed with an exact-LOO calibration. Current state is "intellectually compelling direction with insufficient empirical and theoretical grounding to commit fully."

---

## Core Findings Summary

| Finding | Severity | Actionable? |
|---------|----------|-------------|
| TRV validates against LDS, but LDS is acknowledged as unreliable — circularity | High | Yes: use exact LOO on small calibration set |
| IF and RepSim have different estimands; fusion in low-TRV regime is theoretically unmotivated | High | Yes: reframe Phase 2 or derive estimand alignment condition |
| SI-TRV gap: Wasserstein Lipschitz bound does not equal Jaccard rank stability | Medium-High | Yes: derive bridging theorem or scope claim empirically |
| Daunce/BIF orthogonality is asserted, not demonstrated | Medium | Yes: run empirical correlation or formalize condition |
| CIFAR-10/ResNet-18 probe scope too narrow for target venue | Medium | Yes: add GPT-2 or ViT sanity check |
| RA-TDA algorithm underspecified | Medium | Yes: specify before probe runs |
| Trivial baseline (prediction confidence vs TRV) never controlled for | Medium | Yes: add to probe Step 3, ~10 lines of code |
| Phase 1-only contribution (no RA-TDA) is poster-level at best | Low-Medium | Awareness: set expectations accordingly |
