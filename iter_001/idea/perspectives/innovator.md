# Innovator Perspective: AURA — Beyond Point Estimates

**Agent**: sibyl-innovator
**Date**: 2026-03-17
**Topic**: TDA Robustness Value (TRV) and Sensitivity-Aware Training Data Attribution

---

## Executive Summary

The Probe results exposed a fundamental flaw in the original AURA conception: TRV defined as Jaccard@k across Hessian approximation tiers is a **model-instance property**, not a test-point property (cross-seed Spearman rho ~ 0). Rather than treating this as a fatal blow, I propose three unconventional angles that **reframe the cross-seed instability as the signal itself**, drawing on multi-fidelity optimization, conformal prediction, and Mixture-of-Experts routing. Each angle preserves Phase 1's core diagnostic value while opening genuinely novel research directions that no existing work (Daunce, BIF, Natural W-TRAK) addresses.

---

## Angle 1: Spectral Fingerprint Routing — From Hessian Sensitivity to Per-Sample Expert Selection

### Core Idea

Instead of computing TRV as a scalar stability score, decompose each test point's gradient into its projection onto the **top-k Hessian eigenvectors** (the "spectral fingerprint") and use this fingerprint as a **routing signal** to select the optimal attribution method per test point. This borrows directly from Mixture-of-Experts (MoE) gating: the spectral fingerprint is the "token representation" and the attribution methods (IF variants, RepSim, TRAK) are the "experts."

### Hypothesis

**H-Inn1**: The spectral fingerprint — defined as the vector of squared projections of the test gradient onto the top-k eigenvectors of the Fisher/GGN — is a **stable, per-test-point property** (cross-seed Spearman rho > 0.5), even when the scalar TRV is not. Rationale: individual eigenvector directions are unstable across seeds, but the **energy distribution across eigenvalue magnitude buckets** (bulk vs. outlier) is determined by the data geometry, not the training seed (supported by Ghorbani et al., 2019; Papyan, 2020 on class-cluster structure of Hessian outliers).

**H-Inn2**: Test points whose spectral fingerprints concentrate energy in the outlier eigenspace (high spectral concentration ratio, SCR) have more reliable IF-based attributions, while test points with energy spread across the bulk are better served by representation-based methods. This follows from the observation that outlier eigenvectors correspond to class-discriminative directions where the linear approximation underlying IF is most valid.

### Why This Is Novel

- **Daunce** (ICML 2025) perturbs model parameters and measures attribution variance — it cannot distinguish *why* some points are unstable (spectral misalignment vs. parameter noise).
- **BIF** (ICML 2025) uses posterior variance — it conflates epistemic uncertainty with approximation sensitivity.
- **Natural W-TRAK** (2512.09103) uses scalar SI as a Lipschitz bound — the Probe showed SI-TRV correlation is ~0, precisely because SI collapses the spectral structure into a single number.
- **Spectral Fingerprint Routing** preserves the full spectral decomposition, enabling per-test-point method selection rather than a single reliability score.

### Experimental Plan

| Step | Description | Time | Compute |
|------|-------------|------|---------|
| 1 | Train 5 ResNet-18 on CIFAR-10 (different seeds) | 2.5h | 1 GPU |
| 2 | Compute GGN top-50 eigenvectors for each seed | 1h | 1 GPU |
| 3 | For 500 test points, compute spectral fingerprint (energy in top-10 outlier vs. bulk) | 30min | 1 GPU |
| 4 | Test cross-seed stability of fingerprint (Spearman rho on SCR ranking) | 10min | CPU |
| 5 | Compute IF (EK-FAC) and RepSim attributions for same 500 test points | 3h | 2 GPUs |
| 6 | Evaluate: does SCR predict which method gives higher LDS per test point? | 20min | CPU |
| **Total** | | **~7h** | **4 GPU-hours** |

### Success Criteria

- Cross-seed SCR ranking Spearman rho > 0.5 (stability)
- SCR-high subgroup IF LDS > SCR-low subgroup IF LDS with Cohen's d > 0.5
- SCR-low subgroup RepSim LDS > SCR-low subgroup IF LDS (complementarity)

### Failure Modes

1. **Eigenvector subspace rotation across seeds**: Even bucket-level energy distribution may rotate. Mitigation: use eigenvalue-magnitude bins (0-1, 1-10, 10-100, 100+) rather than specific eigenvector indices.
2. **Class structure dominates**: If spectral fingerprint simply reflects class membership, it adds no information beyond class-conditional attribution. Control: measure within-class SCR variance.
3. **Compute cost**: Full eigendecomposition of GGN for large models is expensive. Fallback: use stochastic Lanczos quadrature to estimate spectral density without full decomposition.

### Computational Cost Estimate

- ResNet-18/CIFAR-10: full GGN eigendecomposition is tractable (~11M params, but GGN has structure exploitable by Kronecker factorization)
- GPT-2: stochastic Lanczos (50 iterations) suffices for spectral density estimation, ~2 GPU-hours
- **Success probability**: 55% (eigenvalue bucket stability is theoretically grounded but empirically untested for TDA routing)

---

## Angle 2: Conformal Attribution Sets — Distribution-Free Reliability Guarantees

### Core Idea

Replace the point-estimate TRV with a **conformal prediction set** over attribution rankings. For each test point, produce not a single top-k attribution list but a **set of plausible top-k lists** with coverage guarantee: "with probability >= 1-alpha, the true attribution ranking (under exact LOO) is contained in this set." The *size* of the conformal set is the new TRV — larger sets mean less reliable attribution.

This is a radical reframing: instead of measuring agreement across Hessian approximations (which is model-instance-dependent), we construct distribution-free prediction intervals using a small calibration set of exact LOO attributions.

### Hypothesis

**H-Inn3**: A conformal attribution set calibrated on ~200 LOO-recomputed test points can produce valid coverage (>= 90%) on held-out test points, with set sizes that vary meaningfully across test points (coefficient of variation > 0.5). The set size serves as a provably valid, distribution-free TRV.

**H-Inn4**: Conformal TRV (set size) correlates with IF-RepSim disagreement (Spearman rho > 0.4), providing a principled bridge between attribution uncertainty and method fusion.

### Why This Is Novel

- **No existing TDA work uses conformal prediction** for attribution reliability. The closest is "Conformal Prediction Sets with Improved Conditional Coverage using Trust Scores" (arxiv 2501.10139, ICLR 2025 submission) which uses trust scores for calibration — but for classification, not attribution.
- **Distribution-free guarantee**: Unlike TRV (which requires computing multiple Hessian approximations) or Daunce (which requires model perturbation ensemble), conformal TRV provides a finite-sample coverage guarantee from standard statistical theory.
- **Solves the cross-seed problem**: Conformal sets are calibrated per model instance, so cross-seed instability is not a concern — each model gets its own calibration.

### Experimental Plan

| Step | Description | Time | Compute |
|------|-------------|------|---------|
| 1 | Train ResNet-18 on CIFAR-10 (1 seed) | 30min | 1 GPU |
| 2 | Compute exact LOO attributions for 300 random test points (calibration + test) | 8h | 4 GPUs |
| 3 | For same 300 points, compute EK-FAC IF attributions | 1h | 1 GPU |
| 4 | Define nonconformity score: Kendall tau distance between IF ranking and LOO ranking | 15min | CPU |
| 5 | Calibrate conformal threshold on 200 points, evaluate coverage on 100 held-out | 15min | CPU |
| 6 | Measure conformal set size distribution across test points | 15min | CPU |
| **Total** | | **~10h** | **~12 GPU-hours** |

**Key bottleneck**: LOO recomputation for 300 test points. On CIFAR-10 with 50K training points and ResNet-18 (~2min/retrain on 1 GPU), exact LOO for 1 test point requires 50K retrains — **infeasible**.

**Practical workaround**: Use TRAK/Datamodel scores (computed from ~50 retrained models on random subsets) as the "ground truth" calibration signal instead of exact LOO. This is standard practice (Park et al., 2023). Cost: ~50 retrains x 30min = 25 GPU-hours for the calibration set, then amortized.

**Alternative**: Calibrate on a reduced training set (5K subset of CIFAR-10) where exact LOO is feasible (~5K retrains x 1min = ~83 GPU-hours on 4 GPUs = ~21 wall-clock hours). This is borderline feasible within our budget.

### Success Criteria

- Marginal coverage >= 88% (allowing 2% slack from nominal 90%)
- Conformal set size CV > 0.5 across test points
- Conformal set size correlates with EK-FAC vs. K-FAC disagreement (Spearman rho > 0.3)

### Failure Modes

1. **Exchangeability violation**: LOO attributions across test points may not be exchangeable if test points are drawn non-i.i.d. Mitigation: use split conformal prediction (Vovk et al.) which only requires exchangeability within the calibration set.
2. **Calibration cost**: Even with the reduced training set, LOO recomputation is expensive. Failure here means the method is theoretically clean but impractical. Fallback: use Datamodels as calibration oracle.
3. **Uniform set sizes**: If all test points get similarly sized conformal sets, the diagnostic value is zero. This would mean IF approximation error is uniformly distributed — a valid negative result that confirms the Probe finding in a distribution-free framework.

### Computational Cost Estimate

- Reduced-set approach: ~25 GPU-hours (feasible on 4x RTX 4090 in ~7 wall-clock hours)
- Datamodel approach: ~25 GPU-hours for 50 random-subset retrains
- **Success probability**: 40% (conformal prediction for ranking is underexplored; the nonconformity score design is non-trivial)

---

## Angle 3: Multi-Fidelity Attribution Ladder — Treating Hessian Approximations as Fidelity Levels

### Core Idea

Reframe the Hessian approximation chain (Identity -> Diagonal -> K-FAC -> EK-FAC -> Block-GGN -> Full GGN -> Hessian) as a **multi-fidelity surrogate hierarchy**, analogous to multi-fidelity optimization in engineering (low-fidelity CFD -> high-fidelity CFD). For each test point, adaptively select the **minimum-fidelity approximation that is "good enough"** — where "good enough" is defined as: the attribution ranking at this fidelity level is within epsilon of the next-higher fidelity level (measured by Jaccard@k).

This reframes TRV from "how unstable is the attribution" to "what is the minimum computational cost to get a reliable attribution for this test point." Some test points need only K-FAC (cheap); others require full GGN (expensive). The per-test-point **fidelity requirement** is the new TRV.

### Hypothesis

**H-Inn5**: The minimum required fidelity level varies significantly across test points (at least 3 distinct fidelity tiers each covering >15% of test points), and this variation is **more stable across seeds** than scalar TRV. Rationale: the fidelity requirement depends on which eigenvalue range the test gradient projects into, which is an eigenvalue-magnitude property (more stable) rather than an eigenvector-direction property (less stable).

**H-Inn6**: A cheap diagnostic (gradient norm, loss value, or prediction entropy) can predict the required fidelity level with AUROC > 0.7, enabling an **adaptive attribution pipeline** that allocates compute proportionally to difficulty.

### Why This Is Novel

- **Multi-fidelity optimization** (Peherstorfer et al., 2018; NASA Sage, 2025) is a mature field in engineering but has **never been applied to TDA**. The Hessian approximation chain is a natural fidelity hierarchy with known cost-accuracy tradeoffs (Hong et al., 2509.23437).
- **Adaptive compute allocation for TDA** is entirely new. All existing methods apply the same approximation uniformly to all test points. Even ASTRA (NeurIPS 2025), which optimizes the Neumann series for IF, uses the same number of iterations for all test points.
- **Directly addresses the cross-seed problem**: The fidelity requirement is defined relative to adjacent tiers (K-FAC vs. EK-FAC agreement), not absolute stability — this relative measure may be more stable across seeds than absolute TRV.

### Experimental Plan

| Step | Description | Time | Compute |
|------|-------------|------|---------|
| 1 | Train 3 ResNet-18 on CIFAR-10 (3 seeds) | 1.5h | 1 GPU |
| 2 | Compute attributions at 4 fidelity levels (Identity, K-FAC, EK-FAC, Full GGN) for 200 test points x 3 seeds | 6h | 4 GPUs |
| 3 | For each test point, determine minimum fidelity where Jaccard@10 with next tier > 0.8 | 20min | CPU |
| 4 | Analyze cross-seed stability of fidelity requirement (agreement rate) | 15min | CPU |
| 5 | Train logistic regression: cheap features (grad norm, loss, entropy, class) -> fidelity level | 15min | CPU |
| 6 | Evaluate: does adaptive fidelity selection achieve >90% of full-GGN LDS at <50% compute? | 30min | CPU |
| **Total** | | **~9h** | **~8 GPU-hours** |

### Success Criteria

- At least 3 fidelity tiers each covering >15% of test points (variation exists)
- Cross-seed fidelity-level agreement > 60% (more stable than scalar TRV)
- Adaptive pipeline achieves >90% of full-GGN LDS at <50% of full-GGN compute
- Cheap diagnostic predicts fidelity level with AUROC > 0.7

### Failure Modes

1. **Monotonic collapse at bottom**: Probe showed Diagonal ~ Damped Identity ~ Identity in last-layer setting. If full-model Hessian chain also collapses at the bottom, the effective fidelity hierarchy shrinks to 2 levels (K-FAC vs. EK-FAC/GGN), reducing the adaptive range. This is actually the most likely failure mode.
2. **Fidelity requirement is class-determined**: If all "cat" images need EK-FAC and all "car" images are fine with K-FAC, the diagnostic is just a class detector. Control: measure within-class fidelity requirement variance.
3. **Hong et al. code limitations**: The Probe found that last-layer Hessian chain has insufficient resolution. Full-model chain code from 2509.23437 needs adaptation. Risk: code may not support per-test-point ranking extraction.

### Computational Cost Estimate

- 4-level attribution computation for 200 points x 3 seeds: dominated by EK-FAC and Full GGN (~2h each)
- Total: ~8 GPU-hours, well within 4x RTX 4090 budget
- **Success probability**: 50% (multi-fidelity framing is natural; the cross-seed stability improvement is the main uncertainty)

---

## Comparative Assessment

| Criterion | Angle 1: Spectral Routing | Angle 2: Conformal Sets | Angle 3: Multi-Fidelity Ladder |
|-----------|--------------------------|------------------------|-------------------------------|
| **Novelty** | High (MoE x TDA cross-domain) | Very High (conformal x TDA, no prior work) | High (multi-fidelity x TDA, no prior work) |
| **Addresses cross-seed problem** | Yes (eigenvalue buckets, not directions) | Sidesteps it (per-model calibration) | Partially (relative stability may help) |
| **Practical value** | Method selection per test point | Provable reliability guarantee | Compute-optimal attribution |
| **Compute cost** | ~4 GPU-hours (pilot) | ~25 GPU-hours (pilot) | ~8 GPU-hours (pilot) |
| **Success probability** | 55% | 40% | 50% |
| **Independent from Phase 2?** | Yes | Yes | Yes |
| **Paper contribution type** | New method | New framework + guarantees | New framework + efficiency |
| **Fallback if fails** | Reduces to class-conditional analysis | Valid negative result in conformal framework | Reduces to cost-accuracy tradeoff analysis |

---

## Recommended Strategy

**Primary**: Angle 3 (Multi-Fidelity Attribution Ladder) — best balance of novelty, feasibility, and directness of response to the Probe findings. It reframes the Hessian approximation chain (which is AURA's existing infrastructure) into a multi-fidelity hierarchy and asks a practical question: "how much compute does this test point need for reliable attribution?" This preserves Phase 1's diagnostic narrative while adding genuine practical value (compute savings).

**Secondary**: Angle 1 (Spectral Fingerprint Routing) — strongest theoretical motivation and most likely to produce stable per-test-point signals. Should be pursued in parallel if GPU budget permits. The spectral fingerprint is a natural evolution of the failed SI proxy (which collapsed spectral structure to a scalar).

**Tertiary/Future Work**: Angle 2 (Conformal Attribution Sets) — highest novelty ceiling but also highest compute cost and risk. Best positioned as a theoretical contribution in the Discussion section or as a separate short paper.

---

## Integration with Existing AURA Framework

All three angles are **compatible with and complementary to** the existing AURA contribution structure:

- **C0 (TRV as diagnostic)**: Angles 1 and 3 provide alternative TRV operationalizations that may resolve the cross-seed instability. Angle 2 provides a distribution-free TRV.
- **C1 (Empirical characterization)**: All angles produce rich empirical characterizations of per-test-point attribution reliability.
- **C3 (RA-TDA fusion)**: Angle 1 directly provides a routing signal for adaptive fusion. Angle 3's fidelity requirement can inform fusion weights (low-fidelity-sufficient points = IF reliable = higher IF weight).
- **C4 ("Stable != Correct")**: Angle 2's conformal sets directly address this by calibrating against ground truth, not just internal consistency.

---

## Key References Informing These Proposals

**Cross-domain sources (not in existing reference list)**:
- Ghorbani et al. (2019), "An Investigation into Neural Net Optimization via Hessian Eigenvalue Density" — Hessian eigenspectrum structure (bulk + outlier), supports Angle 1's stability hypothesis
- Papyan (2020), "Traces of Class/Cross-Class Structure Pervade Deep Learning Spectra" — class-cluster structure of Hessian outlier eigenvectors, supports SCR stability
- Peherstorfer et al. (2018), "Survey of Multifidelity Methods in Uncertainty Quantification, Inference, and Optimization" — theoretical foundation for Angle 3
- Vovk et al. (2005), "Algorithmic Learning in a Random World" — conformal prediction theory for Angle 2
- Fedus et al. (2022), "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" — MoE routing analogy for Angle 1
- Kaiser & Herzog (2025), "A Tutorial on Distribution-Free Uncertainty Quantification Using Conformal Prediction" — recent CP tutorial
- "Spectral Alignment as Predictor of Loss Explosion in Neural Network Training" (arxiv 2510.04202, 2025) — spectral alignment dynamics

**Existing references leveraged**:
- Hong et al. (2509.23437) — Hessian approximation hierarchy = fidelity ladder (Angle 3)
- Li et al. (2512.09103) — SI failure motivates spectral fingerprint (Angle 1)
- Park et al. (2303.12922) — LDS as evaluation metric + Datamodels as calibration oracle (Angle 2)
- Li et al. (2409.19998) — IF failure modes define routing targets (Angle 1)

---

## Literature Search Log

1. **arXiv/Web: conformal prediction + data attribution** — No direct prior work found combining conformal prediction with TDA reliability. Closest: trust-score calibrated conformal sets (2501.10139) for classification. Gap confirmed for Angle 2.
2. **Web: multi-fidelity optimization + surrogate model selection** — Rich literature in engineering (NASA Sage, SAMOO frameworks) but zero application to TDA. Gap confirmed for Angle 3.
3. **Web: spectral analysis + gradient alignment + Hessian** — Strong theoretical foundation: outlier eigenspace alignment with class structure is well-established (Ghorbani 2019, Papyan 2020, Papyan et al. AAAI 2021). Supports Angle 1's stability hypothesis.
4. **Web: MoE adaptive gating per-sample routing** — Mature field with per-token routing (SeqMoE, Feature-Gating MoE). Cross-domain transfer to TDA is novel.
5. **Web: ensemble disagreement adaptive fusion** — FDSNet (2025) uses feature disagreement scoring for sensor fusion; information-theoretic multi-model fusion (2602.03319) balances consensus/disagreement. Neither applied to TDA.
6. **Web: random matrix theory + condition number + per-sample Hessian** — RMT analysis of Hessian spectra is well-studied but per-sample analysis is not. Standard condition number has Tracy-Widom-like distribution. Supports the theoretical basis of Angle 1.
