# Pragmatist Perspective: AURA — Beyond Point Estimates

**Agent**: sibyl-pragmatist
**Date**: 2026-03-17
**Topic**: TDA Robustness Value (TRV) and Sensitivity-Aware Training Data Attribution

---

## Executive Summary

The Probe results delivered a clear engineering verdict: the original TRV (Jaccard@k across Hessian tiers) is unstable across seeds (Spearman rho ~ 0) and has insufficient per-point variance (std 0.053-0.082). As a pragmatist, I focus on **what can actually be built and validated on 4x RTX 4090 within the iteration budget**, leveraging existing open-source tooling (dattri, pyDVL, TRAK library, Hong et al. code). I propose three angles ordered by engineering feasibility, each targeting completion within 1 hour per pilot experiment task.

The central pragmatic insight: **stop trying to make a single scalar TRV stable across seeds. Instead, use cheap multi-method disagreement as the per-test-point diagnostic signal, and validate it against ground truth (LDS or Datamodel scores).**

---

## Angle 1: Disagreement-Based Diagnostic — The Simplest Thing That Could Work

### Core Idea

Abandon the Hessian approximation chain as the TRV axis. Instead, compute attributions using 2-3 **different TDA methods** (TRAK, EK-FAC IF, RepSim) on the **same trained model**, and define the per-test-point diagnostic as the **pairwise ranking disagreement** (Kendall tau or Jaccard@k) between methods. This is the simplest, cheapest, most engineering-friendly diagnostic: no Hessian eigendecomposition, no multi-seed training, no spectral analysis.

### Why This Is Practical

1. **All methods already implemented**: TRAK has a battle-tested PyTorch library ([MadryLab/trak](https://github.com/MadryLab/trak)), pyDVL provides EK-FAC IF ([aai-institute/pyDVL](https://github.com/aai-institute/pydvl)), RepSim is a ~50-line cosine similarity computation. dattri ([TRAIS-Lab/dattri](https://github.com/trais-lab/dattri)) provides unified benchmarking.
2. **Single model, single seed**: No cross-seed instability by design. The diagnostic is defined per model instance.
3. **Known empirical basis**: Kowal et al. (2602.14869) already showed IF-RepSim correlation is 0.37-0.45. We operationalize this low correlation as a diagnostic signal.
4. **Cheap**: TRAK scores for CIFAR-10/ResNet-18 take ~30 min on 1 GPU. EK-FAC IF takes ~1-2h. RepSim takes ~10 min.

### Hypothesis

**H-Prag1**: The per-test-point TRAK-IF disagreement (Kendall tau distance) correlates with the test point's actual attribution quality gap between methods (measured by LDS against Datamodel ground truth), with Spearman rho > 0.3.

**H-Prag2**: Test points with high cross-method disagreement (top 20%) have significantly higher LDS variance across methods (Cohen's d > 0.5) compared to low-disagreement points, confirming that disagreement identifies points where method selection matters.

### Experimental Plan

| Step | Description | Time | Compute |
|------|-------------|------|---------|
| 1 | Train 1 ResNet-18 on CIFAR-10 (1 seed) | 30min | 1 GPU |
| 2 | Compute TRAK scores (50 random-subset models, 500 test points) using MadryLab/trak library | 45min | 2 GPUs |
| 3 | Compute EK-FAC IF scores for same 500 test points using pyDVL | 1.5h | 1 GPU |
| 4 | Compute RepSim (penultimate layer cosine sim) for same 500 test points | 10min | 1 GPU |
| 5 | Compute pairwise Kendall tau / Jaccard@10 for each test point across 3 methods | 15min | CPU |
| 6 | Compute per-test-point LDS for each method against TRAK ground truth | 15min | CPU |
| 7 | Analyze: disagreement vs LDS gap correlation; high/low disagreement LDS comparison | 15min | CPU |
| **Total** | | **~3.5h** | **~4 GPU-hours** |

**Pilot (15 min)**: Train 1 ResNet-18, compute TRAK + RepSim for 100 test points on CIFAR-10 subset (5K training). If Jaccard@10(TRAK, RepSim) variance across test points is < 0.01, the diagnostic has no discriminative power — stop here.

### Success Criteria

- Per-test-point Kendall tau variance (across test points) > 0.05
- Spearman(disagreement, LDS gap) > 0.3
- High-disagreement subset LDS variance > low-disagreement subset (Cohen's d > 0.5)

### Failure Modes

1. **Uniform disagreement**: All test points have the same level of cross-method disagreement (tau ~ 0.5 uniformly). This would mean IF and RepSim are always equally different regardless of test point — possible given that they capture different information axes globally. **Mitigation**: Use disagreement at different top-k cutoffs (k=5, 10, 50) — fine-grained disagreement may vary even if coarse disagreement is uniform.
2. **No correlation with LDS quality**: Disagreement is large but random w.r.t. which method is actually correct. This means cross-method disagreement is not informative for method selection. **Mitigation**: This is still a valid finding — it implies no cheap diagnostic can guide adaptive fusion, supporting a "use both methods + report disagreement as uncertainty" approach.
3. **TRAK ground truth quality**: TRAK with 50 models may be noisy. Fallback: use LDS directly from dattri's benchmark suite.

### Computational Cost Estimate

- ~4 GPU-hours total, well within 4x RTX 4090 budget
- No custom Hessian code needed — all from existing libraries
- **Success probability**: 45% (the key uncertainty is whether disagreement is informative, not whether it can be computed)

---

## Angle 2: Ensemble TRV with Efficient Ensembles — Stabilizing TRV at Minimal Cost

### Core Idea

The Probe's fatal finding was cross-seed TRV instability (rho ~ 0). The brute-force fix — train many seeds and average TRV — is expensive. But Deng et al. (2405.17293, "Efficient Ensembles Improve Training Data Attribution") showed that **Dropout Ensemble** and **LoRA Ensemble** can replace independent retraining with up to 80% training cost reduction while maintaining attribution quality. Apply the same trick to TRV: compute TRV for each ensemble member, then average across members. The ensemble-averaged TRV should be more stable because it marginalizes over training randomness.

### Why This Is Practical

1. **Efficient Ensembles paper provides recipes**: Dropout Ensemble (training cost: 1x model + K forward passes with different dropout masks) and LoRA Ensemble (training cost: 1x base model + K LoRA heads) are well-specified and implementable.
2. **Modest compute**: 10 dropout ensemble members = 1 training run + 10 forward passes per test point. Compare to 10 independent seeds = 10 training runs.
3. **Directly addresses the Probe failure**: The Probe showed TRV is (model-instance, test-point) — averaging over model instances via ensemble extracts the test-point component.
4. **dattri library support**: dattri already implements ensemble methods for TRAK; adapting for IF is straightforward.

### Hypothesis

**H-Prag3**: Dropout Ensemble TRV (averaged over 10 dropout masks) has cross-member Spearman rho > 0.6 for TRV rankings, compared to rho ~ 0 for independent seeds. The ensemble averaging reduces the model-instance-specific noise.

**H-Prag4**: Ensemble TRV retains diagnostic power — the high/low TRV subgroup LDS difference (Cohen's d) is > 0.3, comparable to or better than single-seed TRV.

### Experimental Plan

| Step | Description | Time | Compute |
|------|-------------|------|---------|
| 1 | Train 1 ResNet-18 on CIFAR-10 | 30min | 1 GPU |
| 2 | Generate 10 Dropout Ensemble members (10 forward passes with MC Dropout) | 15min | 1 GPU |
| 3 | Compute EK-FAC IF + K-FAC IF for 200 test points x 10 ensemble members | 3h | 2 GPUs |
| 4 | Compute Jaccard@10(EK-FAC, K-FAC) per test point per member = 10 TRV values per point | 15min | CPU |
| 5 | Average TRV across members → Ensemble TRV | 5min | CPU |
| 6 | Stability test: split 10 members into two groups of 5, compute Spearman rho of averaged TRV | 10min | CPU |
| 7 | Diagnostic test: high vs low Ensemble TRV subgroups, compare LDS | 15min | CPU |
| **Total** | | **~4.5h** | **~7 GPU-hours** |

**Pilot (15 min)**: Use 3 dropout members, 50 test points, K-FAC only (cheapest). Check if TRV variance across test points is measurably different from single-seed. If dropout doesn't change TRV distribution shape at all, the approach is flawed.

### Success Criteria

- Split-half Ensemble TRV Spearman rho > 0.6 (stability)
- Ensemble TRV std across test points > 0.08 (diagnostic power preserved)
- High vs low Ensemble TRV LDS Cohen's d > 0.3

### Failure Modes

1. **Dropout doesn't create sufficient model diversity**: MC Dropout may produce ensemble members that are too similar (all dropout masks share 95%+ of activations), so ensemble TRV converges to single-seed TRV. **Mitigation**: Use aggressive dropout rate (0.3-0.5) or switch to LoRA Ensemble which creates more structural diversity.
2. **Ensemble averaging kills the signal**: Averaging TRV across members washes out genuine per-test-point differences, producing uniform ensemble TRV. **Mitigation**: Use median instead of mean (more robust to outlier members), or use the inter-member TRV variance as the diagnostic signal itself.
3. **Hessian chain bottom-collapse persists**: The Probe found Diagonal ~ Damped Identity ~ Identity in last-layer setting. If this persists in full-model setting, the effective TRV range is too narrow. **Mitigation**: Use only the EK-FAC vs K-FAC pair (the largest single gap per Hong et al.), or switch to full-model Hessian chain if budget allows.

### Computational Cost Estimate

- ~7 GPU-hours for full experiment, ~1 GPU-hour for pilot
- Main bottleneck: iHVP computation for 10 ensemble members
- **Success probability**: 35% (dropout ensemble may not create enough diversity to stabilize TRV; this is the most uncertain angle)

---

## Angle 3: Lightweight Adaptive Fusion via Difficulty Proxy — Skip the TRV, Use What's Free

### Core Idea

The most pragmatic approach to IF-RepSim fusion: **skip TRV entirely** and use freely available per-test-point features (prediction confidence, loss value, gradient norm, entropy, margin) to learn when IF outperforms RepSim and vice versa. Train a simple logistic regression or small decision tree on a calibration set where both methods' LDS are known, then deploy it to select the better method per test point at inference time.

This is the "boring but works" approach. No new diagnostic metric, no Hessian sensitivity analysis — just supervised method selection using cheap features.

### Why This Is Practical

1. **Features are free**: Confidence, loss, entropy, and gradient norm are byproducts of a single forward/backward pass. No additional Hessian computation needed.
2. **Small calibration set suffices**: With 5 features and binary target (IF better vs RepSim better), a logistic regression needs ~200 calibration points.
3. **Ground truth available**: dattri's benchmark provides LDS for multiple methods on standard datasets. Or compute TRAK-based ground truth for 200 calibration points (~2 GPU-hours).
4. **Interpretable**: The learned selector reveals *which features predict method reliability* — this is itself a publishable finding.

### Hypothesis

**H-Prag5**: A logistic regression using (confidence, loss, gradient norm, entropy, margin) can predict whether IF or RepSim gives higher LDS for a given test point with AUROC > 0.65. This significantly outperforms random selection (AUROC 0.50) and is competitive with oracle selection.

**H-Prag6**: The adaptive selector improves average LDS by > 3% absolute over the worse single method (IF or RepSim), closing > 30% of the gap to oracle method selection.

### Connection to AURA Narrative

This angle can be framed as an empirical validation of the AURA thesis: "per-test-point attribution reliability varies and can be predicted." Even if TRV itself is unstable, the finding that cheap features predict method quality supports the broader claim. The learned features can also inform which test-point properties drive reliability differences — contributing to C1 (empirical characterization).

### Experimental Plan

| Step | Description | Time | Compute |
|------|-------------|------|---------|
| 1 | Train 1 ResNet-18 on CIFAR-10 | 30min | 1 GPU |
| 2 | Compute TRAK ground truth for 500 test points (50 subset models) | 45min | 2 GPUs |
| 3 | Compute EK-FAC IF attributions for 500 test points | 1.5h | 1 GPU |
| 4 | Compute RepSim attributions for 500 test points | 10min | 1 GPU |
| 5 | Compute per-test-point LDS for IF and RepSim against TRAK ground truth | 15min | CPU |
| 6 | Label each test point: "IF better" vs "RepSim better" | 5min | CPU |
| 7 | Extract features: confidence, loss, gradient norm, entropy, margin | 10min | 1 GPU |
| 8 | Train logistic regression + decision tree on 300 calibration points | 5min | CPU |
| 9 | Evaluate on 200 held-out points: AUROC, LDS improvement | 10min | CPU |
| **Total** | | **~3.5h** | **~5 GPU-hours** |

**Pilot (10 min)**: Use CIFAR-10 subset (5K training), compute RepSim + identity-IF (cheapest proxy) for 100 test points. Check if the fraction of points where IF > RepSim is between 20-80%. If one method dominates on > 90% of points, there's no room for adaptive selection.

### Success Criteria

- Method selection AUROC > 0.65
- Adaptive selector LDS > max(IF LDS, RepSim LDS) by > 1% absolute
- Adaptive selector closes > 30% of gap to oracle selection
- At least 2 features with |coefficient| > 0.3 in logistic regression (interpretability)

### Failure Modes

1. **One method dominates globally**: If IF outperforms RepSim on > 85% of test points (or vice versa), adaptive selection adds negligible value. **Mitigation**: This is actually the most likely scenario for CIFAR-10/ResNet-18 (a well-behaved, in-distribution setting). Test on a harder setting: CIFAR-10 with 10% label noise, or a fine-grained classification task where some classes are underrepresented.
2. **Features don't predict**: The test-point features may not carry information about method reliability if the reliability depends on global Hessian properties (which is what the Probe suggested). **Mitigation**: Add interaction features (confidence x gradient norm) and polynomial features. If still AUROC < 0.55, report as a valid negative result.
3. **TRAK ground truth is noisy**: 50-model TRAK may be insufficient for reliable per-point LDS. **Mitigation**: Use rank-correlation with a larger ensemble as sanity check, or use the dattri benchmark results directly.

### Computational Cost Estimate

- ~5 GPU-hours total, dominant cost is TRAK computation
- No custom code beyond standard sklearn + existing TDA libraries
- **Success probability**: 40% (the key risk is that cheap features don't capture what drives method-level reliability)

---

## Comparative Assessment

| Criterion | Angle 1: Disagreement Diagnostic | Angle 2: Ensemble TRV | Angle 3: Lightweight Selector |
|-----------|----------------------------------|----------------------|------------------------------|
| **Novelty** | Medium (operationalizes known low correlation) | Medium (applies efficient ensembles to TRV) | Low-Medium (supervised method selection) |
| **Addresses Probe failure** | Sidesteps it (no TRV needed) | Directly (stabilizes TRV via ensemble) | Sidesteps it (no TRV needed) |
| **Practical value** | High (uncertainty flag per test point) | Medium (stable TRV if it works) | Very High (directly improves attribution) |
| **Implementation effort** | Low (all libraries exist) | Medium (need Hessian chain + ensemble) | Low (sklearn + existing TDA) |
| **Compute cost (pilot)** | ~1 GPU-hour | ~1 GPU-hour | ~1 GPU-hour |
| **Compute cost (full)** | ~4 GPU-hours | ~7 GPU-hours | ~5 GPU-hours |
| **Success probability** | 45% | 35% | 40% |
| **Paper contribution type** | Diagnostic tool + empirical finding | Stabilized metric | Adaptive method + empirical characterization |
| **Fallback if fails** | Report disagreement distribution as descriptive finding | Valid negative: efficient ensembles don't help TRV stability | Report feature analysis as interpretability contribution |

---

## Recommended Strategy

**Primary: Angle 1 (Disagreement-Based Diagnostic)** — lowest engineering risk, uses only existing libraries, produces a useful diagnostic whether or not it predicts LDS quality. The multi-method disagreement signal is the most natural operationalization of "attribution uncertainty" from an engineering perspective. Even if H-Prag1 fails, the per-test-point disagreement distribution is a novel empirical characterization (contributing to C1).

**Secondary: Angle 3 (Lightweight Selector)** — run in parallel with Angle 1 since they share the same attribution computations (TRAK, IF, RepSim for 500 test points). Angle 3 adds only ~30 min of feature extraction + logistic regression on top of Angle 1's outputs. This is the angle most likely to produce a *directly useful* tool (adaptive method selection), even if the underlying mechanism is "boring."

**Tertiary: Angle 2 (Ensemble TRV)** — pursue only if Angles 1+3 fail and the reviewer consensus demands stabilizing the original TRV definition. Highest engineering risk due to MC Dropout + Hessian chain interaction complexity.

**Concrete recommendation for next 24 hours**:
1. Run Angle 1 pilot (15 min): TRAK + RepSim on 100 CIFAR-10 test points → check disagreement variance
2. Run Angle 3 pilot (10 min): identity-IF + RepSim on 100 points → check if method dominance is < 85%
3. If both pilots pass, proceed to full Angle 1 + Angle 3 joint experiment (shared attribution computation, total ~5 GPU-hours)

---

## Integration with Existing AURA Framework

### Synergy with Innovator Proposals

- **Angle 1 (Disagreement) + Innovator Angle 1 (Spectral Routing)**: If spectral fingerprint is stable but disagreement is also informative, combine them — spectral fingerprint predicts *why* methods disagree, disagreement quantifies *how much* they disagree. This creates a two-dimensional diagnostic (mechanism + magnitude).
- **Angle 3 (Selector) + Innovator Angle 3 (Multi-Fidelity Ladder)**: The lightweight selector can include "minimum required fidelity" as an additional feature, potentially improving AUROC. Conversely, if the selector's learned coefficients reveal that gradient norm is the dominant feature, this corroborates the multi-fidelity hypothesis (gradient norm ~ spectral energy ~ fidelity requirement).

### Mapping to Contribution Structure

- **C0 (TRV diagnostic)**: Angle 1 redefines TRV as cross-method disagreement (avoids Hessian-specific instability). Angle 2 stabilizes the original TRV definition.
- **C1 (Empirical characterization)**: All three angles produce per-test-point empirical characterizations of attribution reliability.
- **C3 (RA-TDA fusion)**: Angle 3 directly provides adaptive fusion (via method selection). Angle 1's disagreement can weight the fusion.
- **C4 ("Stable != Correct")**: Angle 3's analysis of when IF vs RepSim is correct directly addresses this — if high-confidence points favor IF but low-confidence points favor RepSim, it characterizes the reliability boundary.

### Engineering Reuse Plan

All three angles share these components (implement once, reuse):
1. ResNet-18/CIFAR-10 training pipeline (from Probe code)
2. TRAK score computation (MadryLab/trak library)
3. EK-FAC IF computation (pyDVL or Hong et al. code)
4. RepSim computation (~50 lines custom)
5. Per-test-point LDS computation (~30 lines custom)

Total new code needed: ~200-300 lines for disagreement analysis + feature extraction + logistic regression. No custom CUDA kernels, no eigendecomposition, no multi-fidelity framework.

---

## Key Open-Source Resources and References

**Libraries directly usable**:
- [MadryLab/trak](https://github.com/MadryLab/trak): TRAK scores with custom CUDA kernel, CIFAR-10 tutorial included
- [TRAIS-Lab/dattri](https://github.com/trais-lab/dattri): Unified TDA benchmark, IF/TRAK/TracIn implementations (NeurIPS 2024)
- [aai-institute/pyDVL](https://github.com/aai-institute/pydvl): IF with EK-FAC, K-FAC, CG, Arnoldi, Nystrom approximations
- Probe experiment code: `~/Research/AURA/codes/probe_experiment/` (already validated)

**Key papers informing these proposals**:
- Deng et al. (2405.17293), "Efficient Ensembles Improve TDA": Dropout/LoRA ensemble for TDA — directly applicable to Angle 2
- Park et al. (2303.12922), "TRAK": Ground truth computation + LDS evaluation — infrastructure for all angles
- Kowal et al. (2602.14869), "Concept Influence": IF-RepSim correlation 0.37-0.45 — empirical basis for Angle 1
- Yang et al. (2405.17490), "Revisit Hessian-Free IF": Identity matrix substitution for Hessian — supports cheap IF proxy in Angle 3 pilot
- Hong et al. (2509.23437), "Better Hessians Matter": Hessian hierarchy evidence — infrastructure for Angle 2
- Daunce (2505.23223, ICML 2025): Perturbed model ensemble attribution — alternative uncertainty source, complementary to all angles
- BIF (2509.26544, ICML 2025): Bayesian posterior variance for IF — another uncertainty dimension, potentially combinable with Angle 1's disagreement

**Negative evidence to report honestly**:
- Probe cross-seed TRV instability (rho ~ 0) must be reported regardless of which angle succeeds
- SI-TRV null correlation (H4 falsified) must be reported
- If Angle 3's selector achieves AUROC < 0.55, report as evidence that per-test-point features don't predict method reliability

---

## Literature Search Log

1. **Web: "training data attribution robustness ensemble fusion GitHub 2025 2026"** — Found Deng et al. (2405.17293) on efficient ensembles for TDA; dattri library (NeurIPS 2024) for unified benchmarking. No prior work combining cross-method disagreement as diagnostic. Gap confirmed for Angle 1.
2. **Web: "influence function Hessian approximation sensitivity per-sample adaptive 2025"** — Found Yang et al. (2405.17490) on Hessian-free IF (identity substitution); pyDVL provides 5+ Hessian approximation options. Confirmed identity-IF as viable cheap proxy for pilot experiments. No adaptive per-sample Hessian selection found — gap confirmed for Angles 2+3.
3. **Web: "TRAK influence function cheap proxy diagnostic reliability"** — Confirmed TRAK library maturity: 2-3 orders of magnitude cheaper than Datamodels, battle-tested PyTorch API, CIFAR-10 tutorials available. Suitable as ground truth source for all angles.
4. **Web: "dattri library data attribution benchmark comparison"** — Confirmed dattri implements IF, TRAK, TracIn, RPS families with unified API. Benchmark shows TRAK dominates in non-trivial settings. Key engineering asset.
5. **Web: "Daunce ICML 2025 code GitHub"** — No public code repository found as of 2026-03-17. Method description available from ICML proceedings. Daunce's uncertainty is complementary to (not competing with) our proposed diagnostics — they measure model perturbation variance, we measure cross-method disagreement.
6. **Web: "adaptive ensemble method selection per-sample difficulty uncertainty routing 2025"** — Found active research in adaptive ensemble selection for classification (ELSA, Smart Adaptive Ensemble) but none applied to TDA method selection. Gap confirmed for Angle 3.
