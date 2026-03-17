# Interdisciplinary Perspective: AURA — Beyond Point Estimates

**Agent**: sibyl-interdisciplinary
**Date**: 2026-03-17
**Topic**: TDA Robustness Value (TRV) and Sensitivity-Aware Training Data Attribution

---

## Executive Summary

The Probe results revealed two critical structural failures: (1) cross-seed TRV instability (Spearman rho ~ 0), exposing TRV as a model-instance property rather than a test-point intrinsic; (2) SI-TRV decorrelation, showing that spectral leverage and Hessian approximation sensitivity are orthogonal dimensions. These are not mere empirical inconveniences — they are symptoms of deeper structural issues that become transparent when viewed through the lenses of **robust statistics**, **Bayesian neuroscience (precision-weighted predictive coding)**, and **metrology (measurement uncertainty propagation)**. Each field has independently solved analogous problems of quantifying estimator reliability, adaptively fusing unreliable information sources, and propagating method-choice uncertainty into final estimates.

I propose three interdisciplinary angles that directly address the Probe failures, grounded in concrete structural correspondences rather than surface metaphors.

---

## Angle 1: Robust Statistics — Hampel's Influence Function and Gross Error Sensitivity

### Source Field: Robust Statistics (Hampel, Ronchetti, Rousseeuw, Stahel)

### The Structural Correspondence

The connection between AURA's TRV and robust statistics is not metaphorical — it is a precise mathematical isomorphism that the debate synthesis already flagged as a "zero-cost theoretical upgrade."

**In robust statistics**: The Influence Function (IF) of an estimator T at distribution F measures the effect of an infinitesimal contamination at point x:

$$IF(x; T, F) = \lim_{\epsilon \to 0} \frac{T((1-\epsilon)F + \epsilon \delta_x) - T(F)}{\epsilon}$$

The **Gross Error Sensitivity (GES)** is the supremum of the IF over all possible contamination points:

$$\gamma^* = \sup_x ||IF(x; T, F)||$$

Critically, Hampel defined **local sensitivity** measures that quantify the instability of the estimator at specific data points. The **sensitivity curve** SC(x) = n[T(x_1,...,x_{n-1}, x) - T(x_1,...,x_{n-1})] is the finite-sample analogue that varies across observations.

**In AURA's TDA context**: Replace "estimator T" with "attribution ranking function A_H" (parameterized by Hessian approximation H), "distribution F" with "the specific Hessian choice," and "contamination" with "Hessian approximation perturbation." Then:

- **TRV(z)** = sensitivity of attribution ranking at test point z to Hessian approximation changes = a discrete analogue of Hampel's sensitivity curve
- The **cross-seed instability** maps to the well-known phenomenon that the sensitivity curve depends on the specific sample (model instance), not just the data point — Hampel's influence function resolves this by taking the asymptotic limit

### What This Transplant Solves

**The cross-seed instability problem** has a precise analogue and solution in robust statistics:

1. **From sensitivity curve to influence function**: The finite-sample sensitivity curve (analogous to per-model TRV) is unstable across samples (seeds). The resolution is to move to the asymptotic influence function, which depends on the underlying distribution, not the specific sample. For AURA, this means defining TRV not relative to a single trained model but relative to the *distribution over models* induced by training randomness — i.e., **Ensemble TRV** is exactly the right move, and it has theoretical backing from Hampel's framework.

2. **Breakdown point as a complementary diagnostic**: Hampel's breakdown point (the smallest fraction of contamination that makes the estimator arbitrarily bad) provides a *global* robustness measure complementary to the *local* sensitivity curve. For TDA, a "breakdown point" would quantify: "what fraction of the Hessian eigenspectrum can be arbitrarily wrong before the top-k attribution ranking changes completely?" This is directly computable from the probe data.

3. **Minimax optimal B-robust estimators**: Hampel's theory shows that for a given GES bound, there exists an optimal estimator that minimizes asymptotic variance while satisfying the robustness constraint. Applied to TDA: given a budget for Hessian approximation quality (e.g., "I can afford K-FAC but not full GGN"), what is the minimax optimal attribution procedure? This frames RA-TDA's adaptive fusion as a **B-robust estimator design problem** with known optimality theory.

### Grounding in Existing Cross-Disciplinary Work

- Koh & Liang (2017, ICML) explicitly borrowed the term "influence function" from robust statistics, but imported only the definition — not the full robustness theory (breakdown point, GES, B-robustness). AURA can complete the transplant.
- Basu et al. (2020, "Influence Functions in Deep Learning Are Fragile") empirically showed IF fragility without connecting to Hampel's theoretical framework for diagnosing and bounding fragility.
- Broderick et al. (2020, "An Automatic Finite-Sample Robustness Metric") developed AMIP (Approximate Maximum Influence Perturbation), which is the closest existing bridge — they define a "robustness value" analogous to TRV but for Bayesian posteriors, not TDA rankings. AMIP's structural insight: replace leave-one-out with "what is the worst-case subset removal," and express the answer via the influence function.

**Key references**:
- Hampel, Ronchetti, Rousseeuw & Stahel, *Robust Statistics: The Approach Based on Influence Functions* (Wiley, 1986) — foundational theory of sensitivity curves, GES, and B-robustness
- Broderick, Giordano & Meager (2020), "An Automatic Finite-Sample Robustness Metric: Can Dropping a Few Observations Change the Conclusion?" — AMIP framework
- Basu et al. (2020), "Influence Functions in Deep Learning Are Fragile" — empirical IF fragility without Hampel connection

### Concrete Experimental Plan

**Experiment 1 (Ensemble IF as asymptotic TRV)**:
- Train M=10 models with different seeds on CIFAR-10/ResNet-18
- For each test point z, compute attribution rankings under Full GGN and K-FAC for all 10 models
- Define Ensemble-TRV(z) = mean Jaccard@10 across models (the empirical analogue of Hampel's influence function)
- **Testable prediction**: Ensemble-TRV cross-validation stability (leave-one-model-out Spearman rho) > 0.6, even though single-model TRV rho ~ 0
- **Computational cost**: 10 models x 2 approximations x 100 test points = ~20 GPU-hours (4x probe cost)
- **Success probability**: 65% — the ensemble averaging should stabilize, but the number of seeds needed for convergence is unknown

**Experiment 2 (Attribution breakdown point)**:
- For each test point z, progressively perturb the GGN eigenspectrum (replace top-k eigenvalues with K-FAC values, sweeping k from 1 to d)
- Measure the fraction k*/d at which Jaccard@10 drops below 0.5 — this is the "attribution breakdown point" BP(z)
- **Testable prediction**: BP(z) should be highly variable across test points (std > 0.15) and correlate with Ensemble-TRV (Spearman > 0.5)
- **Computational cost**: ~5 GPU-hours (eigendecomposition reuse)
- **Success probability**: 70% — the spectral perturbation sweep is well-defined and should reveal per-point structure

---

## Angle 2: Bayesian Neuroscience — Precision-Weighted Predictive Coding

### Source Field: Computational Neuroscience (Friston, Rao & Ballard, Clark)

### The Structural Correspondence

The brain faces an almost identical problem to AURA: it receives information from multiple sensory channels (vision, audition, proprioception) that differ in reliability depending on context, and must adaptively weight these channels to form coherent percepts. The solution — **precision-weighted predictive coding** — provides a principled, biologically validated framework for adaptive information fusion under channel-specific reliability estimation.

**In predictive coding**: The brain maintains a generative model that predicts sensory inputs. Prediction errors (the difference between expected and actual signals) drive belief updates, but crucially, these errors are **weighted by their precision** (inverse variance):

$$\text{belief update} \propto \sum_i \pi_i \cdot \epsilon_i$$

where $\pi_i$ is the precision (reliability) of channel i, and $\epsilon_i$ is the prediction error from channel i. When a channel is unreliable (low precision), its prediction errors are attenuated; when reliable (high precision), they are amplified. The precision estimates themselves are learned through a second-order optimization that tracks the variance of prediction errors over time.

**In AURA's TDA context**: Map the structural correspondence:

| Neuroscience | AURA |
|---|---|
| Sensory channel (vision, audition) | Attribution method (IF, RepSim) |
| Prediction error | Attribution score / ranking |
| Precision (inverse variance) | TRV (attribution stability) |
| Multisensory integration | RA-TDA adaptive fusion |
| Context-dependent reliability | Per-test-point method reliability |
| Precision estimation via error variance | TRV estimation via Hessian tier agreement |

The correspondence is not superficial — it shares the same mathematical structure of **inverse-variance-weighted combination of noisy estimates**. The key neuroscience insight that AURA can import is the **hierarchical precision estimation** mechanism.

### What This Transplant Solves

**The "precision of precision" problem** (how reliable is TRV itself?):

1. **Hierarchical precision**: In predictive coding, the brain doesn't just estimate precision — it estimates the *reliability of its precision estimates* (meta-precision or "precision of precision"). Volatile environments (where reliability changes rapidly) require higher meta-precision. For AURA: the cross-seed instability of TRV means TRV's own precision is low. The neuroscience solution: estimate **meta-TRV** — the variance of TRV across model instances — and use it to modulate how strongly TRV influences fusion weights. When meta-TRV is high (TRV is stable across seeds), trust TRV for routing; when meta-TRV is low, fall back to uniform weighting.

2. **The "inverse effectiveness" principle**: A robust finding in multisensory neuroscience is that combining modalities is *more* beneficial when individual channel quality is low (the "principle of inverse effectiveness" — Stein & Meredith, 1993). Applied to AURA: the benefit of IF+RepSim fusion should be greatest precisely when individual methods are unreliable (low TRV for IF, low semantic match for RepSim). This predicts a specific interaction effect testable in experiments.

3. **Volatility-driven adaptation**: The brain adjusts its learning rate for precision based on environmental volatility (Mathys et al., 2011, Hierarchical Gaussian Filter). For AURA across different model architectures or datasets: some settings have stable TRV (low volatility → trust TRV), others have volatile TRV (high volatility → rely less on TRV, more on prior about method reliability).

### Grounding in Existing Cross-Disciplinary Work

- **Friston (2010), "The free-energy principle: a unified brain theory?"** — Foundational framework for precision-weighted prediction error minimization. The mathematical formalism (variational Bayes with precision-weighted prediction errors) is directly applicable to AURA's fusion problem.
- **Ernst & Banks (2002, Nature), "Humans integrate visual and haptic information in a statistically optimal fashion"** — Empirical demonstration that human multisensory integration follows Maximum Likelihood Estimation (MLE) with inverse-variance weighting. The exact same mathematical structure applies to combining IF and RepSim when their variances are known.
- **Mathys et al. (2011, Frontiers in Human Neuroscience), "A Bayesian foundation for individual learning under uncertainty in changing environments"** — The Hierarchical Gaussian Filter provides a computational mechanism for tracking time-varying precision, directly applicable to tracking TRV stability across model instances.
- **Uncertainty estimation with prediction-error circuits (Nature Communications, 2025)** — Recent work showing hierarchical prediction-error networks can estimate both sensory and prediction uncertainty, validating the neural implementation of precision estimation.

### Concrete Experimental Plan

**Experiment 3 (Precision-weighted TDA fusion)**:
- Define $\pi_{IF}(z) = 1/\text{Var}_{seeds}[\text{rank}_{IF}(z)]$ as the precision of IF attribution at test point z
- Define $\pi_{Rep}(z) = 1/\text{Var}_{seeds}[\text{rank}_{Rep}(z)]$ as the precision of RepSim attribution
- Fused attribution: $\text{rank}_{fused}(z) = \frac{\pi_{IF}(z) \cdot \text{rank}_{IF}(z) + \pi_{Rep}(z) \cdot \text{rank}_{Rep}(z)}{\pi_{IF}(z) + \pi_{Rep}(z)}$
- Compare against: (a) naive 0.5:0.5 ensemble, (b) TRV-weighted ensemble, (c) individual methods
- **Testable prediction**: Precision-weighted fusion should outperform naive ensemble by >3% LDS, and the gain should be largest on test points where one method has high precision and the other has low precision (the "inverse effectiveness" prediction)
- **Model**: GPT-2 / BERT-base (small enough for multi-seed training)
- **Computational cost**: 10 seeds x 2 methods x 200 test points = ~30 GPU-hours
- **Success probability**: 55% — the principle is sound but the empirical benefit depends on whether IF and RepSim actually have complementary failure modes (H2 from the debate)

**Experiment 4 (Meta-TRV as fusion moderator)**:
- Compute meta-TRV(z) = std of TRV(z) across 10 seeds
- Test whether conditioning fusion weights on meta-TRV improves robustness: use precision-weighting when meta-TRV < median (TRV is reliable), use uniform weights when meta-TRV > median (TRV is unreliable)
- **Testable prediction**: This two-tier strategy should match or exceed flat precision-weighting, with the advantage concentrated on high meta-TRV test points
- **Computational cost**: Marginal (reuses Experiment 3 data)
- **Success probability**: 50% — conceptually compelling but may require more than a binary threshold

---

## Angle 3: Metrology — GUM Uncertainty Propagation and Sensitivity Coefficients

### Source Field: Measurement Science / Metrology (BIPM, ISO GUM)

### The Structural Correspondence

Metrology — the science of measurement — has spent decades developing rigorous frameworks for quantifying how **method choices** (instrument selection, calibration procedure, environmental conditions) propagate into **measurement uncertainty**. The ISO Guide to the Expression of Uncertainty in Measurement (GUM) provides a mature, standardized methodology that maps remarkably well onto AURA's problem.

**In metrology (GUM framework)**: A measurand Y depends on input quantities $X_1, ..., X_N$ through a measurement model $Y = f(X_1, ..., X_N)$. Each input has uncertainty $u(x_i)$. The combined standard uncertainty is:

$$u_c^2(y) = \sum_{i=1}^N \left(\frac{\partial f}{\partial x_i}\right)^2 u^2(x_i) + 2 \sum_{i<j} \frac{\partial f}{\partial x_i} \frac{\partial f}{\partial x_j} u(x_i, x_j)$$

The **sensitivity coefficients** $c_i = \partial f / \partial x_i$ quantify how much each input uncertainty contributes to the output uncertainty. The critical insight: uncertainty propagation is **not uniform** — it depends on the sensitivity coefficients, which vary by measurement configuration.

**In AURA's TDA context**:

| Metrology | AURA |
|---|---|
| Measurand Y | Attribution ranking for test point z |
| Input quantities $X_i$ | Method choices (Hessian approximation, damping, projection dim) |
| Input uncertainty $u(x_i)$ | Approximation quality gap |
| Sensitivity coefficient $c_i$ | Per-test-point sensitivity to approximation choice |
| Combined uncertainty $u_c(y)$ | TRV (attribution uncertainty) |
| Uncertainty budget | Decomposition of TRV by source |

### What This Transplant Solves

**TRV decomposition and the "which approximation matters most" question**:

1. **Uncertainty budget for attribution**: GUM's framework naturally decomposes total uncertainty into contributions from each source. For TDA: decompose TRV(z) into contributions from (a) Kronecker factorization error, (b) eigenvalue damping, (c) projection dimensionality, (d) training seed randomness. The probe already showed that Full GGN -> K-FAC is the dominant source — GUM formalizes this as "the sensitivity coefficient for Kronecker factorization is largest." This decomposition tells practitioners *which* approximation choice to improve for maximum TRV gain.

2. **Type A vs Type B uncertainty**: GUM distinguishes Type A uncertainty (estimated from repeated measurements — analogous to multi-seed variance) from Type B uncertainty (estimated from prior knowledge — analogous to theoretical bounds like the spectral amplification factor kappa). The cross-seed instability is a Type A uncertainty; the kappa-based SI bound is a Type B uncertainty. GUM combines both using the same formalism. For AURA: **TRV should be a combined Type A + Type B uncertainty**, using multi-seed empirical variance *and* theoretical spectral bounds jointly.

3. **Expanded uncertainty and coverage factor**: GUM defines expanded uncertainty U = k * u_c, where k is a coverage factor (typically 2 for 95% confidence). For TDA: the "expanded TRV" would give a confidence interval for the attribution ranking — "with 95% confidence, the true top-10 attribution set contains at least 6 of these 10 samples." This is a directly actionable output for practitioners.

### Grounding in Existing Cross-Disciplinary Work

- **BIPM JCGM 100:2008 (GUM)** — The international standard for measurement uncertainty evaluation. The law of propagation of uncertainty and sensitivity coefficient methodology is mature (40+ years of development).
- **JCGM GUM-6:2020** — Extended GUM for cases with multiple measurands, directly applicable when attribution rankings are multivariate outputs.
- **Forbes & Soares (2019), "The GUM tree calculator for uncertainty analysis"** — Software implementation of GUM that demonstrates practical decomposition of uncertainty budgets, providing a template for AURA's implementation.
- **Giordano et al. (2019), "A Swiss Army Infinitesimal Jackknife"** (AISTATS) — The closest existing ML-metrology bridge. Uses infinitesimal jackknife (closely related to influence functions) to propagate parameter uncertainty into downstream predictions. Does not address Hessian approximation choice uncertainty.

### Concrete Experimental Plan

**Experiment 5 (TRV uncertainty budget)**:
- Using the probe data (3 seeds, 5 Hessian tiers, 100 test points), compute a formal uncertainty budget for each test point:
  - Source 1: Kronecker factorization (Full GGN vs K-FAC contribution)
  - Source 2: Diagonal approximation (K-FAC vs Diagonal contribution)
  - Source 3: Hessian removal (Diagonal vs Identity contribution)
  - Source 4: Training seed variance (across 3 seeds)
- For each test point, compute the relative contribution of each source to total TRV variance
- **Testable prediction**: Source 1 (Kronecker factorization) should dominate the budget for >70% of test points, consistent with Hong et al.'s finding that K-FAC eigenvalue mismatch is the dominant error source. But the remaining ~30% where other sources dominate are the diagnostically interesting cases.
- **Computational cost**: Near-zero (reuses probe data)
- **Success probability**: 85% — this is essentially a data analysis exercise on existing probe outputs

**Experiment 6 (Combined Type A + Type B TRV)**:
- Type A component: Empirical variance of Jaccard@10 across 10 training seeds
- Type B component: Theoretical bound based on spectral amplification — define kappa(z) = ||g_z||^2 / (lambda_min * ||g_z||_H^2), where the ratio measures how much test gradient z projects onto poorly-conditioned Hessian modes
- Combined TRV: u_c(z) = sqrt(u_A(z)^2 + u_B(z)^2) following GUM formalism
- **Testable prediction**: Combined TRV should have higher cross-seed stability than pure Type A TRV (addressing the probe's critical failure), because the Type B component provides a model-independent anchor
- **Computational cost**: ~25 GPU-hours (10 seeds)
- **Success probability**: 60% — depends on whether the Type B (spectral) component adds meaningful information beyond Type A (empirical) variance

---

## Synthesis: A Unified Framework from Three Fields

The three interdisciplinary angles converge on a coherent research program that directly addresses the Probe failures:

### Convergent Insights

1. **The cross-seed problem is a finite-sample artifact** (all three fields agree):
   - Robust statistics: sensitivity curve vs. influence function (finite vs. asymptotic)
   - Neuroscience: single-trial precision vs. learned precision (instantaneous vs. integrated)
   - Metrology: Type A uncertainty reduces with repeated measurements
   - **Resolution**: Ensemble TRV (averaging over model instances) is the principled solution, supported by independent theoretical traditions

2. **Adaptive fusion should be precision-weighted, not threshold-based** (neuroscience + metrology):
   - The original RA-TDA design uses TRV as a routing signal (IF when stable, RepSim when unstable). This is a hard threshold.
   - Both neuroscience (precision-weighted prediction errors) and metrology (sensitivity-coefficient-weighted combination) suggest a **continuous, inverse-variance weighting**: $w_{IF}(z) \propto 1/\text{Var}[\text{attr}_{IF}(z)]$
   - This sidesteps the "TRV definition" problem entirely — you don't need a pre-defined TRV metric, just empirical variance estimates from ensemble training

3. **Decomposition before aggregation** (metrology + robust statistics):
   - Don't just compute a single TRV number — decompose it into source-specific sensitivity coefficients (GUM) and assess which sources have the highest breakdown-point risk (Hampel)
   - This decomposition provides actionable guidance: "for this test point, improve the Kronecker factorization quality" vs. "for this test point, the attribution is robust to all approximation choices"

### The Proposed Interdisciplinary Framework: **AURA-PC** (Precision-Calibrated Attribution)

Rename/reconceptualize AURA's framework using precision-weighted predictive coding as the organizing metaphor:

1. **Precision Estimation Layer** (metrology-inspired):
   - Train M models (M >= 5)
   - For each test point z, compute attribution rankings under each model and each Hessian tier
   - Estimate per-method precision: $\pi_{IF}(z) = 1/\text{Var}_{models}[\text{rank}_{IF}(z)]$, $\pi_{Rep}(z) = 1/\text{Var}_{models}[\text{rank}_{Rep}(z)]$
   - Decompose precision into source-specific components (GUM uncertainty budget)

2. **Adaptive Fusion Layer** (neuroscience-inspired):
   - Precision-weighted combination: $\text{attr}_{fused}(z) = (\pi_{IF}(z) \cdot \text{attr}_{IF}(z) + \pi_{Rep}(z) \cdot \text{attr}_{Rep}(z)) / (\pi_{IF}(z) + \pi_{Rep}(z))$
   - Meta-precision gating: when meta-TRV (variance of precision across cross-validation folds) is high, shrink toward uniform weights

3. **Robustness Certificate Layer** (robust-statistics-inspired):
   - For each test point, compute the attribution breakdown point BP(z) — how much can the Hessian approximation degrade before top-k changes completely?
   - Report expanded uncertainty: "top-10 attribution is reliable at 95% confidence for k >= BP(z) * d eigenvalue perturbations"
   - B-robust optimal fusion: choose the fusion weights that minimize worst-case attribution error subject to computational budget

### Advantages Over Original AURA Design

| Issue | Original AURA | AURA-PC (Interdisciplinary) |
|---|---|---|
| Cross-seed instability | Critical failure | Resolved by ensemble precision (Hampel's IF) |
| SI-TRV decorrelation | No cheap proxy | Replaced by empirical precision (no proxy needed) |
| Discrete TRV levels | Coarse 3-category routing | Continuous precision weighting |
| Fusion justification | Ad hoc TRV threshold | Principled inverse-variance weighting (neuroscience + metrology) |
| Robustness guarantee | None | Breakdown point certificate (robust statistics) |
| Uncertainty decomposition | None | GUM source-specific budget |
| Theoretical grounding | Weak (loose SI bound) | Strong (Hampel optimality + GUM standards + precision-weighted PE) |

### Risks and Limitations

1. **Computational cost escalation**: Ensemble TRV requires M >= 5 trained models, increasing cost by 5-10x. Mitigation: use TRAK's random projection to reduce per-model attribution cost; M=5 may suffice if convergence is fast.

2. **"Precision of precision" circularity**: Estimating precision from finite ensembles introduces its own uncertainty. Metrology handles this with degrees-of-freedom corrections (Student's t coverage factor when M is small). Must implement this correctly.

3. **The cross-disciplinary transplant may be too neat**: Each field's framework was developed for its specific domain with specific distributional assumptions (e.g., GUM assumes approximately Gaussian uncertainty propagation). TDA attribution rankings are discrete, high-dimensional, and non-Gaussian. The transplant works at the structural level but may fail at the distributional level. Empirical validation is essential.

4. **Ensemble training is not always feasible**: For large models (Llama-7B), training 5+ independent models is prohibitively expensive. Fallback: use checkpoint ensembles (different training epochs of the same run) or perturbation ensembles (Daunce-style noise injection) as cheaper approximations to the full seed ensemble.

---

## Priority Ranking of Experiments

| Priority | Experiment | Cost (GPU-hrs) | Success Prob. | Addresses |
|---|---|---|---|---|
| 1 | Exp 5: TRV uncertainty budget | ~0 (reuses probe) | 85% | Decomposition, diagnostic value |
| 2 | Exp 1: Ensemble TRV stability | ~20 | 65% | Cross-seed critical failure |
| 3 | Exp 2: Attribution breakdown point | ~5 | 70% | Robustness certificate |
| 4 | Exp 6: Combined Type A+B TRV | ~25 | 60% | Precision calibration |
| 5 | Exp 3: Precision-weighted fusion | ~30 | 55% | Adaptive fusion mechanism |
| 6 | Exp 4: Meta-TRV moderation | ~0 (reuses Exp 3) | 50% | Hierarchical precision |

**Total estimated cost**: ~80 GPU-hours (10x probe, but addresses all critical failures)
**Recommended minimum**: Experiments 1, 2, 5 (~25 GPU-hours, addresses cross-seed stability + decomposition + robustness certificate)

---

## Key Takeaway for the AURA Team

The Probe's cross-seed instability and SI-TRV decorrelation are not dead ends — they are **exactly the problems that robust statistics, Bayesian neuroscience, and metrology have mature solutions for**. The core pivot is from "TRV as a single-model diagnostic" to "TRV as a precision estimate derived from model ensembles." This pivot is independently justified by three separate fields:

- **Hampel**: Finite-sample sensitivity curves are unstable; the asymptotic influence function (= ensemble average) is the principled object
- **Predictive coding**: Single-trial reliability is noisy; precision is estimated from prediction-error variance over repeated exposures (= seed ensemble)
- **GUM**: Type A uncertainty (= empirical variance from repeated measurements) is the gold standard when available

The deepest insight is that the original AURA framing asked the wrong question. The question is not "is this test point's attribution stable?" (which is model-instance-dependent and unstable). The right question, from all three fields, is: **"what is the precision of this test point's attribution, estimated from the distribution over plausible models?"** This reframing dissolves the cross-seed problem by construction.
