# Theoretical Perspective: AURA — Beyond Point Estimates

**Agent**: sibyl-theoretical
**Date**: 2026-03-17
**Topic**: TDA Robustness Value (TRV) and Sensitivity-Aware Training Data Attribution

---

## Executive Summary

The Probe results revealed that the original TRV (Jaccard@k across Hessian approximation tiers) has cross-seed Spearman rho approximately 0, and SI-TRV correlation is null. From a theoretical standpoint, these are not merely empirical inconveniences — they expose a fundamental mathematical structure: the TRV as originally defined conflates two orthogonal sources of variation (Hessian eigenspectrum structure vs. random seed-specific eigenvector rotation), and SI captures the wrong projection of the spectral information.

I propose three theoretically grounded angles that address these root causes with provable or provably characterizable guarantees, each building on distinct mathematical frameworks: (1) operator perturbation theory for bounding attribution sensitivity, (2) semiparametric efficiency theory for principled method fusion, and (3) information-geometric decomposition of attribution uncertainty. Each angle provides formal guarantees that the Innovator's and Pragmatist's proposals lack, while remaining experimentally feasible on GPT-2/ResNet-18 scale.

---

## Angle 1: Operator Perturbation Bounds for Per-Test-Point Attribution Sensitivity

### Core Idea

Define attribution sensitivity through the lens of **operator perturbation theory** (Bauer-Fike, Weyl inequalities). For a test point z, the influence function attribution is:

$$\alpha(z, z_i) = \nabla_\theta L(z)^T H^{-1} \nabla_\theta L(z_i)$$

When H is replaced by an approximation H_tilde (e.g., K-FAC instead of GGN), the attribution error is:

$$|\alpha(z, z_i) - \tilde{\alpha}(z, z_i)| = |\nabla_\theta L(z)^T (H^{-1} - \tilde{H}^{-1}) \nabla_\theta L(z_i)|$$

Using the identity H^{-1} - H_tilde^{-1} = H^{-1}(H_tilde - H)H_tilde^{-1}, the per-test-point attribution error satisfies:

$$|\Delta\alpha(z, z_i)| \leq \|H^{-1}\|_2 \cdot \|H_tilde - H\|_2 \cdot \|\tilde{H}^{-1}\|_2 \cdot \|\nabla_\theta L(z)\|_2 \cdot \|\nabla_\theta L(z_i)\|_2$$

The **per-test-point factor** is ||nabla_theta L(z)||_2 — the test gradient norm. But this is precisely why SI failed as a TRV proxy: SI = phi(z)^T Q^{-1} phi(z) conflates the gradient norm with the spectral amplification, yielding a quantity dominated by whichever factor has higher variance. The key theoretical insight is to **decompose the test gradient into its spectral components** relative to the Hessian eigenbasis and derive per-spectral-band sensitivity bounds.

### Theoretical Framework

**Definition (Spectral Sensitivity Profile, SSP)**. For a test point z with gradient g = nabla_theta L(z), and the Hessian H with eigendecomposition H = V Lambda V^T, define the Spectral Sensitivity Profile as:

$$SSP(z) = \left( \frac{(V_k^T g)^2}{\lambda_k^2} \right)_{k=1}^{d}$$

where V_k is the k-th eigenvector and lambda_k the k-th eigenvalue. SSP(z) captures how much attribution error each spectral mode contributes for test point z.

**Proposition 1 (Spectral Decomposition of Attribution Error)**. Let H and H_tilde share the same eigenvectors but differ in eigenvalues (lambda_k vs. tilde_lambda_k). Then:

$$\Delta\alpha(z, z_i) = \sum_{k=1}^{d} \left(\frac{1}{\lambda_k} - \frac{1}{\tilde{\lambda}_k}\right) (V_k^T g_z)(V_k^T g_{z_i})$$

The per-test-point contribution to this error is:

$$\text{Sens}(z) = \sum_{k=1}^{d} \left|\frac{1}{\lambda_k} - \frac{1}{\tilde{\lambda}_k}\right| (V_k^T g_z)^2$$

This is the test point's "sensitivity" — how much its attribution changes when Hessian eigenvalues are perturbed.

**Proposition 2 (Eigenvalue Bucket Stability)**. Under the RMT framework (Marchenko-Pastur for bulk + deterministic outliers from class structure), the eigenvalue magnitude distribution {lambda_k} is determined by the data distribution and network architecture, not by the training seed. Specifically:

- **Outlier eigenvalues** (top C eigenvalues for C classes) have magnitude O(N * class_separation^2), stable across seeds up to O(1/sqrt(N)) fluctuations (Ghorbani et al., 2019; Papyan, 2020).
- **Bulk eigenvalues** follow a Marchenko-Pastur distribution with support determined by the overparameterization ratio, stable across seeds.

Therefore, the **eigenvalue perturbation factors** |1/lambda_k - 1/tilde_lambda_k| in Sens(z) are seed-stable (they depend on the approximation scheme H vs. H_tilde, not on training randomness). The seed-dependent part is the eigenvector rotation V, but if we aggregate SSP into eigenvalue-magnitude buckets (outlier vs. edge vs. bulk), the per-bucket energy is stable because it depends on the test gradient's alignment with the class structure — a data-geometric property.

**Definition (Bucketed Spectral Sensitivity, BSS)**. Partition eigenvalues into K buckets B_1, ..., B_K by magnitude (e.g., outlier: lambda > 100, edge: 10 < lambda < 100, bulk: lambda < 10). Define:

$$BSS_j(z) = \sum_{k \in B_j} \left|\frac{1}{\lambda_k} - \frac{1}{\tilde{\lambda}_k}\right| (V_k^T g_z)^2$$

The vector BSS(z) = (BSS_1(z), ..., BSS_K(z)) is the **Bucketed Spectral Sensitivity** — a per-test-point diagnostic that decomposes attribution sensitivity by spectral region.

### Hypothesis

**H-Th1**: BSS(z) is more stable across training seeds than scalar TRV (Spearman rho > 0.5 for BSS ranking vs. rho ~ 0 for TRV). Rationale: per-bucket energy depends on test gradient alignment with class-discriminative subspaces (data property), while scalar TRV depends on top-k overlap (sensitive to individual eigenvector rotation).

**H-Th2**: The outlier-bucket sensitivity BSS_outlier(z) predicts IF reliability (correlation with LDS, Spearman rho > 0.4), while the bulk-bucket sensitivity BSS_bulk(z) is uninformative. Rationale: EK-FAC to K-FAC eigenvalue mismatch (the dominant error source per Hong et al., accounting for 41-65% of total error) is concentrated in the outlier eigenspace where Kronecker factorization is most inaccurate.

### Why This Is Novel

- **Natural W-TRAK** (2512.09103): SI = phi(z)^T Q^{-1} phi(z) is a scalar that collapses the spectral structure. The Probe showed SI-TRV correlation is null, precisely because SI sums over all spectral modes indiscriminately. BSS decomposes by spectral region, preserving the per-mode information.
- **Daunce** (ICML 2025): Measures model perturbation variance without spectral decomposition — cannot identify *which spectral modes* cause instability.
- **BIF** (ICML 2025): Uses posterior variance, which is a single scalar per test point. BSS provides a vector diagnostic with richer information.
- **Hong et al.** (2509.23437): Decomposes error by approximation step (H -> GGN -> EK-FAC -> K-FAC) globally. BSS decomposes by spectral region *per test point*.
- **Innovator's Angle 1 (Spectral Fingerprint Routing)**: Closely related but uses energy distribution (V_k^T g)^2 / ||g||^2 without the perturbation-theoretic weighting |1/lambda_k - 1/tilde_lambda_k|. BSS is the theoretically motivated version: the perturbation weights are crucial because they capture *where the approximation actually introduces error*, not just where the test gradient has energy.

### Connection to Existing Literature

The BSS framework builds on:
- **Bauer-Fike theorem**: Eigenvalue perturbation bounds for diagonalizable matrices — provides the foundation for bounding |1/lambda_k - 1/tilde_lambda_k| given ||H - H_tilde||.
- **Weyl's inequality**: Bounds eigenvalue shifts under symmetric perturbations — applicable when H and H_tilde are both symmetric PSD.
- **Ghorbani et al. (2019)**: "An Investigation into Neural Net Optimization via Hessian Eigenvalue Density" — establishes the bulk+outlier spectral structure.
- **Papyan (2020)**: "Traces of Class/Cross-Class Structure Pervade Deep Learning Spectra" — proves outlier eigenvalues correspond to class-discriminative directions.
- **Sagun et al. (2018)**: "Empirical Analysis of the Hessian of Over-Parametrized Neural Networks" — bulk eigenvalue concentration.
- **Martin & Mahoney (2021)**: "Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory and Implications for Training" — RMT framework for weight matrices.

### Experimental Plan

| Step | Description | Time | Compute |
|------|-------------|------|---------|
| 1 | Train 5 ResNet-18 on CIFAR-10 (different seeds) | 2.5h | 1 GPU |
| 2 | Compute GGN top-100 eigenvalues/eigenvectors via Lanczos for each seed | 1.5h | 1 GPU |
| 3 | Compute EK-FAC and K-FAC eigenvalues for each seed | 1h | 1 GPU |
| 4 | For 300 test points: compute BSS(z) with 3 buckets (outlier/edge/bulk) | 30min | 1 GPU |
| 5 | Cross-seed stability: Spearman rho of BSS_outlier ranking across 5 seeds | 15min | CPU |
| 6 | Compute IF (EK-FAC) and IF (K-FAC) attributions; compute per-point LDS | 3h | 2 GPUs |
| 7 | Test BSS_outlier as predictor of IF reliability (LDS correlation) | 15min | CPU |
| **Total** | | **~9h** | **~7 GPU-hours** |

### Success Criteria

- BSS_outlier cross-seed Spearman rho > 0.5 (vs. scalar TRV rho ~ 0)
- Spearman(BSS_outlier, per-point LDS) > 0.4
- BSS_bulk has significantly lower cross-seed stability (rho < 0.3), confirming spectral decomposition adds information

### Failure Modes

1. **Eigenvector subspace rotation invalidates bucketing**: If outlier eigenvectors rotate so much across seeds that energy bucketing is unstable, BSS inherits TRV's instability. Mitigation: use eigenvalue-magnitude thresholds (which are stable) rather than eigenvector-index thresholds. The energy in the "lambda > 100" bucket is well-defined regardless of which specific eigenvectors span that subspace.
2. **Perturbation factors |1/lambda_k - 1/tilde_lambda_k| are uniform**: If EK-FAC-to-K-FAC eigenvalue mismatch is proportionally constant across all spectral modes, the perturbation weights add no information beyond gradient norm. Check: plot the eigenvalue ratio lambda_k / tilde_lambda_k vs. k. Hong et al. Figure 3 suggests the mismatch is non-uniform (concentrated in outlier modes), but this needs per-test-point verification.
3. **Lanczos convergence issues**: Top-100 eigenvectors of GGN for ResNet-18 (~11M params) may require many Lanczos iterations. Mitigation: use the Kronecker-factored structure (EK-FAC already provides an approximate eigendecomposition) rather than explicit Lanczos on the full GGN.

### Computational Cost Estimate

- ResNet-18/CIFAR-10: Kronecker-factored eigendecomposition is O(d_in^3 + d_out^3) per layer, tractable.
- GPT-2: Stochastic Lanczos quadrature for spectral density estimation, ~2 GPU-hours.
- **Success probability**: 50%. The theoretical framework is sound, but the empirical question is whether eigenvalue bucketing provides sufficient resolution to beat scalar TRV. The key uncertainty is whether outlier vs. bulk decomposition captures the relevant structure.

---

## Angle 2: Semiparametric Efficiency Theory for Principled IF-RepSim Fusion

### Core Idea

The fundamental theoretical weakness of the AURA Phase 2 proposal (and what 4/6 debate perspectives flagged) is the lack of a common estimand for IF and RepSim. Without a shared target parameter, "fusion" is ill-defined in a statistical sense. I propose to resolve this using **semiparametric efficiency theory**: define a common estimand tau(z) for which IF and RepSim are two different (possibly misspecified) estimators, derive the **efficient influence function** and the **semiparametric efficiency bound**, and show that a properly constructed fusion estimator achieves lower asymptotic variance than either component — under conditions that can be empirically verified.

### Theoretical Framework

**Step 1: Common Estimand via Distributional TDA.**

Following Mlodozeniec et al. (2506.12965), define the attribution of training point z_i to test point z as:

$$\tau(z, z_i) = \mathbb{E}_\theta[\nabla_\theta L(z)^T (H_\theta)^{-1} \nabla_\theta L(z_i)]$$

where the expectation is over the posterior/training distribution of theta. This is the "distributional influence" — the expected counterfactual effect of removing z_i, averaged over training randomness. IF approximates this by plugging in a point estimate theta_hat; RepSim approximates the gradient inner product structure without the Hessian inverse.

**Step 2: IF and RepSim as RAL estimators.**

- **IF estimator**: tau_IF(z, z_i) = g_z^T H_hat^{-1} g_{z_i}, where H_hat is a Hessian approximation. This is **regular** (admits a linear representation) but potentially **biased** due to Hessian approximation error.
- **RepSim estimator**: tau_RS(z, z_i) = sim(phi(z), phi(z_i)), where phi extracts a representation. This is a **nonparametric** estimator with lower variance but potential **model misspecification bias** (it estimates a different functional of the data distribution).

**Step 3: Efficiency bound and optimal fusion.**

Under the semiparametric model where tau is the target functional, the efficient influence function (EIF) determines the minimum achievable asymptotic variance for any regular estimator. The EIF for tau(z, z_i) can be derived as:

$$\psi_{eff}(z, z_i; \theta) = \nabla_\theta L(z)^T H^{-1} \nabla_\theta L(z_i) - \tau(z, z_i) + \text{correction terms}$$

**Proposition 3 (Optimal Fusion Weights)**. Consider the fusion estimator:

$$\hat{\tau}_{fuse}(z, z_i) = w(z) \cdot \hat{\tau}_{IF}(z, z_i) + (1 - w(z)) \cdot \hat{\tau}_{RS}(z, z_i)$$

If IF has per-test-point bias b_IF(z) and variance v_IF(z), and RepSim has bias b_RS(z) and variance v_RS(z), then the MSE-optimal fusion weight is:

$$w^*(z) = \frac{b_{RS}(z)^2 + v_{RS}(z)}{b_{IF}(z)^2 + v_{IF}(z) + b_{RS}(z)^2 + v_{RS}(z)}$$

This is **not** doubly robust in the classical sense (which requires consistent estimation of either the outcome model or the propensity score). Instead, it is **MSE-optimal adaptive fusion** — a well-studied concept in semiparametric statistics (e.g., Chernozhukov et al., 2018; Hansen, 2007 on model averaging).

**Proposition 4 (BSS as Bias Proxy)**. Under the spectral decomposition framework from Angle 1:

$$b_{IF}(z)^2 \propto \text{Sens}(z) = \sum_k \left|\frac{1}{\lambda_k} - \frac{1}{\tilde{\lambda}_k}\right| (V_k^T g_z)^2 = \|BSS(z)\|_1$$

Therefore, BSS(z) provides a **theoretically motivated proxy for the per-test-point IF bias**, which directly informs the optimal fusion weight w*(z). This creates a principled bridge from Angle 1's diagnostic (BSS) to Angle 2's fusion framework.

### Hypothesis

**H-Th3**: The MSE-optimal fusion weight w*(z) varies significantly across test points (coefficient of variation > 0.3), and the BSS-based proxy weight correlates with the oracle optimal weight (Spearman rho > 0.4).

**H-Th4**: The BSS-guided fusion estimator achieves lower mean squared error than both uniform fusion (w = 0.5) and either individual method, closing > 40% of the gap to oracle-weight fusion.

### Why This Is Novel

- **No existing work derives semiparametric efficiency bounds for TDA fusion.** The closest is Chernozhukov et al. (1608.00060) for treatment effect estimation, but the TDA setting differs fundamentally: the "treatment" is training data inclusion/exclusion, the "outcome" is model loss, and the nuisance parameters include the Hessian approximation.
- **Connects BSS (Angle 1) to fusion weights through a formal statistical framework**, rather than ad hoc heuristics. The Pragmatist's Angle 3 (lightweight selector via logistic regression) lacks this theoretical grounding — our framework shows *why* certain features should predict method quality (they proxy for b_IF(z)).
- **Structure-agnostic optimality**: Recent work (arxiv 2402.14264) proves doubly robust estimators achieve minimax optimal MSE rates *without* assuming specific functional forms. We extend this philosophy to TDA fusion: the fusion weights need not be parametrically specified — they are determined by the bias-variance tradeoff of each component.

### Connection to Existing Literature

- **Chernozhukov et al. (2018)**: Double/debiased machine learning — general framework for semiparametric inference with nuisance parameters. TDA's Hessian approximation plays the role of nuisance parameter estimation.
- **Hansen (2007)**: "Least Squares Model Averaging" — optimal combination of estimators minimizing MSE. Direct analog for IF+RepSim fusion.
- **Cinelli & Hazlett (2020)**: Sensitivity analysis framework — provides tools for quantifying how much unmeasured confounding (here: Hessian approximation error) can change conclusions.
- **Mlodozeniec et al. (2506.12965)**: d-TDA distributional framework — provides the common estimand definition.
- **Semiparametric efficient fusion** (arxiv 2210.00200): Fuses individual data and summary statistics optimally — methodological template for our IF+RepSim fusion.

### Experimental Plan

| Step | Description | Time | Compute |
|------|-------------|------|---------|
| 1 | Compute BSS(z) for 500 test points (from Angle 1 infrastructure) | 30min | 1 GPU |
| 2 | Compute IF (EK-FAC) and RepSim attributions for 500 test points | 2h | 2 GPUs |
| 3 | Compute TRAK ground truth (50 random-subset models) | 45min | 2 GPUs |
| 4 | Estimate per-point bias: b_IF(z)^2 = (IF_ranking - TRAK_ranking)^2, similarly b_RS(z)^2 | 15min | CPU |
| 5 | Compute oracle optimal weights w*(z) from bias+variance estimates | 15min | CPU |
| 6 | Compute BSS-based proxy weights: w_BSS(z) = sigmoid(-a * ||BSS(z)||_1 + b) | 15min | CPU |
| 7 | Evaluate: oracle fusion vs. BSS fusion vs. uniform fusion vs. single methods (LDS) | 15min | CPU |
| 8 | Compute Spearman(w_BSS, w*) and MSE improvement over uniform | 10min | CPU |
| **Total** | | **~4h** | **~5 GPU-hours** |

### Success Criteria

- w*(z) coefficient of variation > 0.3 (fusion weights vary meaningfully)
- Spearman(w_BSS, w*) > 0.4 (BSS predicts optimal weights)
- BSS fusion LDS > uniform fusion LDS by > 2% absolute
- BSS fusion closes > 40% of gap between best single method and oracle fusion

### Failure Modes

1. **Common estimand is vacuous**: If the distributional TDA estimand tau(z, z_i) cannot be consistently estimated by either IF or RepSim (both are misspecified for different reasons), the MSE-optimal fusion may not converge to anything meaningful. Mitigation: the framework still provides finite-sample MSE reduction even without consistency — we are minimizing empirical MSE, not claiming consistency.
2. **RepSim bias is dominant and constant**: If b_RS(z)^2 >> v_RS(z) for all z and b_RS is constant across test points, the optimal weight is approximately constant (w* ~ constant), and adaptive fusion adds nothing. This is the same failure mode as "one method dominates." Mitigation: the Probe results showing IF-RepSim correlation of 0.37-0.45 suggest significant disagreement, implying non-trivial w* variation.
3. **TRAK ground truth is noisy**: 50-model TRAK may be insufficient for reliable per-point bias estimation. Mitigation: use TRAK only for relative bias ordering (which is more robust than absolute bias estimation), and evaluate using rank-based metrics.

### Computational Cost Estimate

- ~5 GPU-hours, sharing infrastructure with Angle 1
- Theoretical derivations (Propositions 3-4) are pen-and-paper, zero compute
- **Success probability**: 45%. The theoretical framework is solid, but the empirical question is whether BSS is a sufficiently accurate bias proxy to beat uniform fusion by a meaningful margin.

---

## Angle 3: Information-Geometric Decomposition of Attribution Uncertainty

### Core Idea

The three sources of attribution uncertainty identified in the AURA problem statement — Hessian approximation sensitivity (TRV), training randomness (Daunce), and Bayesian parameter uncertainty (BIF) — are not merely "different perspectives." They correspond to **distinct geometric quantities** in the information manifold of the model parameter space. I propose a unified framework using **information geometry** (Amari, 2016) that decomposes total attribution uncertainty into orthogonal components, each with a distinct per-test-point signature.

### Theoretical Framework

**Setup.** The parameter space Theta equipped with the Fisher information metric G(theta) = E[nabla log p(y|x,theta) nabla log p(y|x,theta)^T] is a Riemannian manifold. The influence function attribution alpha(z, z_i) = g_z^T G^{-1} g_{z_i} is the **natural inner product** on the tangent space at theta, using the Fisher metric (note: G = H for log-likelihood loss at the MLE, so the Hessian inverse is the Fisher inverse metric).

**Decomposition.** Total attribution uncertainty for test point z can be decomposed into three geometrically distinct components:

1. **Metric approximation uncertainty** (AURA's TRV): When G is replaced by an approximation G_tilde, the attribution changes because the **metric on the tangent space changes**. This is the uncertainty from using a different Riemannian metric — analogous to computing distances in a curved space with an approximate metric tensor. Per-test-point sensitivity:

$$U_{metric}(z) = \sup_{\|g_{z_i}\| \leq 1} |g_z^T (G^{-1} - \tilde{G}^{-1}) g_{z_i}| = \|(G^{-1} - \tilde{G}^{-1}) g_z\|_2$$

2. **Geodesic uncertainty** (Daunce's training randomness): Different training seeds produce different theta_hat, which corresponds to different points on the manifold. Attribution uncertainty from this source is the **geodesic dispersion** of the attribution functional across the posterior:

$$U_{geodesic}(z) = \text{Var}_{\theta \sim p(\theta|D)}[\alpha(z, z_i; \theta)]$$

3. **Curvature uncertainty** (BIF's Bayesian uncertainty): The posterior uncertainty in theta induces curvature-dependent attribution uncertainty. In regions of high curvature (small eigenvalues of G), small parameter perturbations cause large attribution changes. Per-test-point:

$$U_{curvature}(z) = g_z^T G^{-1} \text{Cov}(\theta|D) G^{-1} g_z$$

**Proposition 5 (Orthogonality of Uncertainty Components)**. Under mild regularity conditions (Fisher information well-defined, theta in the interior of Theta), the three uncertainty components satisfy an approximate variance decomposition:

$$\text{Var}[\hat{\alpha}(z, z_i)] \approx U_{metric}(z) + U_{geodesic}(z) + U_{curvature}(z) + \text{cross terms}$$

The cross terms vanish asymptotically (as N -> infinity) because:
- U_metric depends on the approximation scheme (a deterministic choice), independent of training randomness
- U_geodesic depends on the training trajectory (random), independent of the Hessian approximation choice
- U_curvature depends on the posterior width (N^{-1} scaling), asymptotically decoupled from finite-sample training randomness

**Proposition 6 (Per-Test-Point Uncertainty Profile)**. Define the Attribution Uncertainty Profile (AUP) as the triple:

$$AUP(z) = (U_{metric}(z), U_{geodesic}(z), U_{curvature}(z))$$

AUP(z) provides a **complete characterization** of why attributions for test point z are unreliable. Different points in AUP space require different mitigation strategies:
- High U_metric, low others: Use better Hessian approximation (ASTRA) or RepSim bypass
- High U_geodesic, low others: Use ensemble/distributional TDA (d-TDA)
- High U_curvature, low others: Use regularization or informative priors

### Hypothesis

**H-Th5**: The three uncertainty components (U_metric, U_geodesic, U_curvature) are empirically near-orthogonal: pairwise Spearman rho < 0.3 across test points. This confirms that TRV, Daunce, and BIF measure genuinely different aspects of attribution reliability.

**H-Th6**: AUP(z) predicts attribution quality (LDS) better than any single component alone. Specifically, a linear model using all three components achieves R^2 > 0.3 for predicting per-point LDS, while any single component achieves R^2 < 0.15.

### Why This Is Novel

- **Unifies TRV, Daunce, and BIF** under a single geometric framework. Currently these are treated as competing or parallel approaches; our framework shows they are complementary components of a complete uncertainty decomposition.
- **Information geometry applied to TDA** is entirely new. Amari's framework has been applied to optimization (natural gradient), generalization bounds (PAC-Bayes via KL divergence), and model selection — but never to characterize per-test-point attribution reliability.
- **Provides a theoretical foundation for the AURA "sensitivity analysis" narrative** that the current proposal lacks. Instead of ad hoc TRV definitions, the AUP gives a principled decomposition grounded in differential geometry.
- **Goes beyond the Probe findings**: The Probe showed TRV is seed-unstable and SI-TRV is null. The AUP framework explains *why*: scalar TRV conflates U_metric with seed-specific eigenvector noise (which should be captured by U_geodesic), and SI captures a mix of U_metric and U_curvature that is dominated by whichever is larger.

### Connection to Existing Literature

- **Amari (2016)**: "Information Geometry and Its Applications" — foundational framework. The Fisher metric defines the natural Riemannian structure on the statistical manifold.
- **Martens (2020)**: "New Insights and Perspectives on the Natural Gradient Method" — connects Fisher information, natural gradient, and Kronecker-factored approximations. Directly relevant to understanding U_metric for K-FAC/EK-FAC.
- **Kunstner et al. (2019)**: "Limitations of the Empirical Fisher Approximation" — the empirical Fisher != true Fisher, creating an additional source of U_metric that is often ignored.
- **Ritter et al. (2018)**: "A Scalable Laplace Approximation for Neural Networks" — Kronecker-factored Laplace approximation provides U_curvature estimates.
- **Khan & Rue (2023)**: "The Bayesian Learning Rule" — unifies variational inference and natural gradient, connecting U_curvature to U_geodesic through the learning dynamics.

### Experimental Plan

| Step | Description | Time | Compute |
|------|-------------|------|---------|
| 1 | Compute U_metric(z) = BSS(z) from Angle 1 for 300 test points | (shared) | (shared) |
| 2 | Train 10 ResNet-18 seeds, compute IF attributions for each seed, estimate U_geodesic(z) as cross-seed attribution variance per test point | 5h | 2 GPUs |
| 3 | Compute Laplace approximation posterior covariance (Kronecker-factored), estimate U_curvature(z) per test point | 1h | 1 GPU |
| 4 | Compute pairwise Spearman rho between U_metric, U_geodesic, U_curvature | 10min | CPU |
| 5 | Fit linear model: LDS ~ U_metric + U_geodesic + U_curvature, evaluate R^2 | 10min | CPU |
| 6 | Compare with single-component models (R^2 of each alone) | 10min | CPU |
| **Total** | | **~6.5h** | **~8 GPU-hours** |

### Success Criteria

- Pairwise Spearman rho between components < 0.3 (orthogonality)
- Joint R^2 > 0.3 for LDS prediction (components are jointly informative)
- Each single component R^2 < 0.15 (no single component suffices)
- AUP-based method selection (route to RepSim when U_metric is high) improves LDS by > 2% over uniform strategy

### Failure Modes

1. **Components are highly correlated**: If U_metric and U_curvature correlate strongly (both involve the Fisher/Hessian inverse), the decomposition reduces to a two-component model. This would still be valuable (distinguishing training randomness from approximation+curvature effects) but less impactful. Mitigation: use orthogonalized components (regress each on the others).
2. **Laplace approximation for U_curvature is inaccurate**: The Kronecker-factored Laplace approximation may be too crude for reliable per-test-point U_curvature estimation. Mitigation: use MC dropout as a computationally cheaper (but theoretically less clean) proxy.
3. **10 seeds insufficient for U_geodesic**: Training variance estimation with 10 samples has high uncertainty. Mitigation: use efficient ensembles (Deng et al., 2405.17293) to generate 50 pseudo-seeds from 1 training run.
4. **Joint R^2 is low**: If none of the three components (nor their combination) predicts LDS, it means attribution quality is driven by factors outside our decomposition (e.g., dataset-specific artifacts, evaluation metric noise). This is a valid negative result that constrains the space of possible explanations.

### Computational Cost Estimate

- Dominant cost: 10-seed training (5 GPU-hours) + per-seed IF computation (3 GPU-hours)
- Total: ~8 GPU-hours
- **Success probability**: 35%. This is the most theoretically ambitious angle. The mathematical framework is clean, but empirically demonstrating near-orthogonality requires precise estimation of each component, which may be noisy at ResNet-18 scale.

---

## Comparative Assessment

| Criterion | Angle 1: Operator Perturbation (BSS) | Angle 2: Semiparametric Fusion | Angle 3: Info-Geometric Decomposition |
|-----------|--------------------------------------|-------------------------------|--------------------------------------|
| **Novelty** | High (spectral decomposition of TDA sensitivity) | Very High (efficiency theory for TDA fusion) | Very High (information geometry for TDA) |
| **Theoretical depth** | Medium-High (perturbation theory, well-established) | High (semiparametric efficiency, minimax optimality) | Very High (Riemannian geometry, novel application) |
| **Addresses cross-seed problem** | Yes (eigenvalue buckets are seed-stable) | Indirectly (optimal weights are per-model) | Yes (decomposes seed-dependence as U_geodesic) |
| **Provable guarantees** | Yes (perturbation bounds are exact) | Yes (MSE-optimal fusion, finite-sample) | Approximate (asymptotic orthogonality) |
| **Practical output** | BSS diagnostic vector per test point | Optimal fusion weights per test point | Complete uncertainty profile per test point |
| **Compute cost** | ~7 GPU-hours | ~5 GPU-hours (shares Angle 1 infra) | ~8 GPU-hours |
| **Success probability** | 50% | 45% | 35% |
| **Paper contribution type** | Diagnostic tool + theoretical bound | Principled fusion method + efficiency theory | Unifying framework + decomposition |
| **Fallback if fails** | Perturbation bounds still hold; BSS characterization is descriptive | Uniform fusion with theoretical MSE analysis | Orthogonality analysis as empirical finding |

---

## Recommended Strategy

**Primary**: Angle 1 (Operator Perturbation Bounds / BSS) — provides the strongest theoretical foundation with the most direct empirical test. BSS is the theoretically motivated version of the Innovator's Spectral Fingerprint Routing: both decompose test-point sensitivity by spectral region, but BSS weights by actual perturbation factors rather than raw energy. This angle has the highest success probability and produces a clean, publishable diagnostic even if fusion (Angle 2) fails.

**Secondary**: Angle 2 (Semiparametric Efficiency for Fusion) — builds directly on Angle 1's BSS output to provide *principled* fusion weights. This resolves the debate's strongest criticism (lack of common estimand for IF-RepSim fusion) by grounding fusion in MSE-optimal adaptive combination theory. Shares all infrastructure with Angle 1, so marginal compute cost is only ~2 GPU-hours.

**Tertiary/Future Work**: Angle 3 (Information-Geometric Decomposition) — highest ceiling but also highest risk. Best positioned as a "unifying perspective" in the Discussion section of the paper, or as the theoretical backbone of a follow-up paper that includes empirical comparison with Daunce and BIF. The 10-seed training cost is the main bottleneck; if budget permits, running the orthogonality analysis would significantly strengthen the paper's theoretical narrative.

---

## Integration with Other Perspectives

### Synergy with Innovator Proposals

- **Innovator Angle 1 (Spectral Fingerprint Routing) vs. our Angle 1 (BSS)**: BSS is the theoretically motivated generalization. The Innovator's SCR (Spectral Concentration Ratio) is SSP projected to a scalar; BSS adds the perturbation weighting |1/lambda_k - 1/tilde_lambda_k| which accounts for where the approximation actually introduces error. Recommendation: present BSS as the theoretical framework, SCR as the cheap approximation.
- **Innovator Angle 3 (Multi-Fidelity Ladder)**: The multi-fidelity concept maps directly to our semiparametric framework. Each fidelity level (K-FAC, EK-FAC, GGN) produces a different estimator with different bias-variance profile. Angle 2's optimal fusion extends naturally to multi-estimator combination, not just IF+RepSim binary fusion.

### Synergy with Pragmatist Proposals

- **Pragmatist Angle 1 (Disagreement Diagnostic)**: Cross-method disagreement is an empirical proxy for the MSE gap between methods. Our Angle 2 provides the theoretical framework: disagreement should be weighted by bias estimates (via BSS), not treated as a uniform signal. This elevates the Pragmatist's approach from heuristic to principled.
- **Pragmatist Angle 3 (Lightweight Selector)**: The Pragmatist's logistic regression features (confidence, loss, gradient norm, entropy) should correlate with BSS components — gradient norm is the dominant term in BSS when perturbation factors are uniform. Our framework predicts *which* features should be informative and *why*, adding interpretability to the Pragmatist's black-box selector.

### Mapping to AURA Contribution Structure

- **C0 (TRV diagnostic)**: Angle 1 provides a theoretically grounded replacement for scalar TRV: BSS, a vector diagnostic with perturbation-theoretic justification and predicted seed-stability.
- **C1 (Empirical characterization)**: Angle 3's AUP gives a complete uncertainty decomposition — richer than TRV distribution alone.
- **C2 (SI-TRV bridge)**: SI is a collapsed version of BSS. The theoretical framework explains *why* SI-TRV correlation is null (SI sums indiscriminately over spectral modes with conflicting perturbation sensitivities) and proposes BSS as the corrected diagnostic.
- **C3 (RA-TDA fusion)**: Angle 2 provides MSE-optimal fusion weights derived from semiparametric theory, replacing the ad hoc TRV-guided lambda.
- **C4 ("Stable != Correct")**: The AUP framework distinguishes metric stability (TRV) from correctness (which also depends on U_geodesic and U_curvature). A point can be metrically stable (low U_metric) but geodesically unstable (high U_geodesic), meaning the attribution is consistent across Hessian approximations but wrong due to training randomness.

---

## Literature Search Log

1. **Web: "information-theoretic bounds training data attribution sensitivity Hessian approximation 2025 2026"** — Found ASTRA (2507.14740, NeurIPS 2025) using EK-FAC preconditioner on Neumann iterations; "Learning to Weight Parameters for Data Attribution" (2506.05647) acknowledging Hessian approximation as a fundamental limitation; Data Shapley in One Training Run (ICLR 2025) for efficient gradient-Hessian-gradient products. No existing work derives per-test-point sensitivity bounds from operator perturbation theory. Gap confirmed for Angle 1.

2. **Web: "conformal prediction ranking stability distribution-free guarantees attribution 2025 2026"** — Found "Distribution-informed Efficient Conformal Prediction for Full Ranking" (2601.23128) extending CP to ranking tasks; "Conformal Tradeoffs: Guarantees Beyond Coverage" (2602.18045) examining operational guarantees. No application of CP to TDA attribution reliability. Relevant for Innovator's Angle 2 (Conformal Attribution Sets); informs our framework as an alternative distribution-free approach.

3. **Web: "Fisher information matrix eigenspectrum per-sample sensitivity bounds convergence proof 2025"** — Found "Towards Quantifying the Hessian Structure of Neural Networks" (2025) showing channel count drives block-diagonal structure; "Sensitivity-Preserving of Fisher Information Matrix" (2409.15906). Confirms FIM spectral structure is architecture-determined, supporting our stability claims for eigenvalue buckets.

4. **Web: "Cramer-Rao bound influence function estimation error rate optimal convergence data attribution"** — No direct connection found between Cramer-Rao bounds and TDA error analysis. Gap confirmed: per-test-point estimation efficiency bounds for TDA are unstudied.

5. **Web: "random matrix theory Hessian neural network bulk outlier eigenvalue stability across training seeds 2024 2025"** — Found RMT analysis confirming Marchenko-Pastur bulk + class-structure outliers (Ghorbani 2019, Martin & Mahoney 2021). "Towards Quantifying the Hessian Structure" (2025) uses RMT framework. Confirms eigenvalue magnitude distribution is seed-stable, supporting BSS bucketing approach.

6. **Web: "doubly robust estimation semiparametric efficiency bound minimax optimal fusion adaptive 2025"** — Found "Structure-agnostic Optimality of Doubly Robust Learning" (2402.14264) proving minimax optimality of DR estimators; "Semiparametric Efficient Fusion" (2210.00200) for combining individual data and summary statistics. Provides direct methodological templates for Angle 2's fusion framework.

---

## Key References (Theoretical Foundations)

**Perturbation Theory (Angle 1)**:
- Bauer & Fike (1960), "Norms and exclusion theorems" — eigenvalue perturbation bounds
- Weyl (1912), "Das asymptotische Verteilungsgesetz der Eigenwerte linearer partieller Differentialgleichungen" — eigenvalue inequalities for symmetric matrices
- Stewart & Sun (1990), "Matrix Perturbation Theory" — comprehensive treatment
- Ghorbani et al. (2019), "An Investigation into Neural Net Optimization via Hessian Eigenvalue Density" — bulk+outlier structure
- Papyan (2020), "Traces of Class/Cross-Class Structure Pervade Deep Learning Spectra" — class-discriminative outliers

**Semiparametric Efficiency (Angle 2)**:
- Chernozhukov et al. (2018, 1608.00060), "Double/Debiased ML" — semiparametric inference framework
- Hansen (2007), "Least Squares Model Averaging" — optimal estimator combination
- Robins et al. (1994), "Estimation of Regression Coefficients When Some Regressors Are Not Always Observed" — foundational DR estimation
- Newey (1990), "Semiparametric Efficiency Bounds" — efficiency bound derivations
- arxiv 2402.14264 (2024), "Structure-agnostic Optimality of Doubly Robust Learning" — minimax optimality

**Information Geometry (Angle 3)**:
- Amari (2016), "Information Geometry and Its Applications" — foundational framework
- Martens (2020), "New Insights and Perspectives on the Natural Gradient Method" — Fisher metric and Kronecker approximations
- Kunstner et al. (2019), "Limitations of the Empirical Fisher Approximation" — empirical vs. true Fisher
- Khan & Rue (2023), "The Bayesian Learning Rule" — unifying variational inference and natural gradient
- Ritter et al. (2018), "A Scalable Laplace Approximation for Neural Networks" — Kronecker-factored posterior

**TDA-Specific**:
- Hong et al. (2509.23437), "Better Hessians Matter" — Hessian hierarchy, eigenvalue mismatch analysis
- Li et al. (2512.09103), "Natural W-TRAK" — SI as Lipschitz bound, spectral amplification
- Mlodozeniec et al. (2506.12965), "d-TDA" — distributional framework, common estimand
- Kowal et al. (2602.14869), "Concept Influence" — IF-RepSim low correlation (0.37-0.45)
- Deng et al. (2405.17293), "Efficient Ensembles Improve TDA" — efficient ensemble methods
