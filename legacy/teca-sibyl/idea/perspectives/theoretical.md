# Theoretical Perspective: TECA Research Proposals

**Agent**: sibyl-theoretical
**Date**: 2026-03-17
**Topic**: Using model editing (ROME/MEMIT) parameter update directions as independent validation signals for TDA attribution directions; proposing the TECS (TDA-Editing Consistency Score) metric to probe knowledge geometry in parameter space.

---

## Executive Summary

The TECS proposal connects two independently motivated operations on transformer MLP weights -- rank-one editing (ROME) and gradient-based training data attribution (TDA) -- by comparing their directions in parameter space. From a theoretical standpoint, this connection is not coincidental: both operations can be derived as solutions to related optimization problems over the same linear associative memory. This perspective develops three theoretically grounded angles that formalize when and why TECS should (or should not) be nonzero, with provable bounds and information-geometric structure.

**Central theoretical claim**: ROME's rank-one update and the aggregated TDA gradient both solve constrained optimization problems over the same MLP weight matrix, but with different objective functions and different constraint sets. TECS measures the angular discrepancy between these two solution directions. Under the linear associative memory model that underlies ROME's derivation, we can derive closed-form expressions for TECS as a function of the key-value covariance structure, yielding testable predictions about when alignment should be high versus low.

---

## Angle 1: TECS Through the Lens of Linear Associative Memory Theory

### Core Insight (Improve Existing)

ROME's theoretical foundation rests on modeling MLP layers as **linear associative memories**: W*k = v, where k is the subject key vector and v is the factual value vector (Meng et al., 2022, arXiv 2202.05262). The rank-one edit computes:

```
delta_W_E = (v* - W*k*) * k*^T * C^{-1}
```

where C = E[k*k^T] is the key covariance matrix and v* is the target value. Meanwhile, the TDA gradient for a training example z_i that teaches the same fact is:

```
g_TDA(z_i) = nabla_W L(z_i; theta) = - partial_L/partial_v_i * k_i^T
```

where the loss gradient decomposes through the MLP's linear structure into an outer product of the value-space error signal and the key vector.

**Key theoretical observation**: Both delta_W_E and g_TDA are **rank-one matrices** in the same d_v x d_k space, each expressible as an outer product of a value-direction component and a key-direction component. Their cosine similarity therefore decomposes into two independent alignment terms:

```
TECS = cos(vec(delta_W_E), vec(g_TDA))
     = cos(v* - W*k*, partial_L/partial_v_i) * cos(C^{-1}*k*, k_i) / normalization
```

This decomposition reveals that TECS is high if and only if **both** conditions hold simultaneously:
1. **Value alignment**: The editing target direction (v* - W*k*) aligns with the loss gradient in value space for the training example.
2. **Key alignment**: The whitened subject key C^{-1}*k* aligns with the training example's key vector k_i.

### Mathematical Framework

**Definition 1** (Exact TECS under linear associative memory). Let W in R^{d_v x d_k} be the MLP weight matrix, C = (1/N) sum_i k_i k_i^T the empirical key covariance, and (k*, v*) the target fact. For training example z_i with key k_i and value v_i = W*k_i + epsilon_i, define:

```
TECS(z_i) = <vec(delta_W_E), vec(g_i)> / (||vec(delta_W_E)|| * ||vec(g_i)||)
```

**Proposition 1** (Rank-one decomposition). Under the linear associative memory model, TECS(z_i) decomposes as:

```
TECS(z_i) = sign(alpha_i) * cos(angle(C^{-1} k*, k_i)) * cos(angle(v* - W k*, d_v_i))
```

where alpha_i is a positive scalar and d_v_i = -partial L / partial (W k_i) is the value-space gradient direction.

*Proof sketch*: Both delta_W_E and g_i are rank-one matrices. For rank-one matrices A = a b^T and C = c d^T, cos(vec(A), vec(C)) = cos(a,c) * cos(b,d) * sign(...). The decomposition follows from the outer-product structure of both the ROME update and the backpropagation gradient through a linear layer.

**Proposition 2** (TECS bound under isotropic keys). If keys are drawn i.i.d. from N(0, sigma^2 I_{d_k}), then as N -> infinity, C -> sigma^2 I, and the key alignment term simplifies to cos(k*, k_i). For d_k >> 1, random key pairs satisfy E[cos(k*, k_i)^2] = 1/d_k, so:

```
E[TECS(z_random)^2] ~ 1/d_k * E[cos(value alignment)^2]
```

For GPT-2-XL with d_k = 1600, random TECS has standard deviation ~ 1/40 of the value alignment signal. This provides the **null distribution** for TECS and a lower bound on the effect size needed for detection.

**Proposition 3** (TECS amplification for memorized facts). If training example z_i directly teaches fact (k*, v*), then k_i is correlated with k* (same subject) and d_v_i points toward v* - W k_i (same factual error direction). Under a simple model where cos(k*, k_i) = rho_k and cos(value directions) = rho_v:

```
E[TECS(z_relevant)] ~ rho_k * rho_v
```

The ratio TECS(z_relevant) / TECS(z_random) scales as rho_k * rho_v * sqrt(d_k), which for d_k = 1600 and moderate correlations rho_k = rho_v = 0.3, gives a signal-to-noise ratio of ~3.6. **This predicts that TECS should be detectable with ~100 facts at p < 0.01.**

### Hypothesis

**H1** (Rank-one decomposition testability): The empirically measured TECS will correlate strongly (Spearman rho > 0.7) with the product cos(key alignment) * cos(value alignment), confirming the theoretical decomposition. Deviations would indicate nonlinear effects not captured by the associative memory model.

**H2** (SNR prediction): The Cohen's d between TECS and Null-A will scale approximately as rho_k * rho_v * sqrt(d_k / N_facts). For N=100 facts on GPT-2-XL, the theory predicts d > 0.3 if rho_k * rho_v > 0.08 (very weak alignment suffices).

### Why This is Novel

- Zhou et al. (2025, arXiv 2508.16082) proved task vectors approximate first-epoch negative gradients, establishing a gradient-editing bridge for fine-tuning. Our Proposition 1 establishes the analogous bridge for rank-one editing specifically, exploiting the outer-product structure unique to ROME.
- The Approximate Fisher Influence Function (Lev & Wilson, 2024, arXiv 2407.08169) reformulates influence estimation via information geometry but does not connect to knowledge editing directions.
- Wang et al. (2025, arXiv 2509.26030) analyze the associative memory model for optimizer comparison (Muon vs Adam) and show that the outer-product structure of linear associative memories governs learning dynamics. Our work applies the same structural insight to the editing-attribution comparison rather than optimizer design.
- Li et al. (2026, arXiv 2602.05725) derive scaling laws for associative memory learning, confirming that the frequency spectrum of key-value pairs governs convergence rates. This directly supports our Proposition 3's dependence on key correlation structure.

### Experimental Plan

| Step | Description | Model | Time |
|------|-------------|-------|------|
| 1 | Compute ROME delta_W for 100 CounterFact facts; extract rank-one components (value direction, whitened key) | GPT-2-XL | 20 min |
| 2 | Compute per-sample TDA gradients for top-20 BM25-retrieved documents; extract rank-one components | GPT-2-XL | 25 min |
| 3 | Compute full TECS and decomposed TECS (key alignment * value alignment) separately | - | 5 min |
| 4 | Test Proposition 1: Spearman correlation between full TECS and decomposed product | - | 2 min |
| 5 | Test Proposition 2: compare empirical Null-A distribution against theoretical 1/sqrt(d_k) prediction | - | 3 min |

**Total**: ~55 min. **Success probability**: 60%. The linear associative memory model is an idealization; real MLP layers have nonlinear activations (GELU) that break the exact rank-one decomposition. However, Meng et al. (2022) showed that this model works well enough for ROME to succeed, so moderate departures from exact rank-one structure are expected but should not destroy the signal.

### Failure Modes

- **GELU nonlinearity breaks rank-one structure**: The gradient through GELU is not a clean outer product. Mitigation: compare TECS decomposition accuracy between the linear path (skip connection) and the nonlinear path (GELU). If decomposition works for the linear component but not the full gradient, this isolates the nonlinearity's effect.
- **Key covariance C is ill-conditioned**: C^{-1} amplifies noise in low-variance key directions. This is exactly the numerical instability that r-ROME (arXiv 2403.07175) addressed with Tikhonov regularization. Mitigation: use the same regularization for both ROME and our theoretical analysis; report condition number of C.
- **BM25 retrieval misses the actual training documents**: If k_i does not correlate with k* because the retrieved document is topically related but encodes the fact differently, rho_k will be low. This is a retrieval quality issue, not a theoretical failure. Mitigation: report TECS separately for high-BM25-score vs low-BM25-score documents.

---

## Angle 2: Information-Geometric TECS -- Fisher-Weighted Alignment

### Core Insight (Cross-Domain Transfer from Information Geometry)

The standard TECS computes cosine similarity in **Euclidean** parameter space. But parameter space has a natural Riemannian geometry defined by the **Fisher Information Matrix** (FIM): F = E[nabla_theta log p(x|theta) * nabla_theta log p(x|theta)^T]. Directions that are "close" in Euclidean space may be far apart in Fisher geometry (and vice versa), because the FIM accounts for the model's sensitivity to parameter perturbations.

Li et al. (2025, arXiv 2512.09103) showed that using the **Natural Wasserstein metric** (which incorporates model feature covariance) dramatically improves TDA robustness. The NeurIPS 2024 work on diagonal Fisher estimators (Trade-Offs of Diagonal Fisher Information Matrix Estimators) established practical bounds for approximating the FIM. We propose extending TECS to a **Fisher-weighted** version:

```
TECS_Fisher = <vec(delta_W_E), F_W * vec(g_TDA)> / (||delta_W_E||_F * ||g_TDA||_F)
```

where F_W is the (block of the) Fisher information matrix corresponding to weight matrix W.

### Mathematical Framework

**Definition 2** (Fisher-TECS). For MLP weight W at layer l, define the per-layer Fisher block:

```
F_W = E_{x~D}[vec(nabla_W log p(x|theta)) * vec(nabla_W log p(x|theta))^T]
```

Then Fisher-TECS is:

```
TECS_F = cos_F(delta_W_E, g_TDA) = <delta_W_E, g_TDA>_F / (||delta_W_E||_F * ||g_TDA||_F)
```

where <A, B>_F = vec(A)^T F_W vec(B) is the Fisher inner product.

**Proposition 4** (Fisher-TECS under Kronecker approximation). Under the standard KFAC approximation F_W ~ A otimes G where A = E[k k^T] (key covariance) and G = E[delta_v delta_v^T] (value gradient covariance), Fisher-TECS decomposes as:

```
TECS_F ~ cos_A(key component of delta_W_E, key component of g_TDA) * cos_G(value component of delta_W_E, value component of g_TDA)
```

This is analogous to Proposition 1 but measures alignment in the **natural geometry** of the model rather than Euclidean geometry. Crucially, A = C (the same covariance used in ROME's formula), so the key alignment term under Fisher geometry becomes:

```
cos_A(C^{-1} k*, k_i) = <C^{-1} k*, k_i>_C / (||C^{-1} k*||_C * ||k_i||_C) = <k*, k_i> / (||k*||_{C^{-1}} * ||k_i||_C)
```

**Theorem 1** (Fisher-TECS equals Mahalanobis-corrected Euclidean TECS). Under KFAC approximation, Fisher-TECS for ROME is equivalent to computing Euclidean TECS after applying Mahalanobis normalization to both the key and value components independently. Specifically:

```
TECS_F = cos(C^{-1/2} k*, C^{1/2} k_i) * cos(G^{-1/2} v_E, G^{1/2} v_g)
```

This reveals a **symmetry-breaking** prediction: ROME's update already incorporates C^{-1} (whitening the keys), while the raw TDA gradient does not. Fisher-TECS corrects for this asymmetry.

### Hypothesis

**H3** (Fisher correction improves signal): Fisher-TECS will show higher Cohen's d than Euclidean TECS, because the Euclidean metric conflates high-variance and low-variance parameter directions. The improvement should be largest when the key covariance C has high condition number (kappa(C) > 100).

**H4** (Spectral band selectivity emerges from Fisher geometry): The Innovator's spectral TECS hypothesis (alignment peaks in mid-range singular bands) can be derived as a consequence of the Fisher metric: mid-range singular values of W correspond to directions where the Fisher information is moderate (neither too large nor too small), which is where both editing and attribution signals are most informative.

### Connection to Zhang et al. (2026, arXiv 2601.11042)

REVIVE's spectral analysis showed that dominant singular directions of W encode general abilities and are disrupted by editing. In our Fisher framework, dominant singular directions correspond to **high Fisher information** directions -- exactly the directions the model is most sensitive to. This explains why editing disrupts them: ROME's rank-one update, projected onto high-Fisher directions, causes disproportionate functional change. Our Fisher-TECS provides a quantitative measure of how much of the editing update falls in high-sensitivity vs low-sensitivity directions.

### Experimental Plan

| Step | Description | Model | Time |
|------|-------------|-------|------|
| 1 | Estimate diagonal Fisher (100 samples from OpenWebText) at target layer | GPT-2-XL | 15 min |
| 2 | Estimate KFAC factors A, G (reuse key covariance C from ROME) | GPT-2-XL | 5 min |
| 3 | Compute Fisher-TECS for 100 facts using KFAC approximation | - | 5 min |
| 4 | Compare Cohen's d: Euclidean TECS vs Fisher-TECS vs Null baselines | - | 5 min |
| 5 | Decompose Fisher-TECS by spectral band of W (top-10, 10-50, 50-200, tail) | - | 10 min |

**Total**: ~40 min (reusing Angle 1 computations). **Success probability**: 50%. The KFAC approximation may be too crude for a single layer; the full Fisher is intractable. But even the diagonal Fisher should show improvement over Euclidean if the key covariance structure matters.

### Failure Modes

- **KFAC approximation is inaccurate for transformer MLPs**: KFAC assumes key and value gradient statistics are independent. For transformers with residual connections, this assumption may be violated. Mitigation: compare KFAC Fisher-TECS against diagonal Fisher-TECS as a sanity check.
- **Fisher estimation requires too many samples**: 100 samples may underestimate the Fisher at a single layer. Mitigation: use the empirical Fisher (gradient outer products on training data) rather than the true Fisher; this is standard practice in natural gradient methods.
- **Improvement over Euclidean TECS is marginal**: If the key covariance is approximately isotropic (kappa(C) ~ 1), Fisher correction adds nothing. Report kappa(C) to diagnose this case.

---

## Angle 3: Subspace Geometry and Provable Incommensurability Bounds

### Core Insight (New Method)

If TECS is near zero (the "negative result" scenario), the theoretical question becomes: **can we prove that the editing subspace and the attribution subspace are fundamentally misaligned, and if so, by how much?** This angle develops rigorous tools for characterizing the geometric relationship between two low-dimensional subspaces in high-dimensional parameter space.

Steele (2026, arXiv 2603.02224) recently proved that catastrophic forgetting in LoRA is governed by the **minimum principal angle** between task gradient subspaces: F = alpha(1 - cos^2(theta_min)) + beta. This establishes that principal angles between gradient subspaces have direct functional consequences. We apply the same framework to characterize the editing-attribution relationship.

### Mathematical Framework

**Definition 3** (Editing and Attribution subspaces). Given N facts, define:
- **Editing subspace** S_E = span{vec(delta_W_E^{(1)}), ..., vec(delta_W_E^{(N)})} in R^{d_v * d_k}
- **Attribution subspace** S_A = span{vec(g_TDA^{(1)}), ..., vec(g_TDA^{(N)})} in R^{d_v * d_k}

Both are at most rank-N subspaces in a d_v * d_k = 1600 * 6400 ~ 10^7 dimensional space.

**Definition 4** (Principal angles). The principal angles theta_1 <= theta_2 <= ... <= theta_min(dim(S_E), dim(S_A)) between S_E and S_A are defined recursively as:

```
cos(theta_i) = max_{u in S_E, v in S_A} <u, v>   s.t.  ||u||=||v||=1, u perp u_1,...,u_{i-1}, v perp v_1,...,v_{i-1}
```

**Proposition 5** (TECS as projection onto principal components). The mean TECS across N facts can be bounded in terms of principal angles:

```
(1/N) sum_i TECS(i) <= (1/N) sum_i cos(theta_i) <= cos(theta_1)
```

with equality when editing and attribution directions are individually aligned with the principal vectors.

**Proposition 6** (Rank-one subspace structure). Since both ROME updates and TDA gradients are (approximately) rank-one matrices, S_E and S_A have special structure. Under the associative memory model:

```
S_E = {(v* - W k*) (C^{-1} k*)^T : (k*, v*) in Facts}
S_A = {d_v_i k_i^T : z_i in TrainingData}
```

Both subspaces are contained in the rank-one manifold of R^{d_v x d_k}. The principal angles between S_E and S_A on the rank-one manifold can be decomposed into key-space and value-space components:

```
cos(theta_j) ~ cos(theta_j^{key}) * cos(theta_j^{value})
```

where theta_j^{key} and theta_j^{value} are the principal angles between the key-space and value-space projections respectively.

**Theorem 2** (Incommensurability bound). If the key vectors {C^{-1} k*^{(i)}} (whitened editing keys) and {k_j} (training keys) are drawn from distributions with covariance Sigma_E and Sigma_A respectively, then the expected minimum principal angle satisfies:

```
E[cos^2(theta_1)] <= tr(Sigma_E Sigma_A) / (||Sigma_E||_F * ||Sigma_A||_F)
```

When Sigma_E and Sigma_A are nearly orthogonal (whitening decorrelates the key distribution from the raw training key distribution), this bound approaches zero, providing a **provable lower bound on the geometric gap** between editing and attribution.

**Corollary** (Whitening-induced misalignment). ROME's use of C^{-1} whitening systematically rotates the editing key space away from the natural training key space. The angular gap introduced by whitening is:

```
theta_whitening = arccos(tr(C^{-1} C) / (||C^{-1}||_F * ||C||_F)) = arccos(d_k / (||C^{-1}||_F * ||C||_F))
```

For a covariance matrix with condition number kappa, this angle increases with kappa. **This provides a concrete, testable prediction**: the gap between TECS and chance level should decrease if we remove the C^{-1} whitening from ROME's formula (at the cost of editing quality).

### Hypothesis

**H5** (Principal angle prediction): The minimum principal angle between S_E and S_A at the ROME editing layer l* will be significantly smaller than at non-editing layers (l* +/- 5), confirming layer-specific geometric alignment. This directly tests Null-B at the subspace level rather than the per-fact level.

**H6** (Whitening-induced gap): Computing TECS with an "un-whitened" ROME update (delta_W_raw = (v* - W k*) k*^T, without C^{-1}) will yield **higher** TECS values but **worse** editing success. This dissociation would prove that ROME sacrifices alignment with the natural knowledge geometry (TDA direction) in exchange for better functional editing performance.

**H7** (Forgetting connection): By analogy with Steele (2026), facts where the editing-attribution principal angle is small (high alignment) should show better editing locality preservation, because the edit "follows" the natural parameter structure rather than cutting across it.

### Experimental Plan

| Step | Description | Model | Time |
|------|-------------|-------|------|
| 1 | Collect 100 editing directions and 100 attribution directions at layers l*, l*-5, l*+5 | GPT-2-XL | 0 min (cached from Angle 1) |
| 2 | Compute SVD of each subspace matrix; extract principal angles via scipy.linalg.subspace_angles | - | 5 min |
| 3 | Compare principal angle distributions across layers (test H5) | - | 5 min |
| 4 | Compute "un-whitened" ROME TECS; compare with standard TECS (test H6) | GPT-2-XL | 10 min |
| 5 | Correlate minimum principal angle per-fact with editing locality score (test H7) | - | 5 min |
| 6 | Compute theoretical bounds from Theorem 2; compare with empirical angles | - | 5 min |

**Total**: ~30 min (mostly cached). **Success probability**: 65%. The subspace analysis is mathematically rigorous and produces meaningful results regardless of whether TECS is positive or negative -- it characterizes the geometry either way.

### Failure Modes

- **Subspace dimensionality is too low for meaningful principal angles**: With 100 facts, each subspace has at most rank 100 in a ~10^7-dimensional space. The principal angles may all be near 90 degrees simply due to dimensionality. Mitigation: use effective dimensionality (number of principal angles < 45 degrees) as the metric rather than the full angle distribution; compare against random subspace baselines.
- **Rank-one structure of gradients is violated**: Real backpropagation gradients through GELU are not exactly rank-one. Mitigation: measure the effective rank of each gradient matrix (ratio of nuclear norm to operator norm); if effective rank > 2, the rank-one decomposition in Proposition 6 is invalid and we fall back to full subspace angle analysis.
- **100 facts is too few for stable subspace estimation**: The estimated subspace may be noisy. Mitigation: use bootstrap resampling (draw 80 facts, compute angles, repeat 100 times) to obtain confidence intervals on principal angles.

---

## Computational Budget Summary

| Angle | GPU Time | CPU Time | Total | Shared with |
|-------|----------|----------|-------|-------------|
| 1. Associative Memory Theory | 45 min | 10 min | 55 min | - |
| 2. Fisher-Geometric TECS | 20 min | 20 min | 40 min | Angle 1 |
| 3. Subspace Incommensurability | 10 min | 20 min | 30 min | Angles 1-2 |
| **Combined (with caching)** | **~50 min** | **~30 min** | **~80 min** | |

All experiments fit on a single RTX 4090. The theoretical analysis itself (proving propositions, deriving bounds) requires no GPU time and can be done in parallel with computation.

---

## Recommended Priority

1. **Start with Angle 1** (Associative Memory Theory): This provides the mathematical foundation for the entire paper. The rank-one decomposition (Proposition 1) is the central theoretical insight -- if confirmed empirically, it explains *why* TECS works and predicts *when* it should be high. If the decomposition fails, it means the linear associative memory model is insufficient, which is itself an important finding.

2. **Then Angle 3** (Subspace Geometry): This provides the safety net. Whether TECS is positive or zero, the principal angle analysis yields quantitative characterization of the editing-attribution geometry. The whitening-induced misalignment prediction (Corollary to Theorem 2) is a concrete, falsifiable claim that does not require TECS to be nonzero.

3. **Angle 2** (Fisher-TECS) is the most speculative: it depends on the KFAC approximation being reasonable and the key covariance having high condition number. Run this last, primarily as a robustness check on Angle 1's Euclidean results.

---

## Key Literature Discovered (Theoretical-Specific)

These references were found through targeted searches on information geometry, subspace analysis, and associative memory theory:

| Paper | arXiv ID | Theoretical Relevance |
|-------|----------|----------------------|
| The Approximate Fisher Influence Function | 2407.08169 | Uses information geometry to reformulate influence estimation; provides the theoretical basis for our Fisher-TECS (Angle 2) |
| Subspace Geometry Governs Catastrophic Forgetting in Low-Rank Adaptation | 2603.02224 | Proves forgetting is governed by principal angles between gradient subspaces; directly analogous framework for our editing-attribution subspace analysis (Angle 3) |
| On Task Vectors and Gradients | 2508.16082 | Proves task vectors ~ first-epoch gradients with bounded second-order error; establishes the gradient-editing bridge we extend to rank-one editing |
| Muon Outperforms Adam in Tail-End Associative Memory Learning | 2509.26030 | Analyzes outer-product structure of linear associative memories; confirms that key-value frequency spectrum governs learning dynamics |
| Muon in Associative Memory Learning: Training Dynamics and Scaling Laws | 2602.05725 | Derives scaling laws for associative memory; provides theoretical backing for our Proposition 3's dependence on key correlation |
| Spectral Characterization of Sequential Editing Collapse (REVIVE) | 2601.11042 | Proves dominant singular directions encode general abilities; our Fisher-TECS explains why editing disrupts these specific directions |
| Natural Geometry of Robust Data Attribution | 2512.09103 | Proposes Natural Wasserstein metric for TDA; theoretical foundation for replacing Euclidean cosine with Fisher-weighted cosine in TECS |
| Gradient-Sign Masking for Task Vector Transport (GradFix) | 2510.09658 | Proves gradient-sign alignment ensures first-order descent; analogous guarantee for our "un-whitened TECS" direction matching |

---

## Risk Assessment (Theory-Focused)

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Linear associative memory model is too idealized for GPT-2-XL MLPs | 35% | High | Treat model as generating hypotheses; test decomposition empirically; report deviations as evidence of nonlinear knowledge encoding |
| Fisher/KFAC approximation is too crude | 40% | Medium | Use diagonal Fisher as fallback; report condition numbers; Fisher-TECS is Angle 2, not the core contribution |
| Principal angle analysis is trivially near 90 degrees due to high dimensionality | 30% | Medium | Compare against random subspace baseline; use effective alignment dimensionality rather than raw angles |
| Theoretical bounds are too loose to be informative | 25% | Low | Bounds provide qualitative predictions (scaling with d_k, kappa(C)); even loose bounds guide experimental design |
| Propositions hold but TECS is still near zero empirically | 30% | Low | The theory predicts TECS ~ rho_k * rho_v; near-zero TECS implies rho_k * rho_v is small, which is a substantive finding about knowledge geometry (editing and attribution access different aspects of knowledge encoding) |

**Overall probability of at least one theoretically grounded publishable finding**: ~75%.

The key theoretical insight is that **even a null result has theoretical content**: near-zero TECS under our framework implies that ROME's C^{-1} whitening systematically rotates the editing direction away from the natural training gradient geometry. This "whitening-induced incommensurability" is a provable, novel claim about the relationship between knowledge editing and knowledge attribution.
