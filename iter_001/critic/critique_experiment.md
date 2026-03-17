# Experiment Critique — AURA Phase 1 Probe

**Critic Agent** | 2026-03-17 | Updated post-debate comprehensive review

---

## Critical Issues

### 1. Cross-Seed Instability Is Foundational, Not Fixable by Redefinition

Cross-seed TRV Spearman rho = -0.006 against a threshold of > 0.6. This is not a calibration or resolution problem — it means the *ranking* of test points by sensitivity is random across model instances.

**Why proposed fixes may not work:**

- **Ensemble TRV (multi-seed average)**: Averaging noise does not create signal. If individual TRV rankings are random across seeds, the mean converges to a constant, not a stable diagnostic. Ensemble averaging only helps when individual estimates are noisy but unbiased estimators of a COMMON quantity. The Probe suggests there is no common quantity to estimate.

- **BSS / Spectral Fingerprint**: These decompose TRV into spectral components, but the test gradient's projection onto eigenvalue buckets (V_k^T g_z)^2 depends on learned features, which differ across seeds. If feature representations rotate/permute across seeds (even within the same eigenvalue-magnitude range), BSS inherits TRV's instability.

- **Disagreement Diagnostic**: Uses cross-method disagreement instead of cross-approximation disagreement. This sidesteps the Hessian hierarchy but does not escape the fundamental question: is attribution sensitivity a data property or a model-instance property?

**The question that must be answered first:** Is there ANY data-geometric feature of test points that predicts Hessian sensitivity, or is sensitivity entirely stochastic? The class-conditional variance decomposition (Action Item #1) answers this.

### 2. Hessian Hierarchy Collapse to Binary Signal

5 levels collapse to 2 effective levels (Full GGN vs. everything else). Diagonal approx Damped Identity approx Identity (Jaccard differences < 0.02). This makes TRV essentially binary: changed-at-KFAC or not.

**Implications:**
- TRV has at most 1 bit of information per test point in the current setup
- The "3 levels >10%" pass criterion is met only because Level 5 (identity) happens to separate from Level 2 (KFAC), but the separation is driven by the Full-GGN-vs-KFAC cliff, not by a graded hierarchy
- Continuous-damping TRV (Action Item #12) could test whether this is a discretization artifact or an intrinsic limitation

**Unsubstantiated claims about full-model Hessian:** The assumption that full-model Hessian will produce more fine-grained TRV is speculation. K-FAC's Kronecker factorization errors may be larger in full-model setting (more layers, spatial correlations in conv layers violating Kronecker structure), but the per-test-point variation in these errors is unpredictable.

### 3. Missing Pre-Registered Experiment: TRV-LDS Comparison

Probe Step 8 (TRV-high vs. TRV-low LDS comparison) was pre-registered but results were never reported. This is the ONLY experiment that validates TRV's practical utility — the link between stability and correctness. Without it:

- TRV could be "stably wrong" (all approximations agree on incorrect attribution)
- The project cannot claim any diagnostic value for TRV
- Phase 2's adaptive fusion has no empirical motivation

This experiment MUST be completed before any further investment.

### 4. Sample Size Cannot Support the Conclusions Being Drawn

N=100 ID test points:
- Spearman rho SE approx 0.1 -> true rho plausibly in [-0.2, +0.2]
- TRV Level 2 at "11%" has 95% CI approx [4%, 22%]
- Within-seed analyses comparing 50-point subgroups have very low statistical power

N=20 OOD points:
- Spearman rho SE approx 0.22 -> essentially uninformative
- OOD "PASS" conclusion is not statistically supported

All claims from the Probe should be prefixed with "pilot-level evidence suggests..." rather than treated as definitive.

---

## Major Issues

### 5. Last-Layer-Only Scope Bias

All Hessian computations used only the fc layer (512->10, 5130 params). This is the simplest possible setting:
- K-FAC is exact for the linear final layer's Kronecker structure
- Non-linearities in earlier layers (which drive most model behavior) are absent
- The Jaccard collapse at lower hierarchy levels may be a last-layer-specific phenomenon

The generalizability of ALL Probe findings to full-model or convolutional layers is unknown.

### 6. Jaccard@10 Metric Choice Unjustified and Potentially Artifactual

k=10 out of 50,000 training points is extremely sensitive to small score perturbations. At the top-10 boundary, attributions scores may differ by epsilon, yet a rank swap changes Jaccard by 0.1. The observed TRV volatility may be partially a k-sensitivity artifact.

Jaccard also ignores score magnitudes: two test points with identical Jaccard@10 could have wildly different attribution score stability. Rank-weighted metrics (weighted Kendall tau, NDCG) or score-based metrics (cosine similarity of full attribution vectors) would provide complementary and potentially more stable information.

### 7. Non-Random Test Set Selection

The 50 high-confidence + 50 low-confidence stratified selection may introduce systematic bias. High-confidence points likely cluster near class centroids (simpler Hessian geometry); low-confidence points likely span decision boundaries (more complex). The observed TRV distribution and its cross-seed stability may not generalize to uniform random sampling.

---

## Minor Issues

### 8. Only 3 Training Seeds

Three seeds (42, 123, 456) provide minimal statistical power for cross-seed analyses. The correlation between any two seeds has a single estimate; the mean of three pairwise correlations has very wide uncertainty. Five or more seeds would substantially strengthen cross-seed claims.

### 9. No Computation Time Reporting

Wall-clock times and GPU utilization are not reported. For a diagnostic that competes with "just use the best approximation," computational overhead is a primary practical concern and must be quantified.

### 10. SVHN as Sole OOD Proxy

SVHN represents extreme distribution shift from CIFAR-10. Near-OOD (CIFAR-100, corrupted CIFAR-10) would test whether TRV's OOD behavior is specific to shift magnitude. The current single-OOD design cannot distinguish this.
