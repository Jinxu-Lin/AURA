---
version: "1.1"
created: "2026-03-16"
last_modified: "2026-03-25"
entry_mode: "fr_revise"
iteration_major: 1
iteration_minor: 1
---

# Problem Statement: Per-Test-Point Hessian Sensitivity in TDA

## 1. Gap Definition

### 1.1 Existing Methods

Training Data Attribution (TDA) quantifies the influence of each training sample on model predictions. Two major families exist:

**Parameter-space methods (Influence Functions and variants)**: Estimate training sample influence via Hessian inverse-vector products (iHVP). Key works: TRAK (2303.14186, ICML 2023), MAGIC (2504.16430), SOURCE (2405.12186), ASTRA (2507.14740, NeurIPS 2025), LoGra (2405.13954), TrackStar (2410.17413, ICLR 2025). These have clear counterfactual semantics but accuracy depends critically on Hessian approximation quality. Hong et al. (2509.23437) proved the strict hierarchy H >= GGN >> EK-FAC >> K-FAC, with the K-FAC-to-EK-FAC eigenvalue mismatch accounting for 41-65% of total error.

**Representation-space methods (RepSim and variants)**: Measure training-test similarity via intermediate representations. Key works: RepSim (2409.19998), Concept Influence (2602.14869), AirRep (2505.18513). Hessian-free, superior in low-SNR settings (IF 0-7% vs RepSim 96-100% per Li et al.), but lack counterfactual causal interpretation.

**Attribution uncertainty quantification**: Two ICML 2025 works touch attribution reliability from different angles:
- **Daunce (2505.23223)**: Perturbed model ensemble covariance -- measures training randomness dimension
- **Bayesian IF (BIF, 2509.26544)**: Posterior variance as attribution uncertainty -- measures Bayesian uncertainty dimension

Neither addresses **Hessian approximation choice sensitivity** -- the dimension AURA targets.

### 1.2 Gap Statement

**One sentence**: No per-test-point diagnostic exists for Hessian-approximation-induced attribution instability -- practitioners cannot judge whether attribution rankings for a specific test point depend on the Hessian approximation choice rather than true data influence.

**Evidence**: Two methods' attribution rankings correlate at only 0.37-0.45 (Kowal et al. 2602.14869). Jaccard@10 drops from 1.0 (full GGN self-agreement) to ~0.48 (K-FAC) in our probe. ANOVA shows 77.5% of J10 variance is residual per-test-point variation after controlling class and gradient norm.

**Differentiation from Daunce/BIF**: These measure theoretically orthogonal uncertainty dimensions (model perturbation vs. Bayesian posterior). AURA measures method-choice sensitivity: "If I switch from EK-FAC to K-FAC, do my attributions change?" This is the most directly actionable question for practitioners.

### 1.3 Root Cause Analysis

**Root cause**: The TDA community treats Hessian approximation error as a global property ("method A beats method B on average LDS"). Per-test-point sensitivity variation is invisible under global metrics.

**Why per-test-point variation exists**: Different test-point gradients project differently onto the Hessian error spectrum. Points whose gradients align with high-error eigenspaces (where K-FAC eigenvalue mismatch is largest) have unstable attributions.

**Why prior diagnostics failed**: TRV (scalar Jaccard@10 stability) has cross-seed Spearman rho ~ 0 because it depends on individual eigenvector directions, which rotate across seeds. SI (Self-Influence) is orthogonal to Hessian sensitivity (rho ~ 0 in probe). Both rely on seed-unstable quantities.

**Key insight**: Eigenvalue *magnitude* distributions are seed-stable (RMT: outlier count = number of classes, magnitudes determined by class separation, stable to O(1/sqrt(N))). BSS exploits this by bucketing spectral energy by magnitude rather than tracking individual eigenvectors.

### 1.4 Gap Assessment

| Dimension | Rating | Evidence |
|-----------|--------|----------|
| Importance | **High** | Serves all IF/TRAK/SOURCE/ASTRA users; sits at the intersection of TDA evaluation methodology crises (LDS miss-relation, attribution-influence misalignment, distributional TDA) |
| Novelty | **Medium-High** | Daunce/BIF explore attribution uncertainty from different angles; no work addresses Hessian approximation sensitivity per test point; cross-domain transfer (sensitivity analysis -> TDA) has no precedent |
| Solvability | **Medium-High** | Phase 1 confirmed phenomenon (77.5% residual); BSS has RMT-grounded theoretical basis; computation feasible for ResNet-18/CIFAR-10 |

## 2. Research Questions

**RQ1 (CONFIRMED)**: After controlling for class label and log(gradient norm), what fraction of attribution sensitivity variance is residual per-test-point variation?
- **Result**: Residual J10 = 77.5%, LDS = 51.6%. Gate threshold was >30%. **PASS**.

**RQ2 (TESTING)**: Does BSS provide a seed-stable, per-test-point diagnostic? Specifically:
- Cross-seed BSS ranking Spearman rho > 0.5 (vs. TRV rho ~ 0)
- Within-class BSS variance > 25% (not a class detector)
- Partial correlation with attribution metrics > 0.15 after gradient-norm control

**RQ3 (PLANNED)**: Does MRC soft combining (BSS + disagreement guided) Pareto-dominate uniform strategies by > 2% absolute LDS at equal compute budget?

## 3. Attack Angle

### 3.1 Selected Approach: BSS + MRC Soft Combining

The approach has three components in progressive gating order:

**C1: Attribution Variance Decomposition** (COMPLETED)
- Two-way ANOVA decomposing J10/tau/LDS into class, gradient-norm, and residual
- Confirms per-test-point phenomenon is real, not a confound artifact

**C2: Bucketed Spectral Sensitivity (BSS)** (IN PROGRESS)
- Decompose per-test-point attribution error by Hessian eigenvalue magnitude buckets
- BSS_j(z) = sum_{k in B_j} |1/lambda_k - 1/tilde_lambda_k|^2 * (V_k^T g)^2
- Buckets: outlier (top eigenvalues), edge (transition), bulk (noise)
- Theoretical grounding: RMT predicts eigenvalue magnitude stability across seeds
- Must verify: (a) cross-seed stability, (b) not a class detector, (c) adds info beyond gradient norm

**C3: MRC Soft Combining** (PLANNED)
- Weight function: w(z) = sigma(a * BSS_partial + b * disagreement + c)
- Route between IF and RepSim per test point
- Evaluate on Pareto frontier (LDS vs. GPU-hours)

### 3.2 Addressing Known Risks

**RIF/BIF as orthogonal improvements**: Daunce and BIF measure different uncertainty dimensions (training randomness, Bayesian posterior). AURA's BSS measures method-choice sensitivity. These are complementary, not competing.

**W-TRAK as mandatory baseline**: Natural W-TRAK (2512.09103) provides theoretical attribution stability bounds. W-TRAK is included as a baseline in the Pareto comparison.

**CIFAR-10 scale limitation**: Full-model Hessian analysis is tractable on CIFAR-10/ResNet-18 but not ImageNet-scale. Acknowledged as primary limitation; theory extends to larger models via Kronecker approximation.

**BSS-gradient norm correlation**: Pilot shows rho=0.906. Addressed via partial BSS (regress out gradient norm) and BSS_ratio (BSS_outlier / BSS_total).

## 4. Metadata

- **Target**: NeurIPS 2026
- **GPU Budget**: ~42 GPU-hours total
- **Knowledge base references**: Gaps & Assumptions (G-BHM1, H-BHM1, G4), Cross-Paper Connections (C54, C55), Methods Bank (#10 RepSim, #15 TRAK, #7 EK-FAC IF)
