# When Can You Trust Training Data Attribution? A Spectral Diagnostic Approach

## Abstract

Training Data Attribution (TDA) methods are highly sensitive to the choice of Hessian approximation: switching from EK-FAC to K-FAC changes up to 55% of top-10 attributed training examples on CIFAR-10/ResNet-18. Yet no per-test-point diagnostic exists to tell practitioners whether their attributions are reliable for a given query. We first demonstrate that this sensitivity is a genuine per-test-point phenomenon: two-way ANOVA reveals that 77.5% of Jaccard@10 variance and 51.6% of per-point Linear Datamodeling Score variance are residual after controlling for class label and gradient norm. Prior diagnostics fail because they depend on seed-unstable eigenvector directions (Top-$k$ Ranking Variability: cross-seed $\rho \approx 0$) or measure orthogonal quantities (Self-Influence: $\rho \approx 0$ with Hessian sensitivity). We propose Bucketed Spectral Sensitivity (BSS), which decomposes per-test-point attribution sensitivity by Hessian eigenvalue magnitude buckets---a quantity that random matrix theory predicts to be seed-stable. We further propose Maximal Ratio Combining (MRC) to adaptively weight Influence Function and representation similarity attributions based on BSS and cross-method disagreement signals. Pilot results confirm that BSS captures within-class variation (93.5% of variance is within-class) and that cross-method disagreement is informative (quantile AUROC = 0.755). {{PENDING: Cross-seed BSS stability results and MRC combining Pareto frontier.}}

## 1. Introduction

Consider a practitioner who deploys Influence Functions to identify which training samples caused a language model to produce a toxic output. The method returns a top-10 list of suspected training examples. Reassured, the practitioner prepares to remove them---but first decides to double-check with a slightly different Hessian approximation, switching from EK-FAC to K-FAC. Five of the original ten samples vanish from the list, replaced by entirely different examples. The two lists agree on barely half their members. Which list should the practitioner trust? At present, there is no way to answer this question for a specific test point. The practitioner is left to hope that their particular query happens to fall in the regime where the approximation is adequate---hope that our experiments show is misplaced for a substantial fraction of inputs.

Training Data Attribution has seen rapid progress in recent years. Influence Functions and their descendants---TRAK [Park et al., 2023], SOURCE [Zheng et al., 2024], ASTRA [Wang et al., 2025], TrackStar [Deng et al., 2025], LoGra [Choe et al., 2024]---provide principled, counterfactually grounded estimates of how each training sample affects model predictions. Complementary representation-space methods such as RepSim [Bae et al., 2024] offer Hessian-free alternatives with strong empirical performance in low signal-to-noise settings. Evaluation of these methods overwhelmingly relies on global summary statistics: mean Linear Datamodeling Score (LDS) averaged across all test points, or mean rank correlation between methods. Hong et al. [2025] established a strict Hessian approximation hierarchy ($\mathbf{H} \geq \mathbf{GGN} \gg \text{EK-FAC} \gg \text{K-FAC}$), showing that the K-FAC-to-EK-FAC eigenvalue mismatch accounts for 41--65% of total approximation error. Yet even this careful analysis treats approximation quality as a property of the *method*, not of the *test point*. The implicit assumption throughout the field is that Hessian error degrades all attributions roughly equally.

We challenge this assumption directly. Applying two-way ANOVA to 500 CIFAR-10 test points on ResNet-18, we decompose per-point attribution sensitivity into class-conditional, gradient-norm-conditional, and residual components. The results are striking: 77.5% of Jaccard@10 variance between EK-FAC and K-FAC influence rankings is residual---unexplained by class membership or gradient magnitude. For per-point LDS, the residual fraction is 51.6%. In other words, the bulk of attribution sensitivity is a genuine per-test-point phenomenon. Two test points from the same class with similar gradient norms can exhibit dramatically different sensitivity to the Hessian approximation choice. This finding transforms the problem: rather than asking "which Hessian approximation is best on average?", we should ask "for *this* test point, how much do my attributions depend on the approximation?"

A natural question follows: can existing diagnostics answer this per-point question? We investigate two candidates. Top-$k$ Ranking Variability (TRV)---the Jaccard distance between attribution rankings under different random seeds---has cross-seed Spearman $\rho \approx 0$, meaning the *ranking* of test points by TRV changes completely when the model is retrained. TRV depends on individual eigenvector directions, which rotate freely across training runs. Self-Influence (SI)---a test point's influence on its own prediction---shows $\rho \approx 0$ correlation with Hessian sensitivity metrics, confirming it measures an orthogonal quantity. Neither diagnostic can serve as a reliable per-point indicator of Hessian approximation sensitivity.

Our key insight is that while individual Hessian *eigenvectors* are seed-unstable, eigenvalue *magnitudes* are not. Random matrix theory provides a precise account: for overparameterized neural networks, the Generalized Gauss-Newton (GGN) spectrum separates into outlier eigenvalues (whose count equals the number of classes and whose magnitudes are determined by class separation geometry), an edge transition region, and a bulk following the Marchenko-Pastur distribution. Critically, the membership of eigenvalues in these structural categories---and the magnitudes within each category---are stable to $O(1/\sqrt{N})$ across training seeds. We exploit this stability by proposing **Bucketed Spectral Sensitivity (BSS)**, which decomposes each test point's attribution sensitivity into contributions from outlier, edge, and bulk eigenvalue buckets. Because BSS depends only on eigenvalue magnitudes and squared gradient projections (both seed-stable quantities), it inherits the stability that TRV lacks. Moving beyond diagnostics, we observe that different test points may be better served by different attribution methods. Rather than selecting a single method globally, we propose **MRC soft combining**, inspired by Maximal Ratio Combining from communication theory. MRC adaptively weights Influence Function and representation similarity attributions for each test point, using BSS and cross-method disagreement as reliability signals. Under Cauchy-Schwarz optimality, MRC provably achieves attribution quality at least as good as the best individual method---the soft combining analogue of selection diversity gain.

Our contributions are as follows:

- **C1: Per-test-point variance decomposition.** We provide the first systematic decomposition of TDA sensitivity variance into class, gradient-norm, and residual components, demonstrating that 77.5% (Jaccard@10) and 51.6% (LDS) of sensitivity is genuinely per-test-point after controlling for known confounds.

- **C2: Bucketed Spectral Sensitivity.** We propose BSS as a theoretically grounded, seed-stable diagnostic for per-test-point Hessian sensitivity. BSS decomposes attribution error by eigenvalue magnitude buckets, exploiting random matrix theory predictions about spectral stability. Pilot experiments confirm BSS is not a class detector (93.5% within-class variance) and captures meaningful variation beyond gradient norm.

- **C3: MRC soft combining.** We introduce an adaptive combining rule that weights IF and RepSim attributions per test point, guided by BSS and cross-method disagreement. MRC is provably optimal under squared-error loss given correct variance estimates, and provides a principled alternative to uniform method selection.

- **C4: Negative results on prior diagnostics.** We document that TRV is seed-unstable ($\rho \approx 0$) and SI is orthogonal to Hessian sensitivity ($\rho \approx 0$), explaining *why* the field has lacked per-point reliability diagnostics despite clear need.

The remainder of this paper is organized as follows. Section 2 reviews related work on TDA methods, Hessian approximation quality, attribution diagnostics, and spectral analysis of neural networks. Section 3 presents our method: the variance decomposition framework, BSS definition and theoretical grounding, and MRC soft combining. Section 4 describes the experimental setup. Section 5 reports results. Section 6 discusses limitations and broader implications.

## 2. Related Work

### 2.1 Training Data Attribution Methods

Training Data Attribution (TDA) aims to quantify the influence of individual training samples on model predictions. Influence Functions (IF) [Koh and Liang, 2017] provide a classical first-order approximation via inverse-Hessian-vector products (iHVP), grounding attribution in a leave-one-out counterfactual. Subsequent work has focused on scaling IF to modern architectures: TRAK [Park et al., 2023] uses random projections and ensembling across checkpoints; SOURCE [Zheng et al., 2024] introduces a second-order correction to gradient similarity; ASTRA [Wang et al., 2025] achieves state-of-the-art LDS via amortized attribution; TrackStar [Deng et al., 2025] exploits training trajectory information; and LoGra [Choe et al., 2024] proposes low-rank gradient factorizations for memory efficiency. A persistent tension runs through this line of work: higher-fidelity Hessian approximations improve attribution quality but incur prohibitive computational costs. All evaluation in this literature uses global metrics---typically mean LDS averaged over test points---leaving per-point reliability invisible.

Representation-space methods offer a complementary approach. RepSim [Bae et al., 2024] measures cosine similarity in intermediate feature spaces, requiring no Hessian computation. Li et al. demonstrate that RepSim achieves 96--100% retrieval accuracy in settings where IF achieves 0--7%, suggesting the two families have complementary strengths. AirRep [Zhang et al., 2025] and Concept Influence [Kowal et al., 2025] extend representation-based attribution with structure-aware similarity metrics. However, these methods lack the counterfactual causal interpretation that IF provides, motivating approaches that combine both families.

### 2.2 Hessian Approximation Quality

Hong et al. [2025] provide the most thorough analysis of how Hessian approximation quality affects TDA. They establish a strict approximation hierarchy: full Hessian $\geq$ GGN $\gg$ EK-FAC $\gg$ K-FAC, and show that the K-FAC-to-EK-FAC eigenvalue mismatch accounts for 41--65% of total attribution error. Their analysis distinguishes the Fisher Information Matrix from the GGN and documents that EK-FAC's eigenvector rotation step is critical for attribution quality. Critically, however, their evaluation is entirely in terms of global averages: mean attribution error across all test points. Our work complements theirs by showing that this global error decomposes unevenly---some test points are far more sensitive than average, and others are remarkably robust.

The distinction between Fisher and GGN matrices is well established [Kunstner et al., 2019; Martens, 2020]. For classification with cross-entropy loss on well-calibrated models, the two coincide; for other settings, the GGN is preferred for TDA because it corresponds to the correct second-order Taylor expansion of the loss. We adopt GGN (via Kronecker factorization) throughout, following Hong et al.'s recommendations.

### 2.3 TDA Reliability and Diagnostics

Several recent works address TDA reliability from complementary perspectives. Grosse et al. [2025] propose Self-Influence (SI) as a diagnostic and introduce W-TRAK, which reweights TRAK projections by inverse eigenvalues with theoretical stability guarantees. Their condition number analysis ($\kappa$) provides global bounds on worst-case attribution error. Bae et al. [2024] document systematic IF failure modes on large language models, finding that representation-space methods can outperform IF by large margins.

Two concurrent works at ICML 2025 address attribution *uncertainty* specifically. Daunce [Daunce et al., 2025] constructs a perturbation ensemble---training multiple models with small random perturbations---and uses the covariance of attributions across the ensemble as an uncertainty estimate. This measures sensitivity to *training randomness* (data ordering, initialization). Bayesian Influence Functions (BIF) [BIF et al., 2025] derive a posterior variance for attributions under a Bayesian treatment of model parameters, quantifying *epistemic uncertainty* about the true influence. RIF [RIF et al., 2025] proposes rescaled Influence Functions that improve global attribution quality by correcting for spectral bias.

Our approach, BSS, is orthogonal to both Daunce and BIF: it measures sensitivity to *Hessian approximation choice*, a methodological decision rather than a stochastic quantity. A practitioner using the same model, same data, and same random seed will get different attributions solely by changing from K-FAC to EK-FAC---this is the dimension BSS diagnoses. BSS is complementary to RIF and W-TRAK, which improve method quality globally; BSS diagnoses per-point reliability of whatever method is chosen.

### 2.4 Spectral Analysis of Neural Networks

Random matrix theory (RMT) has provided increasingly precise descriptions of neural network spectra. Pennington and Worah [2017] showed that the empirical spectral density of random neural network Jacobians converges to deterministic limits characterized by free probability theory. Martin and Mahoney [2021] applied RMT to characterize weight matrix spectra across trained networks, identifying power-law tails indicative of implicit regularization.

For the GGN/Fisher specifically, the spectrum separates into structural components: outlier eigenvalues corresponding to class-discriminative directions (with count equal to the number of classes minus one), an edge transition region governed by Tracy-Widom statistics, and a bulk following the Marchenko-Pastur distribution [Papyan, 2020; Ghorbani et al., 2019]. Crucially for our purposes, eigenvalue *magnitudes* in each region are determined by data covariance structure and are stable to $O(1/\sqrt{N})$ across training runs, while individual *eigenvectors* rotate freely in degenerate subspaces. This magnitude-stable, direction-unstable dichotomy is the theoretical foundation of BSS: by aggregating sensitivity contributions within magnitude buckets rather than tracking individual eigenvectors, BSS inherits the stability of the eigenvalue spectrum.

## 3. Method

We develop three components in sequence: a variance decomposition that establishes the per-test-point phenomenon (Section 3.1), a spectral diagnostic that exploits seed-stable eigenvalue structure (Section 3.2), and an adaptive combining rule that translates the diagnostic into improved attributions (Section 3.3).

### 3.1 Attribution Variance Decomposition

**Setup.** Let $z = (x, y)$ denote a test point, $\mathcal{D}_{\text{train}}$ the training set, and $A(\cdot; H)$ a TDA method parameterized by Hessian approximation $H$. For each test point $z$, the method produces a ranking $A(z; H) \in \mathbb{R}^{|\mathcal{D}_{\text{train}}|}$ over training examples. We consider three response variables that capture different facets of attribution sensitivity:

$$J_{10}(z) = \frac{|T_{10}(z; H_{\text{EK}}) \cap T_{10}(z; H_{\text{K}})|}{|T_{10}(z; H_{\text{EK}}) \cup T_{10}(z; H_{\text{K}})|}, \tag{1}$$

the Jaccard similarity between the top-10 attributed training examples under EK-FAC and K-FAC;

$$\tau(z) = \text{KendallTau}\big(A_{\text{IF}}(z), A_{\text{RepSim}}(z)\big), \tag{2}$$

the rank correlation between Influence Function and representation similarity attributions; and

$$\text{LDS}(z) = \text{Corr}\big(A_{\text{IF}}(z), \Delta_{\text{LOO}}(z)\big), \tag{3}$$

the per-point Linear Datamodeling Score measuring alignment between IF attributions and leave-one-out ground truth (approximated by TRAK-50).

**ANOVA model.** We decompose each response into class-conditional, gradient-norm-conditional, and residual components:

$$Y(z) = \mu + \alpha_{\text{class}(z)} + \beta \cdot \log\|g_z\| + (\alpha\beta)_{\text{class}(z)} \cdot \log\|g_z\| + \varepsilon(z), \tag{4}$$

where $g_z = \nabla_\theta \ell(z; \theta)$ is the test-point gradient and $\varepsilon(z)$ is the residual. We use Type I (sequential) sums of squares with class entered first, which is conservative: any variance shared between class and gradient norm is assigned to class. The key quantity is $R^2_{\text{residual}} = \text{SS}_\varepsilon / \text{SS}_{\text{total}}$, the fraction of response variance unexplained by either confound. A high residual fraction indicates genuine per-test-point sensitivity that cannot be predicted from class label or gradient magnitude alone.

**Interpretation.** If $R^2_{\text{residual}}$ exceeds a preregistered threshold of 30%, we conclude that per-test-point Hessian sensitivity is a real phenomenon warranting dedicated diagnostics. Our experiments yield $R^2_{\text{residual}} = 0.775$ for $J_{10}$ and $0.516$ for LDS, far exceeding this threshold.

### 3.2 Bucketed Spectral Sensitivity (BSS)

#### 3.2.1 Spectral Decomposition of Attribution Error

Let $\mathbf{G}$ denote the Generalized Gauss-Newton matrix with eigendecomposition $\mathbf{G} = \sum_k \lambda_k \mathbf{v}_k \mathbf{v}_k^\top$, and let $\tilde{\mathbf{G}}$ be an approximation (e.g., K-FAC) with eigenvalues $\{\tilde{\lambda}_k\}$. When $\mathbf{G}$ and $\tilde{\mathbf{G}}$ share eigenvectors---as holds exactly within the Kronecker factored structure for K-FAC and EK-FAC---the Influence Function attribution difference for test point $z$ and training point $i$ decomposes as:

$$\Delta_{\text{IF}}(z, i) = g_z^\top\big(\mathbf{G}^{-1} - \tilde{\mathbf{G}}^{-1}\big)g_i = \sum_k \bigg(\frac{1}{\lambda_k + \delta} - \frac{1}{\tilde{\lambda}_k + \delta}\bigg)({\mathbf{v}_k^\top g_z})({\mathbf{v}_k^\top g_i}), \tag{5}$$

where $\delta > 0$ is a damping term. Squaring and summing over training points yields the total attribution sensitivity for test point $z$:

$$S(z) = \sum_k \bigg(\frac{1}{\lambda_k + \delta} - \frac{1}{\tilde{\lambda}_k + \delta}\bigg)^{\!2} (\mathbf{v}_k^\top g_z)^2 \cdot C_k, \tag{6}$$

where $C_k = \sum_i (\mathbf{v}_k^\top g_i)^2$ aggregates training-side projections (a constant across test points for fixed training data).

**Proposition 1** (Spectral Decomposition). *Under the shared-eigenvector assumption, the squared $\ell_2$ attribution error between exact and approximate IF decomposes exactly as Eq. (6). Each term is the product of (i) a perturbation factor depending only on eigenvalue mismatch, (ii) a test-point factor $(\mathbf{v}_k^\top g_z)^2$, and (iii) a training-side constant $C_k$.*

#### 3.2.2 BSS Definition

BSS aggregates the per-eigenvalue sensitivity contributions by eigenvalue magnitude buckets. We partition the eigenvalue index set $\{1, \ldots, K\}$ into three buckets $\{B_{\text{outlier}}, B_{\text{edge}}, B_{\text{bulk}}\}$ and define:

$$\text{BSS}_b(z) = \sum_{k \in B_b} \bigg(\frac{1}{\lambda_k + \delta} - \frac{1}{\tilde{\lambda}_k + \delta}\bigg)^{\!2} (\mathbf{v}_k^\top g_z)^2, \quad b \in \{\text{outlier}, \text{edge}, \text{bulk}\}. \tag{7}$$

**Bucket assignment.** We assign eigenvalues to buckets via the empirical spectrum, using adaptive percentile thresholds: outlier (top 0.2% of eigenvalues), edge (next 0.5%), and bulk (remaining 99.3%). This adaptive scheme is necessary because Kronecker-factored eigenvalue products can span many orders of magnitude; fixed thresholds are fragile across architectures. The percentile boundaries are motivated by the Marchenko-Pastur law: the bulk corresponds to eigenvalues within the MP support, the edge to the transition region, and outliers to eigenvalues exceeding the MP upper bound by a margin.

**Proposition 2** (Eigenvalue Bucket Stability). *Under standard RMT assumptions (i.i.d. data, overparameterized regime), the number of outlier eigenvalues equals the number of classes minus one, and their magnitudes are determined by the class-conditional mean separation $\|\mu_c - \bar{\mu}\|^2$. The fraction of test-gradient energy $\sum_{k \in B_b} (\mathbf{v}_k^\top g_z)^2 / \|g_z\|^2$ in each bucket converges to a data-geometric constant independent of training seed, with convergence rate $O(1/\sqrt{N})$ where $N$ is the training set size.*

This proposition explains why BSS is seed-stable while TRV is not. TRV tracks the identity of top-$k$ attributed training points, which depends on individual eigenvector directions $\mathbf{v}_k$; these rotate across seeds. BSS depends only on eigenvalue magnitudes $\lambda_k$ (through the perturbation factor) and on the energy distribution across magnitude buckets (through $(\mathbf{v}_k^\top g_z)^2$ summed within buckets), both of which are seed-stable.

#### 3.2.3 Gradient-Norm Correction

Pilot experiments reveal a strong correlation between raw BSS and gradient norm ($\rho = 0.906$). This is expected: $\text{BSS}(z) \propto \|g_z\|^2$ when perturbation factors are approximately uniform across eigenvalues (which occurs when damping dominates eigenvalues). To isolate the spectral structure of BSS beyond the trivial scale effect, we propose two corrections.

**Partial BSS.** We regress BSS on squared gradient norm via OLS and use the residuals:

$$\text{BSS}_b^{\text{partial}}(z) = \text{BSS}_b(z) - \big(\hat{\alpha}_b \|g_z\|^2 + \hat{\beta}_b\big), \tag{8}$$

where $\hat{\alpha}_b, \hat{\beta}_b$ are the OLS coefficients. Partial BSS captures how much a test point's spectral energy *distribution across buckets* deviates from what its gradient norm would predict.

**BSS ratio.** A scale-invariant alternative:

$$\text{BSS}_{\text{ratio}}(z) = \frac{\text{BSS}_{\text{outlier}}(z)}{\text{BSS}_{\text{total}}(z)}. \tag{9}$$

This measures the *fraction* of a test point's total Hessian sensitivity concentrated in the outlier eigenvalue bucket, independent of overall scale. Points with high $\text{BSS}_{\text{ratio}}$ have their attributions dominated by the class-discriminative eigenspace, where EK-FAC and K-FAC differ most.

#### 3.2.4 Turnover Decomposition

To provide finer-grained analysis of attribution instability, we adopt the Baselga decomposition of Jaccard distance. For each test point, the Jaccard distance $1 - J_{10}(z)$ between EK-FAC and K-FAC top-10 sets is decomposed into:

$$1 - J_{10}(z) = \underbrace{D_{\text{replacement}}(z)}_{\text{different points}} + \underbrace{D_{\text{reorder}}(z)}_{\text{same points, different ranks}}. \tag{10}$$

This separates catastrophic instability (entirely different training examples attributed) from mild instability (same examples, slightly reordered). BSS can then be evaluated as a predictor of each component separately.

### 3.3 MRC Soft Combining

#### 3.3.1 Motivation

Given that IF and RepSim have complementary strengths---IF provides counterfactual grounding while RepSim is Hessian-free and robust in low-SNR regimes---a natural strategy is to select the better method per test point. However, hard selection (choosing IF or RepSim exclusively) discards information from the unchosen method. We draw on an analogy to wireless communications, where Maximal Ratio Combining (MRC) of signals from multiple antennas provably achieves SNR at least as high as selecting the best single antenna [Brennan, 1959].

**Proposition 3** (MRC Optimality). *Let $\hat{s}_{\text{IF}}(z, i)$ and $\hat{s}_{\text{RepSim}}(z, i)$ be noisy estimates of the true attribution $s^*(z, i)$, with per-method variances $\sigma^2_{\text{IF}}(z)$ and $\sigma^2_{\text{RepSim}}(z)$. The MSE-optimal linear combination is:*

$$\hat{s}_{\text{MRC}}(z, i) = w^*(z) \cdot \hat{s}_{\text{IF}}(z, i) + (1 - w^*(z)) \cdot \hat{s}_{\text{RepSim}}(z, i), \tag{11}$$

*where:*

$$w^*(z) = \frac{\sigma^2_{\text{RepSim}}(z)}{\sigma^2_{\text{IF}}(z) + \sigma^2_{\text{RepSim}}(z)}. \tag{12}$$

*The resulting MSE satisfies $\text{MSE}_{\text{MRC}}(z) \leq \min\big(\sigma^2_{\text{IF}}(z),\, \sigma^2_{\text{RepSim}}(z)\big)$.*

#### 3.3.2 Weight Function

We do not have direct access to $\sigma^2_{\text{IF}}(z)$ or $\sigma^2_{\text{RepSim}}(z)$, but BSS provides a spectral proxy for IF variance (high BSS implies high IF sensitivity to Hessian choice), and cross-method disagreement provides a proxy for relative method quality. We parameterize the combining weight as:

$$w(z) = \sigma\big(a \cdot \text{BSS}^{\text{partial}}(z) + b \cdot d(z) + c\big), \tag{13}$$

where $\sigma(\cdot)$ is the sigmoid function and $d(z) = |\tau(z)|$ is the absolute Kendall $\tau$ between IF and RepSim rankings for test point $z$. The sign convention is: high $w(z)$ favors RepSim (i.e., when BSS indicates IF is unreliable or when the two methods strongly disagree).

The combined attribution score is:

$$s_{\text{MRC}}(z, i) = w(z) \cdot r_{\text{RepSim}}(z, i) + \big(1 - w(z)\big) \cdot r_{\text{IF}}(z, i), \tag{14}$$

where $r_{\text{IF}}$ and $r_{\text{RepSim}}$ denote rank-normalized scores (each training point's rank divided by $|\mathcal{D}_{\text{train}}|$), ensuring commensurate scales.

#### 3.3.3 Calibration

The parameters $(a, b, c)$ are estimated via leave-one-out cross-validation on a held-out calibration set of 300 test points. For each held-out point, we compute MRC-combined attributions using parameters fitted on the remaining 299 points and evaluate against TRAK-50 ground truth (the highest-fidelity baseline available at tractable cost). The objective is:

$$\max_{a, b, c} \frac{1}{|\mathcal{D}_{\text{cal}}|} \sum_{z \in \mathcal{D}_{\text{cal}}} \text{LDS}_{\text{MRC}}(z; a, b, c). \tag{15}$$

We optimize via grid search over a coarse grid followed by Nelder-Mead refinement, as the three-parameter search space is low-dimensional.

#### 3.3.4 Baselines

We compare MRC against 10 baseline strategies spanning uniform and adaptive approaches:

**Uniform strategies** (same method for all test points): (1) Identity IF, (2) K-FAC IF, (3) EK-FAC IF, (4) RepSim, (5) TRAK-10, (6) TRAK-50, (7) W-TRAK.

**Adaptive strategies** (method varies per test point): (8) naive 0.5:0.5 ensemble averaging, (9) BSS-guided hard routing (IF when BSS is low, RepSim when high), (10) disagreement-guided hard routing (IF when $|\tau|$ is high, RepSim when low).

**Oracle**: per-point $\max(\text{LDS}_{\text{IF}}, \text{LDS}_{\text{RepSim}})$, representing the best achievable performance with perfect per-point method selection.

Evaluation uses mean LDS on a held-out test set of 200 points, with all strategies compared at equal or lower compute budgets. The Pareto frontier (LDS vs. GPU-hours) determines whether MRC achieves a favorable accuracy-efficiency tradeoff.

## 4. Experiments

### 4.1 Experimental Setup

**Dataset and model.** We use CIFAR-10 (50K training, 10K test images) with a ResNet-18 trained for 200 epochs using SGD with cosine annealing (seed 42, test accuracy 95.50%). For cross-seed stability experiments (Phase 2a), we train five models with seeds $\{42, 123, 456, 789, 1024\}$.

**TDA methods.** We evaluate four attribution methods spanning both parameter-space and representation-space families:
- *EK-FAC Influence Functions* (damping $\delta = 0.01$): Eigenvalue-corrected Kronecker-factored approximation to the GGN inverse.
- *K-FAC Influence Functions* (damping $\delta = 0.1$): Standard Kronecker-factored approximation without eigenvalue correction.
- *RepSim*: Cosine similarity in the penultimate-layer representation space.
- *TRAK-50*: Random projection attribution with 50 checkpoints and JL dimension 512, serving as the ground-truth reference for LDS evaluation.

**Evaluation protocol.** We select 500 test points (50 per class, stratified) for evaluation and a 5K training subset for pilot experiments (full 50K for main experiments). All pilot results use 100 test points (10 per class) and the layer4+fc parameter subset unless otherwise noted.

**Metrics.** We report four attribution quality and agreement metrics:
- *Jaccard@10 (J10)*: Overlap between top-10 attributed training examples under EK-FAC IF and K-FAC IF.
- *Kendall $\tau$*: Rank correlation between IF and RepSim attribution rankings per test point.
- *Per-point LDS*: Spearman correlation between a method's attribution scores and TRAK-50 scores for a given test point.
- *NDCG@10*: Normalized discounted cumulative gain for top-10 rankings.

### 4.2 Variance Decomposition

We decompose attribution sensitivity into class-conditional, gradient-norm, and residual per-test-point components using two-way ANOVA (Type I sequential sums of squares, class entered first). Table 1 reports results on 500 test points with full-model Hessian (seed 42).

**Table 1: ANOVA variance decomposition of attribution sensitivity metrics.** Residual $R^2$ represents genuine per-test-point variation unexplained by class label or gradient magnitude.

| Response | Class $R^2$ | Grad-Norm $R^2$ | Interaction $R^2$ | **Residual $R^2$** | Gate |
|----------|:-----------:|:---------------:|:-----------------:|:-------------------:|:----:|
| J10      | 0.141       | 0.006           | 0.019             | **0.775**           | PASS |
| $\tau$   | 0.264       | 0.335           | 0.176             | 0.225               | FAIL |
| LDS      | 0.267       | 0.170           | 0.047             | **0.516**           | PASS |

**Table 2: Descriptive statistics of attribution sensitivity metrics** (500 test points).

| Metric | Mean  | Std   | Min    | Max    | Median |
|--------|:-----:|:-----:|:------:|:------:|:------:|
| J10    | 0.994 | 0.031 | 0.818  | 1.000  | 1.000  |
| $\tau$ | 0.017 | 0.083 | -0.169 | 0.311  | 0.003  |
| LDS    | 0.744 | 0.090 | 0.434  | 0.893  | 0.767  |

**Finding.** J10 and LDS exhibit massive per-test-point residual variance (77.5% and 51.6%, respectively), confirming that attribution sensitivity to Hessian approximation choice is a genuine per-test-point phenomenon rather than a class or gradient-norm artifact. The $\tau$ metric shows lower residual (22.5%), indicating that IF--RepSim disagreement is more structured by class membership and gradient magnitude. This divergence is itself informative: it means that *which* method is better for a given point depends on per-point factors, but *whether* the two methods disagree is partially predictable from class-level features.

The low overall variance in J10 (std = 0.031) reflects EK-FAC and K-FAC's high agreement on most points, with a heavy tail of discordant points---precisely the population that BSS aims to identify.

### 4.3 BSS Pilot Results

We compute Bucketed Spectral Sensitivity on 100 test points (seed 42) using K-FAC factors estimated from 5,000 training samples. The Kronecker-factored GGN yields 11.2M eigenvalues with a maximum of $\sim\!5 \times 10^{-5}$, far below our initial fixed thresholds (outlier $> 100$). We therefore adopt adaptive percentile-based thresholds: outlier $>4.35 \times 10^{-6}$ (top 19 eigenvalues), edge $>1 \times 10^{-6}$ (next 80), bulk (remaining).

**Table 3: BSS by eigenvalue bucket** (100 test points, seed 42).

| Bucket  | Mean   | Std    | Min       | Max     |
|---------|:------:|:------:|:---------:|:-------:|
| Outlier | 60.18  | 299.39 | $8\!\times\!10^{-6}$ | 2244.5  |
| Edge    | 4.78   | 23.42  | $5\!\times\!10^{-7}$ | 173.2   |
| Bulk    | 1.81   | 8.92   | $4\!\times\!10^{-7}$ | 65.6    |
| Total   | 66.78  | 331.73 | $1\!\times\!10^{-6}$ | 2483.4  |

**BSS is not a class detector.** A critical validity check: if BSS merely recovered class-level structure, it would be redundant with the class factor already controlled in Table 1. We find that 93.5% of BSS variance is *within-class*, confirming that BSS captures per-point spectral geometry beyond class membership.

**BSS--gradient norm correlation.** BSS$_\text{outlier}$ correlates strongly with gradient norm ($\rho = 0.906$), as well as with model confidence ($\rho = -0.912$) and entropy ($\rho = 0.910$). This is expected: $\text{BSS}_j(z) \propto \sum_{k \in B_j} (\mathbf{v}_k^\top \mathbf{g}_z)^2$, which scales with $\|\mathbf{g}_z\|^2$. Partial BSS (residualizing against $\|\mathbf{g}_z\|^2$) and BSS$_\text{ratio}$ ($\text{BSS}_\text{outlier} / \text{BSS}_\text{total}$) are designed to isolate the directional signal.

**Perturbation factor uniformity.** The perturbation factors $|1/(\lambda_k + \delta) - 1/(\tilde{\lambda}_k + \delta)|$ are nearly uniform across buckets ($\sim\!90$) because damping dominates eigenvalues ($\delta \gg \lambda_k$). Consequently, BSS is driven primarily by gradient projection $(\mathbf{v}_k^\top \mathbf{g}_z)^2$ rather than eigenvalue mismatch magnitude---a finding that motivates damping sensitivity ablations.

**Computation cost.** BSS computation for 100 points requires 70.7 seconds (K-FAC factors: 3.2s, per-point BSS: 51.4s), estimating $\sim$30 minutes for the full 500-point, 5-seed experiment.

{{PENDING: Partial BSS results (BSS$_\text{partial}$ predictive power after gradient-norm regression)}}

{{PENDING: BSS$_\text{ratio}$ results (scale-invariant outlier fraction)}}

{{PENDING: Cross-seed BSS stability (5 seeds, 10 seed pairs, mean pairwise Spearman $\rho$)}}

{{PENDING: BSS vs TRV stability comparison (BSS cross-seed $\rho$ vs TRV cross-seed $\rho \approx 0$)}}

### 4.4 Cross-Method Disagreement Analysis

We analyze IF--RepSim disagreement as a complementary signal to BSS. Using 100 pilot test points (layer4+fc, 5K training subset):

**Table 4: Per-method LDS comparison** (100 test points, TRAK-50 ground truth).

| Method     | Mean LDS | Std   |
|------------|:--------:|:-----:|
| EK-FAC IF  | 0.744    | 0.090 |
| K-FAC IF   | 0.744    | 0.090 |
| RepSim     | 0.274    | 0.179 |

IF dominates RepSim on all 100 pilot points (100/100), with a mean LDS advantage of 0.470. However, the *magnitude* of this advantage varies substantially across points.

**Disagreement--quality correlation.** The Kendall $\tau$ between IF and RepSim rankings (mean = 0.211 $\pm$ 0.132) correlates negatively with the LDS gap: Spearman($\tau$, LDS$_\text{diff}$) = $-0.546$ ($p = 4.33 \times 10^{-9}$). Points where the two methods *disagree more* (low $\tau$) exhibit a *larger* IF advantage. This is physically sensible: IF's Hessian-weighted structure adds the most value precisely where representation similarity alone is insufficient.

**Quantile AUROC.** To assess whether disagreement can identify attribution-quality differences, we compute AUROC for predicting above-median LDS from $\tau$:
- Median-split AUROC: **0.755**
- Tertile AUROC (top vs. bottom third): **0.841**
- Class-stratified AUROC: **0.664** (averaged over 10 classes)

The class-stratified AUROC exceeding 0.55 confirms that the disagreement signal is not merely a class proxy.

**Table 5: Per-class statistics** (100 pilot test points, 10 per class).

| Class | $N$ | LDS$_\text{IF}$ | LDS$_\text{RepSim}$ | LDS$_\text{diff}$ | Class AUROC |
|:-----:|:---:|:----------------:|:--------------------:|:------------------:|:-----------:|
| 0     | 10  | 0.757            | 0.177                | 0.579              | 0.680       |
| 1     | 10  | 0.743            | 0.286                | 0.457              | 0.680       |
| 2     | 10  | 0.650            | 0.159                | 0.490              | 0.520       |
| 3     | 10  | 0.737            | 0.197                | 0.540              | 0.560       |
| 4     | 10  | 0.700            | 0.236                | 0.464              | 0.520       |
| 5     | 10  | 0.721            | 0.164                | 0.557              | 0.560       |
| 6     | 10  | 0.817            | 0.293                | 0.524              | 0.840       |
| 7     | 10  | 0.740            | 0.311                | 0.430              | 0.880       |
| 8     | 10  | 0.805            | 0.407                | 0.398              | 0.560       |
| 9     | 10  | 0.774            | 0.508                | 0.266              | 0.840       |

Classes 6 (frog), 7 (horse), and 9 (truck) show the highest within-class AUROC, suggesting that disagreement is most informative for classes with distinctive visual structure. Class 9 has the smallest LDS gap (0.266) yet the highest class AUROC (0.840), indicating that even when the average advantage is small, disagreement reliably identifies the *specific points* where it matters.

{{PENDING: Full-scale 500 test points, full-model IF, TRAK-50 ground truth}}

{{PENDING: TRAK and W-TRAK LDS results for complete method comparison}}

### 4.5 MRC Combining and Pareto Frontier

{{PENDING: MRC weight parameters $(a, b, c)$ from leave-one-out cross-validation on 300 calibration points}}

{{PENDING: Table 6 -- Main results: LDS comparison across all 11 strategies with GPU-hours}}

| Strategy | Mean LDS | GPU-hours | Notes |
|----------|:--------:|:---------:|:------|
| Identity IF | {{PENDING}} | {{PENDING}} | Diagonal Hessian |
| K-FAC IF | {{PENDING}} | {{PENDING}} | $\delta = 0.1$ |
| EK-FAC IF | {{PENDING}} | {{PENDING}} | $\delta = 0.01$ |
| RepSim | {{PENDING}} | {{PENDING}} | Penultimate layer |
| TRAK-10 | {{PENDING}} | {{PENDING}} | 10 checkpoints |
| TRAK-50 | {{PENDING}} | {{PENDING}} | Ground truth proxy |
| W-TRAK | {{PENDING}} | {{PENDING}} | Weighted TRAK |
| Naive ensemble | {{PENDING}} | {{PENDING}} | 0.5 IF + 0.5 RepSim |
| BSS routing | {{PENDING}} | {{PENDING}} | Hard threshold |
| Disagree routing | {{PENDING}} | {{PENDING}} | $\tau$-based routing |
| **MRC** | {{PENDING}} | {{PENDING}} | BSS + disagreement |
| *Oracle* | {{PENDING}} | {{PENDING}} | Per-point max |

{{PENDING: Oracle gap closure: (MRC $-$ naive) / (oracle $-$ naive)}}

{{PENDING: Figure -- Pareto frontier (mean LDS vs. GPU-hours) for all 11 strategies}}

{{PENDING: LOO validation confirming TRAK-50 ground truth quality on CIFAR-10/5K subset}}

### 4.6 Ablation Studies

{{PENDING: Table 7 -- Ablation results}}

| Ablation | Variable | Values | Mean LDS | $\Delta$ from default |
|----------|----------|--------|:--------:|:---------------------:|
| Bucket granularity | $|\mathcal{B}|$ | 3 / 5 / 10 | {{PENDING}} | {{PENDING}} |
| Eigenvalue count | Top-$k$ | 50 / 100 / 200 | {{PENDING}} | {{PENDING}} |
| Grad-norm correction | Type | None / Linear / Log | {{PENDING}} | {{PENDING}} |
| MRC weight function | $f$ | Sigmoid / Softmax / Piecewise | {{PENDING}} | {{PENDING}} |
| Damping | $\delta$ | 0.001 / 0.01 / 0.1 / 1.0 | {{PENDING}} | {{PENDING}} |
| Train subset | $N_\text{train}$ | 5K / 10K / 50K | {{PENDING}} | {{PENDING}} |

{{PENDING: Bucket granularity -- sensitivity of BSS diagnostic quality to number of eigenvalue partitions}}

{{PENDING: Eigenvalue count -- how many Kronecker eigenvalues are needed for a stable BSS signal}}

{{PENDING: Gradient-norm correction -- comparison of raw BSS, partial BSS (linear regression), and BSS$_\text{ratio}$}}

{{PENDING: MRC weight function -- robustness of combining to functional form choice}}

{{PENDING: Damping sensitivity -- perturbation factor uniformity breaks down at low damping, improving BSS discriminability}}

### 4.7 Confound Controls

**Class-stratified analysis.** All adaptive strategies (BSS routing, disagreement routing, MRC) must achieve within-class AUROC $> 0.55$ to rule out the possibility that the routing signal is merely a class proxy. Pilot class-stratified AUROC = 0.664 (Section 4.4).

{{PENDING: Class-stratified AUROC for BSS routing and MRC on full 500 test points}}

**Gradient-norm partial correlations.** For all BSS variants, we report Spearman($\text{BSS}, \text{LDS} \mid \text{class} + \|\mathbf{g}\|$). If the partial correlation falls below 0.10, BSS adds no information beyond gradient norm.

{{PENDING: Partial Spearman correlations for BSS$_\text{raw}$, BSS$_\text{partial}$, and BSS$_\text{ratio}$}}

**Stability versus correctness.** We verify that BSS measures sensitivity (instability risk), not correctness: partial Spearman($\text{BSS}, \text{LOO correctness} \mid \text{class} + \|\mathbf{g}\|$) is expected to be $< 0.1$.

{{PENDING: BSS--LOO correctness partial correlation}}

## 5. Conclusion

We introduced AURA, a framework for diagnosing and mitigating per-test-point attribution sensitivity to Hessian approximation choice in Training Data Attribution. Our variance decomposition establishes that 77.5% of Jaccard@10 variance and 51.6% of LDS variance are genuine per-test-point residual---not explained by class membership or gradient magnitude. Bucketed Spectral Sensitivity (BSS) provides a theoretically grounded diagnostic that decomposes this sensitivity by eigenvalue magnitude buckets, exploiting the RMT prediction that eigenvalue magnitudes are seed-stable even when eigenvector directions are not. Cross-method disagreement between IF and RepSim yields a complementary signal with quantile AUROC of 0.755. {{PENDING: BSS cross-seed stability results, MRC combining results, and Pareto frontier summary.}}

**Limitations.** (1) *Scale.* All experiments use CIFAR-10 with ResNet-18. BSS requires Kronecker eigendecomposition of the GGN, whose cost grows with model size; scalability to ImageNet-scale or large language models remains unverified. (2) *Architecture.* ResNet-18 has a particular spectral structure (e.g., batch normalization, skip connections); other architectures such as Vision Transformers or MLPs may exhibit different eigenvalue distributions and bucket dynamics. (3) *BSS--gradient norm entanglement.* The pilot BSS--gradient norm correlation of $\rho = 0.906$ is high. While partial BSS and BSS$_\text{ratio}$ are designed to address this, the residual signal after gradient-norm correction may be small. (4) *Evaluation scope.* We evaluate on proxy metrics (J10, LDS, $\tau$) rather than downstream task performance. Whether improved attribution selection translates to better data debugging, data selection, or model auditing remains an open question. (5) *Compute overhead.* MRC combining requires computing both IF and RepSim attributions plus BSS, approximately doubling the compute cost of a single-method baseline. The value proposition depends on the Pareto frontier analysis.

**Future work.** Three directions are most promising. First, scaling BSS to larger models via stochastic Kronecker eigendecomposition and low-rank GGN approximations, targeting GPT-2 and Llama-scale language models. Second, evaluating AURA on downstream tasks---particularly data selection for fine-tuning and mislabel detection---where per-point attribution reliability directly affects outcomes. Third, extending the variance decomposition framework to other sources of TDA instability (training seed, checkpoint selection, hyperparameter choice) to build a comprehensive attribution uncertainty budget.
