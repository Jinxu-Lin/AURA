# 3. Method

We develop three components in sequence: a variance decomposition that establishes the per-test-point phenomenon (Section 3.1), a spectral diagnostic that exploits seed-stable eigenvalue structure (Section 3.2), and an adaptive combining rule that translates the diagnostic into improved attributions (Section 3.3).

## 3.1 Attribution Variance Decomposition

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

## 3.2 Bucketed Spectral Sensitivity (BSS)

### 3.2.1 Spectral Decomposition of Attribution Error

Let $\mathbf{G}$ denote the Generalized Gauss-Newton matrix with eigendecomposition $\mathbf{G} = \sum_k \lambda_k \mathbf{v}_k \mathbf{v}_k^\top$, and let $\tilde{\mathbf{G}}$ be an approximation (e.g., K-FAC) with eigenvalues $\{\tilde{\lambda}_k\}$. When $\mathbf{G}$ and $\tilde{\mathbf{G}}$ share eigenvectors---as holds exactly within the Kronecker factored structure for K-FAC and EK-FAC---the Influence Function attribution difference for test point $z$ and training point $i$ decomposes as:

$$\Delta_{\text{IF}}(z, i) = g_z^\top\big(\mathbf{G}^{-1} - \tilde{\mathbf{G}}^{-1}\big)g_i = \sum_k \bigg(\frac{1}{\lambda_k + \delta} - \frac{1}{\tilde{\lambda}_k + \delta}\bigg)({\mathbf{v}_k^\top g_z})({\mathbf{v}_k^\top g_i}), \tag{5}$$

where $\delta > 0$ is a damping term. Squaring and summing over training points yields the total attribution sensitivity for test point $z$:

$$S(z) = \sum_k \bigg(\frac{1}{\lambda_k + \delta} - \frac{1}{\tilde{\lambda}_k + \delta}\bigg)^{\!2} (\mathbf{v}_k^\top g_z)^2 \cdot C_k, \tag{6}$$

where $C_k = \sum_i (\mathbf{v}_k^\top g_i)^2$ aggregates training-side projections (a constant across test points for fixed training data).

**Proposition 1** (Spectral Decomposition). *Under the shared-eigenvector assumption, the squared $\ell_2$ attribution error between exact and approximate IF decomposes exactly as Eq. (6). Each term is the product of (i) a perturbation factor depending only on eigenvalue mismatch, (ii) a test-point factor $(\mathbf{v}_k^\top g_z)^2$, and (iii) a training-side constant $C_k$.*

### 3.2.2 BSS Definition

BSS aggregates the per-eigenvalue sensitivity contributions by eigenvalue magnitude buckets. We partition the eigenvalue index set $\{1, \ldots, K\}$ into three buckets $\{B_{\text{outlier}}, B_{\text{edge}}, B_{\text{bulk}}\}$ and define:

$$\text{BSS}_b(z) = \sum_{k \in B_b} \bigg(\frac{1}{\lambda_k + \delta} - \frac{1}{\tilde{\lambda}_k + \delta}\bigg)^{\!2} (\mathbf{v}_k^\top g_z)^2, \quad b \in \{\text{outlier}, \text{edge}, \text{bulk}\}. \tag{7}$$

**Bucket assignment.** We assign eigenvalues to buckets via the empirical spectrum, using adaptive percentile thresholds: outlier (top 0.2% of eigenvalues), edge (next 0.5%), and bulk (remaining 99.3%). This adaptive scheme is necessary because Kronecker-factored eigenvalue products can span many orders of magnitude; fixed thresholds are fragile across architectures. The percentile boundaries are motivated by the Marchenko-Pastur law: the bulk corresponds to eigenvalues within the MP support, the edge to the transition region, and outliers to eigenvalues exceeding the MP upper bound by a margin.

**Proposition 2** (Eigenvalue Bucket Stability). *Under standard RMT assumptions (i.i.d. data, overparameterized regime), the number of outlier eigenvalues equals the number of classes minus one, and their magnitudes are determined by the class-conditional mean separation $\|\mu_c - \bar{\mu}\|^2$. The fraction of test-gradient energy $\sum_{k \in B_b} (\mathbf{v}_k^\top g_z)^2 / \|g_z\|^2$ in each bucket converges to a data-geometric constant independent of training seed, with convergence rate $O(1/\sqrt{N})$ where $N$ is the training set size.*

This proposition explains why BSS is seed-stable while TRV is not. TRV tracks the identity of top-$k$ attributed training points, which depends on individual eigenvector directions $\mathbf{v}_k$; these rotate across seeds. BSS depends only on eigenvalue magnitudes $\lambda_k$ (through the perturbation factor) and on the energy distribution across magnitude buckets (through $(\mathbf{v}_k^\top g_z)^2$ summed within buckets), both of which are seed-stable.

### 3.2.3 Gradient-Norm Correction

Pilot experiments reveal a strong correlation between raw BSS and gradient norm ($\rho = 0.906$). This is expected: $\text{BSS}(z) \propto \|g_z\|^2$ when perturbation factors are approximately uniform across eigenvalues (which occurs when damping dominates eigenvalues). To isolate the spectral structure of BSS beyond the trivial scale effect, we propose two corrections.

**Partial BSS.** We regress BSS on squared gradient norm via OLS and use the residuals:

$$\text{BSS}_b^{\text{partial}}(z) = \text{BSS}_b(z) - \big(\hat{\alpha}_b \|g_z\|^2 + \hat{\beta}_b\big), \tag{8}$$

where $\hat{\alpha}_b, \hat{\beta}_b$ are the OLS coefficients. Partial BSS captures how much a test point's spectral energy *distribution across buckets* deviates from what its gradient norm would predict.

**BSS ratio.** A scale-invariant alternative:

$$\text{BSS}_{\text{ratio}}(z) = \frac{\text{BSS}_{\text{outlier}}(z)}{\text{BSS}_{\text{total}}(z)}. \tag{9}$$

This measures the *fraction* of a test point's total Hessian sensitivity concentrated in the outlier eigenvalue bucket, independent of overall scale. Points with high $\text{BSS}_{\text{ratio}}$ have their attributions dominated by the class-discriminative eigenspace, where EK-FAC and K-FAC differ most.

### 3.2.4 Turnover Decomposition

To provide finer-grained analysis of attribution instability, we adopt the Baselga decomposition of Jaccard distance. For each test point, the Jaccard distance $1 - J_{10}(z)$ between EK-FAC and K-FAC top-10 sets is decomposed into:

$$1 - J_{10}(z) = \underbrace{D_{\text{replacement}}(z)}_{\text{different points}} + \underbrace{D_{\text{reorder}}(z)}_{\text{same points, different ranks}}. \tag{10}$$

This separates catastrophic instability (entirely different training examples attributed) from mild instability (same examples, slightly reordered). BSS can then be evaluated as a predictor of each component separately.

## 3.3 MRC Soft Combining

### 3.3.1 Motivation

Given that IF and RepSim have complementary strengths---IF provides counterfactual grounding while RepSim is Hessian-free and robust in low-SNR regimes---a natural strategy is to select the better method per test point. However, hard selection (choosing IF or RepSim exclusively) discards information from the unchosen method. We draw on an analogy to wireless communications, where Maximal Ratio Combining (MRC) of signals from multiple antennas provably achieves SNR at least as high as selecting the best single antenna [Brennan, 1959].

**Proposition 3** (MRC Optimality). *Let $\hat{s}_{\text{IF}}(z, i)$ and $\hat{s}_{\text{RepSim}}(z, i)$ be noisy estimates of the true attribution $s^*(z, i)$, with per-method variances $\sigma^2_{\text{IF}}(z)$ and $\sigma^2_{\text{RepSim}}(z)$. The MSE-optimal linear combination is:*

$$\hat{s}_{\text{MRC}}(z, i) = w^*(z) \cdot \hat{s}_{\text{IF}}(z, i) + (1 - w^*(z)) \cdot \hat{s}_{\text{RepSim}}(z, i), \tag{11}$$

*where:*

$$w^*(z) = \frac{\sigma^2_{\text{RepSim}}(z)}{\sigma^2_{\text{IF}}(z) + \sigma^2_{\text{RepSim}}(z)}. \tag{12}$$

*The resulting MSE satisfies $\text{MSE}_{\text{MRC}}(z) \leq \min\big(\sigma^2_{\text{IF}}(z),\, \sigma^2_{\text{RepSim}}(z)\big)$.*

### 3.3.2 Weight Function

We do not have direct access to $\sigma^2_{\text{IF}}(z)$ or $\sigma^2_{\text{RepSim}}(z)$, but BSS provides a spectral proxy for IF variance (high BSS implies high IF sensitivity to Hessian choice), and cross-method disagreement provides a proxy for relative method quality. We parameterize the combining weight as:

$$w(z) = \sigma\big(a \cdot \text{BSS}^{\text{partial}}(z) + b \cdot d(z) + c\big), \tag{13}$$

where $\sigma(\cdot)$ is the sigmoid function and $d(z) = |\tau(z)|$ is the absolute Kendall $\tau$ between IF and RepSim rankings for test point $z$. The sign convention is: high $w(z)$ favors RepSim (i.e., when BSS indicates IF is unreliable or when the two methods strongly disagree).

The combined attribution score is:

$$s_{\text{MRC}}(z, i) = w(z) \cdot r_{\text{RepSim}}(z, i) + \big(1 - w(z)\big) \cdot r_{\text{IF}}(z, i), \tag{14}$$

where $r_{\text{IF}}$ and $r_{\text{RepSim}}$ denote rank-normalized scores (each training point's rank divided by $|\mathcal{D}_{\text{train}}|$), ensuring commensurate scales.

### 3.3.3 Calibration

The parameters $(a, b, c)$ are estimated via leave-one-out cross-validation on a held-out calibration set of 300 test points. For each held-out point, we compute MRC-combined attributions using parameters fitted on the remaining 299 points and evaluate against TRAK-50 ground truth (the highest-fidelity baseline available at tractable cost). The objective is:

$$\max_{a, b, c} \frac{1}{|\mathcal{D}_{\text{cal}}|} \sum_{z \in \mathcal{D}_{\text{cal}}} \text{LDS}_{\text{MRC}}(z; a, b, c). \tag{15}$$

We optimize via grid search over a coarse grid followed by Nelder-Mead refinement, as the three-parameter search space is low-dimensional.

### 3.3.4 Baselines

We compare MRC against 10 baseline strategies spanning uniform and adaptive approaches:

**Uniform strategies** (same method for all test points): (1) Identity IF, (2) K-FAC IF, (3) EK-FAC IF, (4) RepSim, (5) TRAK-10, (6) TRAK-50, (7) W-TRAK.

**Adaptive strategies** (method varies per test point): (8) naive 0.5:0.5 ensemble averaging, (9) BSS-guided hard routing (IF when BSS is low, RepSim when high), (10) disagreement-guided hard routing (IF when $|\tau|$ is high, RepSim when low).

**Oracle**: per-point $\max(\text{LDS}_{\text{IF}}, \text{LDS}_{\text{RepSim}})$, representing the best achievable performance with perfect per-point method selection.

Evaluation uses mean LDS on a held-out test set of 200 points, with all strategies compared at equal or lower compute budgets. The Pareto frontier (LDS vs. GPU-hours) determines whether MRC achieves a favorable accuracy-efficiency tradeoff.
