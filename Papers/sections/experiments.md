# 4. Experiments

## 4.1 Experimental Setup

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

## 4.2 Variance Decomposition

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

## 4.3 BSS Pilot Results

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

## 4.4 Cross-Method Disagreement Analysis

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

## 4.5 MRC Combining and Pareto Frontier

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

## 4.6 Ablation Studies

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

## 4.7 Confound Controls

**Class-stratified analysis.** All adaptive strategies (BSS routing, disagreement routing, MRC) must achieve within-class AUROC $> 0.55$ to rule out the possibility that the routing signal is merely a class proxy. Pilot class-stratified AUROC = 0.664 (Section 4.4).

{{PENDING: Class-stratified AUROC for BSS routing and MRC on full 500 test points}}

**Gradient-norm partial correlations.** For all BSS variants, we report Spearman($\text{BSS}, \text{LDS} \mid \text{class} + \|\mathbf{g}\|$). If the partial correlation falls below 0.10, BSS adds no information beyond gradient norm.

{{PENDING: Partial Spearman correlations for BSS$_\text{raw}$, BSS$_\text{partial}$, and BSS$_\text{ratio}$}}

**Stability versus correctness.** We verify that BSS measures sensitivity (instability risk), not correctness: partial Spearman($\text{BSS}, \text{LOO correctness} \mid \text{class} + \|\mathbf{g}\|$) is expected to be $< 0.1$.

{{PENDING: BSS--LOO correctness partial correlation}}
