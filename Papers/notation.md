# Notation Reference

| Symbol | Definition |
|--------|-----------|
| $z = (x, y)$ | Test point (input $x$, label $y$) |
| $z_i = (x_i, y_i)$ | Training point $i$ |
| $\theta$ | Model parameters |
| $\mathbf{g}_z = \nabla_\theta \ell(z; \theta)$ | Gradient of loss at test point $z$ |
| $\|\mathbf{g}_z\|$ | Gradient norm of test point $z$ |
| $H$ | Hessian of the training loss, $\nabla^2_\theta \mathcal{L}(\theta)$ |
| $G$ | Generalized Gauss-Newton (GGN) matrix |
| $\tilde{H}$ | Approximate Hessian (e.g., K-FAC, EK-FAC) |
| $\lambda_k$ | $k$-th eigenvalue of $G$ (or EK-FAC approximation) |
| $\tilde{\lambda}_k$ | $k$-th eigenvalue of $\tilde{H}$ (K-FAC approximation) |
| $\mathbf{v}_k$ | $k$-th eigenvector of $G$ |
| $\delta$ | Damping parameter (K-FAC: $\delta = 0.1$; EK-FAC: $\delta = 0.01$) |
| $B_j$ | Eigenvalue bucket $j \in \{\text{outlier, edge, bulk}\}$ |
| $\text{BSS}_j(z)$ | Bucketed Spectral Sensitivity of test point $z$ in bucket $j$: $\sum_{k \in B_j} \left\lvert \frac{1}{\lambda_k + \delta} - \frac{1}{\tilde{\lambda}_k + \delta} \right\rvert^2 (\mathbf{v}_k^\top \mathbf{g}_z)^2$ |
| $\text{BSS}_\text{partial}(z)$ | Partial BSS: residual of $\text{BSS}_j(z)$ regressed on $\|\mathbf{g}_z\|^2$ |
| $\text{BSS}_\text{ratio}(z)$ | BSS ratio: $\text{BSS}_\text{outlier}(z) / \text{BSS}_\text{total}(z)$ |
| $\text{J10}(z)$ | Jaccard@10: overlap of top-10 attributed training points between EK-FAC IF and K-FAC IF for test point $z$ |
| $\tau(z)$ | Kendall rank correlation between IF and RepSim attribution rankings for test point $z$ |
| $\text{LDS}(z)$ | Linear Datamodeling Score: Spearman correlation between a method's attribution scores and TRAK-50 scores for test point $z$ |
| $\text{NDCG@}k$ | Normalized Discounted Cumulative Gain at rank $k$ |
| $w(z)$ | MRC weight for test point $z$: $\sigma(a \cdot \text{BSS}_\text{partial}(z) + b \cdot d(z) + c)$ |
| $d(z)$ | Cross-method disagreement: $\lvert \tau(z) \rvert$ (absolute Kendall $\tau$ between IF and RepSim) |
| $a, b, c$ | MRC weight parameters (calibrated via LOO cross-validation) |
| $\sigma(\cdot)$ | Sigmoid function |
| $s_\text{MRC}(z, z_i)$ | MRC attribution score: $w(z) \cdot \text{RepSim}(z, z_i) + (1 - w(z)) \cdot \text{IF}(z, z_i)$ |
| $N$ | Number of training points |
| $K$ | Number of eigenvalues retained (top-$K$ Kronecker eigendecomposition) |
| $\rho$ | Spearman rank correlation coefficient |
| $R^2$ | Coefficient of determination (fraction of variance explained) |
| AUROC | Area Under the Receiver Operating Characteristic curve |
| TDA | Training Data Attribution |
| IF | Influence Functions |
| GGN | Generalized Gauss-Newton |
| K-FAC | Kronecker-Factored Approximate Curvature |
| EK-FAC | Eigenvalue-corrected K-FAC |
| RMT | Random Matrix Theory |
| MRC | Maximal Ratio Combining |
| BSS | Bucketed Spectral Sensitivity |
| TRV | Top-$k$ Ranking Variability |
| SI | Self-Influence |
| LOO | Leave-One-Out |
| LDS | Linear Datamodeling Score |
