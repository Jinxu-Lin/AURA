## [Theorist] 理论家视角

### 逻辑闭合

**Gap → Root Cause chain**: The problem statement identifies two independent gaps: (1) geometric incommensurability of editing and attribution (TECA, resolved), and (2) per-test-point Hessian sensitivity lacking diagnostics (AURA, active). For the AURA thread, the root cause analysis is sound: TRV fails because eigenvector directions are seed-unstable, while eigenvalue magnitudes are seed-stable per RMT. The Gap → RC → BSS chain is logically complete.

**BSS theoretical foundation**: The core formula BSS_j(z) = sum_{k in B_j} (g^T q_k)^2 * |1/lambda_k - 1/tilde_lambda_k|^2 is mathematically well-defined. However, two theoretical gaps exist:

1. **RMT applicability**: Marchenko-Pastur and Tracy-Widom laws apply to matrices with i.i.d. or invariant entries. The Kronecker-factored GGN violates this: Kronecker structure imposes block structure that breaks the universality conditions. The claim "eigenvalue magnitude distributions are seed-stable" needs qualification: it holds for the TOP (outlier) eigenvalues that correspond to class-discriminative directions (Papyan 2020), but may NOT hold for edge/bulk eigenvalues where Kronecker approximation errors dominate.

2. **Perturbation theory**: The |1/lambda - 1/tilde_lambda| factor assumes eigenvalues are directly comparable between EK-FAC and K-FAC. But these two approximations may have DIFFERENT eigenvectors, making the per-eigenvalue comparison ill-defined. The correct perturbation theory should use principal angles between eigenspaces, not individual eigenvalue differences.

**Assessment**: Logically sound at the high level. Two technical gaps need addressing but do not block the experimental validation.

### 组件必要性

Three components (variance decomposition, BSS diagnostic, adaptive selection) are well-motivated and non-redundant:
- Variance decomposition is a prerequisite (establishes phenomenon existence)
- BSS is the core contribution (provides diagnostic)
- Adaptive selection is the application (operationalizes diagnostic)

The gated design ensures no unnecessary computation: each component is conditional on the previous passing.

### 理论正确性

**BSS formula**: Correct for the case where EK-FAC and K-FAC share eigenvectors. When eigenvectors differ (the realistic case), the formula is an approximation. The approximation quality depends on the alignment between the two eigenspaces, which is not characterized.

**RMT predictions**: The claim that outlier eigenvalue COUNT equals the number of classes (10 for CIFAR-10) is well-established (Papyan 2020). The claim that magnitudes are O(1/sqrt(N))-stable is theoretically sound but may not hold for the Kronecker factorization.

**Normalization**: BSS / ||g||^2 is the correct normalization to remove gradient magnitude effects. However, this may introduce division instability for low-norm gradients. Should include a floor: BSS / (||g||^2 + epsilon).

### 与探针一致性

BSS_outlier pilot shows -0.42 correlation with J10 (higher BSS → more sensitive → lower J10), which is the expected direction. The 0.906 correlation with gradient norm is concerning but expected: larger gradients have larger projections onto all eigenspaces. The partial BSS approach (regressing out gradient norm) is the correct mitigation.

**Score**: Pass, with recommendations to address the eigenvector alignment gap and add numerical stability.
