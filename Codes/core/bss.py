# Component: bss
# Source: research/method-design.md §2.2 (BSS definition), §2.3 (partial BSS, BSS_ratio)
# Ablation config key: N/A (analysis utility, always active)
"""
AURA BSS (Bucketed Spectral Sensitivity) computation.

BSS_j(z) = sum_{k in B_j} |1/(lambda_k + delta) - 1/(tilde_lambda_k + delta)|^2 * (v_k^T g_z)^2

Includes: eigendecomposition, adaptive percentile bucket partitioning,
partial BSS (gradient-norm correction), BSS_ratio, randomized-bucket variant.

Based on: Codes/experiments/phase2a_bss_analysis.py patterns.
"""

import numpy as np
import torch
from scipy.stats import spearmanr


def adaptive_bucket_partition(
    eigenvalues: np.ndarray,
    outlier_percentile: float = 0.998,
    edge_percentile: float = 0.993,
) -> dict[str, np.ndarray]:
    """Partition eigenvalue indices into outlier/edge/bulk buckets using adaptive percentiles.

    Per method-design.md §2.2: Adaptive percentile-based thresholds required because
    Kronecker eigenvalue products are extremely small (max ~5e-05).

    Args:
        eigenvalues: Sorted descending eigenvalue array, shape (n_eigen,).
        outlier_percentile: Percentile cutoff for outlier bucket (top 0.2%).
        edge_percentile: Percentile cutoff for edge bucket boundary.

    Returns:
        Dict with: outlier_idx, edge_idx, bulk_idx (boolean masks),
                   outlier_threshold, edge_threshold.
    """
    n = len(eigenvalues)

    # Percentile-based thresholds
    outlier_threshold = np.percentile(eigenvalues, outlier_percentile * 100)
    edge_threshold = np.percentile(eigenvalues, edge_percentile * 100)

    outlier_mask = eigenvalues >= outlier_threshold
    edge_mask = (eigenvalues >= edge_threshold) & (eigenvalues < outlier_threshold)
    bulk_mask = eigenvalues < edge_threshold

    return {
        "outlier_idx": outlier_mask,
        "edge_idx": edge_mask,
        "bulk_idx": bulk_mask,
        "outlier_threshold": float(outlier_threshold),
        "edge_threshold": float(edge_threshold),
        "n_outlier": int(outlier_mask.sum()),
        "n_edge": int(edge_mask.sum()),
        "n_bulk": int(bulk_mask.sum()),
    }


def compute_bss(
    eigenvalues: np.ndarray,
    eigenvalues_approx: np.ndarray,
    gradient_projections: np.ndarray,
    damping: float = 0.01,
    bucket_masks: dict[str, np.ndarray] | None = None,
    outlier_percentile: float = 0.998,
    edge_percentile: float = 0.993,
) -> dict[str, np.ndarray]:
    """Compute BSS for a set of test points.

    BSS_j(z) = sum_{k in B_j} |1/(lambda_k + delta) - 1/(tilde_lambda_k + delta)|^2 * (v_k^T g_z)^2

    Args:
        eigenvalues: True eigenvalues, shape (n_eigen,).
        eigenvalues_approx: Approximate eigenvalues, shape (n_eigen,).
        gradient_projections: (v_k^T g_z)^2 per test point, shape (n_test, n_eigen).
        damping: Damping coefficient delta.
        bucket_masks: Pre-computed bucket masks. If None, computed from eigenvalues.
        outlier_percentile: Outlier percentile for bucket partition.
        edge_percentile: Edge percentile for bucket partition.

    Returns:
        Dict with: bss_outlier, bss_edge, bss_bulk, bss_total (each shape (n_test,)),
                   bucket_info.
    """
    # eigenvalues: (n_eigen,), eigenvalues_approx: (n_eigen,)
    # gradient_projections: (n_test, n_eigen)
    n_test, n_eigen = gradient_projections.shape
    assert eigenvalues.shape == (n_eigen,)
    assert eigenvalues_approx.shape == (n_eigen,)

    # Perturbation factor: |1/(lambda + delta) - 1/(tilde_lambda + delta)|^2
    inv_true = 1.0 / (eigenvalues + damping)
    inv_approx = 1.0 / (eigenvalues_approx + damping)
    perturbation_sq = (inv_true - inv_approx) ** 2  # (n_eigen,)

    # BSS per eigenvalue per test point
    bss_per_eigen = gradient_projections * perturbation_sq[np.newaxis, :]  # (n_test, n_eigen)

    # Bucket partition
    if bucket_masks is None:
        bucket_masks = adaptive_bucket_partition(
            eigenvalues, outlier_percentile, edge_percentile
        )

    bss_outlier = bss_per_eigen[:, bucket_masks["outlier_idx"]].sum(axis=1)
    bss_edge = bss_per_eigen[:, bucket_masks["edge_idx"]].sum(axis=1)
    bss_bulk = bss_per_eigen[:, bucket_masks["bulk_idx"]].sum(axis=1)
    bss_total = bss_per_eigen.sum(axis=1)

    return {
        "bss_outlier": bss_outlier,
        "bss_edge": bss_edge,
        "bss_bulk": bss_bulk,
        "bss_total": bss_total,
        "bucket_info": bucket_masks,
    }


def compute_bss_partial(
    bss_values: np.ndarray,
    gradient_norms_sq: np.ndarray,
) -> np.ndarray:
    """Compute gradient-norm-corrected BSS (partial BSS).

    BSS_partial_j(z) = BSS_j(z) - (alpha * ||g_z||^2 + beta)

    Per method-design.md §2.3: Pilot revealed BSS-gradient_norm rho = 0.906.
    Partial BSS residualizes out the gradient norm dependence.

    Args:
        bss_values: Raw BSS values, shape (n_test,).
        gradient_norms_sq: ||g_z||^2 per test point, shape (n_test,).

    Returns:
        Partial BSS residuals, shape (n_test,).
    """
    # OLS regression: bss = alpha * grad_norm_sq + beta
    X = np.column_stack([gradient_norms_sq, np.ones(len(bss_values))])
    beta, _, _, _ = np.linalg.lstsq(X, bss_values, rcond=None)
    fitted = X @ beta
    return bss_values - fitted


def compute_bss_ratio(
    bss_outlier: np.ndarray,
    bss_total: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """Compute BSS_ratio: BSS_outlier / BSS_total.

    Scale-invariant metric. Per method-design.md §2.3.

    Args:
        bss_outlier: BSS in outlier bucket, shape (n_test,).
        bss_total: Total BSS, shape (n_test,).
        eps: Small constant to avoid division by zero.

    Returns:
        BSS ratio, shape (n_test,).
    """
    return bss_outlier / (bss_total + eps)


def randomized_bucket_control(
    eigenvalues: np.ndarray,
    eigenvalues_approx: np.ndarray,
    gradient_projections: np.ndarray,
    damping: float = 0.01,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Randomized-bucket mechanism control.

    Per design_review binding condition #3: If random bucket assignment produces
    similar BSS structure, the eigenvalue-spectrum structure is irrelevant.

    Permutes eigenvalue-to-bucket assignments and computes the fraction of
    permutations where the outlier-bucket BSS variance exceeds the real assignment.

    Args:
        eigenvalues: True eigenvalues, shape (n_eigen,).
        eigenvalues_approx: Approximate eigenvalues, shape (n_eigen,).
        gradient_projections: Squared projections, shape (n_test, n_eigen).
        damping: Damping coefficient.
        n_permutations: Number of random permutations.
        seed: Random seed.

    Returns:
        Dict with: real_outlier_cv, mean_random_cv, p_value, significant.
    """
    rng = np.random.RandomState(seed)

    # Compute real BSS
    real_bss = compute_bss(eigenvalues, eigenvalues_approx, gradient_projections, damping)
    real_outlier = real_bss["bss_outlier"]
    real_cv = float(real_outlier.std() / (real_outlier.mean() + 1e-10))
    n_outlier = real_bss["bucket_info"]["n_outlier"]

    n_test, n_eigen = gradient_projections.shape

    # Perturbation factor
    inv_true = 1.0 / (eigenvalues + damping)
    inv_approx = 1.0 / (eigenvalues_approx + damping)
    perturbation_sq = (inv_true - inv_approx) ** 2
    bss_per_eigen = gradient_projections * perturbation_sq[np.newaxis, :]

    # Permutation test: randomly assign n_outlier eigenvalues to "outlier" bucket
    random_cvs = np.zeros(n_permutations)
    for p in range(n_permutations):
        perm = rng.permutation(n_eigen)
        random_outlier_idx = perm[:n_outlier]
        random_outlier_bss = bss_per_eigen[:, random_outlier_idx].sum(axis=1)
        random_cvs[p] = random_outlier_bss.std() / (random_outlier_bss.mean() + 1e-10)

    # p-value: fraction of random CVs >= real CV
    p_value = float((random_cvs >= real_cv).mean())

    return {
        "real_outlier_cv": real_cv,
        "mean_random_cv": float(random_cvs.mean()),
        "std_random_cv": float(random_cvs.std()),
        "p_value": p_value,
        "significant": p_value < 0.05,
        "n_outlier": n_outlier,
        "n_permutations": n_permutations,
    }


def compute_gradient_projections(
    gradients: np.ndarray | torch.Tensor,
    eigvecs_A: np.ndarray | torch.Tensor,
    eigvecs_B: np.ndarray | torch.Tensor,
    top_k_a_indices: np.ndarray | torch.Tensor,
    top_k_b_indices: np.ndarray | torch.Tensor,
    in_features: int,
    has_bias: bool = True,
) -> np.ndarray:
    """Compute squared gradient projections (v_k^T g_z)^2 onto Kronecker eigenvectors.

    For the fc layer with Kronecker structure, eigenvector k is:
    v_k = eigvec_B[b_k] ⊗ eigvec_A[a_k]
    Projection: (v_k^T g)^2 = (eigvec_B[b_k]^T G eigvec_A[a_k])^2

    Args:
        gradients: Flat gradient vectors, shape (n_test, n_fc_params).
        eigvecs_A: A covariance eigenvectors, shape (d_a, d_a).
        eigvecs_B: B covariance eigenvectors, shape (d_b, d_b).
        top_k_a_indices: A-dimension indices for top-k eigenvectors.
        top_k_b_indices: B-dimension indices for top-k eigenvectors.
        in_features: Number of input features (512 for ResNet-18 fc).
        has_bias: Whether layer has bias.

    Returns:
        Squared projections, shape (n_test, k).
    """
    # Convert to numpy if torch
    if isinstance(gradients, torch.Tensor):
        gradients = gradients.numpy()
    if isinstance(eigvecs_A, torch.Tensor):
        eigvecs_A = eigvecs_A.numpy()
    if isinstance(eigvecs_B, torch.Tensor):
        eigvecs_B = eigvecs_B.numpy()
    if isinstance(top_k_a_indices, torch.Tensor):
        top_k_a_indices = top_k_a_indices.numpy()
    if isinstance(top_k_b_indices, torch.Tensor):
        top_k_b_indices = top_k_b_indices.numpy()

    n_test = gradients.shape[0]
    out_features = eigvecs_B.shape[0]
    d_a = eigvecs_A.shape[0]  # in_features + (1 if bias)

    # Reshape gradients to matrix form: (n_test, out_features, d_a)
    # fc params = weight (out, in) + bias (out,) = out * in + out = out * (in + 1)
    n_weight = out_features * in_features
    G_weight = gradients[:, :n_weight].reshape(n_test, out_features, in_features)
    if has_bias:
        G_bias = gradients[:, n_weight:n_weight + out_features].reshape(n_test, out_features, 1)
        G = np.concatenate([G_weight, G_bias], axis=2)  # (n_test, out_features, d_a)
    else:
        G = G_weight

    k = len(top_k_a_indices)
    projections_sq = np.zeros((n_test, k))

    for i in range(k):
        a_idx = top_k_a_indices[i]
        b_idx = top_k_b_indices[i]
        # projection = eigvec_B[b_idx]^T @ G @ eigvec_A[a_idx]
        v_b = eigvecs_B[:, b_idx]  # (out_features,)
        v_a = eigvecs_A[:, a_idx]  # (d_a,)
        # For each test point: v_b^T @ G[t] @ v_a
        proj = np.einsum("i,tij,j->t", v_b, G, v_a)
        projections_sq[:, i] = proj ** 2

    return projections_sq
