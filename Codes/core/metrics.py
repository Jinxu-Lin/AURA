# Component: metrics
# Source: research/method-design.md §1 (J10, tau, LDS), §2.4 (Baselga), §2.5 (ICC)
# Ablation config key: N/A (metric utility, always active)
"""
AURA metric utilities: Jaccard@k, Kendall tau, LDS (Spearman), AUROC,
Baselga turnover decomposition, ICC(2,1).

Based on: Codes/probe/phase1_attribution_pilot_v4.py (jaccard_k, tau, LDS patterns).
"""

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score


def jaccard_at_k(
    ranking_a: np.ndarray,
    ranking_b: np.ndarray,
    k: int = 10,
) -> float:
    """Jaccard similarity of top-k sets from two rankings.

    Args:
        ranking_a: Indices sorted by descending score, shape (n,).
        ranking_b: Indices sorted by descending score, shape (n,).
        k: Number of top elements to compare.

    Returns:
        Jaccard similarity in [0, 1].
    """
    set_a = set(ranking_a[:k].tolist())
    set_b = set(ranking_b[:k].tolist())
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def kendall_tau(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> float:
    """Kendall tau rank correlation between two score vectors.

    Args:
        scores_a: Score vector, shape (n,).
        scores_b: Score vector, shape (n,).

    Returns:
        Kendall tau value in [-1, 1]. Returns 0.0 if result is NaN.
    """
    tau, _ = kendalltau(scores_a, scores_b)
    return float(tau) if not np.isnan(tau) else 0.0


def lds(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> float:
    """Linear Datamodeling Score: Spearman correlation between two score vectors.

    Per method-design.md, LDS is the primary evaluation metric.

    Args:
        scores_a: Score vector, shape (n,).
        scores_b: Score vector, shape (n,).

    Returns:
        Spearman rho in [-1, 1]. Returns 0.0 if result is NaN.
    """
    rho, _ = spearmanr(scores_a, scores_b)
    return float(rho) if not np.isnan(rho) else 0.0


def per_point_lds(
    scores_matrix: np.ndarray,
    ground_truth_matrix: np.ndarray,
) -> np.ndarray:
    """Compute per-test-point LDS (Spearman) between score matrices.

    Args:
        scores_matrix: Shape (n_test, n_train), attribution scores.
        ground_truth_matrix: Shape (n_test, n_train), ground truth scores.

    Returns:
        Array of shape (n_test,) with per-point LDS values.
    """
    # scores_matrix: (n_test, n_train)
    # ground_truth_matrix: (n_test, n_train)
    assert scores_matrix.shape == ground_truth_matrix.shape
    n_test = scores_matrix.shape[0]
    result = np.zeros(n_test)
    for i in range(n_test):
        result[i] = lds(scores_matrix[i], ground_truth_matrix[i])
    return result


def auroc(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Area Under ROC Curve.

    Args:
        scores: Predicted scores, shape (n,).
        labels: Binary labels {0, 1}, shape (n,).

    Returns:
        AUROC value in [0, 1]. Returns 0.5 if only one class present.
    """
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def class_stratified_auroc(
    scores: np.ndarray,
    labels: np.ndarray,
    class_labels: np.ndarray,
) -> float:
    """AUROC computed within each class, then averaged.

    This controls for class as a confound. Per design_review binding condition #6.

    Args:
        scores: Predicted scores, shape (n,).
        labels: Binary outcome labels {0, 1}, shape (n,).
        class_labels: Integer class labels, shape (n,).

    Returns:
        Mean within-class AUROC.
    """
    classes = np.unique(class_labels)
    aurocs = []
    for c in classes:
        mask = class_labels == c
        if len(np.unique(labels[mask])) < 2:
            continue
        aurocs.append(auroc(scores[mask], labels[mask]))
    return float(np.mean(aurocs)) if aurocs else 0.5


def baselga_decomposition(
    ranking_a: np.ndarray,
    ranking_b: np.ndarray,
    k: int = 10,
) -> dict[str, float]:
    """Baselga turnover decomposition of Jaccard distance.

    Decomposes Jaccard distance (1 - J) into replacement and reordering components.
    Per method-design.md §2.4.

    Args:
        ranking_a: Indices sorted by descending score, shape (n,).
        ranking_b: Indices sorted by descending score, shape (n,).
        k: Number of top elements.

    Returns:
        Dict with keys: jaccard_distance, replacement, reordering.
    """
    set_a = set(ranking_a[:k].tolist())
    set_b = set(ranking_b[:k].tolist())

    shared = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a

    a = len(shared)  # shared elements
    b = len(only_a)  # in A only
    c = len(only_b)  # in B only

    jaccard_dist = 1.0 - a / (a + b + c) if (a + b + c) > 0 else 1.0

    # Replacement: elements that enter/exit top-k (min(b,c) / (a + min(b,c)))
    # Reordering: Jaccard distance - replacement
    min_bc = min(b, c)
    replacement = min_bc / (a + min_bc) if (a + min_bc) > 0 else 0.0

    # The reordering component captures rank changes among shared elements
    # In Baselga's framework: reordering = jaccard_dist - replacement
    reordering = jaccard_dist - replacement

    return {
        "jaccard_distance": jaccard_dist,
        "replacement": replacement,
        "reordering": reordering,
        "n_shared": a,
        "n_only_a": b,
        "n_only_b": c,
    }


def icc_2_1(
    ratings: np.ndarray,
) -> float:
    """Intraclass Correlation Coefficient ICC(2,1) for absolute agreement.

    Used for cross-seed BSS stability assessment (method-design.md §2.5).

    Args:
        ratings: Shape (n_subjects, n_raters). Each column is one rater's
                 ratings across all subjects (e.g., BSS values from different seeds).

    Returns:
        ICC(2,1) value. Can be negative if between-subject variance is small.
    """
    # ratings: (n_subjects, n_raters)
    n, k = ratings.shape
    assert n > 1 and k > 1, f"Need n>1, k>1, got n={n}, k={k}"

    # Grand mean
    grand_mean = ratings.mean()

    # Mean squares
    row_means = ratings.mean(axis=1)
    col_means = ratings.mean(axis=0)

    # Between-subjects SS
    SS_rows = k * np.sum((row_means - grand_mean) ** 2)
    MS_rows = SS_rows / (n - 1)

    # Between-raters SS
    SS_cols = n * np.sum((col_means - grand_mean) ** 2)
    MS_cols = SS_cols / (k - 1)

    # Residual SS
    SS_residual = np.sum((ratings - row_means[:, None] - col_means[None, :] + grand_mean) ** 2)
    MS_residual = SS_residual / ((n - 1) * (k - 1))

    # ICC(2,1): (MS_rows - MS_residual) / (MS_rows + (k-1)*MS_residual + k*(MS_cols - MS_residual)/n)
    numerator = MS_rows - MS_residual
    denominator = MS_rows + (k - 1) * MS_residual + k * (MS_cols - MS_residual) / n

    if abs(denominator) < 1e-15:
        return 0.0

    return float(numerator / denominator)
