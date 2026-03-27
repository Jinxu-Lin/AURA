# Component: anova
# Source: research/method-design.md §1 (Two-way ANOVA, Type I SS, R^2 decomposition)
# Ablation config key: N/A (analysis utility, always active)
"""
AURA ANOVA utilities: Type I sequential sum of squares, R^2 decomposition,
bootstrap confidence intervals, within-class variance fraction.

Based on: Codes/experiments/phase1_variance_decomposition_full.py and
          Codes/experiments/phase2a_bss_analysis.py patterns.
"""

import numpy as np
from scipy.stats import f_oneway


def type_i_ss(
    response: np.ndarray,
    class_labels: np.ndarray,
    gradient_norms: np.ndarray,
    n_classes: int = 10,
) -> dict[str, float]:
    """Two-way ANOVA with Type I (sequential) sum of squares.

    Enters class first, then gradient norm, then interaction.
    Per method-design.md §1: class entered first.

    Args:
        response: Response variable, shape (n,). E.g., J10, tau, or LDS per test point.
        class_labels: Integer class labels, shape (n,).
        gradient_norms: Gradient norm per test point, shape (n,).
        n_classes: Number of classes.

    Returns:
        Dict with keys: total_ss, class_ss, gradnorm_ss, interaction_ss, residual_ss,
        class_r2, gradnorm_r2, interaction_r2, residual_r2, class_f, class_p.
    """
    # response, class_labels, gradient_norms: (n,)
    n = len(response)
    assert len(class_labels) == n and len(gradient_norms) == n

    grand_mean = response.mean()
    total_ss = np.sum((response - grand_mean) ** 2)

    if total_ss < 1e-15:
        return {
            "total_ss": 0.0, "class_ss": 0.0, "gradnorm_ss": 0.0,
            "interaction_ss": 0.0, "residual_ss": 0.0,
            "class_r2": 0.0, "gradnorm_r2": 0.0,
            "interaction_r2": 0.0, "residual_r2": 0.0,
            "class_f": 0.0, "class_p": 1.0,
        }

    # Step 1: Class effect (entered first)
    class_means = np.zeros(n)
    for c in range(n_classes):
        mask = class_labels == c
        if mask.any():
            class_means[mask] = response[mask].mean()
    class_ss = np.sum((class_means - grand_mean) ** 2)

    # Step 2: Gradient norm effect (entered second, after class)
    # Residualize response w.r.t. class, then regress on gradient norm
    response_resid_class = response - class_means

    # Per-class regression of residual on gradient norm
    fitted_gradnorm = np.zeros(n)
    for c in range(n_classes):
        mask = class_labels == c
        if mask.sum() < 2:
            continue
        gn = gradient_norms[mask]
        r = response_resid_class[mask]
        # Simple OLS: r = a * gn + b
        gn_centered = gn - gn.mean()
        gn_var = np.sum(gn_centered ** 2)
        if gn_var > 1e-15:
            slope = np.sum(gn_centered * r) / gn_var
            intercept = r.mean() - slope * gn.mean()
            fitted_gradnorm[mask] = slope * gn + intercept
        else:
            fitted_gradnorm[mask] = r.mean()

    gradnorm_ss = np.sum(fitted_gradnorm ** 2)

    # Step 3: Interaction (class x gradient norm)
    # Full model with class-specific slopes
    fitted_full = np.zeros(n)
    for c in range(n_classes):
        mask = class_labels == c
        if mask.sum() < 2:
            fitted_full[mask] = response[mask].mean() if mask.any() else grand_mean
            continue
        gn = gradient_norms[mask]
        r = response[mask]
        gn_centered = gn - gn.mean()
        gn_var = np.sum(gn_centered ** 2)
        if gn_var > 1e-15:
            slope = np.sum(gn_centered * (r - r.mean())) / gn_var
            fitted_full[mask] = r.mean() + slope * gn_centered
        else:
            fitted_full[mask] = r.mean()

    model_ss = np.sum((fitted_full - grand_mean) ** 2)
    interaction_ss = max(0.0, model_ss - class_ss - gradnorm_ss)
    residual_ss = max(0.0, total_ss - model_ss)

    # R^2 decomposition
    r2 = lambda ss: float(ss / total_ss) if total_ss > 0 else 0.0

    # F-statistic for class effect
    df_class = n_classes - 1
    df_residual = n - n_classes
    if df_residual > 0 and residual_ss > 0:
        ms_class = class_ss / df_class
        ms_residual = residual_ss / df_residual
        class_f = ms_class / ms_residual if ms_residual > 0 else 0.0
    else:
        class_f = 0.0

    # One-way ANOVA p-value for class
    groups = [response[class_labels == c] for c in range(n_classes) if (class_labels == c).any()]
    if len(groups) >= 2 and all(len(g) >= 1 for g in groups):
        _, class_p = f_oneway(*groups)
        class_p = float(class_p)
    else:
        class_p = 1.0

    return {
        "total_ss": float(total_ss),
        "class_ss": float(class_ss),
        "gradnorm_ss": float(gradnorm_ss),
        "interaction_ss": float(interaction_ss),
        "residual_ss": float(residual_ss),
        "class_r2": r2(class_ss),
        "gradnorm_r2": r2(gradnorm_ss),
        "interaction_r2": r2(interaction_ss),
        "residual_r2": r2(residual_ss),
        "class_f": float(class_f),
        "class_p": class_p,
    }


def within_class_variance_fraction(
    values: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute within-class / total variance fraction.

    Per method-design.md §2.2 gate: within-class variance > 25% means BSS
    captures per-point variation beyond class membership.

    Args:
        values: Per-point values, shape (n,).
        labels: Integer class labels, shape (n,).

    Returns:
        Within-class variance fraction in [0, 1].
    """
    n = len(values)
    total_ss = np.var(values) * n
    if total_ss < 1e-15:
        return 0.0

    classes = np.unique(labels)
    within_ss = sum(
        np.var(values[labels == c]) * np.sum(labels == c)
        for c in classes
        if np.sum(labels == c) > 0
    )
    return float(within_ss / total_ss)


def bootstrap_ci(
    values: np.ndarray,
    statistic_fn,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a statistic.

    Args:
        values: Data array, shape (n,) or (n, k).
        statistic_fn: Callable that takes a resampled array and returns a scalar.
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed.

    Returns:
        (point_estimate, ci_lower, ci_upper) tuple.
    """
    rng = np.random.RandomState(seed)
    n = values.shape[0]

    point_estimate = float(statistic_fn(values))

    bootstrap_stats = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if values.ndim == 1:
            bootstrap_stats[b] = statistic_fn(values[idx])
        else:
            bootstrap_stats[b] = statistic_fn(values[idx])

    alpha = (1 - ci_level) / 2
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha)))

    return point_estimate, ci_lower, ci_upper
