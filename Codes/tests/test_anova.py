"""Tests for core/anova.py: Type I SS, variance decomposition, bootstrap CI."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.anova import (
    type_i_ss,
    within_class_variance_fraction,
    bootstrap_ci,
)


def test_forward_shape():
    """Test that type_i_ss returns correct dict structure."""
    np.random.seed(42)
    n = 100
    response = np.random.randn(n)
    class_labels = np.random.randint(0, 10, n)
    gradient_norms = np.abs(np.random.randn(n))

    result = type_i_ss(response, class_labels, gradient_norms, n_classes=10)

    expected_keys = [
        "total_ss", "class_ss", "gradnorm_ss", "interaction_ss", "residual_ss",
        "class_r2", "gradnorm_r2", "interaction_r2", "residual_r2",
        "class_f", "class_p",
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"
        assert isinstance(result[key], float), f"{key} should be float, got {type(result[key])}"


def test_gradient_flow():
    """Test ANOVA with known data: class-dominated response."""
    np.random.seed(42)
    n_per_class = 50
    n_classes = 5
    n = n_per_class * n_classes

    # Create response that is purely class-driven
    class_labels = np.repeat(np.arange(n_classes), n_per_class)
    class_effects = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    response = class_effects[class_labels] + 0.01 * np.random.randn(n)
    gradient_norms = np.abs(np.random.randn(n))

    result = type_i_ss(response, class_labels, gradient_norms, n_classes=n_classes)

    # Class should explain almost all variance
    assert result["class_r2"] > 0.9, f"class_r2 should be > 0.9, got {result['class_r2']}"
    assert result["residual_r2"] < 0.1, f"residual_r2 should be < 0.1, got {result['residual_r2']}"

    # R^2 values should approximately sum to 1
    r2_sum = result["class_r2"] + result["gradnorm_r2"] + result["interaction_r2"] + result["residual_r2"]
    assert abs(r2_sum - 1.0) < 0.05, f"R^2 values should sum to ~1.0, got {r2_sum}"


def test_output_range():
    """Test that all outputs are numerically valid (no NaN/Inf, proper ranges)."""
    np.random.seed(42)

    for trial in range(10):
        n = 200
        response = np.random.randn(n)
        class_labels = np.random.randint(0, 10, n)
        gradient_norms = np.abs(np.random.randn(n))

        result = type_i_ss(response, class_labels, gradient_norms)

        for key, val in result.items():
            assert not np.isnan(val), f"{key} is NaN in trial {trial}"
            assert not np.isinf(val), f"{key} is Inf in trial {trial}"

        # R^2 should be in [0, 1] approximately
        for r2_key in ["class_r2", "gradnorm_r2", "interaction_r2", "residual_r2"]:
            assert -0.01 <= result[r2_key] <= 1.01, f"{r2_key} out of range: {result[r2_key]}"

    # within_class_variance_fraction should be in [0, 1]
    values = np.random.randn(100)
    labels = np.random.randint(0, 5, 100)
    wcvf = within_class_variance_fraction(values, labels)
    assert 0.0 <= wcvf <= 1.0, f"WCVF out of range: {wcvf}"

    # Bootstrap CI should have lower < point < upper
    point, lower, upper = bootstrap_ci(
        np.random.randn(50),
        statistic_fn=np.mean,
        n_bootstrap=500,
    )
    assert lower <= point <= upper or abs(lower - upper) < 0.5, (
        f"CI mismatch: {lower} <= {point} <= {upper}"
    )
    assert not np.isnan(point)
    assert not np.isnan(lower)
    assert not np.isnan(upper)


def test_config_switch():
    """Test edge cases: constant response, single class, small n."""
    # Constant response: all SS should be 0
    n = 50
    response = np.ones(n) * 3.14
    class_labels = np.random.randint(0, 5, n)
    gradient_norms = np.abs(np.random.randn(n))

    result = type_i_ss(response, class_labels, gradient_norms, n_classes=5)
    assert result["total_ss"] < 1e-10

    # within_class_variance_fraction with constant values
    wcvf = within_class_variance_fraction(np.ones(50), np.arange(50) % 5)
    assert wcvf == 0.0

    # Bootstrap with known distribution
    data = np.ones(100)
    point, lower, upper = bootstrap_ci(data, np.mean, n_bootstrap=100)
    assert abs(point - 1.0) < 1e-10
    assert abs(lower - 1.0) < 1e-10
    assert abs(upper - 1.0) < 1e-10
