"""Tests for core/bss.py: BSS computation, bucket partitioning, partial BSS, randomized control."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.bss import (
    adaptive_bucket_partition,
    compute_bss,
    compute_bss_partial,
    compute_bss_ratio,
    randomized_bucket_control,
    compute_gradient_projections,
)


def test_forward_shape():
    """Test that BSS outputs have correct shapes."""
    np.random.seed(42)
    n_test = 50
    n_eigen = 100

    eigenvalues = np.sort(np.abs(np.random.randn(n_eigen)))[::-1]
    eigenvalues_approx = eigenvalues * (1 + 0.1 * np.random.randn(n_eigen))
    grad_proj = np.abs(np.random.randn(n_test, n_eigen))

    result = compute_bss(eigenvalues, eigenvalues_approx, grad_proj, damping=0.01)

    assert result["bss_outlier"].shape == (n_test,), f"bss_outlier shape: {result['bss_outlier'].shape}"
    assert result["bss_edge"].shape == (n_test,)
    assert result["bss_bulk"].shape == (n_test,)
    assert result["bss_total"].shape == (n_test,)

    # Bucket info
    assert "bucket_info" in result
    info = result["bucket_info"]
    assert info["n_outlier"] + info["n_edge"] + info["n_bulk"] == n_eigen

    # Partial BSS shape
    grad_norms_sq = np.random.rand(n_test) + 0.1
    partial = compute_bss_partial(result["bss_outlier"], grad_norms_sq)
    assert partial.shape == (n_test,)

    # BSS ratio shape
    ratio = compute_bss_ratio(result["bss_outlier"], result["bss_total"])
    assert ratio.shape == (n_test,)


def test_gradient_flow():
    """Test BSS with known inputs: zero perturbation should give zero BSS."""
    np.random.seed(42)
    n_test = 20
    n_eigen = 50

    eigenvalues = np.abs(np.random.randn(n_eigen)) + 0.1
    eigenvalues.sort()
    eigenvalues = eigenvalues[::-1]

    # Same eigenvalues (no approximation error) -> BSS should be 0
    grad_proj = np.abs(np.random.randn(n_test, n_eigen))
    result = compute_bss(eigenvalues, eigenvalues.copy(), grad_proj, damping=0.01)
    np.testing.assert_allclose(result["bss_total"], 0.0, atol=1e-15)

    # Large perturbation should give large BSS
    eigenvalues_bad = eigenvalues * 2.0
    result_bad = compute_bss(eigenvalues, eigenvalues_bad, grad_proj, damping=0.01)
    assert result_bad["bss_total"].mean() > 0, "Perturbed eigenvalues should give nonzero BSS"

    # BSS_total = BSS_outlier + BSS_edge + BSS_bulk
    total_check = result_bad["bss_outlier"] + result_bad["bss_edge"] + result_bad["bss_bulk"]
    np.testing.assert_allclose(result_bad["bss_total"], total_check, rtol=1e-10)


def test_output_range():
    """Test BSS values are non-negative and numerically stable."""
    np.random.seed(42)

    for trial in range(5):
        n_test = 30
        n_eigen = 80
        eigenvalues = np.abs(np.random.randn(n_eigen)) + 0.01
        eigenvalues.sort()
        eigenvalues = eigenvalues[::-1]
        eigenvalues_approx = eigenvalues * (1 + 0.2 * np.random.randn(n_eigen))
        grad_proj = np.abs(np.random.randn(n_test, n_eigen))

        result = compute_bss(eigenvalues, eigenvalues_approx, grad_proj, damping=0.01)

        # BSS should be non-negative (sum of squared terms * squared projections)
        assert (result["bss_total"] >= -1e-10).all(), f"Negative BSS total in trial {trial}"
        assert (result["bss_outlier"] >= -1e-10).all(), f"Negative BSS outlier in trial {trial}"
        assert not np.isnan(result["bss_total"]).any(), f"NaN in BSS total trial {trial}"
        assert not np.isinf(result["bss_total"]).any(), f"Inf in BSS total trial {trial}"

        # BSS_ratio should be in [0, 1]
        ratio = compute_bss_ratio(result["bss_outlier"], result["bss_total"])
        assert (ratio >= -1e-10).all() and (ratio <= 1.0 + 1e-10).all(), (
            f"BSS_ratio out of [0,1]: [{ratio.min()}, {ratio.max()}]"
        )

    # Bucket partition should cover all eigenvalues
    eigenvalues = np.linspace(10, 0.01, 200)
    buckets = adaptive_bucket_partition(eigenvalues)
    total = buckets["n_outlier"] + buckets["n_edge"] + buckets["n_bulk"]
    assert total == 200, f"Bucket partition doesn't cover all eigenvalues: {total}/200"


def test_config_switch():
    """Test randomized-bucket control and gradient projection computation."""
    np.random.seed(42)
    n_test = 20
    n_eigen = 50

    eigenvalues = np.abs(np.random.randn(n_eigen)) + 0.1
    eigenvalues.sort()
    eigenvalues = eigenvalues[::-1]
    eigenvalues_approx = eigenvalues * (1 + 0.3 * np.random.randn(n_eigen))
    grad_proj = np.abs(np.random.randn(n_test, n_eigen))

    # Randomized bucket control should return valid results
    control = randomized_bucket_control(
        eigenvalues, eigenvalues_approx, grad_proj,
        damping=0.01, n_permutations=100, seed=42,
    )
    assert "p_value" in control
    assert 0.0 <= control["p_value"] <= 1.0
    assert "real_outlier_cv" in control
    assert isinstance(control["significant"], bool)

    # Test gradient projections with synthetic Kronecker eigenvectors
    in_features = 10
    out_features = 3
    d_a = in_features + 1  # with bias
    k = 5

    eigvecs_A = np.eye(d_a)  # identity eigenvectors
    eigvecs_B = np.eye(out_features)
    a_indices = np.array([0, 1, 2, 3, 4])
    b_indices = np.array([0, 0, 1, 1, 2])

    # Gradients: (n_test, out_features * in_features + out_features)
    n_params = out_features * in_features + out_features
    gradients = np.random.randn(n_test, n_params)

    proj_sq = compute_gradient_projections(
        gradients, eigvecs_A, eigvecs_B,
        a_indices, b_indices,
        in_features=in_features, has_bias=True,
    )
    assert proj_sq.shape == (n_test, k), f"Projection shape: {proj_sq.shape}"
    assert (proj_sq >= 0).all(), "Squared projections should be non-negative"
    assert not np.isnan(proj_sq).any(), "NaN in gradient projections"
