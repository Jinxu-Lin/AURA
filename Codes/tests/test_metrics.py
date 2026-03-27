"""Tests for core/metrics.py: J10, tau, LDS, AUROC, Baselga, ICC(2,1)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.metrics import (
    jaccard_at_k,
    kendall_tau,
    lds,
    per_point_lds,
    auroc,
    class_stratified_auroc,
    baselga_decomposition,
    icc_2_1,
)


def test_forward_shape():
    """Test that all metrics return correct types and shapes."""
    np.random.seed(42)
    n = 100

    # Jaccard
    ranking_a = np.arange(n)
    ranking_b = np.arange(n)
    j = jaccard_at_k(ranking_a, ranking_b, k=10)
    assert isinstance(j, float)

    # Kendall tau
    tau = kendall_tau(np.random.randn(n), np.random.randn(n))
    assert isinstance(tau, float)

    # LDS
    l = lds(np.random.randn(n), np.random.randn(n))
    assert isinstance(l, float)

    # Per-point LDS
    mat_a = np.random.randn(5, n)
    mat_b = np.random.randn(5, n)
    pp = per_point_lds(mat_a, mat_b)
    assert pp.shape == (5,), f"Expected (5,), got {pp.shape}"

    # AUROC
    a = auroc(np.random.randn(n), np.random.randint(0, 2, n))
    assert isinstance(a, float)

    # Baselga
    bd = baselga_decomposition(ranking_a, ranking_b, k=10)
    assert isinstance(bd, dict)
    assert "jaccard_distance" in bd
    assert "replacement" in bd
    assert "reordering" in bd

    # ICC(2,1)
    ratings = np.random.randn(20, 5)
    icc = icc_2_1(ratings)
    assert isinstance(icc, float)


def test_gradient_flow():
    """Test metrics with known inputs produce correct values."""
    # Identical rankings: Jaccard should be 1.0
    r = np.arange(50)
    assert jaccard_at_k(r, r, k=10) == 1.0

    # Completely disjoint top-k: Jaccard should be 0.0
    r_a = np.arange(50)
    r_b = np.arange(50, 100)
    assert jaccard_at_k(r_a, r_b, k=10) == 0.0

    # Perfect positive correlation
    x = np.arange(100, dtype=float)
    assert kendall_tau(x, x) == 1.0
    assert abs(lds(x, x) - 1.0) < 1e-10

    # Perfect negative correlation
    assert kendall_tau(x, -x) == -1.0

    # AUROC with perfect prediction
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == 1.0

    # AUROC with single class returns 0.5
    assert auroc(np.array([0.1, 0.2]), np.array([0, 0])) == 0.5


def test_output_range():
    """Test that all metrics produce values in valid ranges (no NaN/Inf)."""
    np.random.seed(42)

    for _ in range(10):
        n = 100
        a = np.random.randn(n)
        b = np.random.randn(n)

        j = jaccard_at_k(np.argsort(-a), np.argsort(-b), k=10)
        assert 0.0 <= j <= 1.0, f"Jaccard out of range: {j}"
        assert not np.isnan(j)

        tau = kendall_tau(a, b)
        assert -1.0 <= tau <= 1.0, f"tau out of range: {tau}"
        assert not np.isnan(tau)

        l = lds(a, b)
        assert -1.0 <= l <= 1.0, f"LDS out of range: {l}"
        assert not np.isnan(l)

    # Baselga: components should sum correctly
    r_a = np.random.permutation(100)
    r_b = np.random.permutation(100)
    bd = baselga_decomposition(r_a, r_b, k=20)
    assert abs(bd["jaccard_distance"] - bd["replacement"] - bd["reordering"]) < 1e-10
    assert 0.0 <= bd["jaccard_distance"] <= 1.0
    assert 0.0 <= bd["replacement"] <= 1.0
    assert bd["reordering"] >= -1e-10  # should be non-negative

    # ICC with highly consistent raters should be high
    base = np.random.randn(50)
    ratings = np.column_stack([base + 0.01 * np.random.randn(50) for _ in range(5)])
    icc = icc_2_1(ratings)
    assert icc > 0.9, f"ICC should be high for consistent raters, got {icc}"

    # ICC with random raters should be near zero
    ratings_rand = np.random.randn(50, 5)
    icc_rand = icc_2_1(ratings_rand)
    assert abs(icc_rand) < 0.5, f"ICC should be near zero for random raters, got {icc_rand}"


def test_config_switch():
    """Test class_stratified_auroc and edge cases."""
    np.random.seed(42)
    n = 100
    scores = np.random.randn(n)
    labels = np.random.randint(0, 2, n)
    class_labels = np.random.randint(0, 5, n)

    # Class-stratified AUROC should return a valid value
    cs_auroc = class_stratified_auroc(scores, labels, class_labels)
    assert 0.0 <= cs_auroc <= 1.0, f"Class-stratified AUROC out of range: {cs_auroc}"

    # Per-point LDS with identical matrices should be all 1.0
    mat = np.random.randn(10, 50)
    pp = per_point_lds(mat, mat)
    np.testing.assert_allclose(pp, 1.0, atol=1e-10)

    # Baselga with identical rankings should have zero distance
    r = np.arange(50)
    bd = baselga_decomposition(r, r, k=10)
    assert bd["jaccard_distance"] == 0.0
    assert bd["replacement"] == 0.0
    assert bd["reordering"] == 0.0
    assert bd["n_shared"] == 10
    assert bd["n_only_a"] == 0
    assert bd["n_only_b"] == 0
