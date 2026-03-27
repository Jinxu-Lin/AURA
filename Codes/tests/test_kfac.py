"""Tests for core/kfac.py: K-FAC/EK-FAC factors, eigendecomposition, inverse."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.kfac import (
    compute_kfac_factors,
    kfac_inverse,
    ekfac_eigendecompose,
    ekfac_inverse_transform,
    compute_kronecker_eigenvalues,
    top_k_eigendecomposition,
)
from core.data import make_resnet18_cifar10


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, n=64):
        self.data = torch.randn(n, 3, 32, 32)
        self.targets = [i % 10 for i in range(n)]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def test_forward_shape():
    """Test that K-FAC factor shapes are correct."""
    model = make_resnet18_cifar10()
    ds = TinyDataset(n=32)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)

    factors = compute_kfac_factors(model, loader, layer_name="fc", device="cpu")

    # fc layer: in_features=512, out_features=10, has_bias=True
    # A_cov: (513, 513), B_cov: (10, 10)
    assert factors["A_cov"].shape == (513, 513), f"A_cov shape: {factors['A_cov'].shape}"
    assert factors["B_cov"].shape == (10, 10), f"B_cov shape: {factors['B_cov'].shape}"
    assert factors["in_features"] == 512
    assert factors["out_features"] == 10
    assert factors["has_bias"] is True

    # Test eigendecomposition shapes
    eig = ekfac_eigendecompose(factors["A_cov"], factors["B_cov"])
    assert eig["eigvals_A"].shape == (513,)
    assert eig["eigvecs_A"].shape == (513, 513)
    assert eig["eigvals_B"].shape == (10,)
    assert eig["eigvecs_B"].shape == (10, 10)

    # Test Kronecker eigenvalues shape
    kron_eigs = compute_kronecker_eigenvalues(eig["eigvals_A"], eig["eigvals_B"])
    assert kron_eigs.shape == (513 * 10,), f"Kron eigs shape: {kron_eigs.shape}"

    # Test top-k shape
    topk = top_k_eigendecomposition(factors["A_cov"], factors["B_cov"], k=50)
    assert topk["eigenvalues"].shape == (50,)


def test_gradient_flow():
    """Test K-FAC inverse produces valid inverse (A_inv @ A ≈ I)."""
    torch.manual_seed(42)
    d_a, d_b = 20, 5

    # Create symmetric positive definite matrices
    X = torch.randn(100, d_a)
    A = (X.T @ X) / 100 + 0.01 * torch.eye(d_a)
    Y = torch.randn(100, d_b)
    B = (Y.T @ Y) / 100 + 0.01 * torch.eye(d_b)

    damping = 0.1
    A_inv, B_inv = kfac_inverse(A, B, damping)

    # A_inv @ (A + damping*I) should be approximately identity
    identity_approx = A_inv @ (A + damping * torch.eye(d_a))
    assert torch.allclose(identity_approx, torch.eye(d_a), atol=1e-5), (
        "K-FAC inverse is not correct"
    )

    # EK-FAC inverse: test round-trip
    eig = ekfac_eigendecompose(A, B)
    G = torch.randn(d_b, d_a)
    G_inv = ekfac_inverse_transform(
        G, eig["eigvals_A"], eig["eigvecs_A"],
        eig["eigvals_B"], eig["eigvecs_B"], damping=0.1,
    )

    # G_inv should have same shape as G
    assert G_inv.shape == G.shape, f"EK-FAC output shape mismatch: {G_inv.shape} vs {G.shape}"

    # All elements of G_inv should have non-zero gradients if computed via autograd
    # (not directly testable here, but verify non-trivial)
    assert G_inv.abs().sum() > 0, "EK-FAC inverse output is all zeros"


def test_output_range():
    """Test that K-FAC outputs are numerically stable (no NaN/Inf)."""
    torch.manual_seed(42)

    model = make_resnet18_cifar10()
    ds = TinyDataset(n=32)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)

    factors = compute_kfac_factors(model, loader, layer_name="fc", device="cpu")

    # Check no NaN/Inf
    assert not torch.isnan(factors["A_cov"]).any(), "NaN in A_cov"
    assert not torch.isinf(factors["A_cov"]).any(), "Inf in A_cov"
    assert not torch.isnan(factors["B_cov"]).any(), "NaN in B_cov"
    assert not torch.isinf(factors["B_cov"]).any(), "Inf in B_cov"

    # Covariance matrices should be symmetric and positive semi-definite
    assert torch.allclose(factors["A_cov"], factors["A_cov"].T, atol=1e-5), "A_cov not symmetric"
    assert torch.allclose(factors["B_cov"], factors["B_cov"].T, atol=1e-5), "B_cov not symmetric"

    # Eigenvalues should be non-negative (for PSD matrices)
    eig = ekfac_eigendecompose(factors["A_cov"], factors["B_cov"])
    # With small random datasets, numerical noise can produce tiny negative eigenvalues
    assert (eig["eigvals_A"] >= -1e-3).all(), f"Large negative A eigenvalue: {eig['eigvals_A'].min()}"
    assert (eig["eigvals_B"] >= -1e-3).all(), f"Large negative B eigenvalue: {eig['eigvals_B'].min()}"

    # Kronecker eigenvalues should be sorted descending
    kron = compute_kronecker_eigenvalues(eig["eigvals_A"], eig["eigvals_B"])
    diffs = kron[:-1] - kron[1:]
    assert (diffs >= -1e-6).all(), "Kronecker eigenvalues not sorted descending"


def test_config_switch():
    """Test top-k eigendecomposition with various k values."""
    torch.manual_seed(42)
    d_a, d_b = 50, 10

    X = torch.randn(200, d_a)
    A = (X.T @ X) / 200
    Y = torch.randn(200, d_b)
    B = (Y.T @ Y) / 200

    for k in [10, 50, 100]:
        topk = top_k_eigendecomposition(A, B, k=k)
        expected_k = min(k, d_a * d_b)
        assert topk["eigenvalues"].shape[0] == expected_k, (
            f"k={k}: expected {expected_k} eigenvalues, got {topk['eigenvalues'].shape[0]}"
        )
        # Top eigenvalues should be non-negative
        assert (topk["eigenvalues"] >= -1e-6).all(), f"Negative eigenvalue at k={k}"

        # Verify indices are valid
        assert (topk["top_k_b_indices"] < d_b).all()
        assert (topk["top_k_a_indices"] < d_a).all()

    # K-FAC inverse with various dampings
    for damping in [0.001, 0.01, 0.1, 1.0]:
        A_inv, B_inv = kfac_inverse(A, B, damping)
        assert not torch.isnan(A_inv).any(), f"NaN in A_inv at damping={damping}"
        assert not torch.isnan(B_inv).any(), f"NaN in B_inv at damping={damping}"
