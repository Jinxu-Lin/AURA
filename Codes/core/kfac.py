# Component: kfac
# Source: research/method-design.md §2.2 (eigendecomposition), §2.6 (K-FAC/EK-FAC inverse)
# Ablation config key: N/A (utility module, always active)
"""
AURA K-FAC/EK-FAC factor computation and eigendecomposition.
Based on: Codes/probe/phase1_attribution_pilot_v4.py, refactored for reusability.

Computes Kronecker-factored Fisher approximation for linear layers:
  H_fc ≈ A ⊗ B
where A = E[a a^T] (input covariance) and B = E[g g^T] (output gradient covariance).

K-FAC inverse: (A + λI)^{-1} ⊗ (B + λI)^{-1}
EK-FAC inverse: eigenvalue-corrected Kronecker inverse.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_kfac_factors(
    model: nn.Module,
    train_loader: DataLoader,
    layer_name: str = "fc",
    device: str | torch.device = "cuda",
) -> dict[str, torch.Tensor]:
    """Compute K-FAC factors (A_cov, B_cov) for a linear layer.

    Args:
        model: Trained model.
        train_loader: Training DataLoader for factor estimation.
        layer_name: Name of the linear layer (e.g., "fc").
        device: Device for computation.

    Returns:
        Dict with keys: A_cov, B_cov (covariance matrices on device).
    """
    device = torch.device(device)
    model = model.to(device).eval()
    criterion = nn.CrossEntropyLoss()

    # Get the target layer
    layer = dict(model.named_modules())[layer_name]
    assert isinstance(layer, nn.Linear), f"{layer_name} must be nn.Linear, got {type(layer)}"

    in_features = layer.in_features
    has_bias = layer.bias is not None
    out_features = layer.out_features

    activations = []
    out_grads = []

    def fwd_hook(m, inp, out):
        activations.append(inp[0].detach())

    def bwd_hook(m, grad_input, grad_output):
        out_grads.append(grad_output[0].detach())

    fwd_h = layer.register_forward_hook(fwd_hook)
    bwd_h = layer.register_full_backward_hook(bwd_hook)

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        model.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()

    fwd_h.remove()
    bwd_h.remove()
    model.zero_grad()

    # Concatenate all activations and gradients
    A_list = torch.cat(activations, dim=0)  # (N, in_features)
    B_list = torch.cat(out_grads, dim=0)    # (N, out_features)

    # Add bias term to activations (append column of 1s)
    if has_bias:
        A_list_ext = torch.cat([
            A_list,
            torch.ones(A_list.shape[0], 1, device=device),
        ], dim=1)  # (N, in_features + 1)
    else:
        A_list_ext = A_list

    n = A_list_ext.shape[0]
    A_cov = (A_list_ext.T @ A_list_ext) / n  # (in_features+bias, in_features+bias)
    B_cov = (B_list.T @ B_list) / n          # (out_features, out_features)

    return {
        "A_cov": A_cov,
        "B_cov": B_cov,
        "in_features": in_features,
        "out_features": out_features,
        "has_bias": has_bias,
        "n_samples": n,
    }


def kfac_inverse(
    A_cov: torch.Tensor,
    B_cov: torch.Tensor,
    damping: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute K-FAC inverse: (A + λI)^{-1}, (B + λI)^{-1}.

    Args:
        A_cov: Input covariance, shape (d_a, d_a).
        B_cov: Output gradient covariance, shape (d_b, d_b).
        damping: Damping coefficient λ.

    Returns:
        (A_inv, B_inv) tuple.
    """
    # A_cov: (d_a, d_a), B_cov: (d_b, d_b)
    A_inv = torch.linalg.inv(A_cov + damping * torch.eye(A_cov.shape[0], device=A_cov.device))
    B_inv = torch.linalg.inv(B_cov + damping * torch.eye(B_cov.shape[0], device=B_cov.device))
    return A_inv, B_inv


def ekfac_eigendecompose(
    A_cov: torch.Tensor,
    B_cov: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Eigendecompose K-FAC factors for EK-FAC correction.

    Args:
        A_cov: Input covariance, shape (d_a, d_a).
        B_cov: Output gradient covariance, shape (d_b, d_b).

    Returns:
        Dict with: eigvals_A, eigvecs_A, eigvals_B, eigvecs_B.
    """
    eigvals_A, eigvecs_A = torch.linalg.eigh(A_cov)
    eigvals_B, eigvecs_B = torch.linalg.eigh(B_cov)
    return {
        "eigvals_A": eigvals_A,
        "eigvecs_A": eigvecs_A,
        "eigvals_B": eigvals_B,
        "eigvecs_B": eigvecs_B,
    }


def ekfac_inverse_transform(
    grad_matrix: torch.Tensor,
    eigvals_A: torch.Tensor,
    eigvecs_A: torch.Tensor,
    eigvals_B: torch.Tensor,
    eigvecs_B: torch.Tensor,
    damping: float,
) -> torch.Tensor:
    """Apply EK-FAC inverse to a gradient matrix.

    The EK-FAC inverse rotates to the Kronecker eigenbasis, applies
    corrected damped inverse, then rotates back.

    Args:
        grad_matrix: Gradient reshaped as (out_features, in_features[+1]),
                     matching the fc layer weight layout.
        eigvals_A: Eigenvalues of A_cov.
        eigvecs_A: Eigenvectors of A_cov.
        eigvals_B: Eigenvalues of B_cov.
        eigvecs_B: Eigenvectors of B_cov.
        damping: Damping coefficient δ.

    Returns:
        Inverse-transformed gradient matrix, same shape as input.
    """
    # grad_matrix: (d_b, d_a)
    # Rotate to eigenbasis
    G_eig = eigvecs_B.T @ grad_matrix @ eigvecs_A  # (d_b, d_a)

    # Kronecker eigenvalues: lambda_ab[i,j] = eigvals_B[i] * eigvals_A[j]
    kron_eigvals = eigvals_B.unsqueeze(1) * eigvals_A.unsqueeze(0)  # (d_b, d_a)

    # Apply damped inverse
    G_inv_eig = G_eig / (kron_eigvals + damping)

    # Rotate back
    G_inv = eigvecs_B @ G_inv_eig @ eigvecs_A.T

    return G_inv


def compute_kronecker_eigenvalues(
    eigvals_A: torch.Tensor,
    eigvals_B: torch.Tensor,
) -> torch.Tensor:
    """Compute Kronecker product eigenvalues (outer product of A and B eigenvalues).

    These are the eigenvalues of the Kronecker-factored Hessian approximation.
    Used for BSS bucket partitioning (method-design.md §2.2).

    Args:
        eigvals_A: Eigenvalues of A_cov, shape (d_a,).
        eigvals_B: Eigenvalues of B_cov, shape (d_b,).

    Returns:
        Flattened sorted eigenvalues, shape (d_a * d_b,), descending.
    """
    kron = (eigvals_B.unsqueeze(1) * eigvals_A.unsqueeze(0)).flatten()
    return kron.sort(descending=True).values


def top_k_eigendecomposition(
    A_cov: torch.Tensor,
    B_cov: torch.Tensor,
    k: int = 100,
) -> dict[str, torch.Tensor]:
    """Get top-k Kronecker eigenvalues and corresponding eigenvector indices.

    Per method-design.md §5: top-100 eigendecomposition.

    Args:
        A_cov: Input covariance, shape (d_a, d_a).
        B_cov: Output gradient covariance, shape (d_b, d_b).
        k: Number of top eigenvalues to return.

    Returns:
        Dict with: eigenvalues (top-k), eigvecs_A, eigvecs_B, eigvals_A, eigvals_B,
                   top_k_indices (pairs of (b_idx, a_idx) for each top eigenvalue).
    """
    eigvals_A, eigvecs_A = torch.linalg.eigh(A_cov)
    eigvals_B, eigvecs_B = torch.linalg.eigh(B_cov)

    # Kronecker eigenvalues
    kron = eigvals_B.unsqueeze(1) * eigvals_A.unsqueeze(0)  # (d_b, d_a)
    flat = kron.flatten()
    topk_vals, topk_flat_idx = flat.topk(min(k, flat.numel()))

    # Convert flat indices to (b_idx, a_idx) pairs
    d_a = eigvals_A.shape[0]
    b_indices = topk_flat_idx // d_a
    a_indices = topk_flat_idx % d_a

    return {
        "eigenvalues": topk_vals,
        "eigvecs_A": eigvecs_A,
        "eigvecs_B": eigvecs_B,
        "eigvals_A": eigvals_A,
        "eigvals_B": eigvals_B,
        "top_k_b_indices": b_indices,
        "top_k_a_indices": a_indices,
    }
