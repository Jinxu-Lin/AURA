#!/usr/bin/env python3
"""
Sanity checks for AURA codebase:
  2a. Overfit check — 2 samples, ~50 steps, loss -> 0
  2b. Gradient check — all fc-layer gradients non-zero, non-NaN
  2c. Shape check — full forward pass shape consistency through BSS pipeline
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data import make_resnet18_cifar10
from core.kfac import compute_kfac_factors, top_k_eigendecomposition
from core.bss import compute_bss, compute_bss_partial, compute_bss_ratio, compute_gradient_projections
from core.training import compute_per_sample_gradients
from core.utils import set_reproducibility


class OverfitDataset(torch.utils.data.Dataset):
    """Tiny fixed dataset (2 samples) for overfit check."""
    def __init__(self, n=2, n_classes=10):
        torch.manual_seed(42)
        self.data = torch.randn(n, 3, 32, 32)
        self.targets = list(range(min(n, n_classes)))[:n]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class SmallDataset(torch.utils.data.Dataset):
    """Small synthetic dataset (20 samples, 10 classes) for shape/gradient checks."""
    def __init__(self, n=20, n_classes=10):
        torch.manual_seed(42)
        self.data = torch.randn(n, 3, 32, 32)
        self.targets = [i % n_classes for i in range(n)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# =========================================================================
# 2a. Overfit Check
# =========================================================================

def test_overfit_check():
    """Train on 2 samples for 50 steps and verify loss approaches 0."""
    set_reproducibility(42)

    ds = OverfitDataset(n=2)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)

    model = make_resnet18_cifar10()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for step in range(50):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    initial_loss = losses[0]
    final_loss = losses[-1]

    print(f"  Overfit check: initial_loss={initial_loss:.4f}, final_loss={final_loss:.6f}")

    # Loss should decrease substantially and be near zero
    assert final_loss < 0.1, (
        f"Overfit check FAILED: final_loss={final_loss:.4f} > 0.1 after 50 steps on 2 samples"
    )
    assert final_loss < initial_loss * 0.01, (
        f"Loss did not decrease sufficiently: {initial_loss:.4f} -> {final_loss:.4f}"
    )

    # Verify model can perfectly predict the 2 training samples
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            assert correct == 2, f"Model should predict both samples correctly, got {correct}/2"

    print("  Overfit check PASSED: loss converged to near-zero, 100% train accuracy")


# =========================================================================
# 2b. Gradient Check
# =========================================================================

def test_gradient_check():
    """Verify all fc-layer parameter gradients are non-zero and non-NaN."""
    set_reproducibility(42)

    ds = SmallDataset(n=10)
    model = make_resnet18_cifar10()
    model.train()
    criterion = nn.CrossEntropyLoss()

    # Forward + backward on a single sample
    x, y = ds[0]
    x = x.unsqueeze(0)
    model.zero_grad()
    out = model(x)
    loss = criterion(out, torch.tensor([y]))
    loss.backward()

    # Check ALL named parameters
    zero_grad_params = []
    nan_grad_params = []
    inf_grad_params = []
    grad_stats = {}

    for name, param in model.named_parameters():
        if param.grad is None:
            zero_grad_params.append(name)
            continue
        g = param.grad
        if torch.isnan(g).any():
            nan_grad_params.append(name)
        if torch.isinf(g).any():
            inf_grad_params.append(name)
        if (g == 0).all():
            zero_grad_params.append(name)
        grad_stats[name] = {
            "shape": tuple(g.shape),
            "norm": g.norm().item(),
            "max_abs": g.abs().max().item(),
            "frac_zero": (g == 0).float().mean().item(),
        }

    print(f"  Gradient check: {len(grad_stats)} params with gradients")
    assert len(nan_grad_params) == 0, f"NaN gradients in: {nan_grad_params}"
    assert len(inf_grad_params) == 0, f"Inf gradients in: {inf_grad_params}"
    assert len(zero_grad_params) == 0, f"Zero/None gradients in: {zero_grad_params}"

    # Specifically check fc layer (the layer used for K-FAC)
    for name in ["fc.weight", "fc.bias"]:
        assert name in grad_stats, f"Missing gradient for {name}"
        assert grad_stats[name]["norm"] > 0, f"Zero gradient norm for {name}"
        print(f"    {name}: norm={grad_stats[name]['norm']:.6f}, "
              f"max_abs={grad_stats[name]['max_abs']:.6f}")

    # Also verify per-sample gradient computation works
    grads = compute_per_sample_gradients(
        model, ds, list(range(5)),
        target_param_names=["fc.weight", "fc.bias"],
        device="cpu",
    )
    assert grads.shape[0] == 5
    assert not torch.isnan(grads).any(), "NaN in per-sample gradients"
    assert not torch.isinf(grads).any(), "Inf in per-sample gradients"
    # Each sample should have a non-zero gradient
    per_sample_norms = grads.norm(dim=1)
    assert (per_sample_norms > 0).all(), (
        f"Zero per-sample gradient found: norms = {per_sample_norms.tolist()}"
    )

    print(f"  Per-sample gradient norms: {per_sample_norms.tolist()}")
    print("  Gradient check PASSED: all parameters have valid non-zero gradients")


# =========================================================================
# 2c. Shape Check
# =========================================================================

def test_shape_check():
    """Verify full forward-pass shape consistency through the BSS pipeline.

    Pipeline: model -> K-FAC factors -> eigendecomp -> per-sample grads
              -> gradient projections -> BSS (raw, partial, ratio)
    """
    set_reproducibility(42)

    ds = SmallDataset(n=20, n_classes=10)
    loader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=False)
    n_test = 8
    test_indices = list(range(n_test))
    top_k = 30

    # Step 1: Model forward pass
    model = make_resnet18_cifar10()
    model.eval()
    with torch.no_grad():
        sample_x = ds[0][0].unsqueeze(0)
        out = model(sample_x)
        assert out.shape == (1, 10), f"Model output shape: expected (1, 10), got {out.shape}"
    print(f"  Model output shape: {out.shape} -- OK")

    # Step 2: K-FAC factor computation
    factors = compute_kfac_factors(model, loader, layer_name="fc", device="cpu")
    d_a = factors["in_features"] + (1 if factors["has_bias"] else 0)  # 512 + 1 = 513
    d_b = factors["out_features"]  # 10
    assert factors["A_cov"].shape == (d_a, d_a), (
        f"A_cov shape: expected ({d_a}, {d_a}), got {factors['A_cov'].shape}"
    )
    assert factors["B_cov"].shape == (d_b, d_b), (
        f"B_cov shape: expected ({d_b}, {d_b}), got {factors['B_cov'].shape}"
    )
    print(f"  K-FAC factors: A_cov={factors['A_cov'].shape}, B_cov={factors['B_cov'].shape} -- OK")

    # Step 3: Eigendecomposition
    eigen = top_k_eigendecomposition(factors["A_cov"], factors["B_cov"], k=top_k)
    assert eigen["eigenvalues"].shape == (top_k,), (
        f"Eigenvalues shape: expected ({top_k},), got {eigen['eigenvalues'].shape}"
    )
    assert eigen["eigvecs_A"].shape == (d_a, d_a), (
        f"eigvecs_A shape: expected ({d_a}, {d_a}), got {eigen['eigvecs_A'].shape}"
    )
    assert eigen["eigvecs_B"].shape == (d_b, d_b), (
        f"eigvecs_B shape: expected ({d_b}, {d_b}), got {eigen['eigvecs_B'].shape}"
    )
    assert eigen["top_k_a_indices"].shape == (top_k,)
    assert eigen["top_k_b_indices"].shape == (top_k,)
    print(f"  Eigendecomp: eigenvalues={eigen['eigenvalues'].shape}, "
          f"eigvecs_A={eigen['eigvecs_A'].shape}, eigvecs_B={eigen['eigvecs_B'].shape} -- OK")

    # Step 4: Per-sample gradients
    fc_params = ["fc.weight", "fc.bias"]
    grads = compute_per_sample_gradients(
        model, ds, test_indices, target_param_names=fc_params, device="cpu"
    )
    n_fc_params = 10 * 512 + 10  # weight + bias
    assert grads.shape == (n_test, n_fc_params), (
        f"Gradients shape: expected ({n_test}, {n_fc_params}), got {grads.shape}"
    )
    print(f"  Per-sample gradients: {grads.shape} -- OK")

    # Step 5: Gradient projections
    proj_sq = compute_gradient_projections(
        grads.numpy(),
        eigen["eigvecs_A"].numpy(),
        eigen["eigvecs_B"].numpy(),
        eigen["top_k_a_indices"].numpy(),
        eigen["top_k_b_indices"].numpy(),
        in_features=512,
        has_bias=True,
    )
    assert proj_sq.shape == (n_test, top_k), (
        f"Projections shape: expected ({n_test}, {top_k}), got {proj_sq.shape}"
    )
    assert not np.isnan(proj_sq).any(), "NaN in gradient projections"
    assert (proj_sq >= 0).all(), "Negative squared projections"
    print(f"  Gradient projections: {proj_sq.shape} -- OK")

    # Step 6: BSS computation
    eigs = eigen["eigenvalues"].numpy()
    eigs_approx = np.zeros_like(eigs)
    bss_result = compute_bss(eigs, eigs_approx, proj_sq, damping=0.01)
    assert bss_result["bss_outlier"].shape == (n_test,)
    assert bss_result["bss_edge"].shape == (n_test,)
    assert bss_result["bss_bulk"].shape == (n_test,)
    assert bss_result["bss_total"].shape == (n_test,)
    assert not np.isnan(bss_result["bss_total"]).any()
    # Total should equal sum of buckets
    bucket_sum = bss_result["bss_outlier"] + bss_result["bss_edge"] + bss_result["bss_bulk"]
    np.testing.assert_allclose(bss_result["bss_total"], bucket_sum, rtol=1e-5,
                               err_msg="BSS total != outlier + edge + bulk")
    print(f"  BSS raw: outlier={bss_result['bss_outlier'].shape}, "
          f"total={bss_result['bss_total'].shape} -- OK")

    # Step 7: Partial BSS
    grad_norms_sq = (grads ** 2).sum(dim=1).numpy()
    assert grad_norms_sq.shape == (n_test,)
    bss_partial = compute_bss_partial(bss_result["bss_outlier"], grad_norms_sq)
    assert bss_partial.shape == (n_test,)
    assert not np.isnan(bss_partial).any()
    # Partial BSS should have mean ~0 (residuals of regression)
    assert abs(bss_partial.mean()) < 1e-8, (
        f"Partial BSS mean should be ~0, got {bss_partial.mean():.6e}"
    )
    print(f"  BSS partial: {bss_partial.shape}, mean={bss_partial.mean():.2e} -- OK")

    # Step 8: BSS ratio
    bss_ratio = compute_bss_ratio(bss_result["bss_outlier"], bss_result["bss_total"])
    assert bss_ratio.shape == (n_test,)
    assert not np.isnan(bss_ratio).any()
    assert (bss_ratio >= 0).all() and (bss_ratio <= 1.0 + 1e-6).all(), (
        f"BSS ratio out of [0, 1]: min={bss_ratio.min():.4f}, max={bss_ratio.max():.4f}"
    )
    print(f"  BSS ratio: {bss_ratio.shape}, range=[{bss_ratio.min():.4f}, {bss_ratio.max():.4f}] -- OK")

    print("  Shape check PASSED: all pipeline stages produce correct shapes")


if __name__ == "__main__":
    print("=" * 60)
    print("AURA Sanity Checks")
    print("=" * 60)

    print("\n--- 2a. Overfit Check ---")
    test_overfit_check()

    print("\n--- 2b. Gradient Check ---")
    test_gradient_check()

    print("\n--- 2c. Shape Check ---")
    test_shape_check()

    print("\n" + "=" * 60)
    print("ALL SANITY CHECKS PASSED")
    print("=" * 60)
