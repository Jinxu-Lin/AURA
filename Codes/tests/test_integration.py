"""Integration test: full pipeline from model -> K-FAC -> BSS -> metrics -> ANOVA.

Verifies that all core components work together end-to-end.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.utils import set_reproducibility, partial_correlation, make_class_dummies
from core.data import make_resnet18_cifar10, make_dataloader
from core.training import train_resnet18, compute_per_sample_gradients
from core.kfac import (
    compute_kfac_factors,
    kfac_inverse,
    ekfac_eigendecompose,
    ekfac_inverse_transform,
    top_k_eigendecomposition,
)
from core.bss import (
    compute_bss,
    compute_bss_partial,
    compute_bss_ratio,
    compute_gradient_projections,
    randomized_bucket_control,
)
from core.metrics import (
    jaccard_at_k,
    kendall_tau,
    lds,
    per_point_lds,
    auroc,
    baselga_decomposition,
    icc_2_1,
)
from core.anova import type_i_ss, within_class_variance_fraction, bootstrap_ci


class TinyDataset(torch.utils.data.Dataset):
    """Tiny synthetic CIFAR-10-like dataset for integration testing."""
    def __init__(self, n=64, n_classes=10):
        self.data = torch.randn(n, 3, 32, 32)
        self.targets = [i % n_classes for i in range(n)]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def test_full_forward_backward():
    """4a. Full pipeline: model -> train -> K-FAC -> BSS -> metrics -> ANOVA."""
    set_reproducibility(42)

    # Create tiny dataset
    ds = TinyDataset(n=64)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)

    # Train model (2 epochs, tiny dataset)
    with tempfile.TemporaryDirectory() as tmpdir:
        result = train_resnet18(
            train_loader=loader,
            test_loader=loader,
            seed=42,
            epochs=2,
            checkpoint_dir=tmpdir,
            device="cpu",
            verbose=False,
        )
    model = result["model"]
    model.eval()

    # Compute K-FAC factors
    factors = compute_kfac_factors(model, loader, layer_name="fc", device="cpu")
    assert not torch.isnan(factors["A_cov"]).any()
    assert not torch.isnan(factors["B_cov"]).any()

    # Eigendecomposition
    topk = top_k_eigendecomposition(factors["A_cov"], factors["B_cov"], k=50)
    assert topk["eigenvalues"].shape[0] == 50

    # Per-sample gradients for fc layer
    test_indices = list(range(10))
    fc_param_names = ["fc.weight", "fc.bias"]
    grads = compute_per_sample_gradients(
        model, ds, test_indices, target_param_names=fc_param_names, device="cpu"
    )
    assert grads.shape[0] == 10

    # Gradient projections
    proj_sq = compute_gradient_projections(
        grads.numpy(),
        topk["eigvecs_A"].numpy(),
        topk["eigvecs_B"].numpy(),
        topk["top_k_a_indices"].numpy(),
        topk["top_k_b_indices"].numpy(),
        in_features=512,
        has_bias=True,
    )
    assert proj_sq.shape == (10, 50)
    assert not np.isnan(proj_sq).any()

    # BSS computation
    eigs = topk["eigenvalues"].numpy()
    eigs_approx = eigs * (1 + 0.1 * np.random.randn(50))  # synthetic approx
    bss_result = compute_bss(eigs, eigs_approx, proj_sq, damping=0.01)
    assert bss_result["bss_total"].shape == (10,)
    assert not np.isnan(bss_result["bss_total"]).any()

    # Partial BSS
    grad_norms_sq = (grads ** 2).sum(dim=1).numpy()
    partial = compute_bss_partial(bss_result["bss_outlier"], grad_norms_sq)
    assert partial.shape == (10,)
    assert not np.isnan(partial).any()

    # BSS ratio
    ratio = compute_bss_ratio(bss_result["bss_outlier"], bss_result["bss_total"])
    assert ratio.shape == (10,)

    # Metrics
    scores_a = np.random.randn(10, 30)
    scores_b = np.random.randn(10, 30)
    pp_lds = per_point_lds(scores_a, scores_b)
    assert pp_lds.shape == (10,)

    labels = np.array([i % 10 for i in range(10)])
    j10 = jaccard_at_k(np.argsort(-scores_a[0]), np.argsort(-scores_b[0]), k=5)
    assert 0.0 <= j10 <= 1.0

    # ANOVA
    response = bss_result["bss_outlier"]
    class_labels = labels
    gradient_norms = np.sqrt(grad_norms_sq)
    anova_result = type_i_ss(response, class_labels, gradient_norms, n_classes=10)
    assert "class_r2" in anova_result
    assert not np.isnan(anova_result["class_r2"])

    # Within-class variance
    wcvf = within_class_variance_fraction(response, class_labels)
    assert 0.0 <= wcvf <= 1.0

    # Partial correlation
    dummies = make_class_dummies(labels)
    covariates = np.column_stack([dummies, gradient_norms])
    rho, p = partial_correlation(response, pp_lds, covariates)
    assert -1.0 <= rho <= 1.0

    print("Integration test PASSED: full pipeline works end-to-end")


def test_all_components_independently():
    """4b. Verify each component can be independently disabled/replaced."""
    np.random.seed(42)

    # Each metric can be called independently
    a = np.random.randn(50)
    b = np.random.randn(50)
    assert isinstance(lds(a, b), float)
    assert isinstance(kendall_tau(a, b), float)

    rankings = np.argsort(-a)
    bd = baselga_decomposition(rankings, np.argsort(-b), k=10)
    assert abs(bd["jaccard_distance"] - bd["replacement"] - bd["reordering"]) < 1e-10

    # ICC can be used independently
    ratings = np.random.randn(20, 5)
    assert isinstance(icc_2_1(ratings), float)

    # Bootstrap CI can be used independently
    point, lower, upper = bootstrap_ci(a, np.mean, n_bootstrap=100)
    assert isinstance(point, float)

    # ANOVA can be used independently
    result = type_i_ss(a, np.random.randint(0, 5, 50), np.abs(b), n_classes=5)
    r2_sum = result["class_r2"] + result["gradnorm_r2"] + result["interaction_r2"] + result["residual_r2"]
    assert abs(r2_sum - 1.0) < 0.1, f"R^2 should sum to ~1.0, got {r2_sum}"

    # BSS can be used independently with synthetic data
    eigs = np.abs(np.random.randn(30)) + 0.1
    eigs.sort()
    eigs = eigs[::-1]
    eigs_approx = eigs * 1.2
    proj = np.abs(np.random.randn(10, 30))
    bss = compute_bss(eigs, eigs_approx, proj, damping=0.01)
    assert bss["bss_total"].shape == (10,)

    print("Component independence test PASSED")


def test_memory_footprint():
    """4c. Verify memory usage is reasonable for the test pipeline."""
    # This test checks that the pipeline can run on CPU without excessive memory.
    # No GPU available in test environment, so we just verify execution completes.
    import gc

    set_reproducibility(42)
    ds = TinyDataset(n=32)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)

    model = make_resnet18_cifar10()
    # Verify model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    # ResNet-18 CIFAR10: ~11.2M params
    assert total_params > 10_000_000 and total_params < 15_000_000, (
        f"Unexpected param count: {total_params}"
    )

    # Verify K-FAC factor memory is bounded
    factors = compute_kfac_factors(model, loader, layer_name="fc", device="cpu")
    a_mem = factors["A_cov"].element_size() * factors["A_cov"].nelement()
    b_mem = factors["B_cov"].element_size() * factors["B_cov"].nelement()
    # A: 513*513*4 ≈ 1MB, B: 10*10*4 ≈ 400B
    assert a_mem < 2_000_000, f"A_cov too large: {a_mem} bytes"
    assert b_mem < 1000, f"B_cov too large: {b_mem} bytes"

    gc.collect()
    print(f"Memory test PASSED: model={total_params} params, A_cov={a_mem}B, B_cov={b_mem}B")
