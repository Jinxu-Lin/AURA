"""Tests for core/data.py: CIFAR-10 loading, stratified sampling, model creation."""

import sys
from pathlib import Path

import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data import (
    get_cifar10_transforms,
    make_resnet18_cifar10,
    stratified_test_indices,
    stratified_train_indices,
    make_dataloader,
    CIFAR10_MEAN,
    CIFAR10_STD,
)


def test_forward_shape():
    """Test model output shape and transform pipeline shapes."""
    model = make_resnet18_cifar10()
    # CIFAR-10 input: (B, 3, 32, 32) -> output: (B, 10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"

    # Test transforms produce correct tensor shape
    train_tf = get_cifar10_transforms(train=True)
    test_tf = get_cifar10_transforms(train=False)
    # Create a dummy PIL image
    from PIL import Image
    img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    t_train = train_tf(img)
    t_test = test_tf(img)
    assert t_train.shape == (3, 32, 32), f"Expected (3, 32, 32), got {t_train.shape}"
    assert t_test.shape == (3, 32, 32), f"Expected (3, 32, 32), got {t_test.shape}"


def test_gradient_flow():
    """Test that ResNet-18 CIFAR10 model has proper gradient flow."""
    model = make_resnet18_cifar10()
    model.train()
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 5])
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Grad is None for {name}"
            assert not torch.all(p.grad == 0), f"Grad is all zeros for {name}"


def test_output_range():
    """Test model output is numerically stable (no NaN/Inf)."""
    model = make_resnet18_cifar10()
    model.eval()

    # Test with various input scales
    for scale in [0.1, 1.0, 10.0]:
        x = torch.randn(4, 3, 32, 32) * scale
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any(), f"NaN in output at scale {scale}"
        assert not torch.isinf(out).any(), f"Inf in output at scale {scale}"

    # Test normalization constants are reasonable
    assert all(0.0 < m < 1.0 for m in CIFAR10_MEAN), "Mean values should be in (0, 1)"
    assert all(0.0 < s < 1.0 for s in CIFAR10_STD), "Std values should be in (0, 1)"


def test_config_switch():
    """Test stratified sampling produces correct counts and is reproducible."""
    # Create a mock dataset-like object for testing stratified indices
    class MockDataset:
        def __init__(self, n=1000, n_classes=10):
            self.targets = [i % n_classes for i in range(n)]
            self.data = list(range(n))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return torch.randn(3, 32, 32), self.targets[idx]

    mock_test = MockDataset(n=1000, n_classes=10)
    mock_train = MockDataset(n=5000, n_classes=10)

    # Test stratified test indices
    indices = stratified_test_indices(mock_test, n_per_class=10, n_classes=10, seed=42)
    assert len(indices) == 100, f"Expected 100 indices, got {len(indices)}"

    # Verify stratification: each class should have 10 points
    labels = [mock_test.targets[i] for i in indices]
    for c in range(10):
        count = labels.count(c)
        assert count == 10, f"Class {c} has {count} points, expected 10"

    # Test reproducibility
    indices2 = stratified_test_indices(mock_test, n_per_class=10, n_classes=10, seed=42)
    assert indices == indices2, "Same seed should produce same indices"

    # Different seed should produce different indices
    indices3 = stratified_test_indices(mock_test, n_per_class=10, n_classes=10, seed=123)
    assert indices != indices3, "Different seeds should produce different indices"

    # Test stratified train indices
    train_idx = stratified_train_indices(mock_train, n_per_class=50, n_classes=10, seed=42)
    assert len(train_idx) == 500, f"Expected 500 train indices, got {len(train_idx)}"

    # Test make_dataloader with subset
    dl = make_dataloader(mock_test, batch_size=16, indices=list(range(32)))
    batch = next(iter(dl))
    assert batch[0].shape[0] <= 16, f"Batch size should be <= 16, got {batch[0].shape[0]}"
