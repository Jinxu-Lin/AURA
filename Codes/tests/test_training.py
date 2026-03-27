"""Tests for core/training.py: ResNet-18 training, per-sample gradients."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.training import train_resnet18, compute_per_sample_gradients
from core.data import make_resnet18_cifar10


class TinyDataset(torch.utils.data.Dataset):
    """Small synthetic dataset for fast testing."""
    def __init__(self, n=64, n_classes=10):
        self.data = torch.randn(n, 3, 32, 32)
        self.targets = [i % n_classes for i in range(n)]
        self.labels = torch.tensor(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def test_forward_shape():
    """Test train_resnet18 returns correct result structure."""
    ds = TinyDataset(n=32)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)

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

    assert "model" in result
    assert "best_acc" in result
    assert "training_time_sec" in result
    assert isinstance(result["best_acc"], float)
    assert result["best_acc"] >= 0.0

    # Test output shape of the trained model
    model = result["model"]
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"


def test_gradient_flow():
    """Test that per-sample gradients are non-zero and correctly shaped."""
    ds = TinyDataset(n=16)
    model = make_resnet18_cifar10()

    # Compute gradients for first 4 samples, only fc layer
    grads = compute_per_sample_gradients(
        model=model,
        dataset=ds,
        indices=[0, 1, 2, 3],
        target_param_names=["fc.weight", "fc.bias"],
        device="cpu",
    )

    # fc.weight: (10, 512) = 5120, fc.bias: (10,) = 10 => total = 5130
    assert grads.shape == (4, 5130), f"Expected (4, 5130), got {grads.shape}"

    # All gradients should be non-zero
    for i in range(4):
        assert grads[i].abs().sum() > 0, f"Gradient for sample {i} is all zeros"

    # Gradients should differ between samples (different inputs -> different grads)
    assert not torch.allclose(grads[0], grads[1], atol=1e-6), (
        "Different samples should have different gradients"
    )


def test_output_range():
    """Test that training produces numerically stable outputs."""
    ds = TinyDataset(n=32)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)

    result = train_resnet18(
        train_loader=loader,
        test_loader=loader,
        seed=42,
        epochs=3,
        device="cpu",
        verbose=False,
    )

    model = result["model"]
    model.eval()

    # Check no NaN/Inf in model parameters
    for name, p in model.named_parameters():
        assert not torch.isnan(p).any(), f"NaN in parameter {name}"
        assert not torch.isinf(p).any(), f"Inf in parameter {name}"

    # Check no NaN/Inf in model output
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any(), "NaN in model output after training"
    assert not torch.isinf(out).any(), "Inf in model output after training"


def test_config_switch():
    """Test checkpoint saving/loading round-trip."""
    ds = TinyDataset(n=32)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)

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

        # Checkpoint should exist
        ckpt_path = Path(result["checkpoint_path"])
        assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"

        # Load checkpoint and verify
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "seed" in ckpt
        assert ckpt["seed"] == 42

        # Load into new model
        from core.data import load_checkpoint
        model2 = make_resnet18_cifar10()
        load_checkpoint(model2, ckpt_path, device="cpu")

        # Verify outputs match
        x = torch.randn(2, 3, 32, 32)
        result["model"].eval()
        model2.eval()
        with torch.no_grad():
            out1 = result["model"](x)
            out2 = model2(x)
        assert torch.allclose(out1, out2, atol=1e-6), "Loaded model should match trained model"
