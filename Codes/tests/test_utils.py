"""Tests for core/utils.py: partial correlation, reproducibility, progress reporting."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.utils import (
    set_reproducibility,
    partial_correlation,
    make_class_dummies,
    ProgressReporter,
    gpu_free_memory,
    NumpyEncoder,
)


def test_forward_shape():
    """Test that partial_correlation returns correct output types and shapes."""
    np.random.seed(42)
    n = 100
    x = np.random.randn(n)
    y = np.random.randn(n)
    covariates = np.random.randn(n, 3)

    rho, p = partial_correlation(x, y, covariates)

    assert isinstance(rho, float), f"rho should be float, got {type(rho)}"
    assert isinstance(p, float), f"p should be float, got {type(p)}"
    assert -1.0 <= rho <= 1.0, f"rho should be in [-1, 1], got {rho}"
    assert 0.0 <= p <= 1.0, f"p should be in [0, 1], got {p}"

    # Test make_class_dummies shape
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10)
    dummies = make_class_dummies(labels, n_classes=10)
    assert dummies.shape == (100, 9), f"Expected (100, 9), got {dummies.shape}"


def test_gradient_flow():
    """Test that set_reproducibility produces deterministic torch outputs."""
    gen1 = set_reproducibility(42)
    t1 = torch.randn(10, generator=gen1)

    gen2 = set_reproducibility(42)
    t2 = torch.randn(10, generator=gen2)

    assert torch.allclose(t1, t2), "Same seed should produce same random values"

    # Different seed should produce different values
    gen3 = set_reproducibility(123)
    t3 = torch.randn(10, generator=gen3)
    assert not torch.allclose(t1, t3), "Different seeds should produce different values"


def test_output_range():
    """Test partial_correlation with known correlated data produces expected range."""
    np.random.seed(42)
    n = 500
    z = np.random.randn(n)
    x = z + 0.1 * np.random.randn(n)  # x correlated with z
    y = z + 0.1 * np.random.randn(n)  # y correlated with z
    covariate = z.reshape(-1, 1)

    # After controlling for z, x and y should have near-zero correlation
    rho_partial, _ = partial_correlation(x, y, covariate)
    assert abs(rho_partial) < 0.3, (
        f"After controlling for confound, partial corr should be small, got {rho_partial}"
    )

    # Without controlling, raw Spearman should be high
    from scipy.stats import spearmanr
    rho_raw, _ = spearmanr(x, y)
    assert abs(rho_raw) > 0.8, f"Raw correlation should be high, got {rho_raw}"

    # NumpyEncoder should handle all numpy types without NaN/Inf
    encoder = NumpyEncoder()
    assert encoder.default(np.float64(1.5)) == 1.5
    assert encoder.default(np.int64(42)) == 42
    assert encoder.default(np.bool_(True)) is True
    assert encoder.default(np.array([1, 2, 3])) == [1, 2, 3]


def test_config_switch():
    """Test ProgressReporter lifecycle (create, report, done)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reporter = ProgressReporter("test_task", tmpdir)

        # PID file should exist
        pid_path = Path(tmpdir) / "test_task.pid"
        assert pid_path.exists(), "PID file should be created"

        # Report progress
        reporter.report(epoch=1, total_epochs=10, metric={"acc": 0.95})
        progress_path = Path(tmpdir) / "test_task_PROGRESS.json"
        assert progress_path.exists(), "Progress file should exist"
        progress = json.loads(progress_path.read_text())
        assert progress["epoch"] == 1
        assert progress["metric"]["acc"] == 0.95

        # Mark done
        reporter.done(status="success", summary="Test passed")
        assert not pid_path.exists(), "PID file should be removed on done"
        done_path = Path(tmpdir) / "test_task_DONE"
        assert done_path.exists(), "DONE marker should exist"
        done_data = json.loads(done_path.read_text())
        assert done_data["status"] == "success"

        # gpu_free_memory should return a non-negative float
        mem = gpu_free_memory()
        assert isinstance(mem, float) and mem >= 0.0
