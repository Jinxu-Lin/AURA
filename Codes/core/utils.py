# Component: utils
# Source: research/method-design.md §2.3 (partial correlation), Codes/CLAUDE.md §4
# Ablation config key: N/A (utility module, always active)
"""
AURA shared utilities: partial correlation, progress reporting, reproducibility setup.
Reused patterns from probe code (phase2a_bss_analysis.py), refactored for reusability.
"""

import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from numpy.linalg import lstsq
from scipy.stats import spearmanr


def set_reproducibility(seed: int = 42) -> torch.Generator:
    """Set all random seeds for reproducibility.

    Args:
        seed: Random seed value.

    Returns:
        torch.Generator seeded with the given seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def partial_correlation(
    x: np.ndarray,
    y: np.ndarray,
    covariates: np.ndarray,
) -> tuple[float, float]:
    """Compute partial Spearman correlation between x and y, controlling for covariates.

    Uses the residual method: regress out covariates from both x and y via OLS,
    then compute Spearman correlation on the residuals.

    Args:
        x: Array of shape (n,).
        y: Array of shape (n,).
        covariates: Array of shape (n, p) with p covariates.

    Returns:
        (rho, p_value) tuple.
    """
    # x, y: (n,)
    # covariates: (n, p)
    assert x.ndim == 1 and y.ndim == 1, f"x and y must be 1D, got {x.ndim}D and {y.ndim}D"
    assert x.shape[0] == y.shape[0] == covariates.shape[0], "All inputs must have same length"

    # Add intercept
    C = np.column_stack([covariates, np.ones(len(x))])

    # Residualize x
    beta_x, _, _, _ = lstsq(C, x, rcond=None)
    resid_x = x - C @ beta_x

    # Residualize y
    beta_y, _, _, _ = lstsq(C, y, rcond=None)
    resid_y = y - C @ beta_y

    rho, p = spearmanr(resid_x, resid_y)
    return float(rho), float(p)


def make_class_dummies(labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Create class dummy variables (drop last class for identifiability).

    Args:
        labels: Integer class labels, shape (n,).
        n_classes: Total number of classes.

    Returns:
        Dummy matrix of shape (n, n_classes - 1).
    """
    n = len(labels)
    dummies = np.zeros((n, n_classes - 1))
    for i in range(n):
        cls = int(labels[i])
        if cls < n_classes - 1:
            dummies[i, cls] = 1.0
    return dummies


class ProgressReporter:
    """Progress reporting and task lifecycle management for experiments."""

    def __init__(self, task_id: str, results_dir: str | Path):
        self.task_id = task_id
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Write PID file
        pid_file = self.results_dir / f"{task_id}.pid"
        pid_file.write_text(str(os.getpid()))

    def report(
        self,
        epoch: int = 0,
        total_epochs: int = 0,
        step: int = 0,
        total_steps: int = 0,
        loss: float | None = None,
        metric: dict | None = None,
    ) -> None:
        """Write progress JSON."""
        progress_path = self.results_dir / f"{self.task_id}_PROGRESS.json"
        progress_path.write_text(json.dumps({
            "task_id": self.task_id,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "step": step,
            "total_steps": total_steps,
            "loss": loss,
            "metric": metric or {},
            "updated_at": datetime.now().isoformat(),
        }))

    def done(self, status: str = "success", summary: str = "") -> None:
        """Mark task as done: remove PID, write DONE marker."""
        pid_file = self.results_dir / f"{self.task_id}.pid"
        if pid_file.exists():
            pid_file.unlink()

        progress_file = self.results_dir / f"{self.task_id}_PROGRESS.json"
        final_progress = {}
        if progress_file.exists():
            try:
                final_progress = json.loads(progress_file.read_text())
            except (json.JSONDecodeError, ValueError):
                pass

        marker = self.results_dir / f"{self.task_id}_DONE"
        marker.write_text(json.dumps({
            "task_id": self.task_id,
            "status": status,
            "summary": summary,
            "final_progress": final_progress,
            "timestamp": datetime.now().isoformat(),
        }))


def gpu_free_memory() -> float:
    """Return free GPU memory in GB. Returns 0.0 if no GPU available."""
    if not torch.cuda.is_available():
        return 0.0
    return (
        torch.cuda.get_device_properties(0).total_memory
        - torch.cuda.memory_allocated()
    ) / 1e9


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
