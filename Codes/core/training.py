# Component: training
# Source: research/method-design.md §2.5 (5 seeds), Codes/CLAUDE.md §2 (multi-seed training)
# Ablation config key: N/A (training utility, always active)
"""
AURA training utilities: ResNet-18 training with checkpointing (multi-seed).
Based on: Codes/probe/phase1_train_resnet.py, refactored for multi-seed and config-driven use.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from core.data import make_resnet18_cifar10
from core.utils import set_reproducibility


def train_resnet18(
    train_loader: DataLoader,
    test_loader: DataLoader,
    seed: int = 42,
    epochs: int = 200,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    checkpoint_dir: str | Path | None = None,
    device: str | torch.device = "cuda",
    verbose: bool = True,
    save_every_n: int = 0,
) -> dict[str, Any]:
    """Train ResNet-18 on CIFAR-10 with standard SGD + cosine annealing.

    Args:
        train_loader: Training DataLoader.
        test_loader: Test DataLoader.
        seed: Random seed for reproducibility.
        epochs: Number of training epochs.
        lr: Initial learning rate.
        momentum: SGD momentum.
        weight_decay: L2 regularization.
        checkpoint_dir: Directory to save checkpoints. None = no saving.
        device: Device to train on.
        verbose: Print progress every 10 epochs.
        save_every_n: If > 0, save checkpoint every N epochs (for TRAK multi-checkpoint).

    Returns:
        Dict with: model, best_acc, training_time_sec, checkpoint_path.
    """
    import time

    set_reproducibility(seed)
    device = torch.device(device)

    model = make_resnet18_cifar10().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    best_state = None
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        test_acc = 100.0 * test_correct / test_total

        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {
                "epoch": epoch,
                "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "test_acc": test_acc,
                "train_acc": train_acc,
                "seed": seed,
            }

        # Save periodic checkpoints (for TRAK multi-checkpoint usage)
        if save_every_n > 0 and epoch % save_every_n == 0 and checkpoint_dir is not None:
            ckpt_path = checkpoint_dir / f"resnet18_cifar10_seed{seed}_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "seed": seed,
                "test_acc": test_acc,
            }, str(ckpt_path))

        if verbose and (epoch % 10 == 0 or epoch == 1):
            elapsed = time.time() - start_time
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Train: {train_loss:.4f} / {train_acc:.2f}% | "
                f"Test: {test_acc:.2f}% | Best: {best_acc:.2f}% | "
                f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                f"{elapsed/60:.1f}min"
            )

    training_time = time.time() - start_time

    # Save best checkpoint
    checkpoint_path = None
    if checkpoint_dir is not None and best_state is not None:
        checkpoint_path = checkpoint_dir / f"resnet18_cifar10_seed{seed}.pt"
        torch.save(best_state, str(checkpoint_path))

    # Restore best model weights
    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])
    model = model.to(device)

    return {
        "model": model,
        "best_acc": best_acc,
        "training_time_sec": training_time,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "seed": seed,
        "epochs": epochs,
    }


def train_multi_seed(
    train_loader: DataLoader,
    test_loader: DataLoader,
    seeds: list[int] = (42, 123, 456, 789, 1024),
    checkpoint_dir: str | Path = "checkpoints",
    device: str | torch.device = "cuda",
    **kwargs,
) -> list[dict[str, Any]]:
    """Train ResNet-18 for multiple seeds.

    Args:
        train_loader: Training DataLoader.
        test_loader: Test DataLoader.
        seeds: List of random seeds.
        checkpoint_dir: Base directory for checkpoints.
        device: Device to train on.
        **kwargs: Additional arguments passed to train_resnet18.

    Returns:
        List of result dicts (one per seed).
    """
    results = []
    for i, seed in enumerate(seeds):
        print(f"\n=== Training seed {seed} ({i+1}/{len(seeds)}) ===")
        result = train_resnet18(
            train_loader=train_loader,
            test_loader=test_loader,
            seed=seed,
            checkpoint_dir=checkpoint_dir,
            device=device,
            **kwargs,
        )
        results.append(result)
        print(f"  Seed {seed}: {result['best_acc']:.2f}% in {result['training_time_sec']/60:.1f}min")
    return results


def compute_per_sample_gradients(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    indices: list[int],
    target_param_names: list[str] | None = None,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Compute per-sample gradient vectors via backward loop.

    Args:
        model: Trained model (will be set to eval mode).
        dataset: Source dataset.
        indices: Sample indices to compute gradients for.
        target_param_names: If provided, only include gradients for these named parameters.
                           If None, includes all parameters.
        device: Device for computation.

    Returns:
        Gradient matrix of shape (len(indices), n_params) on CPU.
    """
    device = torch.device(device)
    model = model.to(device).eval()
    criterion = nn.CrossEntropyLoss()

    # Determine target parameters
    if target_param_names is not None:
        target_params = {
            name: p for name, p in model.named_parameters()
            if name in target_param_names
        }
    else:
        target_params = dict(model.named_parameters())

    n_params = sum(p.numel() for p in target_params.values())
    grads = torch.zeros(len(indices), n_params)

    for i, idx in enumerate(indices):
        x, y = dataset[idx]
        x = x.unsqueeze(0).to(device)
        model.zero_grad()
        logits = model(x)
        loss = criterion(logits, torch.tensor([y], device=device))
        loss.backward()

        grad_parts = []
        for name in target_params:
            p = target_params[name]
            if p.grad is not None:
                grad_parts.append(p.grad.detach().flatten().cpu())
            else:
                grad_parts.append(torch.zeros(p.numel()))
        grads[i] = torch.cat(grad_parts)
        model.zero_grad()

    return grads
