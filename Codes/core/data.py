# Component: data
# Source: research/method-design.md §1 (CIFAR-10, 500 test points, 50/class stratified)
# Ablation config key: N/A (data utility, always active)
"""
AURA data utilities: CIFAR-10 loading, stratified test point sampling, seed management.
Based on: Codes/probe/phase1_attribution_pilot_v4.py, refactored per Codes/CLAUDE.md.
"""

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18


# CIFAR-10 normalization constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def get_cifar10_transforms(train: bool = True) -> transforms.Compose:
    """Get standard CIFAR-10 transforms.

    Args:
        train: If True, include random augmentation (crop + flip).

    Returns:
        transforms.Compose pipeline.
    """
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])


def load_cifar10(
    data_dir: str | Path = "~/Resources/Datasets",
    download: bool = True,
) -> tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
    """Load CIFAR-10 train and test sets.

    Args:
        data_dir: Root directory for datasets.
        download: Whether to download if not present.

    Returns:
        (trainset, testset) tuple.
    """
    data_dir = Path(data_dir).expanduser()
    cifar_dir = data_dir / "cifar10"

    trainset = torchvision.datasets.CIFAR10(
        root=str(cifar_dir),
        train=True,
        download=download,
        transform=get_cifar10_transforms(train=True),
    )
    testset = torchvision.datasets.CIFAR10(
        root=str(cifar_dir),
        train=False,
        download=download,
        transform=get_cifar10_transforms(train=False),
    )
    return trainset, testset


def stratified_test_indices(
    testset: torchvision.datasets.CIFAR10,
    n_per_class: int = 50,
    n_classes: int = 10,
    seed: int = 42,
) -> list[int]:
    """Select stratified test point indices (n_per_class per class).

    Args:
        testset: CIFAR-10 test dataset.
        n_per_class: Number of test points per class.
        n_classes: Number of classes.
        seed: Random seed for reproducible selection.

    Returns:
        Sorted list of test indices, length = n_per_class * n_classes.
    """
    rng = np.random.RandomState(seed)

    by_class: dict[int, list[int]] = {c: [] for c in range(n_classes)}
    for i in range(len(testset)):
        y = testset.targets[i]
        by_class[y].append(i)

    selected = []
    for c in range(n_classes):
        candidates = by_class[c]
        chosen = rng.choice(candidates, size=min(n_per_class, len(candidates)), replace=False)
        selected.extend(chosen.tolist())

    return sorted(selected)


def stratified_train_indices(
    trainset: torchvision.datasets.CIFAR10,
    n_per_class: int = 500,
    n_classes: int = 10,
    seed: int = 42,
) -> list[int]:
    """Select stratified training subset indices.

    Args:
        trainset: CIFAR-10 train dataset.
        n_per_class: Number of training points per class.
        n_classes: Number of classes.
        seed: Random seed.

    Returns:
        Sorted list of training indices, length = n_per_class * n_classes.
    """
    rng = np.random.RandomState(seed)

    by_class: dict[int, list[int]] = {c: [] for c in range(n_classes)}
    for i in range(len(trainset)):
        y = trainset.targets[i]
        by_class[y].append(i)

    selected = []
    for c in range(n_classes):
        candidates = by_class[c]
        chosen = rng.choice(candidates, size=min(n_per_class, len(candidates)), replace=False)
        selected.extend(chosen.tolist())

    return sorted(selected)


def make_resnet18_cifar10() -> nn.Module:
    """Create ResNet-18 adapted for CIFAR-10 (32x32 images).

    Modifications from standard ImageNet ResNet-18:
    - conv1: 3x3 kernel, stride 1, padding 1 (instead of 7x7, stride 2)
    - maxpool replaced with Identity

    Returns:
        ResNet-18 model (untrained).
    """
    model = resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> dict:
    """Load model checkpoint.

    Args:
        model: Model to load weights into.
        checkpoint_path: Path to checkpoint file.
        device: Device to map checkpoint to.

    Returns:
        Full checkpoint dict (contains metadata like test_acc, seed, etc.).
    """
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def make_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True,
    indices: Sequence[int] | None = None,
) -> DataLoader:
    """Create a DataLoader, optionally from a subset of indices.

    Args:
        dataset: Source dataset.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for GPU transfer.
        indices: Optional subset indices.

    Returns:
        DataLoader instance.
    """
    if indices is not None:
        dataset = Subset(dataset, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
