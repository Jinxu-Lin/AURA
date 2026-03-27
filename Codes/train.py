#!/usr/bin/env python3
"""
AURA train.py — ResNet-18 training on CIFAR-10 with full reproducibility.

This is NOT the core experiment; it is infrastructure for producing trained
models needed by the BSS diagnostic pipeline. Supports multi-seed training,
checkpointing, dry-run mode, and resume.

Usage:
    python train.py --config configs/base.yaml
    python train.py --config configs/base.yaml --dry-run --max-steps 2
    python train.py --config configs/base.yaml --seed 789
    python train.py --config configs/base.yaml --resume
    python train.py --config configs/pilot.yaml
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add Codes/ to path for core imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from config_loader import add_common_args, load_config, resolve_paths
from core.data import load_cifar10, make_dataloader, make_resnet18_cifar10
from core.utils import NumpyEncoder, ProgressReporter, set_reproducibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AURA: Train ResNet-18 on CIFAR-10")
    add_common_args(parser)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max training steps (for dry-run). Overrides epochs.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint if available")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override seed list (e.g., --seeds 789 1024)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save intermediate checkpoints every N epochs (for TRAK)")
    return parser.parse_args()


def train_single_seed(
    seed: int,
    config: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    """Train ResNet-18 for a single seed. Returns result dict."""
    print(f"\n{'='*60}")
    print(f"Training seed {seed}")
    print(f"{'='*60}")

    set_reproducibility(seed)

    data_dir = Path(config["data_dir"])
    checkpoint_dir = data_dir / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"resnet18_cifar10_seed{seed}.pt"

    # Check for existing checkpoint (skip if already trained)
    if checkpoint_path.exists() and not args.resume:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        existing_acc = ckpt.get("test_acc", 0)
        if existing_acc > 93.0:
            print(f"  Checkpoint exists with {existing_acc:.2f}% acc. Skipping training.")
            return {
                "seed": seed,
                "status": "skipped",
                "test_acc": existing_acc,
                "checkpoint_path": str(checkpoint_path),
            }

    # Load data
    trainset, testset = load_cifar10(data_dir=config.get("dataset_dir", "~/Resources/Datasets"))
    train_loader = make_dataloader(trainset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = make_dataloader(testset, batch_size=256, shuffle=False, num_workers=4)

    # Model
    model = make_resnet18_cifar10().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
    )
    epochs = args.epochs or config.get("train_epochs", 200)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()

    # Resume logic
    start_epoch = 1
    best_acc = 0.0
    best_state = None
    if args.resume and checkpoint_path.exists():
        ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("test_acc", 0)
        print(f"  Resumed from epoch {start_epoch - 1}, acc {best_acc:.2f}%")

    # Dry-run: override epochs
    max_steps = args.max_steps
    if args.dry_run and max_steps is None:
        max_steps = 2

    # Progress reporter
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    reporter = ProgressReporter(f"train_seed{seed}", results_dir)

    start_time = time.time()
    global_step = 0

    for epoch in range(start_epoch, epochs + 1):
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if max_steps is not None and global_step >= max_steps:
                break

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
            global_step += 1

        if max_steps is not None and global_step >= max_steps:
            print(f"  Dry-run: stopped after {global_step} steps")
            break

        train_loss = running_loss / max(len(train_loader), 1)
        train_acc = 100.0 * correct / max(total, 1)

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
        test_acc = 100.0 * test_correct / max(test_total, 1)

        scheduler.step()

        # Track best
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

        # Save intermediate checkpoints (for TRAK multi-checkpoint)
        if args.save_every > 0 and epoch % args.save_every == 0:
            trak_dir = checkpoint_dir / "trak_checkpoints"
            trak_dir.mkdir(parents=True, exist_ok=True)
            trak_path = trak_dir / f"resnet18_seed{seed}_epoch{epoch:03d}.pt"
            torch.save(model.state_dict(), str(trak_path))

        # Report progress
        reporter.report(
            epoch=epoch, total_epochs=epochs,
            loss=train_loss,
            metric={"train_acc": train_acc, "test_acc": test_acc, "best_acc": best_acc},
        )

        if epoch % 10 == 0 or epoch == 1:
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
    if best_state is not None and not args.dry_run:
        torch.save(best_state, str(checkpoint_path))
        print(f"  Saved checkpoint: {checkpoint_path}")

    reporter.done(
        status="success",
        summary=f"Seed {seed}: {best_acc:.2f}% in {training_time/60:.1f}min"
    )

    return {
        "seed": seed,
        "status": "success" if not args.dry_run else "dry_run",
        "test_acc": best_acc,
        "training_time_min": round(training_time / 60, 1),
        "checkpoint_path": str(checkpoint_path) if not args.dry_run else None,
        "epochs_completed": epoch if not (max_steps and global_step >= max_steps) else f"dry_run_{global_step}_steps",
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    resolve_paths(config)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[train.py] Device: {device}")
    if device.type == "cuda":
        print(f"[train.py] GPU: {torch.cuda.get_device_name()}")

    # Seeds to train
    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = config.get("seeds", [42])

    print(f"[train.py] Seeds: {seeds}")
    print(f"[train.py] Dry-run: {args.dry_run}")

    # Train each seed
    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n[train.py] Seed {seed} ({i+1}/{len(seeds)})")
        result = train_single_seed(seed, config, args, device)
        all_results.append(result)
        print(f"[train.py] Seed {seed}: {result['status']} ({result.get('test_acc', 'N/A')}%)")

    # Save summary
    results_dir = Path(config["results_dir"])
    summary = {
        "task": "train_resnet18",
        "seeds": seeds,
        "results": all_results,
        "dry_run": args.dry_run,
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = results_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, cls=NumpyEncoder))
    print(f"\n[train.py] Summary saved to {summary_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'Seed':<8} {'Status':<12} {'Acc':<10} {'Time':<10}")
    print(f"{'='*60}")
    for r in all_results:
        acc = f"{r.get('test_acc', 0):.2f}%" if r.get('test_acc') else "N/A"
        t = f"{r.get('training_time_min', 0):.1f}min" if r.get('training_time_min') else "N/A"
        print(f"{r['seed']:<8} {r['status']:<12} {acc:<10} {t:<10}")


if __name__ == "__main__":
    main()
