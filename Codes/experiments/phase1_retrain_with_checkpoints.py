"""
Phase 1: Retrain ResNet-18 with 50 intermediate checkpoint saves for TRAK-50.
Runs on a single GPU. Saves checkpoints every 4 epochs (200/50 = 4).
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from datetime import datetime

# Configuration
SEED = 42
NUM_EPOCHS = 200
SAVE_EVERY = 4  # Save checkpoint every 4 epochs -> 50 checkpoints
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 128

TASK_ID = "phase1_retrain_ckpts"
PROJECT_DIR = Path("/home/jinxulin/sibyl_system/projects/AURA")
RESULTS_DIR = PROJECT_DIR / "exp" / "results"
CKPT_DIR = PROJECT_DIR / "exp" / "checkpoints" / "trak_checkpoints"

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)

def get_resnet18(num_classes=10):
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    # Adjust for CIFAR-10 (32x32 images): replace first conv and remove maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    set_seed(SEED)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Write PID
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='/home/jinxulin/sibyl_system/shared/datasets/cifar10',
                                             train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=4, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='/home/jinxulin/sibyl_system/shared/datasets/cifar10',
                                            train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False,
                                              num_workers=4, pin_memory=True)

    # Model
    model = get_resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    start_time = time.time()
    saved_checkpoints = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in trainloader:
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

        scheduler.step()
        train_acc = 100. * correct / total

        # Save checkpoint every SAVE_EVERY epochs
        if (epoch + 1) % SAVE_EVERY == 0:
            ckpt_path = CKPT_DIR / f"resnet18_epoch{epoch+1:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            saved_checkpoints.append(str(ckpt_path))

            # Report progress
            progress = {
                "task_id": TASK_ID,
                "epoch": epoch + 1,
                "total_epochs": NUM_EPOCHS,
                "loss": running_loss / len(trainloader),
                "train_acc": train_acc,
                "n_checkpoints_saved": len(saved_checkpoints),
                "updated_at": datetime.now().isoformat(),
            }
            (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps(progress))

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}: loss={running_loss/len(trainloader):.4f}, "
                      f"acc={train_acc:.2f}%, ckpts={len(saved_checkpoints)}")

    # Final test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = 100. * correct / total
    elapsed = time.time() - start_time

    print(f"\nFinal test accuracy: {test_acc:.2f}%")
    print(f"Saved {len(saved_checkpoints)} checkpoints")
    print(f"Elapsed: {elapsed/60:.1f} minutes")

    # Save final checkpoint too (same as the existing one for consistency)
    final_ckpt = PROJECT_DIR / "exp" / "checkpoints" / "resnet18_cifar10_seed42_trak.pt"
    torch.save(model.state_dict(), final_ckpt)

    # Write result
    result = {
        "task_id": TASK_ID,
        "test_acc": test_acc,
        "n_checkpoints": len(saved_checkpoints),
        "checkpoint_dir": str(CKPT_DIR),
        "checkpoint_paths": saved_checkpoints,
        "elapsed_minutes": elapsed / 60,
        "timestamp": datetime.now().isoformat(),
    }
    (RESULTS_DIR / f"{TASK_ID}_result.json").write_text(json.dumps(result, indent=2))

    # Write DONE marker
    pid_file.unlink(missing_ok=True)
    done_marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    done_marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": "success",
        "summary": f"Retrained ResNet-18 to {test_acc:.2f}% test acc, saved {len(saved_checkpoints)} checkpoints for TRAK-50",
        "timestamp": datetime.now().isoformat(),
    }))

if __name__ == "__main__":
    main()
