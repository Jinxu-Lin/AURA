"""
Train ResNet-18 on CIFAR-10 with multiple seeds for the AURA probe experiment.
Saves trained models and training metadata.
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataloaders(batch_size=128, num_workers=4):
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

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return trainloader, testloader


def train_one_epoch(model, trainloader, optimizer, criterion, device):
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

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100. * correct / total


def evaluate(model, testloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100. * correct / total


def train(seed, device, output_dir, epochs=100):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = torchvision.models.resnet18(num_classes=10)
    # Adapt first conv for CIFAR-10 (32x32 images)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)

    trainloader, testloader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Seed {seed} | Epoch {epoch+1}/{epochs} | "
                  f"Train: {train_acc:.1f}% | Test: {test_acc:.1f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(output_dir, f'resnet18_seed{seed}_best.pt'))

    elapsed = time.time() - start_time
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, f'resnet18_seed{seed}_final.pt'))

    metadata = {
        'seed': seed,
        'epochs': epochs,
        'best_test_acc': best_acc,
        'training_time_sec': elapsed,
    }
    with open(os.path.join(output_dir, f'train_meta_seed{seed}.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Seed {seed} done. Best test acc: {best_acc:.2f}%, Time: {elapsed:.0f}s")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./outputs/models')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Training ResNet-18 on CIFAR-10 with seeds {args.seeds}")
    print(f"Device: {device}")

    for seed in args.seeds:
        print(f"\n--- Training seed {seed} ---")
        train(seed, device, args.output_dir, args.epochs)

    print("\nAll training complete.")


if __name__ == '__main__':
    main()
