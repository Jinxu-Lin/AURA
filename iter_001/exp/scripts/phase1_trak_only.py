#!/usr/bin/env python3
"""
Phase 1: TRAK-50 attribution only (RepSim already computed).
Uses existing 50 TRAK checkpoints.
Manual TRAK: project last-layer gradients, average over checkpoints.
"""
import os, json, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from datetime import datetime

TASK_ID = "phase1_trak50"
PROJECT_DIR = Path("/home/jinxulin/sibyl_system/projects/AURA")
RESULTS_DIR = PROJECT_DIR / "exp" / "results"
ATTR_DIR    = PROJECT_DIR / "exp" / "results" / "phase1_attributions"
TRAK_CKPT_DIR = PROJECT_DIR / "exp" / "checkpoints" / "trak_checkpoints"

SEED = 42
N_PER_CLASS = 50
TRAK_PROJ_DIM = 4096

def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s)

def get_resnet18(nc=10):
    m = torchvision.models.resnet18(weights=None, num_classes=nc)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m

def select_stratified(testset, n_per_class, seed):
    rng = np.random.RandomState(seed)
    targets = np.array(testset.targets)
    idx = []
    for c in range(10):
        ci = np.where(targets == c)[0]
        idx.extend(sorted(rng.choice(ci, n_per_class, replace=False)))
    return [int(x) for x in sorted(idx)]

def report(stage, detail, pct):
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID, "stage": stage, "detail": detail,
        "percent": pct, "updated_at": datetime.now().isoformat()}))


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB, "
          f"free: {torch.cuda.mem_get_info(0)[0] / 1e9:.1f} GB")

    set_seed(SEED)
    ATTR_DIR.mkdir(parents=True, exist_ok=True)

    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))
    start_time = time.time()

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='/home/jinxulin/sibyl_system/shared/datasets/cifar10',
        train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root='/home/jinxulin/sibyl_system/shared/datasets/cifar10',
        train=False, download=True, transform=transform)

    test_indices = select_stratified(testset, N_PER_CLASS, SEED)
    test_subset = torch.utils.data.Subset(testset, test_indices)
    n_test, n_train = len(test_indices), len(trainset)
    print(f"Test: {n_test}, Train: {n_train}")

    # Checkpoints
    ckpt_paths = sorted(TRAK_CKPT_DIR.glob("resnet18_epoch*.pt"))
    n_ckpts = len(ckpt_paths)
    print(f"Found {n_ckpts} TRAK checkpoints")
    assert n_ckpts >= 50, f"Need 50 checkpoints, found {n_ckpts}"
    ckpt_paths = ckpt_paths[:50]

    report("trak", f"Computing TRAK-50 ({n_ckpts} ckpts)", 5)

    # Manual TRAK: for each checkpoint, project last-layer gradients
    # and accumulate projected train/test gradients
    proj_dim = TRAK_PROJ_DIM
    fc_dim = 512 * 10 + 10  # fc weight (10, 512) + bias (10) = 5130

    # Accumulate across checkpoints
    train_proj = torch.zeros(n_train, proj_dim, dtype=torch.float32)  # CPU
    test_proj = torch.zeros(n_test, proj_dim, dtype=torch.float32)

    for ci, ckpt_path in enumerate(ckpt_paths):
        model = get_resnet18().to(device)
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(sd, dict) and 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        model.load_state_dict(sd)
        model.eval()

        # Random projection for this checkpoint
        torch.manual_seed(SEED + ci)
        P = torch.randn(fc_dim, proj_dim, device=device) / (proj_dim ** 0.5)

        # Hook for penultimate features
        feat_cache = {}
        hook = model.avgpool.register_forward_hook(
            lambda m, i, o: feat_cache.__setitem__('f', o.detach()))

        # Training projected gradients
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

        offset = 0
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                bs = x.size(0)
                out = model(x)
                feat = feat_cache['f'].flatten(1)  # (bs, 512)

                # Per-sample fc gradient: grad_W = grad_logits^T @ feat
                probs = F.softmax(out, dim=1)
                grad_logits = probs.clone()
                grad_logits.scatter_(1, y.unsqueeze(1), grad_logits.gather(1, y.unsqueeze(1)) - 1)

                # grad_W per sample: (bs, 10, 512) → flatten → (bs, 5120)
                grad_w = torch.bmm(grad_logits.unsqueeze(2), feat.unsqueeze(1))  # (bs, 10, 512)
                grad_b = grad_logits  # (bs, 10)
                grad_flat = torch.cat([grad_w.reshape(bs, -1), grad_b], dim=1)  # (bs, 5130)

                projected = grad_flat @ P  # (bs, proj_dim)
                train_proj[offset:offset+bs] += projected.cpu()
                offset += bs

        # Test projected gradients
        test_loader = torch.utils.data.DataLoader(
            test_subset, batch_size=64, shuffle=False, num_workers=2)

        offset = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                bs = x.size(0)
                out = model(x)
                feat = feat_cache['f'].flatten(1)
                probs = F.softmax(out, dim=1)
                grad_logits = probs.clone()
                grad_logits.scatter_(1, y.unsqueeze(1), grad_logits.gather(1, y.unsqueeze(1)) - 1)
                grad_w = torch.bmm(grad_logits.unsqueeze(2), feat.unsqueeze(1))
                grad_b = grad_logits
                grad_flat = torch.cat([grad_w.reshape(bs, -1), grad_b], dim=1)
                projected = grad_flat @ P
                test_proj[offset:offset+bs] += projected.cpu()
                offset += bs

        hook.remove()
        del model, P
        torch.cuda.empty_cache()
        gc.collect()

        if (ci + 1) % 10 == 0:
            print(f"  Checkpoint {ci+1}/{n_ckpts}")
            report("trak", f"Checkpoint {ci+1}/{n_ckpts}",
                   5 + 80 * (ci+1) / n_ckpts)

    # Average over checkpoints
    train_proj /= n_ckpts
    test_proj /= n_ckpts

    # TRAK scores: dot product of projected gradients
    print("Computing TRAK scores...")
    trak_scores = (test_proj @ train_proj.T).numpy()
    print(f"TRAK scores: {trak_scores.shape}")

    # Save
    np.save(ATTR_DIR / "trak50_scores_fullmodel.npy", trak_scores)
    trak_rank = np.argsort(-trak_scores, axis=1)[:, :100]
    np.save(ATTR_DIR / "trak50_rankings_fullmodel_top100.npy", trak_rank)

    elapsed = time.time() - start_time

    result = {
        "task_id": TASK_ID, "mode": "FULL",
        "n_test": n_test, "n_train": n_train,
        "n_checkpoints": n_ckpts, "proj_dim": proj_dim,
        "shape": list(trak_scores.shape),
        "elapsed_minutes": elapsed / 60,
        "timestamp": datetime.now().isoformat(),
    }
    (ATTR_DIR / "trak50_results.json").write_text(json.dumps(result, indent=2))

    pid_file.unlink(missing_ok=True)
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID, "status": "success",
        "summary": f"TRAK-50 computed ({n_ckpts} ckpts, proj_dim={proj_dim}). {elapsed/60:.1f} min.",
        "timestamp": datetime.now().isoformat(),
    }))

    report("done", f"Done in {elapsed/60:.1f} min", 100)
    print(f"Total: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
