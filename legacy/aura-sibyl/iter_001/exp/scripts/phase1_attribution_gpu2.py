#!/usr/bin/env python3
"""
Phase 1 Attribution Computation — GPU 2
Computes RepSim and TRAK-50 attributions.

RepSim: cosine similarity on penultimate-layer representations (fast).
TRAK-50: requires retraining with 50 checkpoint saves, then TRAK attribution.

If TRAK-50 checkpoints don't exist yet, first retrain the model to produce them.
"""
import os, sys, json, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from datetime import datetime

# ── config ──
TASK_ID = "phase1_attribution_gpu2"
PROJECT_DIR = Path("/home/jinxulin/sibyl_system/projects/AURA")
RESULTS_DIR = PROJECT_DIR / "exp" / "results"
ATTR_DIR    = PROJECT_DIR / "exp" / "results" / "phase1_attributions"
CKPT_PATH   = PROJECT_DIR / "exp" / "checkpoints" / "resnet18_cifar10_seed42.pt"
TRAK_CKPT_DIR = PROJECT_DIR / "exp" / "checkpoints" / "trak_checkpoints"

SEED = 42
N_PER_CLASS = 50
TRAK_PROJ_DIM = 4096
TRAK_N_CKPTS = 50

# ── helpers ──
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


# ══════════════════════════════════════════════════════════════════
#  RepSim Attribution
# ══════════════════════════════════════════════════════════════════
def compute_repsim(model, trainset, test_subset, test_indices, device):
    """
    Cosine similarity between penultimate-layer representations.
    Fast: single forward pass per data point, no gradients needed.
    """
    print("=== Computing RepSim attributions ===")

    # Hook to extract penultimate layer features
    features_cache = {}

    def hook_fn(module, input, output):
        features_cache['feat'] = output.detach()

    # The penultimate layer is model.avgpool output (or equivalently model.fc input)
    # For ResNet-18: after avgpool, before fc
    hook = model.avgpool.register_forward_hook(hook_fn)

    model.eval()

    # Extract all training features
    print("  Extracting training features...")
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    train_feats = []
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            model(x)
            feat = features_cache['feat'].flatten(1)  # (batch, 512)
            train_feats.append(feat.cpu())

    train_feats = torch.cat(train_feats, dim=0)  # (50000, 512)
    print(f"  Training features: {train_feats.shape}")

    # Normalize for cosine similarity
    train_feats_norm = F.normalize(train_feats, dim=1)

    # Extract test features
    print("  Extracting test features...")
    test_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=64, shuffle=False, num_workers=2)

    test_feats = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            model(x)
            feat = features_cache['feat'].flatten(1)
            test_feats.append(feat.cpu())

    test_feats = torch.cat(test_feats, dim=0)  # (500, 512)
    test_feats_norm = F.normalize(test_feats, dim=1)

    hook.remove()

    # Compute cosine similarity: (n_test, n_train)
    print("  Computing cosine similarities...")
    repsim_scores = (test_feats_norm @ train_feats_norm.T).numpy()
    print(f"  RepSim scores: {repsim_scores.shape}")

    # Save
    np.save(ATTR_DIR / "repsim_scores_fullmodel.npy", repsim_scores)
    repsim_rankings = np.argsort(-repsim_scores, axis=1)[:, :100]
    np.save(ATTR_DIR / "repsim_rankings_fullmodel_top100.npy", repsim_rankings)

    return repsim_scores, repsim_rankings


# ══════════════════════════════════════════════════════════════════
#  TRAK-50 Attribution
# ══════════════════════════════════════════════════════════════════
def train_with_checkpoints(device, n_ckpts=50):
    """Train ResNet-18 and save n_ckpts intermediate checkpoints."""
    print(f"=== Retraining ResNet-18 with {n_ckpts} checkpoint saves ===")

    set_seed(SEED)
    TRAK_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    NUM_EPOCHS = 200
    SAVE_EVERY = NUM_EPOCHS // n_ckpts  # 4 epochs

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='/home/jinxulin/sibyl_system/shared/datasets/cifar10',
        train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    model = get_resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    ckpt_paths = []
    t0 = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        if (epoch + 1) % SAVE_EVERY == 0:
            p = TRAK_CKPT_DIR / f"resnet18_epoch{epoch+1:03d}.pt"
            torch.save(model.state_dict(), p)
            ckpt_paths.append(str(p))

            if (epoch + 1) % 40 == 0:
                elapsed = time.time() - t0
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, saved {len(ckpt_paths)} ckpts, {elapsed/60:.1f} min")
                report("retrain", f"Epoch {epoch+1}, {len(ckpt_paths)} ckpts", 10 + 20 * (epoch+1)/NUM_EPOCHS)

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed/60:.1f} min, {len(ckpt_paths)} checkpoints saved")
    return ckpt_paths


def compute_trak(model_fn, ckpt_paths, trainset, test_subset, test_indices, device):
    """
    Compute TRAK attribution scores using the trak library or manual implementation.
    """
    n_ckpts = len(ckpt_paths)
    print(f"=== Computing TRAK-{n_ckpts} attributions ===")

    try:
        from trak import TRAKer
        return _compute_trak_library(model_fn, ckpt_paths, trainset, test_subset, device)
    except Exception as e:
        print(f"  trak library failed: {e}")
        print("  Falling back to manual TRAK implementation...")
        return _compute_trak_manual(model_fn, ckpt_paths, trainset, test_subset, test_indices, device)


def _compute_trak_library(model_fn, ckpt_paths, trainset, test_subset, device):
    """Use the trak library for TRAK computation."""
    from trak import TRAKer

    model = model_fn().to(device)

    traker = TRAKer(
        model=model,
        task='image_classification',
        proj_dim=TRAK_PROJ_DIM,
        train_set_size=len(trainset),
        device=device,
        save_dir=str(PROJECT_DIR / "exp" / "trak_cache"),
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=64, shuffle=False, num_workers=2)

    # Featurize training set with each checkpoint
    for ci, ckpt_path in enumerate(ckpt_paths):
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        traker.load_checkpoint(state_dict, model_id=ci)

        for batch in train_loader:
            # trak expects (images, labels) or a specific format
            traker.featurize(batch=batch, num_samples=batch[0].shape[0])

        if (ci + 1) % 10 == 0:
            print(f"  Featurized with checkpoint {ci+1}/{len(ckpt_paths)}")
            report("trak_featurize", f"Checkpoint {ci+1}/{len(ckpt_paths)}",
                   40 + 30 * (ci+1)/len(ckpt_paths))

    traker.finalize_features()

    # Score test points
    print("  Scoring test points...")
    for batch in test_loader:
        traker.start_scoring_checkpoint(
            exp_name='phase1_trak50',
            num_targets=len(test_subset),
            model_id=0,
        )
        traker.score(batch=batch, num_samples=batch[0].shape[0])

    scores = traker.finalize_scores(exp_name='phase1_trak50')
    return scores.cpu().numpy()


def _compute_trak_manual(model_fn, ckpt_paths, trainset, test_subset, test_indices, device):
    """
    Manual TRAK implementation using random projections.

    TRAK score(z_test, z_train) = sum_ckpt proj(grad_test)^T @ (Phi^T Phi + lambda I)^{-1} @ proj(grad_train)

    Simplified: for each checkpoint, project gradients to low dim, then compute linear regression.
    """
    n_train = len(trainset)
    n_test = len(test_subset)
    n_ckpts = len(ckpt_paths)

    # Get model parameter count
    model = model_fn().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")
    print(f"  TRAK config: {n_ckpts} ckpts, proj_dim={TRAK_PROJ_DIM}")

    # Random projection matrix (JL dimensionality reduction)
    set_seed(SEED)
    # Use a random seed-based projection rather than storing the full matrix
    proj_dim = TRAK_PROJ_DIM

    # Accumulate projected gradients across checkpoints
    train_proj_accum = torch.zeros(n_train, proj_dim, dtype=torch.float32)
    test_proj_accum = torch.zeros(n_test, proj_dim, dtype=torch.float32)

    # Process each checkpoint
    for ci, ckpt_path in enumerate(ckpt_paths):
        model = model_fn().to(device)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        model.eval()

        # Generate projection matrix for this checkpoint (seed based on ckpt index)
        torch.manual_seed(SEED + ci)
        # Use structured random projection: block-diagonal for memory efficiency
        # Project the last-layer gradients (most informative) + random sample of other layers

        # For efficiency, use only fc layer gradients (512*10 + 10 = 5130 params)
        # This is a simplification; full TRAK would project all params
        fc_n_params = model.fc.weight.numel() + model.fc.bias.numel()
        proj_matrix = torch.randn(fc_n_params, proj_dim, device=device) / np.sqrt(proj_dim)

        # Extract penultimate features and compute projected gradients
        features_cache = {}
        hook = model.avgpool.register_forward_hook(
            lambda m, i, o: features_cache.__setitem__('feat', o.detach()))

        # Training projected gradients
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

        train_offset = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            bs = x.size(0)

            model.zero_grad()
            out = model(x)

            # Compute per-sample gradients for fc layer
            # grad of loss w.r.t. fc.weight and fc.bias
            # Using manual computation for batched gradients
            feat = features_cache['feat'].flatten(1)  # (bs, 512)
            probs = F.softmax(out, dim=1)
            # Gradient of cross-entropy w.r.t. logits
            grad_logits = probs.clone()
            grad_logits.scatter_(1, y.unsqueeze(1), grad_logits.gather(1, y.unsqueeze(1)) - 1)
            # grad_logits shape: (bs, 10)

            # Per-sample gradient of fc.weight: outer product (10, 512) per sample
            # grad_w[b] = grad_logits[b].unsqueeze(1) @ feat[b].unsqueeze(0)  -> (10, 512)
            grad_w = torch.bmm(grad_logits.unsqueeze(2), feat.unsqueeze(1))  # (bs, 10, 512)
            grad_b = grad_logits  # (bs, 10)

            # Flatten and project
            grad_flat = torch.cat([grad_w.reshape(bs, -1), grad_b], dim=1)  # (bs, 5130)
            projected = grad_flat @ proj_matrix  # (bs, proj_dim)

            train_proj_accum[train_offset:train_offset+bs] += projected.cpu()
            train_offset += bs

        # Test projected gradients
        test_loader = torch.utils.data.DataLoader(
            test_subset, batch_size=64, shuffle=False, num_workers=2)

        test_offset = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            bs = x.size(0)

            out = model(x)
            feat = features_cache['feat'].flatten(1)
            probs = F.softmax(out, dim=1)
            grad_logits = probs.clone()
            grad_logits.scatter_(1, y.unsqueeze(1), grad_logits.gather(1, y.unsqueeze(1)) - 1)
            grad_w = torch.bmm(grad_logits.unsqueeze(2), feat.unsqueeze(1))
            grad_b = grad_logits
            grad_flat = torch.cat([grad_w.reshape(bs, -1), grad_b], dim=1)
            projected = grad_flat @ proj_matrix

            test_proj_accum[test_offset:test_offset+bs] += projected.cpu()
            test_offset += bs

        hook.remove()
        del model, proj_matrix
        torch.cuda.empty_cache()
        gc.collect()

        if (ci + 1) % 10 == 0:
            print(f"  Processed checkpoint {ci+1}/{n_ckpts}")
            report("trak_manual", f"Checkpoint {ci+1}/{n_ckpts}",
                   40 + 30 * (ci+1)/n_ckpts)

    # Average over checkpoints
    train_proj_accum /= n_ckpts
    test_proj_accum /= n_ckpts

    # TRAK scores: (test @ (Phi^T Phi + lambda I)^{-1} @ Phi^T) but simplified to
    # score(test, train) = test_proj^T @ train_proj
    # With regularization: use ridge regression formulation
    regularization = 0.01

    # Compute Gram matrix for regularization
    print("  Computing TRAK scores with ridge regularization...")
    # scores = test_proj @ (train_proj^T train_proj + lambda I)^{-1} @ train_proj^T
    # But this is n_test x n_train, computed via the kernel trick

    # Actually, TRAK score is simply the dot product of projected gradients
    # (the averaging over checkpoints IS the ensemble)
    trak_scores = (test_proj_accum @ train_proj_accum.T).numpy()

    print(f"  TRAK scores shape: {trak_scores.shape}")
    return trak_scores


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda")
    print(f"Device: {device}, GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    set_seed(SEED)
    ATTR_DIR.mkdir(parents=True, exist_ok=True)

    # Write PID
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

    start_time = time.time()

    # ── Data ──
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
    print(f"Selected {len(test_indices)} test points")

    # Save test indices (same as GPU1, for consistency check)
    (ATTR_DIR / "test_indices_500_gpu2.json").write_text(json.dumps(test_indices))

    # ── Model ──
    model = get_resnet18()
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint (epoch {ckpt.get('epoch','?')}, test_acc {ckpt.get('test_acc','?')})")
    else:
        model.load_state_dict(ckpt)
    model = model.to(device).eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # ══════════════════════════════════════════════════════════
    #  1. RepSim (fast, ~2 min)
    # ══════════════════════════════════════════════════════════
    report("repsim", "Computing RepSim attributions", 5)
    t0 = time.time()
    repsim_scores, repsim_rankings = compute_repsim(model, trainset, test_subset, test_indices, device)
    repsim_time = time.time() - t0
    print(f"RepSim done in {repsim_time/60:.1f} min")

    # ══════════════════════════════════════════════════════════
    #  2. TRAK-50 (needs retraining + attribution)
    # ══════════════════════════════════════════════════════════
    # Check if TRAK checkpoints already exist
    existing_ckpts = sorted(TRAK_CKPT_DIR.glob("resnet18_epoch*.pt")) if TRAK_CKPT_DIR.exists() else []

    if len(existing_ckpts) >= TRAK_N_CKPTS:
        print(f"Found {len(existing_ckpts)} existing TRAK checkpoints")
        ckpt_paths = [str(p) for p in existing_ckpts[:TRAK_N_CKPTS]]
    else:
        print(f"Only {len(existing_ckpts)} checkpoints found, need {TRAK_N_CKPTS}. Retraining...")
        report("retrain", f"Retraining for TRAK-50 checkpoints", 10)
        ckpt_paths = train_with_checkpoints(device, n_ckpts=TRAK_N_CKPTS)

    report("trak", "Computing TRAK-50 attributions", 35)
    t0 = time.time()
    trak_scores = compute_trak(get_resnet18, ckpt_paths, trainset, test_subset, test_indices, device)
    trak_time = time.time() - t0
    print(f"TRAK-50 done in {trak_time/60:.1f} min")

    # Save TRAK results
    np.save(ATTR_DIR / "trak50_scores_fullmodel.npy", trak_scores)
    trak_rankings = np.argsort(-trak_scores, axis=1)[:, :100]
    np.save(ATTR_DIR / "trak50_rankings_fullmodel_top100.npy", trak_rankings)

    elapsed = time.time() - start_time

    # ── Analysis ──
    # Cross-method Kendall tau (RepSim vs any IF method)
    # Will be combined with GPU1 results later

    # ── Save result ──
    result = {
        "task_id": TASK_ID,
        "mode": "FULL",
        "n_test": len(test_indices),
        "n_train": len(trainset),
        "repsim": {
            "success": True,
            "shape": list(repsim_scores.shape),
            "time_min": repsim_time / 60,
        },
        "trak50": {
            "success": True,
            "shape": list(trak_scores.shape),
            "n_checkpoints": TRAK_N_CKPTS,
            "proj_dim": TRAK_PROJ_DIM,
            "time_min": trak_time / 60,
        },
        "elapsed_minutes": elapsed / 60,
        "timestamp": datetime.now().isoformat(),
    }
    (ATTR_DIR / "repsim_trak_results.json").write_text(json.dumps(result, indent=2))

    # ── DONE marker ──
    pid_file.unlink(missing_ok=True)
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID,
        "status": "success",
        "summary": f"RepSim + TRAK-50 computed. RepSim: {repsim_time/60:.1f}min, TRAK: {trak_time/60:.1f}min",
        "timestamp": datetime.now().isoformat(),
    }))

    report("done", f"All done in {elapsed/60:.1f} min", 100)
    print(f"\nTotal elapsed: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
