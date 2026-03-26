"""
Phase 1: Full-model EK-FAC IF attribution for 500 test points.
Uses dattri IFAttributorEKFAC with module_name=None (full model).
Runs on a single GPU (requires ~16-20GB).
"""
import os
import sys
import json
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from datetime import datetime

TASK_ID = "phase1_ekfac_if"
PROJECT_DIR = Path("/home/jinxulin/sibyl_system/projects/AURA")
RESULTS_DIR = PROJECT_DIR / "exp" / "results"
ATTR_DIR = PROJECT_DIR / "exp" / "results" / "phase1_attributions"
CKPT_PATH = PROJECT_DIR / "exp" / "checkpoints" / "resnet18_cifar10_seed42.pt"

SEED = 42
N_TEST = 500  # 50 per class
N_TRAIN = 50000  # Full training set
DAMPING = 0.01
BATCH_SIZE_TRAIN = 64  # For K-FAC factor computation
BATCH_SIZE_TEST = 16   # For attribution computation

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_resnet18(num_classes=10):
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

def select_stratified_test_indices(testset, n_per_class=50, seed=42):
    """Select n_per_class test samples from each of 10 classes."""
    rng = np.random.RandomState(seed)
    targets = np.array(testset.targets)
    indices = []
    for c in range(10):
        class_indices = np.where(targets == c)[0]
        chosen = rng.choice(class_indices, size=n_per_class, replace=False)
        indices.extend(sorted(chosen))
    return sorted(indices)

def report_progress(task_id, stage, detail, percent=0):
    progress_file = RESULTS_DIR / f"{task_id}_PROGRESS.json"
    progress_file.write_text(json.dumps({
        "task_id": task_id,
        "stage": stage,
        "detail": detail,
        "percent": percent,
        "updated_at": datetime.now().isoformat(),
    }))

def main():
    device = torch.device("cuda")
    print(f"Device: {device}, GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    set_seed(SEED)
    ATTR_DIR.mkdir(parents=True, exist_ok=True)

    # Write PID
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

    start_time = time.time()

    # Data transforms (no augmentation for attribution computation)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='/home/jinxulin/sibyl_system/shared/datasets/cifar10',
        train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root='/home/jinxulin/sibyl_system/shared/datasets/cifar10',
        train=False, download=False, transform=transform)

    # Select 500 stratified test points
    test_indices = select_stratified_test_indices(testset, n_per_class=50, seed=SEED)
    print(f"Selected {len(test_indices)} test points (50/class)")

    # Save test indices for other methods to use
    test_indices_file = ATTR_DIR / "test_indices_500.json"
    test_indices_file.write_text(json.dumps(test_indices))

    # Create subset datasets
    test_subset = torch.utils.data.Subset(testset, test_indices)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE_TRAIN, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=2, pin_memory=True)

    # Load model
    model = get_resnet18(num_classes=10)
    state_dict = torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {CKPT_PATH}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    report_progress(TASK_ID, "setup", "Model loaded, computing EK-FAC factors", 5)

    # Define loss function for dattri
    def loss_fn(params, data_target_pair):
        """Loss function compatible with dattri's functional API."""
        x, y = data_target_pair
        import torch.nn.functional as F
        # This will be called by dattri internally
        output = torch.func.functional_call(model, params, x)
        return F.cross_entropy(output, y)

    # Try using dattri EK-FAC attributor
    try:
        from dattri.task import AttributionTask
        from dattri.algorithm import IFAttributorEKFAC

        # Create attribution task
        task = AttributionTask(
            loss_func=loss_fn,
            model=model,
            checkpoints=CKPT_PATH.as_posix(),
        )

        report_progress(TASK_ID, "ekfac_init", "Initializing EK-FAC attributor (full model)", 10)

        # Initialize EK-FAC with full model (module_name=None)
        attributor = IFAttributorEKFAC(
            task=task,
            module_name=None,  # Full model, not just last layer
            device=str(device),
            damping=DAMPING,
        )

        report_progress(TASK_ID, "ekfac_cache", "Computing K-FAC factors (full training set)", 15)

        # Cache K-FAC factors on full training set
        print("Computing K-FAC factors on full training set...")
        cache_start = time.time()
        attributor.cache(full_train_dataloader=train_loader)
        cache_time = time.time() - cache_start
        print(f"K-FAC factor computation took {cache_time/60:.1f} minutes")

        report_progress(TASK_ID, "ekfac_attribute", "Computing EK-FAC attributions for 500 test points", 40)

        # Compute attributions
        print("Computing EK-FAC attributions...")
        attr_start = time.time()

        # Process in batches to manage memory
        # Use a smaller train_loader for attribution (dattri computes train-test pairs)
        attr_train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=BATCH_SIZE_TRAIN, shuffle=False, num_workers=4, pin_memory=True)

        ekfac_scores = attributor.attribute(
            train_dataloader=attr_train_loader,
            test_dataloader=test_loader,
        )
        attr_time = time.time() - attr_start
        print(f"EK-FAC attribution took {attr_time/60:.1f} minutes")
        print(f"EK-FAC scores shape: {ekfac_scores.shape}")

        # Save scores
        ekfac_scores_np = ekfac_scores.cpu().numpy()
        np.save(ATTR_DIR / "ekfac_scores_fullmodel.npy", ekfac_scores_np)

        # Save top-100 rankings
        ekfac_rankings = np.argsort(-ekfac_scores_np, axis=1)[:, :100]
        np.save(ATTR_DIR / "ekfac_rankings_fullmodel_top100.npy", ekfac_rankings)

        ekfac_success = True
        ekfac_info = {
            "success": True,
            "shape": list(ekfac_scores_np.shape),
            "cache_time_min": cache_time / 60,
            "attr_time_min": attr_time / 60,
            "damping": DAMPING,
            "module_name": "None (full model)",
        }
        print("EK-FAC IF attribution completed successfully!")

    except Exception as e:
        print(f"EK-FAC via dattri failed: {e}")
        import traceback
        traceback.print_exc()
        ekfac_success = False
        ekfac_info = {"success": False, "error": str(e)}

        # Fallback: Manual K-FAC implementation for full model
        print("\n=== Falling back to manual full-model K-FAC/EK-FAC ===")
        try:
            ekfac_scores_np, ekfac_info = manual_ekfac_attribution(
                model, trainset, test_subset, test_indices, device)
            ekfac_success = True
        except Exception as e2:
            print(f"Manual EK-FAC also failed: {e2}")
            traceback.print_exc()
            ekfac_info = {"success": False, "error": f"dattri: {e}, manual: {e2}"}

    # Also compute test point features (gradient norms, confidence, entropy)
    report_progress(TASK_ID, "features", "Computing test point features", 85)
    features = compute_test_features(model, test_subset, test_indices, testset, device)

    elapsed = time.time() - start_time

    # Write result
    result = {
        "task_id": TASK_ID,
        "method": "ekfac_if_fullmodel",
        "n_test": len(test_indices),
        "n_train": N_TRAIN,
        "ekfac": ekfac_info,
        "features_computed": True,
        "elapsed_minutes": elapsed / 60,
        "timestamp": datetime.now().isoformat(),
    }
    (RESULTS_DIR / f"{TASK_ID}_result.json").write_text(json.dumps(result, indent=2))

    # Write DONE marker
    pid_file.unlink(missing_ok=True)
    status = "success" if ekfac_success else "failed"
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": f"EK-FAC full-model IF: {'OK' if ekfac_success else 'FAILED'}. {elapsed/60:.1f} min.",
        "timestamp": datetime.now().isoformat(),
    }))

    report_progress(TASK_ID, "done", f"Completed in {elapsed/60:.1f} min", 100)
    print(f"\nTotal elapsed: {elapsed/60:.1f} minutes")


def manual_ekfac_attribution(model, trainset, test_subset, test_indices, device):
    """Manual implementation of full-model K-FAC/EK-FAC IF attribution."""
    import torch.nn.functional as F
    from collections import OrderedDict

    report_progress(TASK_ID, "manual_ekfac", "Manual K-FAC factor computation", 20)

    # Collect all layers with parameters
    layers_info = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers_info.append((name, module))
    print(f"Found {len(layers_info)} parameterized layers for K-FAC")

    # K-FAC factor computation
    # For each layer, compute A = E[a*a^T] and G = E[g*g^T]
    # where a = input activation, g = output gradient

    model.eval()

    # Hook-based K-FAC factor accumulation
    kfac_factors = {}  # layer_name -> (A, G)
    activations = {}
    output_grads = {}

    def make_forward_hook(name):
        def hook(module, input, output):
            if isinstance(module, nn.Linear):
                # input shape: (batch, in_features)
                a = input[0].detach()
                if module.bias is not None:
                    a = torch.cat([a, torch.ones(a.shape[0], 1, device=a.device)], dim=1)
                activations[name] = a
            elif isinstance(module, nn.Conv2d):
                # For conv layers, unfold input
                a = input[0].detach()
                activations[name] = a
        return hook

    def make_backward_hook(name):
        def hook(module, grad_input, grad_output):
            g = grad_output[0].detach()
            output_grads[name] = g
        return hook

    hooks = []
    for name, module in layers_info:
        hooks.append(module.register_forward_hook(make_forward_hook(name)))
        hooks.append(module.register_full_backward_hook(make_backward_hook(name)))

    # Accumulate factors over training set
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    n_samples = 0
    factor_A = {name: None for name, _ in layers_info}
    factor_G = {name: None for name, _ in layers_info}

    print("Computing K-FAC factors over training set...")
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        bs = inputs.size(0)

        model.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()

        for name, module in layers_info:
            if name not in activations or name not in output_grads:
                continue

            if isinstance(module, nn.Linear):
                a = activations[name]  # (batch, in+1)
                g = output_grads[name]  # (batch, out)

                A_batch = (a.T @ a) / bs
                G_batch = (g.T @ g) / bs

                if factor_A[name] is None:
                    factor_A[name] = A_batch
                    factor_G[name] = G_batch
                else:
                    factor_A[name] += A_batch
                    factor_G[name] += G_batch
            elif isinstance(module, nn.Conv2d):
                # For conv: unfold and treat as locally connected linear
                a = activations[name]  # (batch, C_in, H, W)
                g = output_grads[name]  # (batch, C_out, H', W')

                # Spatial unfolding for A
                k = module.kernel_size
                s = module.stride
                p = module.padding
                a_unf = F.unfold(a, k, padding=p, stride=s)  # (batch, C_in*k*k, L)
                a_unf = a_unf.mean(dim=2)  # average over spatial locations -> (batch, C_in*k*k)
                if module.bias is not None:
                    a_unf = torch.cat([a_unf, torch.ones(bs, 1, device=device)], dim=1)

                # G: average over spatial locations
                g_avg = g.mean(dim=[2, 3])  # (batch, C_out)

                A_batch = (a_unf.T @ a_unf) / bs
                G_batch = (g_avg.T @ g_avg) / bs

                if factor_A[name] is None:
                    factor_A[name] = A_batch
                    factor_G[name] = G_batch
                else:
                    factor_A[name] += A_batch
                    factor_G[name] += G_batch

        n_samples += bs
        activations.clear()
        output_grads.clear()

        if (batch_idx + 1) % 100 == 0:
            print(f"  K-FAC factors: batch {batch_idx+1}/{len(train_loader)}")
            report_progress(TASK_ID, "manual_kfac_factors",
                           f"Batch {batch_idx+1}/{len(train_loader)}",
                           20 + 30 * (batch_idx + 1) / len(train_loader))

    n_batches = batch_idx + 1
    for name, _ in layers_info:
        if factor_A[name] is not None:
            factor_A[name] /= n_batches
            factor_G[name] /= n_batches

    # Remove hooks
    for h in hooks:
        h.remove()

    print("K-FAC factors computed. Computing eigendecompositions for EK-FAC...")
    report_progress(TASK_ID, "manual_ekfac_eigen", "Eigendecomposition of K-FAC factors", 55)

    # EK-FAC: eigendecompose A and G factors
    eigen_A = {}
    eigen_G = {}
    for name, module in layers_info:
        if factor_A[name] is None:
            continue
        # Eigendecompose A
        A = factor_A[name].float()
        evals_a, evecs_a = torch.linalg.eigh(A)
        eigen_A[name] = (evals_a, evecs_a)

        # Eigendecompose G
        G = factor_G[name].float()
        evals_g, evecs_g = torch.linalg.eigh(G)
        eigen_G[name] = (evals_g, evecs_g)

    print("Eigendecompositions complete.")

    # Now compute EK-FAC IF attributions
    # For each (test, train) pair, IF score = grad_test^T H^{-1} grad_train
    # With EK-FAC: H^{-1} approx = sum_l (Q_A_l kron Q_G_l) diag(1/(lambda_a kron lambda_g + damping)) (Q_A_l kron Q_G_l)^T
    # This means: for each layer l, project gradients into eigenspace and scale by inverse eigenvalues

    damping = DAMPING

    # Compute per-layer IHVP for test points
    print("Computing EK-FAC attributions for 500 test points against 50K training points...")
    report_progress(TASK_ID, "manual_ekfac_attr", "Computing attributions", 60)

    # Strategy: compute test gradients, apply EK-FAC IHVP, then dot with train gradients
    # Memory-efficient: process test points one at a time, batch over training points

    test_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=1, shuffle=False, num_workers=2)

    # We'll compute attributions in chunks to fit in memory
    # For each test point: compute ihvp, then batch dot with all train gradients

    n_test = len(test_subset)
    n_train = len(trainset)

    # Pre-compute all training gradients per layer (memory intensive but faster)
    # Alternative: compute on-the-fly
    # Given 50K training * 11M params -> too much memory
    # Instead: compute ihvp(test_grad) then iterate over training batches

    ekfac_scores = np.zeros((n_test, n_train), dtype=np.float32)

    for test_idx, (test_x, test_y) in enumerate(test_loader):
        test_x, test_y = test_x.to(device), test_y.to(device)

        # Compute test gradient
        model.zero_grad()
        out = model(test_x)
        loss = F.cross_entropy(out, test_y)
        loss.backward()

        # Collect per-layer test gradients and apply EK-FAC inverse
        ihvp_params = {}
        for name, module in layers_info:
            if name not in eigen_A:
                continue

            evals_a, evecs_a = eigen_A[name]
            evals_g, evecs_g = eigen_G[name]

            if isinstance(module, nn.Linear):
                # Weight gradient: (out, in) -> reshape to (out, in)
                grad_w = module.weight.grad.detach().float()
                if module.bias is not None:
                    grad_b = module.bias.grad.detach().float().unsqueeze(1)
                    grad_w = torch.cat([grad_w, grad_b], dim=1)  # (out, in+1)

                # Project: Q_G^T grad Q_A
                projected = evecs_g.T @ grad_w @ evecs_a

                # Scale by inverse eigenvalues: 1 / (lambda_g kron lambda_a + damping)
                lambda_kron = evals_g.unsqueeze(1) * evals_a.unsqueeze(0)  # (out, in)
                scaled = projected / (lambda_kron + damping)

                # Unproject: Q_G scaled Q_A^T
                ihvp_w = evecs_g @ scaled @ evecs_a.T

                if module.bias is not None:
                    ihvp_params[(name, 'weight')] = ihvp_w[:, :-1]
                    ihvp_params[(name, 'bias')] = ihvp_w[:, -1]
                else:
                    ihvp_params[(name, 'weight')] = ihvp_w

            elif isinstance(module, nn.Conv2d):
                # Similar but with spatial averaging
                grad_w = module.weight.grad.detach().float()  # (C_out, C_in, k, k)
                C_out = grad_w.shape[0]
                grad_w_2d = grad_w.reshape(C_out, -1)  # (C_out, C_in*k*k)

                if module.bias is not None:
                    grad_b = module.bias.grad.detach().float().unsqueeze(1)
                    grad_w_2d = torch.cat([grad_w_2d, grad_b], dim=1)

                projected = evecs_g.T @ grad_w_2d @ evecs_a
                lambda_kron = evals_g.unsqueeze(1) * evals_a.unsqueeze(0)
                scaled = projected / (lambda_kron + damping)
                ihvp_w_2d = evecs_g @ scaled @ evecs_a.T

                if module.bias is not None:
                    ihvp_params[(name, 'weight')] = ihvp_w_2d[:, :-1].reshape(grad_w.shape)
                    ihvp_params[(name, 'bias')] = ihvp_w_2d[:, -1]
                else:
                    ihvp_params[(name, 'weight')] = ihvp_w_2d.reshape(grad_w.shape)

        # Now compute dot product with each training gradient
        train_loader_attr = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

        train_idx = 0
        for train_x, train_y in train_loader_attr:
            train_x, train_y = train_x.to(device), train_y.to(device)
            bs = train_x.size(0)

            scores = torch.zeros(bs, device=device)

            # Compute train gradients one-by-one and dot with ihvp
            for b in range(bs):
                model.zero_grad()
                out_b = model(train_x[b:b+1])
                loss_b = F.cross_entropy(out_b, train_y[b:b+1])
                loss_b.backward()

                dot = 0.0
                for name, module in layers_info:
                    if (name, 'weight') not in ihvp_params:
                        continue
                    grad_w = module.weight.grad.detach().float()
                    dot += (ihvp_params[(name, 'weight')] * grad_w).sum().item()
                    if module.bias is not None and (name, 'bias') in ihvp_params:
                        grad_b = module.bias.grad.detach().float()
                        dot += (ihvp_params[(name, 'bias')] * grad_b).sum().item()

                scores[b] = dot

            ekfac_scores[test_idx, train_idx:train_idx+bs] = scores.cpu().numpy()
            train_idx += bs

        if (test_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time if 'start_time' in dir() else 0
            print(f"  EK-FAC attribution: test point {test_idx+1}/{n_test}")
            report_progress(TASK_ID, "manual_ekfac_attr",
                           f"Test point {test_idx+1}/{n_test}",
                           60 + 25 * (test_idx + 1) / n_test)

        # Clear GPU cache periodically
        if (test_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Save results
    np.save(ATTR_DIR / "ekfac_scores_fullmodel.npy", ekfac_scores)
    ekfac_rankings = np.argsort(-ekfac_scores, axis=1)[:, :100]
    np.save(ATTR_DIR / "ekfac_rankings_fullmodel_top100.npy", ekfac_rankings)

    info = {
        "success": True,
        "shape": list(ekfac_scores.shape),
        "method": "manual_ekfac_fullmodel",
        "damping": damping,
        "n_layers": len([n for n in factor_A if factor_A[n] is not None]),
    }
    return ekfac_scores, info


def compute_test_features(model, test_subset, test_indices, testset, device):
    """Compute gradient norms, confidence, entropy for each test point."""
    import torch.nn.functional as F

    model.eval()
    features = []

    test_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=1, shuffle=False, num_workers=2)

    for idx, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)

        # Forward pass
        model.zero_grad()
        out = model(x)
        probs = F.softmax(out, dim=1)
        loss = F.cross_entropy(out, y)
        loss.backward()

        # Gradient norm
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = total_grad_norm ** 0.5

        # Confidence (max prob)
        confidence = probs.max().item()

        # Entropy
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

        # Predicted class
        pred = out.argmax(dim=1).item()

        features.append({
            "test_idx": test_indices[idx],
            "true_label": y.item(),
            "pred_label": pred,
            "correct": pred == y.item(),
            "confidence": confidence,
            "entropy": entropy,
            "grad_norm": grad_norm,
            "log_grad_norm": float(np.log(grad_norm + 1e-10)),
        })

    # Save features
    features_file = ATTR_DIR / "test_features_500.json"
    features_file.write_text(json.dumps(features, indent=2))
    print(f"Saved test features for {len(features)} points")

    return features


if __name__ == "__main__":
    main()
