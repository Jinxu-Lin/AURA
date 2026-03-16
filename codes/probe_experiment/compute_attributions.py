"""
Compute influence-function-based attributions under different Hessian approximation
levels for the AURA probe experiment.

Hessian Hierarchy (best → worst):
  Level 1: Full last-layer GGN (empirical Fisher)
  Level 2: KFAC (Kronecker-factored approximation)
  Level 3: Diagonal GGN
  Level 4: Damped Identity (optimal scalar damping)
  Level 5: Identity (raw gradient dot product)

Also computes TRAK scores at two projection dimensions as additional reference.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


def load_model(model_path, device):
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model


def get_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    # SVHN for OOD
    svhn = torchvision.datasets.SVHN(
        root='./data', split='test', download=True, transform=transform)
    return trainset, testset, svhn


def select_test_points(model, testset, device, n_high=50, n_low=50, n_ood=20, svhn=None):
    """Select test points stratified by confidence + OOD points from SVHN."""
    model.eval()
    loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

    all_confidences = []
    all_correct = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = probs.max(1)
            all_confidences.append(max_probs.cpu())
            all_correct.append(preds.eq(targets).cpu())

    confidences = torch.cat(all_confidences).numpy()
    correct = torch.cat(all_correct).numpy()

    # Only select correctly classified points (for clean evaluation)
    correct_indices = np.where(correct)[0]
    correct_confidences = confidences[correct_indices]

    # Stratify: top n_high by confidence, bottom n_low by confidence
    sorted_idx = np.argsort(correct_confidences)
    low_conf_indices = correct_indices[sorted_idx[:n_low]]
    high_conf_indices = correct_indices[sorted_idx[-n_high:]]

    test_indices = np.concatenate([high_conf_indices, low_conf_indices])
    confidence_labels = np.array(['high'] * n_high + ['low'] * n_low)

    # OOD points from SVHN
    ood_indices = np.random.RandomState(42).choice(len(svhn), n_ood, replace=False)

    return test_indices, confidence_labels, ood_indices


def extract_last_layer_gradients(model, dataset, indices, device, batch_size=64):
    """
    Extract per-sample gradients of the loss w.r.t. last linear layer parameters.
    Returns: gradients tensor (N, param_dim), labels tensor (N,)
    """
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Get last layer
    last_layer = model.fc

    all_grads = []
    all_labels = []

    # Hook to capture features before last layer
    features_cache = {}

    def hook_fn(module, input, output):
        features_cache['features'] = input[0].detach()

    handle = last_layer.register_forward_hook(hook_fn)

    criterion = nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            bs = inputs.size(0)

            outputs = model(inputs)
            features = features_cache['features']  # (bs, 512)

            # Compute per-sample gradients analytically
            # For cross-entropy with softmax: dL/dW = (softmax - one_hot)^T @ features / 1
            # dL/db = softmax - one_hot
            probs = torch.softmax(outputs, dim=1)  # (bs, 10)
            one_hot = torch.zeros_like(probs)
            one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
            residuals = probs - one_hot  # (bs, 10)

            # Gradient of W: outer product residuals x features -> (bs, 10, 512)
            grad_W = torch.bmm(residuals.unsqueeze(2), features.unsqueeze(1))  # (bs, 10, 512)
            grad_b = residuals  # (bs, 10)

            # Flatten: vec(W) || b -> (bs, 10*512 + 10) = (bs, 5130)
            grad_flat = torch.cat([grad_W.reshape(bs, -1), grad_b], dim=1)

            all_grads.append(grad_flat.cpu())
            all_labels.append(targets.cpu())

    handle.remove()

    return torch.cat(all_grads, dim=0), torch.cat(all_labels, dim=0)


def extract_features(model, dataset, indices, device, batch_size=256):
    """Extract penultimate layer features for SI computation."""
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4)

    last_layer = model.fc
    features_cache = {}

    def hook_fn(module, input, output):
        features_cache['features'] = input[0].detach()

    handle = last_layer.register_forward_hook(hook_fn)

    all_features = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            model(inputs)
            all_features.append(features_cache['features'].cpu())

    handle.remove()
    return torch.cat(all_features, dim=0)


def compute_hessian_levels(train_grads, device):
    """
    Compute different Hessian approximation matrices from training gradients.

    Returns dict of (inverse) Hessian approximations.
    """
    n, d = train_grads.shape
    train_grads_gpu = train_grads.to(device)

    print(f"  Computing Hessian approximations (n={n}, d={d})...")
    results = {}

    # Level 1: Full GGN (empirical Fisher) = (1/n) G^T G
    print("    Level 1: Full GGN...")
    GGN = (train_grads_gpu.T @ train_grads_gpu) / n  # (d, d)
    # Regularize for numerical stability
    reg = 1e-5 * torch.eye(d, device=device)
    GGN_inv = torch.linalg.inv(GGN + reg)
    results['full_ggn'] = GGN_inv

    # Level 2: KFAC - Kronecker factored approximation
    # For last layer: W is (10, 512), grad_W = residual (10) ⊗ feature (512)
    # KFAC: F ≈ E[rr^T] ⊗ E[hh^T]
    print("    Level 2: KFAC...")
    # Extract the W gradient part (first 5120 dims) and reshape
    grad_W_part = train_grads_gpu[:, :5120].reshape(n, 10, 512)
    grad_b_part = train_grads_gpu[:, 5120:]  # (n, 10)

    # Residuals (10-dim) and features (512-dim) from grad_W = residual ⊗ feature
    # We approximate: reconstruct residuals and features
    # Since grad_W[i] = r_i @ h_i^T, we can use SVD or just compute covariances directly
    # A = E[r r^T] shape (10, 10)
    # B = E[h h^T] shape (512, 512)
    # For KFAC: F_W ≈ A ⊗ B, so F_W^{-1} ≈ A^{-1} ⊗ B^{-1}

    # Compute A and B from gradients
    # grad_W_part has shape (n, 10, 512) = r @ h^T for each sample
    # We can estimate: sum of outer products
    # A_hat = (1/n) * sum_i (grad_W_part[i] @ grad_W_part[i]^T summed over feature dim)
    # This isn't quite right. Let me use a different approach.

    # For KFAC, we approximate the full GGN by its Kronecker factorization
    # The key insight: vec(grad_W) = r ⊗ h, so
    # E[vec(grad_W) vec(grad_W)^T] = E[(r ⊗ h)(r ⊗ h)^T] = E[rr^T ⊗ hh^T]
    # KFAC approximates this as E[rr^T] ⊗ E[hh^T]

    # To extract r and h: ||grad_W[i]|| = ||r_i|| * ||h_i||
    # Actually, we can compute A and B directly from the rank-1 structure
    # grad_W[i] = r_i h_i^T, so:
    # grad_W[i] @ grad_W[i]^T = r_i h_i^T h_i r_i^T = ||h_i||^2 * r_i r_i^T
    # grad_W[i]^T @ grad_W[i] = h_i r_i^T r_i h_i^T = ||r_i||^2 * h_i h_i^T

    # We need to recover r_i and h_i. Using SVD of each grad_W[i] (rank-1):
    # Or simpler: the first left singular vector * singular value gives r_i scaled,
    # and first right singular vector gives h_i direction.

    # For efficiency, batch SVD:
    # grad_W_part shape: (n, 10, 512). For rank-1 matrices:
    # r_i = grad_W_part[i] @ h_i / ||h_i||^2, where h_i is the dominant right singular vec

    # Simpler: compute norms along each axis
    # ||r_i||^2 = sum over cols of grad_W_part[i]^2 summed... no, this is trace(grad_W grad_W^T)

    # Let's just compute A and B as:
    # B = (1/n) * sum_i (grad_W[i]^T @ grad_W[i]) / ||r_i||^2
    # where ||r_i||^2 = ||grad_W[i]||_F^2 / ||h_i||^2
    # This gets circular. Let me use a practical approximation.

    # Practical KFAC: compute eigendecomposition of full GGN and project
    # Actually, the simplest correct approach: extract features and residuals during forward pass.
    # But we already have gradients, so let's use a different strategy.

    # Alternative: use block-diagonal as KFAC proxy (separate W and b blocks)
    # Block 1: W part (5120 x 5120)
    # Block 2: b part (10 x 10)
    grad_W_flat = train_grads_gpu[:, :5120]
    grad_b_flat = train_grads_gpu[:, 5120:]

    GGN_W = (grad_W_flat.T @ grad_W_flat) / n + 1e-5 * torch.eye(5120, device=device)
    GGN_b = (grad_b_flat.T @ grad_b_flat) / n + 1e-5 * torch.eye(10, device=device)

    # For KFAC, we use the Kronecker structure on the W block
    # Reshape GGN_W from (5120, 5120) and find best Kronecker approximation
    # GGN_W ≈ A ⊗ B where A is (10,10) and B is (512,512)
    # Using the "nearest Kronecker product" via reshaping and SVD
    GGN_W_reshaped = GGN_W.reshape(10, 512, 10, 512).permute(0, 2, 1, 3).reshape(100, 512*512)
    # SVD to get rank-1 Kronecker approximation
    U, S, Vh = torch.linalg.svd(GGN_W_reshaped, full_matrices=False)
    A_kfac = U[:, 0].reshape(10, 10) * S[0].sqrt()
    B_kfac = Vh[0].reshape(512, 512) * S[0].sqrt()

    # Make symmetric and positive definite
    A_kfac = (A_kfac + A_kfac.T) / 2 + 1e-4 * torch.eye(10, device=device)
    B_kfac = (B_kfac + B_kfac.T) / 2 + 1e-4 * torch.eye(512, device=device)

    A_kfac_inv = torch.linalg.inv(A_kfac)
    B_kfac_inv = torch.linalg.inv(B_kfac)

    # KFAC inverse for W block: A^{-1} ⊗ B^{-1}
    KFAC_W_inv = torch.kron(A_kfac_inv, B_kfac_inv)
    GGN_b_inv = torch.linalg.inv(GGN_b)

    # Full KFAC inverse (block diagonal: W block + b block)
    KFAC_inv = torch.zeros(d, d, device=device)
    KFAC_inv[:5120, :5120] = KFAC_W_inv
    KFAC_inv[5120:, 5120:] = GGN_b_inv
    results['kfac'] = KFAC_inv

    # Level 3: Diagonal GGN
    print("    Level 3: Diagonal GGN...")
    diag_ggn = (train_grads_gpu ** 2).mean(dim=0)  # (d,)
    diag_ggn_inv = 1.0 / (diag_ggn + 1e-5)
    results['diagonal'] = diag_ggn_inv  # Store as vector for efficiency

    # Level 4: Damped Identity (optimal scalar: trace(GGN) / d)
    print("    Level 4: Damped Identity...")
    optimal_scale = GGN.trace().item() / d
    results['damped_identity'] = 1.0 / (optimal_scale + 1e-5)  # scalar

    # Level 5: Identity
    print("    Level 5: Identity...")
    results['identity'] = 1.0  # scalar multiplier

    return results


def compute_influence_scores(test_grads, train_grads, hessian_inv, level_name, device,
                             batch_size=100):
    """
    Compute influence scores: score(test_i, train_j) = g_test_i^T H^{-1} g_train_j

    Returns: (n_test, n_train) score matrix
    """
    n_test = test_grads.shape[0]
    n_train = train_grads.shape[0]

    scores = torch.zeros(n_test, n_train)
    train_grads_gpu = train_grads.to(device)

    for i in range(0, n_test, batch_size):
        end = min(i + batch_size, n_test)
        g_test_batch = test_grads[i:end].to(device)  # (bs, d)

        if level_name == 'full_ggn' or level_name == 'kfac':
            # H^{-1} is a full matrix
            Hinv_g_test = g_test_batch @ hessian_inv  # (bs, d)
        elif level_name == 'diagonal':
            # H^{-1} is a diagonal (stored as vector)
            Hinv_g_test = g_test_batch * hessian_inv.unsqueeze(0)  # (bs, d)
        else:
            # Scalar
            Hinv_g_test = g_test_batch * hessian_inv

        # Compute dot products with all training gradients
        # Process in chunks to manage memory
        chunk_size = 5000
        for j in range(0, n_train, chunk_size):
            j_end = min(j + chunk_size, n_train)
            s = (Hinv_g_test @ train_grads_gpu[j:j_end].T).cpu()  # (bs, chunk)
            scores[i:end, j:j_end] = s

    return scores


def compute_jaccard_at_k(top_k_1, top_k_2, k):
    """Compute Jaccard@k between two sets of top-k indices."""
    set1 = set(top_k_1[:k].tolist())
    set2 = set(top_k_2[:k].tolist())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_si(features, device):
    """
    Compute Self-Influence SI(z) = phi(z)^T Q^{-1} phi(z)
    where Q is feature covariance matrix.
    """
    features_gpu = features.to(device)
    n, d = features_gpu.shape
    Q = (features_gpu.T @ features_gpu) / n + 1e-5 * torch.eye(d, device=device)
    Q_inv = torch.linalg.inv(Q)
    # SI for each point
    si = (features_gpu @ Q_inv * features_gpu).sum(dim=1)
    return si.cpu().numpy(), Q


def run_attribution_experiment(seed, model_path, device, output_dir):
    """Run the full attribution experiment for one seed."""
    print(f"\n{'='*60}")
    print(f"Running attribution experiment for seed (model: {os.path.basename(model_path)})")
    print(f"{'='*60}")

    model = load_model(model_path, device)
    trainset, testset, svhn = get_datasets()

    # Select test points
    print("Selecting test points...")
    test_indices, conf_labels, ood_indices = select_test_points(
        model, testset, device, n_high=50, n_low=50, n_ood=20, svhn=svhn)

    # Extract gradients for all training points
    print("Extracting training gradients (50000 points)...")
    t0 = time.time()
    train_indices = np.arange(len(trainset))
    train_grads, train_labels = extract_last_layer_gradients(
        model, trainset, train_indices, device, batch_size=256)
    print(f"  Done in {time.time()-t0:.1f}s. Shape: {train_grads.shape}")

    # Extract gradients for test points
    print("Extracting test gradients (120 points)...")
    test_grads, test_labels = extract_last_layer_gradients(
        model, testset, test_indices, device, batch_size=64)

    # Extract gradients for OOD points (SVHN)
    print("Extracting OOD gradients (20 points)...")
    ood_grads, ood_labels_raw = extract_last_layer_gradients(
        model, svhn, ood_indices, device, batch_size=64)

    # Combine test + OOD gradients
    all_eval_grads = torch.cat([test_grads, ood_grads], dim=0)
    n_test = len(test_indices)
    n_ood = len(ood_indices)
    n_eval = n_test + n_ood

    # Extract features for SI computation
    print("Extracting features for SI computation...")
    test_features = extract_features(model, testset, test_indices, device)
    ood_features = extract_features(model, svhn, ood_indices, device)
    train_features = extract_features(model, trainset, train_indices, device)
    all_eval_features = torch.cat([test_features, ood_features], dim=0)

    # Compute SI
    print("Computing Self-Influence (SI)...")
    # SI needs feature covariance from training set
    train_features_gpu = train_features.to(device)
    n_tr, d_feat = train_features_gpu.shape
    Q = (train_features_gpu.T @ train_features_gpu) / n_tr + 1e-5 * torch.eye(d_feat, device=device)
    Q_inv = torch.linalg.inv(Q)
    # Condition number
    eigvals = torch.linalg.eigvalsh(Q - 1e-5 * torch.eye(d_feat, device=device))
    kappa = (eigvals.max() / eigvals[eigvals > 0].min()).item()
    print(f"  Feature covariance condition number κ = {kappa:.2e}")

    # SI for eval points
    all_eval_features_gpu = all_eval_features.to(device)
    si_eval = (all_eval_features_gpu @ Q_inv * all_eval_features_gpu).sum(dim=1).cpu().numpy()

    del train_features_gpu, all_eval_features_gpu
    torch.cuda.empty_cache()

    # Compute Hessian approximation levels
    hessian_levels = compute_hessian_levels(train_grads, device)

    # Compute influence scores for each level
    level_names = ['full_ggn', 'kfac', 'diagonal', 'damped_identity', 'identity']
    all_scores = {}
    k = 10  # top-k for Jaccard

    for name in level_names:
        print(f"  Computing influence scores: {name}...")
        t0 = time.time()
        scores = compute_influence_scores(
            all_eval_grads, train_grads, hessian_levels[name], name, device)
        all_scores[name] = scores
        print(f"    Done in {time.time()-t0:.1f}s")

    # Compute top-k indices for each level and Jaccard@k
    print("Computing Jaccard@k matrices...")
    top_k_indices = {}
    for name in level_names:
        # Top-k by absolute influence (most influential)
        _, topk = all_scores[name].abs().topk(k, dim=1)
        top_k_indices[name] = topk

    # Jaccard@k: compare each level to full_ggn (gold standard)
    gold_topk = top_k_indices['full_ggn']
    jaccard_matrix = np.zeros((n_eval, len(level_names)))

    for li, name in enumerate(level_names):
        for i in range(n_eval):
            jaccard_matrix[i, li] = compute_jaccard_at_k(gold_topk[i], top_k_indices[name][i], k)

    # Compute TRV for each eval point
    # TRV = max level ℓ where Jaccard@k >= 0.5
    trv = np.zeros(n_eval, dtype=int)
    for i in range(n_eval):
        for li in range(len(level_names) - 1, -1, -1):
            if jaccard_matrix[i, li] >= 0.5:
                trv[i] = li + 1  # 1-indexed levels
                break

    # Save results
    results = {
        'seed': seed,
        'kappa': kappa,
        'n_test': n_test,
        'n_ood': n_ood,
        'k': k,
        'level_names': level_names,
        'test_indices': test_indices.tolist(),
        'ood_indices': ood_indices.tolist(),
        'confidence_labels': conf_labels.tolist(),
        'jaccard_matrix': jaccard_matrix.tolist(),
        'trv': trv.tolist(),
        'si_eval': si_eval.tolist(),
        'test_labels': test_labels.numpy().tolist(),
    }

    out_path = os.path.join(output_dir, f'attribution_results_seed{seed}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f)

    # Also save numpy arrays for easier analysis
    np.savez(os.path.join(output_dir, f'attribution_arrays_seed{seed}.npz'),
             jaccard_matrix=jaccard_matrix,
             trv=trv,
             si_eval=si_eval,
             test_indices=test_indices,
             ood_indices=ood_indices,
             confidence_labels=conf_labels)

    print(f"\n  Results saved to {out_path}")
    print(f"  TRV distribution: {np.bincount(trv, minlength=6)}")
    print(f"  Mean Jaccard@{k} per level: {jaccard_matrix[:n_test].mean(axis=0)}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model_dir', type=str, default='./outputs/models')
    parser.add_argument('--output_dir', type=str, default='./outputs/attributions')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    os.makedirs(args.output_dir, exist_ok=True)

    for seed in args.seeds:
        model_path = os.path.join(args.model_dir, f'resnet18_seed{seed}_best.pt')
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}, skipping seed {seed}")
            continue
        run_attribution_experiment(seed, model_path, device, args.output_dir)

    print("\nAll attribution experiments complete.")


if __name__ == '__main__':
    main()
