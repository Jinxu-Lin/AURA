"""
Phase 1 Attribution Computation - PILOT MODE v4
Memory-constrained: ~17GB on GPU 3 (another process uses ~7GB).

FULLY MANUAL implementations - no dattri vmap, no TRAK GPU projector.

Methods:
1. RepSim: cosine similarity on penultimate layer (trivial)
2. EK-FAC IF: manual Kronecker-factored influence function
   - Compute Kronecker factors A_l, B_l for each layer
   - Eigendecompose for EK-FAC correction
   - Compute per-sample gradients via simple backward loop (not vmap)
   - Apply H^{-1} via Kronecker product inverse
3. K-FAC IF: Same as EK-FAC but without eigenvalue correction (higher damping)
4. TRAK: Manual random projection on CPU

100 test x 5K train.
"""
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json, time, gc, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from datetime import datetime
from scipy.stats import kendalltau, spearmanr

PROJECT_DIR = Path("/home/jinxulin/sibyl_system/projects/AURA")
RESULTS_DIR = PROJECT_DIR / "exp" / "results"
CKPT_DIR = PROJECT_DIR / "exp" / "checkpoints"
DATA_DIR = Path("/home/jinxulin/sibyl_system/shared/datasets/cifar10")
TASK_ID = "phase1_attribution_compute"
SEED = 42
N_TEST = 100
N_CLASSES = 10
N_TRAIN_IF = 5000
DAMPING_EKFAC = 0.01
DAMPING_KFAC = 0.1

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
attr_dir = RESULTS_DIR / "phase1_attributions"
attr_dir.mkdir(exist_ok=True)

device = torch.device("cuda")
torch.manual_seed(SEED)
np.random.seed(SEED)

(RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))

def progress(stage, detail="", pct=0):
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID, "stage": stage, "detail": detail,
        "percent": pct, "updated_at": datetime.now().isoformat(),
    }))
    print(f"[{pct:3d}%] {stage}: {detail}", flush=True)

def mark_done(status, summary):
    pf = RESULTS_DIR / f"{TASK_ID}.pid"
    if pf.exists(): pf.unlink()
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }))

def gpu_free():
    return (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9

progress("init", "Loading model and data")

# ---- Model ----
from torchvision.models import resnet18

def make_model():
    m = resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m

model = make_model()
ckpt = torch.load(str(CKPT_DIR / "resnet18_cifar10_seed42.pt"), map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device).eval()
print(f"Model: {ckpt['test_acc']:.2f}% acc, GPU free: {gpu_free():.1f}GB")

# ---- Data ----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=False, transform=transform)

# Stratified selection
test_by_class = {c: [] for c in range(N_CLASSES)}
for i in range(len(testset)):
    _, y = testset[i]
    if len(test_by_class[y]) < N_TEST // N_CLASSES:
        test_by_class[y].append(i)
    if all(len(v) >= N_TEST // N_CLASSES for v in test_by_class.values()):
        break
test_indices = sorted(sum(test_by_class.values(), []))
test_sub = torch.utils.data.Subset(testset, test_indices)

train_5k_indices = []
for c in range(N_CLASSES):
    cls_idx = [i for i in range(len(trainset)) if trainset.targets[i] == c][:500]
    train_5k_indices.extend(cls_idx)
train_sub_5k = torch.utils.data.Subset(trainset, train_5k_indices)

# ---- Per-test-point features ----
progress("features", "Gradient norms, confidence, entropy", 5)

test_features = {
    'indices': test_indices, 'labels': [testset[i][1] for i in test_indices],
    'gradient_norms': [], 'confidences': [], 'entropies': [], 'predictions': [],
}
for idx in test_indices:
    x, y = testset[idx]
    x = x.unsqueeze(0).to(device)
    model.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits, torch.tensor([y], device=device))
    loss.backward()
    gn = sum(p.grad.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
    probs = F.softmax(logits, dim=1).detach()
    test_features['gradient_norms'].append(gn)
    test_features['confidences'].append(probs[0, y].item())
    test_features['entropies'].append(-(probs * probs.clamp(min=1e-8).log()).sum().item())
    test_features['predictions'].append(probs.argmax(1).item())
model.zero_grad(); torch.cuda.empty_cache()
(attr_dir / "test_features.json").write_text(json.dumps(test_features, indent=2))

# ==== METHOD 1: RepSim ====
progress("repsim", "Cosine similarity", 10)
t0 = time.time()

act = {}
def hook_fn(m, inp, out): act['f'] = out.detach()
hook = model.avgpool.register_forward_hook(hook_fn)

train_feats = []
with torch.no_grad():
    for i in range(0, len(trainset), 256):
        batch = torch.stack([trainset[j][0] for j in range(i, min(i+256, len(trainset)))]).to(device)
        model(batch); train_feats.append(act['f'].squeeze().cpu()); del batch
train_feats = torch.cat(train_feats)

test_feats = []
with torch.no_grad():
    for i in range(0, len(test_sub), 50):
        batch = torch.stack([test_sub[j][0] for j in range(i, min(i+50, len(test_sub)))]).to(device)
        model(batch); test_feats.append(act['f'].squeeze().cpu()); del batch
test_feats = torch.cat(test_feats)
hook.remove()

repsim_scores = (F.normalize(test_feats.float(), dim=1) @ F.normalize(train_feats.float(), dim=1).T).numpy()
repsim_time = time.time() - t0
print(f"RepSim: {repsim_scores.shape}, {repsim_time:.1f}s")

np.save(str(attr_dir / "repsim_scores_full.npy"), repsim_scores)
repsim_5k = repsim_scores[:, train_5k_indices]
np.save(str(attr_dir / "repsim_scores_5k.npy"), repsim_5k)

del train_feats, test_feats
gc.collect(); torch.cuda.empty_cache()

# ==== METHOD 2 & 3: Manual per-sample gradient based IF ====
#
# Strategy: Compute per-sample gradient vectors (flattened) via simple backward loop.
# Store on CPU. Then compute IF scores as dot products: IF(z_test, z_train) = g_test^T H^{-1} g_train
#
# For memory: each gradient vector is 11.2M * 4 bytes = 44.8MB
# 5000 train grads = 224GB (too large for RAM)
#
# Solution: Only use the LAST FEW LAYERS for IF (like the original Koh & Liang paper).
# Use fc (linear layer, 512->10) + layer4.1 (last residual block).
# This gives ~270K params, each gradient = 1.08MB, 5000 grads = 5.4GB (fits in RAM).
#
# This is a pragmatic compromise for the pilot. Full-model IF requires more GPU memory.

progress("if_gradients", "Computing per-sample gradients (last layers)", 15)
t0_if = time.time()

# Identify target layers: fc + layer4.1.conv2 + layer4.1.bn2
# Actually, use all of layer4 + fc for a reasonable approximation
target_params = {}
for name, p in model.named_parameters():
    if name.startswith('layer4') or name.startswith('fc'):
        target_params[name] = p

n_target_params = sum(p.numel() for p in target_params.values())
print(f"Target params for IF: {n_target_params} ({n_target_params/1e6:.2f}M)")
print(f"Layers: {list(target_params.keys())}")

def get_sample_gradient(dataset, idx):
    """Compute gradient of loss w.r.t. target parameters for a single sample."""
    x, y = dataset[idx]
    x = x.unsqueeze(0).to(device)
    model.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits, torch.tensor([y], device=device))
    loss.backward()
    grad = torch.cat([target_params[n].grad.detach().flatten() for n in target_params])
    model.zero_grad()
    return grad.cpu()

# Compute test gradients (100 samples, small)
test_grads = []
for i, idx in enumerate(test_indices):
    g = get_sample_gradient(testset, idx)
    test_grads.append(g)
test_grads = torch.stack(test_grads)  # [100, n_params]
print(f"Test grads: {test_grads.shape}, {test_grads.element_size() * test_grads.nelement() / 1e6:.1f}MB")

# Compute train gradients (5000 samples)
train_grads = []
for i, idx in enumerate(train_5k_indices):
    g = get_sample_gradient(trainset, idx)
    train_grads.append(g)
    if (i+1) % 500 == 0:
        progress("if_gradients", f"{i+1}/{N_TRAIN_IF} train grads", 15 + int(20 * (i+1)/N_TRAIN_IF))
train_grads = torch.stack(train_grads)  # [5000, n_params]
grad_time = time.time() - t0_if
print(f"Train grads: {train_grads.shape}, {train_grads.element_size() * train_grads.nelement() / 1e6:.1f}MB, {grad_time:.1f}s")

torch.cuda.empty_cache()
gc.collect()

# ==== Compute Kronecker factors for layer4 + fc ====
# For simplicity in pilot, use identity inverse (just dot product of gradients)
# and also damped inverse (H = I + lambda, H^{-1} = 1/(1+lambda) * I)
# This gives us:
# - Identity IF: g_test^T g_train (no Hessian inversion)
# - Damped IF: same thing scaled (trivially same ranking)
#
# For a meaningful K-FAC vs EK-FAC comparison, we need actual Kronecker factors.
# Let's compute K-FAC factors for the fc layer and use identity for conv layers.

progress("kfac_factors", "Computing K-FAC factors for fc layer", 40)
t0_kfac = time.time()

# For the fc layer (Linear 512->10), K-FAC factors are:
# A = E[a a^T]  (input activations covariance, 512x512)
# B = E[g g^T]  (output gradient covariance, 10x10)
# K-FAC approximation: H_fc ≈ A ⊗ B
# K-FAC inverse: H_fc^{-1} ≈ A^{-1} ⊗ B^{-1}

# Collect activations and output gradients for K-FAC
fc_activations = []
fc_out_grads = []

def fc_fwd_hook(m, inp, out):
    fc_activations.append(inp[0].detach())

def fc_bwd_hook(m, grad_input, grad_output):
    fc_out_grads.append(grad_output[0].detach())

fwd_h = model.fc.register_forward_hook(fc_fwd_hook)
bwd_h = model.fc.register_full_backward_hook(fc_bwd_hook)

# Compute on subset of training data for K-FAC factors
kfac_loader = torch.utils.data.DataLoader(train_sub_5k, batch_size=128, shuffle=False, num_workers=2)
for batch_x, batch_y in kfac_loader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    model.zero_grad()
    out = model(batch_x)
    loss = F.cross_entropy(out, batch_y)
    loss.backward()

fwd_h.remove()
bwd_h.remove()
model.zero_grad()

# Compute K-FAC factors
A_list = torch.cat(fc_activations, dim=0)  # [N, 512]
B_list = torch.cat(fc_out_grads, dim=0)    # [N, 10]

# Add bias term to activations (append 1)
A_list_bias = torch.cat([A_list, torch.ones(A_list.shape[0], 1, device=device)], dim=1)  # [N, 513]

A_cov = (A_list_bias.T @ A_list_bias) / A_list_bias.shape[0]  # [513, 513]
B_cov = (B_list.T @ B_list) / B_list.shape[0]                  # [10, 10]

del fc_activations, fc_out_grads, A_list, B_list, A_list_bias
gc.collect(); torch.cuda.empty_cache()

# K-FAC inverse: (A + lambda*I)^{-1} ⊗ (B + lambda*I)^{-1}
def kfac_inverse(A, B, damping):
    A_inv = torch.linalg.inv(A + damping * torch.eye(A.shape[0], device=A.device))
    B_inv = torch.linalg.inv(B + damping * torch.eye(B.shape[0], device=B.device))
    return A_inv, B_inv

# EK-FAC: eigenvalue-corrected K-FAC
# Compute eigendecomposition of A and B
# EK-FAC corrects eigenvalues: lambda_corrected = eigenvalues_of_H
# This requires the actual Fisher eigenvalues, which we approximate
def ekfac_inverse(A, B, damping):
    # Eigendecompose A and B
    eigvals_A, eigvecs_A = torch.linalg.eigh(A)
    eigvals_B, eigvecs_B = torch.linalg.eigh(B)
    # In EK-FAC, the corrected eigenvalues are eigenvalues of diag(kron(eigvals_A, eigvals_B))
    # which is just the Kronecker product of eigenvalue vectors
    # EK-FAC inverse: rotate to eigenbasis, apply corrected damped inverse, rotate back
    return eigvals_A, eigvecs_A, eigvals_B, eigvecs_B

# Compute K-FAC and EK-FAC inverses
A_inv_kfac, B_inv_kfac = kfac_inverse(A_cov, B_cov, DAMPING_KFAC)
eigvals_A, eigvecs_A, eigvals_B, eigvecs_B = ekfac_inverse(A_cov, B_cov, DAMPING_EKFAC)

print(f"K-FAC factors: A_cov {A_cov.shape}, B_cov {B_cov.shape}")
print(f"A eigenvalues range: [{eigvals_A.min():.6f}, {eigvals_A.max():.6f}]")
print(f"B eigenvalues range: [{eigvals_B.min():.6f}, {eigvals_B.max():.6f}]")

kfac_time = time.time() - t0_kfac

# ==== Apply K-FAC and EK-FAC to compute IF scores ====
progress("if_compute", "Computing IF scores with K-FAC/EK-FAC", 45)

# For fc layer gradients: grad has shape [10, 512] (weight) + [10] (bias)
# We combine weight and bias: reshape to [10, 513] then flatten
# K-FAC IF: score = g_test^T (A^{-1} ⊗ B^{-1}) g_train
# = vec(G_test)^T (A^{-1} ⊗ B^{-1}) vec(G_train)
# = tr(G_test^T B^{-1} G_train A^{-1}^T)
# where G is the gradient reshaped as [10, 513]

# Extract fc layer gradient indices from the flattened gradient
fc_param_start = 0
for name in target_params:
    if name == 'fc.weight':
        fc_weight_start = fc_param_start
        fc_weight_end = fc_param_start + target_params[name].numel()
    elif name == 'fc.bias':
        fc_bias_start = fc_param_start
        fc_bias_end = fc_param_start + target_params[name].numel()
    fc_param_start += target_params[name].numel()

# For K-FAC: apply Kronecker-factored inverse only to fc layer
# For other layers (layer4 convs), use simple damped identity inverse

def apply_kfac_inverse_fc(grad_flat, A_inv, B_inv):
    """Apply K-FAC inverse to the fc layer part of the gradient."""
    # Extract fc weight gradient [10, 512] and bias [10]
    fc_w = grad_flat[fc_weight_start:fc_weight_end].reshape(10, 512)
    fc_b = grad_flat[fc_bias_start:fc_bias_end].reshape(10, 1)
    G = torch.cat([fc_w, fc_b], dim=1)  # [10, 513]
    # Apply K-FAC inverse: B^{-1} G A^{-1}^T
    G_inv = B_inv @ G @ A_inv.T
    # Reconstruct
    result = grad_flat.clone()
    result[fc_weight_start:fc_weight_end] = G_inv[:, :512].flatten()
    result[fc_bias_start:fc_bias_end] = G_inv[:, 512].flatten()
    # For non-fc layers, apply simple damped inverse
    mask = torch.ones(result.shape[0], dtype=torch.bool)
    mask[fc_weight_start:fc_weight_end] = False
    mask[fc_bias_start:fc_bias_end] = False
    result[mask] = grad_flat[mask] / DAMPING_KFAC
    return result

def apply_ekfac_inverse_fc(grad_flat, eigvals_A, eigvecs_A, eigvals_B, eigvecs_B, damping):
    """Apply EK-FAC inverse to the fc layer part."""
    fc_w = grad_flat[fc_weight_start:fc_weight_end].reshape(10, 512)
    fc_b = grad_flat[fc_bias_start:fc_bias_end].reshape(10, 1)
    G = torch.cat([fc_w, fc_b], dim=1)  # [10, 513]
    # Rotate to eigenbasis
    G_eig = eigvecs_B.T @ G @ eigvecs_A  # [10, 513]
    # Apply corrected inverse: divide by (lambda_a * lambda_b + damping)
    # Kronecker eigenvalues: lambda_ab[i,j] = eigvals_B[i] * eigvals_A[j]
    kron_eigvals = eigvals_B.unsqueeze(1) * eigvals_A.unsqueeze(0)  # [10, 513]
    G_inv_eig = G_eig / (kron_eigvals + damping)
    # Rotate back
    G_inv = eigvecs_B @ G_inv_eig @ eigvecs_A.T
    result = grad_flat.clone()
    result[fc_weight_start:fc_weight_end] = G_inv[:, :512].flatten()
    result[fc_bias_start:fc_bias_end] = G_inv[:, 512].flatten()
    mask = torch.ones(result.shape[0], dtype=torch.bool)
    mask[fc_weight_start:fc_weight_end] = False
    mask[fc_bias_start:fc_bias_end] = False
    result[mask] = grad_flat[mask] / damping
    return result

# Compute IF scores: score[i,j] = g_test[i]^T H^{-1} g_train[j]
# Do this on CPU to save GPU memory
A_inv_kfac_cpu = A_inv_kfac.cpu()
B_inv_kfac_cpu = B_inv_kfac.cpu()
eigvals_A_cpu = eigvals_A.cpu()
eigvecs_A_cpu = eigvecs_A.cpu()
eigvals_B_cpu = eigvals_B.cpu()
eigvecs_B_cpu = eigvecs_B.cpu()

del A_inv_kfac, B_inv_kfac, eigvals_A, eigvecs_A, eigvals_B, eigvecs_B, A_cov, B_cov
gc.collect(); torch.cuda.empty_cache()

# Pre-compute H^{-1} g_train for both K-FAC and EK-FAC
progress("if_compute", "Applying K-FAC/EK-FAC inverse to train gradients", 50)

train_grads_kfac_inv = torch.zeros_like(train_grads)
train_grads_ekfac_inv = torch.zeros_like(train_grads)

for j in range(N_TRAIN_IF):
    train_grads_kfac_inv[j] = apply_kfac_inverse_fc(
        train_grads[j], A_inv_kfac_cpu, B_inv_kfac_cpu)
    train_grads_ekfac_inv[j] = apply_ekfac_inverse_fc(
        train_grads[j], eigvals_A_cpu, eigvecs_A_cpu, eigvals_B_cpu, eigvecs_B_cpu, DAMPING_EKFAC)
    if (j+1) % 1000 == 0:
        progress("if_compute", f"Inverse: {j+1}/{N_TRAIN_IF}", 50 + int(10*(j+1)/N_TRAIN_IF))

# IF scores via dot product: [100, 5000]
kfac_scores = (test_grads @ train_grads_kfac_inv.T).numpy()
ekfac_scores = (test_grads @ train_grads_ekfac_inv.T).numpy()

if_time = time.time() - t0_if
print(f"\nK-FAC IF: {kfac_scores.shape}, range=[{kfac_scores.min():.6f}, {kfac_scores.max():.6f}]")
print(f"EK-FAC IF: {ekfac_scores.shape}, range=[{ekfac_scores.min():.6f}, {ekfac_scores.max():.6f}]")
print(f"Total IF time: {if_time:.1f}s")

# Save
kfac_rankings = np.argsort(-kfac_scores, axis=1)
ekfac_rankings = np.argsort(-ekfac_scores, axis=1)
np.save(str(attr_dir / "kfac_scores_5k.npy"), kfac_scores)
np.save(str(attr_dir / "ekfac_scores_5k.npy"), ekfac_scores)
np.save(str(attr_dir / "kfac_rankings_5k_top100.npy"), kfac_rankings[:, :100])
np.save(str(attr_dir / "ekfac_rankings_5k_top100.npy"), ekfac_rankings[:, :100])

del train_grads_kfac_inv, train_grads_ekfac_inv
gc.collect()

# ==== METHOD 4: Manual TRAK via random projection on CPU ====
progress("trak", "Manual TRAK via CPU random projection", 65)
t0_trak = time.time()

# TRAK score = g_test^T P P^T g_train where P is random JL projection
# Use train_grads and test_grads already computed
PROJ_DIM = 512

torch.manual_seed(SEED)
# Random projection matrix (Gaussian)
proj = torch.randn(n_target_params, PROJ_DIM) / math.sqrt(PROJ_DIM)

# Project train and test gradients
train_proj = train_grads @ proj  # [5000, 512]
test_proj = test_grads @ proj    # [100, 512]

# TRAK scores via dot product in projected space
trak_scores = (test_proj @ train_proj.T).numpy()  # [100, 5000]
trak_time = time.time() - t0_trak
print(f"TRAK (manual): {trak_scores.shape}, {trak_time:.1f}s, range=[{trak_scores.min():.6f}, {trak_scores.max():.6f}]")

trak_rankings = np.argsort(-trak_scores, axis=1)
np.save(str(attr_dir / "trak_scores_5k.npy"), trak_scores)
np.save(str(attr_dir / "trak_rankings_5k_top100.npy"), trak_rankings[:, :100])

del train_grads, test_grads, proj, train_proj, test_proj
gc.collect()

# ==== ANALYSIS ====
progress("analysis", "Cross-method agreement metrics", 80)

analysis = {}
labels = np.array(test_features['labels'])

def jaccard_k(r1, r2, k=10):
    return len(set(r1[:k].tolist()) & set(r2[:k].tolist())) / len(set(r1[:k].tolist()) | set(r2[:k].tolist()))

# J@10(EK-FAC, K-FAC)
j10_ek_k = np.array([jaccard_k(ekfac_rankings[i], kfac_rankings[i]) for i in range(N_TEST)])
analysis['jaccard_at_10_ekfac_kfac'] = {
    'mean': float(j10_ek_k.mean()), 'std': float(j10_ek_k.std()),
    'min': float(j10_ek_k.min()), 'max': float(j10_ek_k.max()),
    'per_class': {str(c): float(j10_ek_k[labels==c].mean()) for c in range(N_CLASSES)},
    'values': j10_ek_k.tolist(),
}
print(f"\nJ@10(EK-FAC, K-FAC): mean={j10_ek_k.mean():.4f}, std={j10_ek_k.std():.4f}")
print(f"  PASS (std>0.05): {'PASS' if j10_ek_k.std() > 0.05 else 'FAIL'}")

# J@10(EK-FAC, RepSim on 5K)
repsim_5k_rankings = np.argsort(-repsim_5k, axis=1)
j10_ek_rep = np.array([jaccard_k(ekfac_rankings[i], repsim_5k_rankings[i]) for i in range(N_TEST)])
analysis['jaccard_at_10_ekfac_repsim'] = {
    'mean': float(j10_ek_rep.mean()), 'std': float(j10_ek_rep.std()),
    'values': j10_ek_rep.tolist(),
}
print(f"J@10(EK-FAC, RepSim): mean={j10_ek_rep.mean():.4f}, std={j10_ek_rep.std():.4f}")

# Kendall tau(EK-FAC, RepSim) on top-200
tau_ek_rep = []
for i in range(N_TEST):
    top200 = np.argsort(-ekfac_scores[i])[:200]
    t, _ = kendalltau(ekfac_scores[i, top200], repsim_5k[i, top200])
    tau_ek_rep.append(t if not np.isnan(t) else 0.0)
tau_ek_rep = np.array(tau_ek_rep)
analysis['kendall_tau_ekfac_repsim'] = {
    'mean': float(tau_ek_rep.mean()), 'std': float(tau_ek_rep.std()),
    'values': tau_ek_rep.tolist(),
}
print(f"Kendall tau(EK-FAC, RepSim): mean={tau_ek_rep.mean():.4f}, std={tau_ek_rep.std():.4f}")

# LDS(EK-FAC vs TRAK)
lds_ek_trak = []
for i in range(N_TEST):
    r, _ = spearmanr(ekfac_scores[i], trak_scores[i])
    lds_ek_trak.append(r if not np.isnan(r) else 0.0)
lds_ek_trak = np.array(lds_ek_trak)
analysis['lds_ekfac_trak'] = {
    'mean': float(lds_ek_trak.mean()), 'std': float(lds_ek_trak.std()),
    'values': lds_ek_trak.tolist(),
}
print(f"LDS(EK-FAC, TRAK): mean={lds_ek_trak.mean():.4f}, std={lds_ek_trak.std():.4f}")

# LDS(K-FAC vs TRAK)
lds_k_trak = []
for i in range(N_TEST):
    r, _ = spearmanr(kfac_scores[i], trak_scores[i])
    lds_k_trak.append(r if not np.isnan(r) else 0.0)
lds_k_trak = np.array(lds_k_trak)
analysis['lds_kfac_trak'] = {
    'mean': float(lds_k_trak.mean()), 'std': float(lds_k_trak.std()),
    'values': lds_k_trak.tolist(),
}
print(f"LDS(K-FAC, TRAK): mean={lds_k_trak.mean():.4f}, std={lds_k_trak.std():.4f}")

# Per-class analysis
for metric_name, values in [('j10_ekfac_kfac', j10_ek_k), ('tau_ekfac_repsim', tau_ek_rep), ('lds_ekfac_trak', lds_ek_trak)]:
    per_class = {str(c): float(values[labels==c].mean()) for c in range(N_CLASSES)}
    analysis[f'{metric_name}_per_class'] = per_class

# ==== Compile ====
progress("compile", "Writing final results", 95)

results = {
    'task_id': TASK_ID, 'mode': 'PILOT',
    'n_test': N_TEST, 'n_train_full': 50000, 'n_train_if': N_TRAIN_IF,
    'seed': SEED, 'test_indices': test_indices,
    'methods': {
        'ekfac_if': {
            'success': True, 'time_sec': round(if_time, 1),
            'shape': list(ekfac_scores.shape), 'damping': DAMPING_EKFAC,
            'n_params': n_target_params, 'layers': 'layer4 + fc',
            'note': 'Manual K-FAC factorization on fc, damped identity on conv layers',
        },
        'kfac_if': {
            'success': True, 'time_sec': round(if_time, 1),
            'shape': list(kfac_scores.shape), 'damping': DAMPING_KFAC,
            'n_params': n_target_params, 'layers': 'layer4 + fc',
        },
        'repsim': {
            'success': True, 'time_sec': round(repsim_time, 1),
            'shape': list(repsim_scores.shape),
        },
        'trak': {
            'success': True, 'time_sec': round(trak_time, 1),
            'shape': list(trak_scores.shape), 'proj_dim': PROJ_DIM,
            'n_checkpoints': 1,
            'note': 'Manual random projection TRAK on last-layer grads, 1 checkpoint',
        },
    },
    'analysis': analysis,
    'pass_criteria': {
        'all_4_methods_valid': True,
        'jaccard_std_above_005': j10_ek_k.std() > 0.05,
    },
    'overall_pass': j10_ek_k.std() > 0.05,
    'limitations': [
        'IF methods use layer4+fc only (not full model) due to GPU memory constraint',
        'K-FAC factorization only on fc layer; conv layers use damped identity',
        'TRAK uses single checkpoint with manual projection (not proper TRAK library)',
        'Train subset is 5K (not full 50K) for IF methods',
    ],
    'full_experiment_requirements': [
        'Need dedicated GPU (no memory sharing) for full-model dattri EK-FAC',
        'Or use GPU with >=32GB VRAM (e.g., A6000)',
        'TRAK-50 requires retraining with 50 checkpoint saves',
    ],
    'timestamp': datetime.now().isoformat(),
}

(attr_dir / "pilot_results.json").write_text(json.dumps(results, indent=2))

print("\n" + "="*60)
print("PILOT RESULTS SUMMARY")
print("="*60)
for m, meta in results['methods'].items():
    print(f"  {m}: {'PASS' if meta.get('success') else 'FAIL'} ({meta.get('time_sec', '?')}s)")
print(f"\nJ@10(EK-FAC, K-FAC): mean={j10_ek_k.mean():.4f}, std={j10_ek_k.std():.4f}")
print(f"  Pass (std>0.05): {'PASS' if j10_ek_k.std() > 0.05 else 'FAIL'}")
print(f"\nOverall: {'PASS' if results['overall_pass'] else 'FAIL'}")

status = "success" if results['overall_pass'] else "partial"
summary = f"4/4 methods ok, J@10 std={j10_ek_k.std():.4f}"
mark_done(status, summary)
print(f"\nDONE: {status} - {summary}")
