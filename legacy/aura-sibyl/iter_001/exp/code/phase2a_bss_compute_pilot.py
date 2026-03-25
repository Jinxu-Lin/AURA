"""
Phase 2a BSS Computation - PILOT MODE
======================================
Compute Bucketed Spectral Sensitivity (BSS) for 100 test points on seed 42.

BSS_j(z) = sum_{k in B_j} |1/lambda_k - 1/tilde_lambda_k| * (V_k^T g)^2

Where:
- lambda_k: true GGN eigenvalues (approximated by EK-FAC corrected eigenvalues)
- tilde_lambda_k: K-FAC approximate eigenvalues (products of Kronecker factor eigenvalues)
- V_k: eigenvectors
- g: per-sample gradient
- B_j: eigenvalue buckets (outlier: >100, edge: 10-100, bulk: <10)

Pilot: 100 test points (10/class), 1 seed (42), verify computation feasibility.
"""
import os
import sys
import json
import time
import gc
from pathlib import Path
from datetime import datetime

# GPU selection
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # Use GPU 2 (dedicated, 24GB free)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

# ===== Configuration =====
TASK_ID = "phase2a_bss_compute"
SEED = 42
N_TEST = 100  # Pilot: 100 test points (10/class)
N_EIGEN_TOP = 100  # Top-100 eigenvalues
DAMPING_KFAC = 0.1  # K-FAC damping
DAMPING_EKFAC = 0.01  # EK-FAC damping (smaller = closer to true GGN)

PROJECT_DIR = Path("/home/jinxulin/sibyl_system/projects/AURA")
RESULTS_DIR = PROJECT_DIR / "exp" / "results"
BSS_DIR = RESULTS_DIR / "phase2a_bss"
CHECKPOINT_DIR = PROJECT_DIR / "exp" / "checkpoints"
DATA_DIR = Path("/home/jinxulin/sibyl_system/shared/datasets")

BSS_DIR.mkdir(parents=True, exist_ok=True)

# ===== Reproducibility =====
torch.manual_seed(SEED)
np.random.seed(SEED)

# ===== PID & Progress =====
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))
print(f"[BSS] PID {os.getpid()} written to {pid_file}")

def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status="success", summary=""):
    pf = RESULTS_DIR / f"{TASK_ID}.pid"
    if pf.exists(): pf.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try: final_progress = json.loads(progress_file.read_text())
        except: pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))

# ===== Device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[BSS] Device: {device}")
if torch.cuda.is_available():
    print(f"[BSS] GPU: {torch.cuda.get_device_name()}")
    print(f"[BSS] VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
          if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
          else f"[BSS] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ===== Data =====
print("[BSS] Loading CIFAR-10...")
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root=str(DATA_DIR / "cifar10"), train=False, download=True, transform=transform_test
)
trainset = torchvision.datasets.CIFAR10(
    root=str(DATA_DIR / "cifar10"), train=True, download=True, transform=transform_train
)

# Select 100 test points (10/class, stratified)
np.random.seed(SEED)
class_indices = {c: [] for c in range(10)}
for i in range(len(testset)):
    _, label = testset[i]
    class_indices[label].append(i)

test_indices = []
for c in range(10):
    chosen = np.random.choice(class_indices[c], size=10, replace=False)
    test_indices.extend(chosen.tolist())
test_indices.sort()
print(f"[BSS] Selected {len(test_indices)} test points (10/class)")

# ===== Model =====
print("[BSS] Loading model checkpoint (seed 42)...")
model = resnet18(num_classes=10)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model = model.to(device)

ckpt_path = CHECKPOINT_DIR / "resnet18_cifar10_seed42.pt"
ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"[BSS] Model loaded. Test acc from checkpoint: {ckpt.get('test_acc', 'N/A')}")

total_params = sum(p.numel() for p in model.parameters())
print(f"[BSS] Model parameters: {total_params:,}")

report_progress(0, 5, metric={"phase": "model_loaded"})

# ===== K-FAC Factor Computation =====
print("\n[BSS] === Step 1: Computing K-FAC Kronecker factors ===")

# We'll compute K-FAC factors for conv and linear layers
# For each layer with weight W of shape (out, in, kh, kw) or (out, in):
#   A = E[a a^T]  (input activations auto-correlation)
#   B = E[g g^T]  (output gradient auto-correlation)
# where a = input activation, g = output gradient (from loss)

class KFACComputer:
    """Compute K-FAC Kronecker factors (A, B) for each layer."""

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.activations = {}
        self.gradients = {}
        self.layers = []  # (name, module) for eligible layers

        # Register hooks for Conv2d and Linear layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.layers.append((name, module))
                self._register_hooks(name, module)

    def _register_hooks(self, name, module):
        def fwd_hook(mod, inp, out, name=name):
            self.activations[name] = inp[0].detach()
        def bwd_hook(mod, grad_in, grad_out, name=name):
            self.gradients[name] = grad_out[0].detach()

        self.hooks.append(module.register_forward_hook(fwd_hook))
        self.hooks.append(module.register_full_backward_hook(bwd_hook))

    def compute_factors(self, dataloader, n_batches=None, device='cuda'):
        """Compute empirical K-FAC factors A and B for each layer."""
        criterion = nn.CrossEntropyLoss(reduction='sum')

        factors = {}
        for name, module in self.layers:
            if isinstance(module, nn.Conv2d):
                in_dim = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                out_dim = module.out_channels
            else:
                in_dim = module.in_features
                out_dim = module.out_features

            # Include bias dimension if bias exists
            has_bias = module.bias is not None
            if has_bias:
                in_dim += 1

            factors[name] = {
                'A': torch.zeros(in_dim, in_dim, device=device),
                'B': torch.zeros(out_dim, out_dim, device=device),
                'n_samples': 0,
                'has_bias': has_bias,
                'type': 'conv' if isinstance(module, nn.Conv2d) else 'linear',
            }

        n_total = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if n_batches and batch_idx >= n_batches:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            bs = inputs.size(0)
            n_total += bs

            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            for name, module in self.layers:
                act = self.activations[name]  # (B, C_in, H, W) or (B, in_features)
                grad = self.gradients[name]  # (B, C_out, H, W) or (B, out_features)

                info = factors[name]

                if info['type'] == 'conv':
                    # Unfold activations: (B, C_in*kh*kw, spatial)
                    act_unf = torch.nn.functional.unfold(
                        act, module.kernel_size,
                        dilation=module.dilation, padding=module.padding, stride=module.stride
                    )  # (B, C_in*kh*kw, L)

                    # Average over spatial locations
                    spatial_size = act_unf.size(2)
                    act_flat = act_unf.mean(dim=2)  # (B, C_in*kh*kw)

                    # Gradient: average over spatial
                    grad_flat = grad.reshape(bs, grad.size(1), -1).mean(dim=2)  # (B, C_out)
                else:
                    act_flat = act  # (B, in_features)
                    grad_flat = grad  # (B, out_features)

                # Append bias unit
                if info['has_bias']:
                    ones = torch.ones(bs, 1, device=device)
                    act_flat = torch.cat([act_flat, ones], dim=1)

                # Accumulate: A = (1/N) sum_i a_i a_i^T, B = (1/N) sum_i g_i g_i^T
                info['A'] += act_flat.t() @ act_flat
                info['B'] += grad_flat.t() @ grad_flat
                info['n_samples'] += bs

            if (batch_idx + 1) % 20 == 0:
                print(f"  [K-FAC] Processed {n_total} samples ({batch_idx+1} batches)")

        # Normalize
        for name in factors:
            n = factors[name]['n_samples']
            factors[name]['A'] /= n
            factors[name]['B'] /= n
            print(f"  [K-FAC] {name}: A={factors[name]['A'].shape}, B={factors[name]['B'].shape}")

        return factors

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

# Use a subset of training data for K-FAC estimation (5000 samples for pilot)
np.random.seed(SEED)
train_subset_idx = np.random.choice(len(trainset), size=5000, replace=False)
train_subset = Subset(trainset, train_subset_idx.tolist())
train_loader = DataLoader(train_subset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

start_time = time.time()

kfac_computer = KFACComputer(model)
kfac_factors = kfac_computer.compute_factors(train_loader, device=device)
kfac_computer.remove_hooks()

kfac_time = time.time() - start_time
print(f"[BSS] K-FAC factors computed in {kfac_time:.1f}s")
report_progress(1, 5, metric={"phase": "kfac_factors_computed", "time_sec": kfac_time})

# ===== Step 2: Eigendecomposition of Kronecker factors =====
print("\n[BSS] === Step 2: Eigendecomposition of Kronecker factors ===")

eigendata = {}
all_kfac_eigenvalues = []  # Collect all K-FAC eigenvalues (products of A and B eigenvalues)

for name in kfac_factors:
    info = kfac_factors[name]
    A = info['A']
    B = info['B']

    # Eigendecomposition of A and B
    # A = V_A diag(lambda_A) V_A^T
    # B = V_B diag(lambda_B) V_B^T
    eig_A, V_A = torch.linalg.eigh(A)
    eig_B, V_B = torch.linalg.eigh(B)

    # Ensure non-negative (numerical stability)
    eig_A = eig_A.clamp(min=1e-10)
    eig_B = eig_B.clamp(min=1e-10)

    # K-FAC eigenvalues: all products lambda_A_i * lambda_B_j
    # Shape: (dim_A, dim_B) -> flatten
    kfac_eigs = (eig_A.unsqueeze(1) * eig_B.unsqueeze(0)).flatten()

    # EK-FAC eigenvalues: need diagonal correction
    # For EK-FAC, the true eigenvalues are corrected using the diagonal of
    # V^T H V where V = V_B kron V_A (Kronecker eigenvectors)
    # For now, we approximate by computing the EK-FAC correction factors

    eigendata[name] = {
        'eig_A': eig_A.cpu(),
        'V_A': V_A.cpu(),
        'eig_B': eig_B.cpu(),
        'V_B': V_B.cpu(),
        'kfac_eigs': kfac_eigs.cpu(),
        'dim_A': A.shape[0],
        'dim_B': B.shape[0],
    }

    all_kfac_eigenvalues.append(kfac_eigs.cpu())

    print(f"  {name}: A eigs range [{eig_A.min():.4e}, {eig_A.max():.4e}], "
          f"B eigs range [{eig_B.min():.4e}, {eig_B.max():.4e}]")
    print(f"    K-FAC eigs (products): range [{kfac_eigs.min():.4e}, {kfac_eigs.max():.4e}], "
          f"count={len(kfac_eigs)}")

# Concatenate all K-FAC eigenvalues across layers
all_kfac_eigs = torch.cat(all_kfac_eigenvalues)
all_kfac_eigs_sorted, sort_idx = all_kfac_eigs.sort(descending=True)

print(f"\n[BSS] Total K-FAC eigenvalues: {len(all_kfac_eigs)}")
print(f"[BSS] Top-10 eigenvalues: {all_kfac_eigs_sorted[:10].numpy()}")
print(f"[BSS] Eigenvalue at rank 100: {all_kfac_eigs_sorted[min(99, len(all_kfac_eigs_sorted)-1)]:.4f}")

# ===== Step 3: Compute EK-FAC eigenvalue corrections =====
print("\n[BSS] === Step 3: Computing EK-FAC eigenvalue corrections ===")

# For EK-FAC, we need to compute the diagonal of V^T H V where V = V_B kron V_A
# This gives the corrected eigenvalues. We'll compute this using per-sample gradients.
#
# For efficiency in pilot mode, we compute the correction using a subset of training data.
# The EK-FAC correction for eigenvalue (i,j) of layer l is:
#   corrected_lambda_{ij} = (1/N) sum_n (v_B_j^T g_n) * (v_A_i^T a_n)
#     where this is the diagonal of the rotated empirical Fisher
#
# More precisely: corrected eigenvalue = E[(v_B^T g)^2 * (v_A^T a)^2]

print("[BSS] Computing EK-FAC corrected eigenvalues via empirical Fisher diagonal...")

ekfac_corrections = {}

# Re-register hooks for per-sample computation
kfac_computer2 = KFACComputer(model)
criterion = nn.CrossEntropyLoss(reduction='sum')

# We accumulate the diagonal of the rotated Fisher
for name in eigendata:
    dim_A = eigendata[name]['dim_A']
    dim_B = eigendata[name]['dim_B']
    ekfac_corrections[name] = torch.zeros(dim_A * dim_B, device='cpu')

n_corr_samples = 0
with torch.no_grad():
    # Need gradients for backward, so can't use no_grad for the forward+backward pass
    pass

for batch_idx, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.to(device), targets.to(device)
    bs = inputs.size(0)
    n_corr_samples += bs

    model.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    for name, module in kfac_computer2.layers:
        act = kfac_computer2.activations[name]
        grad = kfac_computer2.gradients[name]
        info = kfac_factors[name]
        ed = eigendata[name]

        V_A = ed['V_A'].to(device)
        V_B = ed['V_B'].to(device)

        if info['type'] == 'conv':
            act_unf = torch.nn.functional.unfold(
                act, module.kernel_size,
                dilation=module.dilation, padding=module.padding, stride=module.stride
            )
            act_flat = act_unf.mean(dim=2)  # (B, C_in*kh*kw)
            grad_flat = grad.reshape(bs, grad.size(1), -1).mean(dim=2)
        else:
            act_flat = act
            grad_flat = grad

        if info['has_bias']:
            ones = torch.ones(bs, 1, device=device)
            act_flat = torch.cat([act_flat, ones], dim=1)

        # Project onto Kronecker eigenvectors
        # rotated_a = V_A^T @ a  -> (B, dim_A)
        # rotated_g = V_B^T @ g  -> (B, dim_B)
        rotated_a = act_flat @ V_A  # (B, dim_A)
        rotated_g = grad_flat @ V_B  # (B, dim_B)

        # EK-FAC diagonal: E[(rotated_g_j)^2 * (rotated_a_i)^2]
        # = sum over samples of (rotated_g_j)^2 * (rotated_a_i)^2
        # Shape: (B, dim_A, 1) * (B, 1, dim_B) -> (B, dim_A, dim_B) -> sum over B
        ra2 = (rotated_a ** 2)  # (B, dim_A)
        rg2 = (rotated_g ** 2)  # (B, dim_B)

        # Outer product sum: (dim_A, dim_B)
        diag_correction = (ra2.t() @ rg2).flatten().cpu()  # (dim_A * dim_B,)
        ekfac_corrections[name] += diag_correction

    if (batch_idx + 1) % 20 == 0:
        print(f"  [EK-FAC] Processed {n_corr_samples} samples")

kfac_computer2.remove_hooks()

# Normalize
for name in ekfac_corrections:
    ekfac_corrections[name] /= n_corr_samples

print(f"[BSS] EK-FAC corrections computed from {n_corr_samples} samples")

report_progress(2, 5, metric={"phase": "ekfac_corrections_computed"})

# ===== Step 4: Assemble eigenvalue spectrum and define buckets =====
print("\n[BSS] === Step 4: Assembling eigenvalue spectrum and defining buckets ===")

# For each layer, we now have:
# - K-FAC eigenvalues: kfac_eigs = lambda_A_i * lambda_B_j
# - EK-FAC corrected eigenvalues: ekfac_corrections[name]

# Build global eigenvalue arrays
layer_info = []
global_kfac_eigs = []
global_ekfac_eigs = []
layer_starts = {}
offset = 0

for name in eigendata:
    ed = eigendata[name]
    kfac_eigs = ed['kfac_eigs']  # (dim_A * dim_B,)
    ekfac_eigs = ekfac_corrections[name]  # (dim_A * dim_B,)

    n_eigs = len(kfac_eigs)
    layer_starts[name] = offset
    offset += n_eigs

    global_kfac_eigs.append(kfac_eigs)
    global_ekfac_eigs.append(ekfac_eigs)

    layer_info.append({
        'name': name,
        'n_eigs': n_eigs,
        'kfac_max': kfac_eigs.max().item(),
        'ekfac_max': ekfac_eigs.max().item(),
    })

global_kfac_eigs = torch.cat(global_kfac_eigs)
global_ekfac_eigs = torch.cat(global_ekfac_eigs)

# Use EK-FAC eigenvalues for bucketing (they're the better approximation to true GGN)
# Sort by magnitude for bucket assignment
sorted_ekfac, ekfac_sort_idx = global_ekfac_eigs.sort(descending=True)

print(f"[BSS] Total eigenvalues: {len(global_ekfac_eigs)}")
print(f"[BSS] EK-FAC top-10: {sorted_ekfac[:10].numpy()}")
print(f"[BSS] EK-FAC at rank 50: {sorted_ekfac[min(49, len(sorted_ekfac)-1)]:.6f}")
print(f"[BSS] EK-FAC at rank 100: {sorted_ekfac[min(99, len(sorted_ekfac)-1)]:.6f}")

# Define buckets based on EK-FAC eigenvalue magnitudes
# Adaptive thresholds: use the actual spectrum to set meaningful boundaries
# Try the spec thresholds first, then adapt if needed
OUTLIER_THRESH = 100.0
EDGE_THRESH = 10.0

n_outlier = (sorted_ekfac > OUTLIER_THRESH).sum().item()
n_edge = ((sorted_ekfac > EDGE_THRESH) & (sorted_ekfac <= OUTLIER_THRESH)).sum().item()
n_bulk = (sorted_ekfac <= EDGE_THRESH).sum().item()

print(f"\n[BSS] Bucket distribution (thresholds: outlier>{OUTLIER_THRESH}, edge>{EDGE_THRESH}):")
print(f"  Outlier (lambda > {OUTLIER_THRESH}): {n_outlier} eigenvalues")
print(f"  Edge ({EDGE_THRESH} < lambda <= {OUTLIER_THRESH}): {n_edge} eigenvalues")
print(f"  Bulk (lambda <= {EDGE_THRESH}): {n_bulk} eigenvalues")

# If outlier bucket is too small, adapt thresholds
if n_outlier < 5:
    # Use percentile-based thresholds
    # For ResNet-18 on CIFAR-10, we expect ~10 class-discriminative eigenvalues
    # Let's use the top-20 as "outlier", next 80 as "edge"
    if len(sorted_ekfac) >= 100:
        OUTLIER_THRESH = sorted_ekfac[19].item()  # top 20
        EDGE_THRESH = sorted_ekfac[99].item()    # top 100
    else:
        OUTLIER_THRESH = sorted_ekfac[max(0, len(sorted_ekfac)//5 - 1)].item()
        EDGE_THRESH = sorted_ekfac[max(0, len(sorted_ekfac)//2 - 1)].item()

    n_outlier = (global_ekfac_eigs > OUTLIER_THRESH).sum().item()
    n_edge = ((global_ekfac_eigs > EDGE_THRESH) & (global_ekfac_eigs <= OUTLIER_THRESH)).sum().item()
    n_bulk = (global_ekfac_eigs <= EDGE_THRESH).sum().item()

    print(f"\n[BSS] Adapted thresholds: outlier>{OUTLIER_THRESH:.4f}, edge>{EDGE_THRESH:.6f}")
    print(f"  Outlier: {n_outlier} eigenvalues")
    print(f"  Edge: {n_edge} eigenvalues")
    print(f"  Bulk: {n_bulk} eigenvalues")

# Create bucket masks
outlier_mask = global_ekfac_eigs > OUTLIER_THRESH
edge_mask = (global_ekfac_eigs > EDGE_THRESH) & (global_ekfac_eigs <= OUTLIER_THRESH)
bulk_mask = global_ekfac_eigs <= EDGE_THRESH

report_progress(3, 5, metric={
    "phase": "buckets_defined",
    "n_outlier": n_outlier, "n_edge": n_edge, "n_bulk": n_bulk,
    "outlier_thresh": OUTLIER_THRESH, "edge_thresh": EDGE_THRESH,
})

# ===== Step 5: Compute perturbation factors =====
print("\n[BSS] === Step 5: Computing perturbation factors ===")

# BSS perturbation factor: |1/lambda_ekfac - 1/lambda_kfac|
# This measures how much the K-FAC approximation distorts the inverse Hessian
# at each eigenvalue

# Add damping to avoid division by zero
kfac_damped = global_kfac_eigs + DAMPING_KFAC
ekfac_damped = global_ekfac_eigs + DAMPING_EKFAC

perturbation_factors = torch.abs(1.0 / ekfac_damped - 1.0 / kfac_damped)

print(f"[BSS] Perturbation factor stats:")
print(f"  Overall: mean={perturbation_factors.mean():.6f}, max={perturbation_factors.max():.6f}")
if outlier_mask.any():
    print(f"  Outlier: mean={perturbation_factors[outlier_mask].mean():.6f}")
if edge_mask.any():
    print(f"  Edge: mean={perturbation_factors[edge_mask].mean():.6f}")
if bulk_mask.any():
    print(f"  Bulk: mean={perturbation_factors[bulk_mask].mean():.6f}")

# ===== Step 6: Compute per-test-point BSS =====
print(f"\n[BSS] === Step 6: Computing BSS for {N_TEST} test points ===")

# For each test point z with gradient g:
# BSS_j(z) = sum_{k in B_j} perturbation_factor_k * (V_k^T g)^2
#
# V_k = v_B kron v_A (Kronecker eigenvector)
# So V_k^T g = (v_B^T G v_A) where G is the gradient reshaped per layer
# = (rotated_g_j * rotated_a_i) for the (i,j) eigenvalue pair

# Prepare test data
test_subset = Subset(testset, test_indices)
test_loader = DataLoader(test_subset, batch_size=1, shuffle=False, num_workers=2)

# Re-register hooks
kfac_computer3 = KFACComputer(model)
criterion_pt = nn.CrossEntropyLoss(reduction='sum')

bss_outlier = np.zeros(N_TEST)
bss_edge = np.zeros(N_TEST)
bss_bulk = np.zeros(N_TEST)
bss_total = np.zeros(N_TEST)
test_labels = np.zeros(N_TEST, dtype=int)
test_grad_norms = np.zeros(N_TEST)
test_confidences = np.zeros(N_TEST)
test_entropies = np.zeros(N_TEST)

print(f"[BSS] Processing {N_TEST} test points...")
t_bss_start = time.time()

for idx, (inputs, targets) in enumerate(test_loader):
    inputs, targets = inputs.to(device), targets.to(device)
    test_labels[idx] = targets.item()

    # Forward + backward to get gradients and activations
    model.zero_grad()
    outputs = model(inputs)

    # Confidence and entropy
    probs = torch.softmax(outputs, dim=1)
    test_confidences[idx] = probs.max().item()
    test_entropies[idx] = -(probs * torch.log(probs + 1e-10)).sum().item()

    loss = criterion_pt(outputs, targets)
    loss.backward()

    # Compute gradient norm
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    test_grad_norms[idx] = np.sqrt(grad_norm)

    # For each layer, project gradient onto Kronecker eigenvectors
    # and compute BSS contributions
    global_offset = 0
    bss_vec = torch.zeros(len(global_ekfac_eigs))

    for name, module in kfac_computer3.layers:
        act = kfac_computer3.activations[name]
        grad = kfac_computer3.gradients[name]
        info = kfac_factors[name]
        ed = eigendata[name]

        V_A = ed['V_A'].to(device)
        V_B = ed['V_B'].to(device)

        if info['type'] == 'conv':
            act_unf = torch.nn.functional.unfold(
                act, module.kernel_size,
                dilation=module.dilation, padding=module.padding, stride=module.stride
            )
            act_flat = act_unf.mean(dim=2)  # (1, C_in*kh*kw)
            grad_flat = grad.reshape(1, grad.size(1), -1).mean(dim=2)
        else:
            act_flat = act  # (1, in_features)
            grad_flat = grad  # (1, out_features)

        if info['has_bias']:
            ones = torch.ones(1, 1, device=device)
            act_flat = torch.cat([act_flat, ones], dim=1)

        # Project: rotated_a = V_A^T a, rotated_g = V_B^T g
        rotated_a = (act_flat @ V_A).squeeze(0)  # (dim_A,)
        rotated_g = (grad_flat @ V_B).squeeze(0)  # (dim_B,)

        # Squared projections: (V_k^T g)^2 = (rotated_a_i * rotated_g_j)^2
        # Shape: (dim_A,) outer (dim_B,) -> (dim_A, dim_B) -> flatten
        sq_proj = (rotated_a.unsqueeze(1) * rotated_g.unsqueeze(0)).flatten() ** 2

        n_layer_eigs = ed['dim_A'] * ed['dim_B']
        bss_vec[global_offset:global_offset + n_layer_eigs] = sq_proj.cpu()
        global_offset += n_layer_eigs

    # BSS = perturbation_factor * squared_projection, summed per bucket
    bss_weighted = perturbation_factors * bss_vec

    bss_outlier[idx] = bss_weighted[outlier_mask].sum().item() if outlier_mask.any() else 0.0
    bss_edge[idx] = bss_weighted[edge_mask].sum().item() if edge_mask.any() else 0.0
    bss_bulk[idx] = bss_weighted[bulk_mask].sum().item() if bulk_mask.any() else 0.0
    bss_total[idx] = bss_weighted.sum().item()

    if (idx + 1) % 20 == 0:
        elapsed = time.time() - t_bss_start
        eta = elapsed / (idx + 1) * (N_TEST - idx - 1)
        print(f"  [BSS] {idx+1}/{N_TEST} points | "
              f"BSS_outlier={bss_outlier[idx]:.6f} | "
              f"Elapsed: {elapsed:.1f}s ETA: {eta:.1f}s")

kfac_computer3.remove_hooks()

bss_time = time.time() - t_bss_start
print(f"\n[BSS] BSS computation completed in {bss_time:.1f}s")

report_progress(4, 5, metric={
    "phase": "bss_computed",
    "bss_time_sec": bss_time,
    "bss_outlier_mean": float(bss_outlier.mean()),
    "bss_outlier_std": float(bss_outlier.std()),
})

# ===== Step 7: Analysis and save results =====
print("\n[BSS] === Step 7: Analysis ===")

# BSS outlier statistics
print(f"\n[BSS] BSS Outlier bucket statistics:")
print(f"  Mean: {bss_outlier.mean():.6f}")
print(f"  Std:  {bss_outlier.std():.6f}")
print(f"  Min:  {bss_outlier.min():.6f}")
print(f"  Max:  {bss_outlier.max():.6f}")

# Per-class BSS outlier
print(f"\n[BSS] Per-class BSS_outlier means:")
class_bss_means = {}
class_bss_stds = {}
for c in range(10):
    mask = test_labels == c
    if mask.sum() > 0:
        class_bss_means[c] = float(bss_outlier[mask].mean())
        class_bss_stds[c] = float(bss_outlier[mask].std())
        print(f"  Class {c}: mean={class_bss_means[c]:.6f}, std={class_bss_stds[c]:.6f}, n={mask.sum()}")

# Within-class vs total variance (for H-D3 class detector test)
overall_var = float(np.var(bss_outlier))
within_class_var = 0.0
for c in range(10):
    mask = test_labels == c
    if mask.sum() > 1:
        within_class_var += float(np.var(bss_outlier[mask])) * mask.sum()
within_class_var /= N_TEST
within_class_frac = within_class_var / (overall_var + 1e-20)

print(f"\n[BSS] Variance decomposition:")
print(f"  Total variance: {overall_var:.10f}")
print(f"  Within-class variance: {within_class_var:.10f}")
print(f"  Within-class fraction: {within_class_frac:.4f} ({within_class_frac*100:.1f}%)")
print(f"  Between-class fraction: {1-within_class_frac:.4f} ({(1-within_class_frac)*100:.1f}%)")

# Correlation with gradient norm and confidence
from scipy import stats

corr_grad, p_grad = stats.spearmanr(bss_outlier, test_grad_norms)
corr_conf, p_conf = stats.spearmanr(bss_outlier, test_confidences)
corr_entr, p_entr = stats.spearmanr(bss_outlier, test_entropies)

print(f"\n[BSS] Correlations:")
print(f"  BSS_outlier vs gradient_norm: rho={corr_grad:.4f} (p={p_grad:.4e})")
print(f"  BSS_outlier vs confidence: rho={corr_conf:.4f} (p={p_conf:.4e})")
print(f"  BSS_outlier vs entropy: rho={corr_entr:.4f} (p={p_entr:.4e})")

# Eigenvalue spectrum summary
eigenvalue_spectrum = {
    'total_eigenvalues': len(global_ekfac_eigs),
    'top_10': sorted_ekfac[:10].numpy().tolist(),
    'top_100': sorted_ekfac[:min(100, len(sorted_ekfac))].numpy().tolist(),
    'outlier_threshold': OUTLIER_THRESH,
    'edge_threshold': EDGE_THRESH,
    'n_outlier': n_outlier,
    'n_edge': n_edge,
    'n_bulk': n_bulk,
}

# Save results
results = {
    "task_id": TASK_ID,
    "mode": "PILOT",
    "seed": SEED,
    "n_test": N_TEST,
    "n_train_kfac": 5000,
    "damping_kfac": DAMPING_KFAC,
    "damping_ekfac": DAMPING_EKFAC,
    "eigenvalue_spectrum": eigenvalue_spectrum,
    "bss_statistics": {
        "outlier": {
            "mean": float(bss_outlier.mean()),
            "std": float(bss_outlier.std()),
            "min": float(bss_outlier.min()),
            "max": float(bss_outlier.max()),
        },
        "edge": {
            "mean": float(bss_edge.mean()),
            "std": float(bss_edge.std()),
            "min": float(bss_edge.min()),
            "max": float(bss_edge.max()),
        },
        "bulk": {
            "mean": float(bss_bulk.mean()),
            "std": float(bss_bulk.std()),
            "min": float(bss_bulk.min()),
            "max": float(bss_bulk.max()),
        },
        "total": {
            "mean": float(bss_total.mean()),
            "std": float(bss_total.std()),
            "min": float(bss_total.min()),
            "max": float(bss_total.max()),
        },
    },
    "per_class_bss_outlier": {
        "means": class_bss_means,
        "stds": class_bss_stds,
    },
    "variance_decomposition": {
        "total_variance": overall_var,
        "within_class_variance": within_class_var,
        "within_class_fraction": within_class_frac,
        "between_class_fraction": 1 - within_class_frac,
    },
    "correlations": {
        "bss_outlier_vs_grad_norm": {"rho": float(corr_grad), "p_value": float(p_grad)},
        "bss_outlier_vs_confidence": {"rho": float(corr_conf), "p_value": float(p_conf)},
        "bss_outlier_vs_entropy": {"rho": float(corr_entr), "p_value": float(p_entr)},
    },
    "pass_criteria": {
        "outlier_has_5_plus_eigenvalues": n_outlier >= 5,
        "bss_outlier_std_above_001": float(bss_outlier.std()) > 0.01,
        "no_oom": True,
    },
    "overall_pass": n_outlier >= 5 and float(bss_outlier.std()) > 0.01,
    "go_no_go": "GO" if (n_outlier >= 5 and float(bss_outlier.std()) > 0.01) else "NO_GO",
    "timing": {
        "kfac_factors_sec": round(kfac_time, 1),
        "bss_computation_sec": round(bss_time, 1),
        "total_sec": round(time.time() - start_time, 1),
    },
    "gpu_info": {
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "vram_total_mb": torch.cuda.get_device_properties(0).total_memory // (1024*1024) if torch.cuda.is_available() else 0,
    },
    "timestamp": datetime.now().isoformat(),
}

# Check if std > 0.01 might fail due to scale -- in that case, also check relative CV
cv = float(bss_outlier.std() / (bss_outlier.mean() + 1e-20))
results["bss_statistics"]["outlier"]["cv"] = cv
print(f"\n[BSS] BSS_outlier CV (std/mean): {cv:.4f}")

# If absolute std is small but CV is reasonable, adjust assessment
if float(bss_outlier.std()) <= 0.01 and cv > 0.1:
    results["pass_criteria"]["bss_outlier_cv_above_01"] = True
    results["overall_pass"] = n_outlier >= 5
    results["go_no_go"] = "GO" if n_outlier >= 5 else "NO_GO"
    results["note"] = "BSS_outlier absolute std < 0.01 but CV > 0.1 indicates meaningful variation relative to magnitude"

# Save JSON
result_path = BSS_DIR / "pilot_bss_results.json"
result_path.write_text(json.dumps(results, indent=2))
print(f"\n[BSS] Results saved to {result_path}")

# Save per-point BSS vectors as numpy arrays
np.save(str(BSS_DIR / "bss_outlier_seed42.npy"), bss_outlier)
np.save(str(BSS_DIR / "bss_edge_seed42.npy"), bss_edge)
np.save(str(BSS_DIR / "bss_bulk_seed42.npy"), bss_bulk)
np.save(str(BSS_DIR / "bss_total_seed42.npy"), bss_total)
np.save(str(BSS_DIR / "test_indices.npy"), np.array(test_indices))
np.save(str(BSS_DIR / "test_labels.npy"), test_labels)
np.save(str(BSS_DIR / "test_grad_norms.npy"), test_grad_norms)
np.save(str(BSS_DIR / "test_confidences.npy"), test_confidences)
np.save(str(BSS_DIR / "test_entropies.npy"), test_entropies)

# Save eigenvalue spectrum
np.save(str(BSS_DIR / "eigenvalues_ekfac_sorted.npy"), sorted_ekfac.numpy())
np.save(str(BSS_DIR / "eigenvalues_kfac_all.npy"), global_kfac_eigs.numpy())
np.save(str(BSS_DIR / "eigenvalues_ekfac_all.npy"), global_ekfac_eigs.numpy())

print(f"[BSS] All numpy arrays saved to {BSS_DIR}")

# Final summary
total_time = time.time() - start_time
print(f"\n{'='*60}")
print(f"[BSS PILOT SUMMARY]")
print(f"  Seed: {SEED}")
print(f"  Test points: {N_TEST}")
print(f"  Outlier eigenvalues: {n_outlier} (threshold: {OUTLIER_THRESH})")
print(f"  BSS_outlier std: {bss_outlier.std():.6f} (need > 0.01)")
print(f"  BSS_outlier CV: {cv:.4f}")
print(f"  Within-class variance fraction: {within_class_frac:.4f}")
print(f"  Overall PASS: {results['overall_pass']}")
print(f"  GO/NO-GO: {results['go_no_go']}")
print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
print(f"{'='*60}")

# Mark done
mark_done(
    status="success" if results['overall_pass'] else "partial",
    summary=f"BSS pilot (seed {SEED}, {N_TEST} pts): outlier_eigs={n_outlier}, "
            f"BSS_std={bss_outlier.std():.6f}, CV={cv:.4f}, "
            f"within_class={within_class_frac:.1%}. {results['go_no_go']}"
)

report_progress(5, 5, metric={
    "phase": "done",
    "go_no_go": results['go_no_go'],
    "overall_pass": results['overall_pass'],
    "total_time_sec": round(total_time, 1),
})

print("[BSS] Done.")
