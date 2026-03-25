"""
Phase 1: Verify full-model Hessian computation feasibility with pyDVL.
Tests EK-FAC influence on 10 test points using the trained ResNet-18.
"""
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import json
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from pathlib import Path

PROJECT_DIR = Path("/home/jinxulin/sibyl_system/projects/AURA")
CHECKPOINT = PROJECT_DIR / "exp" / "checkpoints" / "resnet18_cifar10_seed42.pt"
DATA_DIR = Path("/home/jinxulin/sibyl_system/shared/datasets/cifar10")
RESULTS_DIR = PROJECT_DIR / "exp" / "results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[HESSIAN] Device: {device} ({torch.cuda.get_device_name()})")

# Load model
model = resnet18(num_classes=10)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
checkpoint = torch.load(str(CHECKPOINT), map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print(f"[HESSIAN] Model loaded: test_acc={checkpoint['test_acc']:.2f}%")

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=False, transform=transform)

# Use small subsets for verification
train_subset = torch.utils.data.Subset(trainset, list(range(2000)))
test_subset = torch.utils.data.Subset(testset, list(range(10)))
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=10, shuffle=False)

results = {}

# === Test 1: pyDVL EkfacInfluence ===
print("\n[TEST 1] pyDVL EkfacInfluence (full-model)...")
try:
    from pydvl.influence.torch import EkfacInfluence
    t0 = time.time()
    ekfac = EkfacInfluence(model, update_diagonal=True, hessian_regularization=0.1)
    ekfac = ekfac.fit(train_loader)
    fit_time = time.time() - t0
    print(f"[TEST 1] EK-FAC fit completed in {fit_time:.1f}s")

    # Compute influences for 10 test points vs 2000 train points
    t1 = time.time()
    test_batch = next(iter(test_loader))
    test_x, test_y = test_batch[0].to(device), test_batch[1].to(device)
    influences = ekfac.influences(test_x, test_y, train_loader)
    inf_time = time.time() - t1
    print(f"[TEST 1] Influences computed in {inf_time:.1f}s, shape={influences.shape}")
    print(f"[TEST 1] Influence range: [{influences.min().item():.6f}, {influences.max().item():.6f}]")
    print(f"[TEST 1] Influence std: {influences.std().item():.6f}")

    results['pydvl_ekfac'] = {
        'success': True,
        'fit_time_sec': round(fit_time, 1),
        'influence_time_sec': round(inf_time, 1),
        'influence_shape': list(influences.shape),
        'influence_range': [influences.min().item(), influences.max().item()],
        'influence_std': influences.std().item(),
    }
except Exception as e:
    print(f"[TEST 1] FAILED: {e}")
    import traceback; traceback.print_exc()
    results['pydvl_ekfac'] = {'success': False, 'error': str(e)}

# === Test 2: pyDVL CgInfluence (K-FAC baseline) ===
print("\n[TEST 2] pyDVL CgInfluence...")
try:
    from pydvl.influence.torch import CgInfluence
    t0 = time.time()
    cg = CgInfluence(model, hessian_regularization=0.1, maxiter=50)
    cg = cg.fit(train_loader)
    fit_time = time.time() - t0
    print(f"[TEST 2] CG fit completed in {fit_time:.1f}s")

    t1 = time.time()
    test_batch = next(iter(test_loader))
    test_x, test_y = test_batch[0].to(device), test_batch[1].to(device)
    influences_cg = cg.influences(test_x, test_y, train_loader)
    inf_time = time.time() - t1
    print(f"[TEST 2] CG influences shape={influences_cg.shape}, time={inf_time:.1f}s")

    results['pydvl_cg'] = {
        'success': True,
        'fit_time_sec': round(fit_time, 1),
        'influence_time_sec': round(inf_time, 1),
        'influence_shape': list(influences_cg.shape),
    }
except Exception as e:
    print(f"[TEST 2] FAILED: {e}")
    results['pydvl_cg'] = {'success': False, 'error': str(e)}

# === Test 3: dattri API check ===
print("\n[TEST 3] dattri API exploration...")
try:
    import dattri
    from dattri.func import hessian as dattri_hessian
    print(f"[TEST 3] dattri.func.hessian exports: {[x for x in dir(dattri_hessian) if not x.startswith('_')]}")

    # Check available influence function implementations
    from dattri import algorithm as dattri_algo
    print(f"[TEST 3] dattri.algorithm exports: {[x for x in dir(dattri_algo) if not x.startswith('_')]}")

    results['dattri'] = {
        'success': True,
        'hessian_exports': [x for x in dir(dattri_hessian) if not x.startswith('_')],
        'algorithm_exports': [x for x in dir(dattri_algo) if not x.startswith('_')],
    }
except Exception as e:
    print(f"[TEST 3] FAILED: {e}")
    results['dattri'] = {'success': False, 'error': str(e)}

# === Test 4: traker (TRAK) ===
print("\n[TEST 4] TRAK library...")
try:
    from traker import TRAKer
    print(f"[TEST 4] TRAKer imported OK")
    results['trak'] = {'success': True, 'available': True}
except Exception as e:
    print(f"[TEST 4] FAILED: {e}")
    results['trak'] = {'success': False, 'error': str(e)}

# === Test 5: Manual per-sample gradient computation ===
print("\n[TEST 5] Manual per-sample gradients (functorch/vmap)...")
try:
    from torch.func import vmap, grad
    # Per-sample gradient via vmap
    criterion = nn.CrossEntropyLoss()

    def compute_loss(params, buffers, sample, target):
        from torch.func import functional_call
        output = functional_call(model, (params, buffers), (sample.unsqueeze(0),))
        return criterion(output, target.unsqueeze(0))

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v for k, v in model.named_buffers()}

    test_batch = next(iter(test_loader))
    test_x, test_y = test_batch[0].to(device), test_batch[1].to(device)

    # Compute gradient for single sample
    ft_compute_grad = grad(compute_loss)
    grads = ft_compute_grad(params, buffers, test_x[0], test_y[0])
    total_grad_elems = sum(g.numel() for g in grads.values())
    print(f"[TEST 5] Per-sample gradient computed: {total_grad_elems:,} elements")

    # Test vmap for batch
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
    batch_grads = ft_compute_sample_grad(params, buffers, test_x[:5], test_y[:5])
    print(f"[TEST 5] vmap batch gradient: {sum(g.shape[0] for g in batch_grads.values())} samples verified")

    results['manual_grad'] = {
        'success': True,
        'grad_elements': total_grad_elems,
        'vmap_works': True,
    }
except Exception as e:
    print(f"[TEST 5] FAILED: {e}")
    import traceback; traceback.print_exc()
    results['manual_grad'] = {'success': False, 'error': str(e)}

# === GPU memory profile ===
print("\n[GPU] Memory usage:")
if torch.cuda.is_available():
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    results['gpu_memory'] = {
        'allocated_gb': round(torch.cuda.memory_allocated() / 1e9, 2),
        'max_allocated_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2),
        'reserved_gb': round(torch.cuda.memory_reserved() / 1e9, 2),
    }

# Save results
output_path = RESULTS_DIR / "phase1_hessian_verification.json"
output_path.write_text(json.dumps(results, indent=2, default=str))
print(f"\n[RESULTS] Saved to {output_path}")

# Summary
print("\n=== HESSIAN VERIFICATION SUMMARY ===")
for method, r in results.items():
    if isinstance(r, dict) and 'success' in r:
        status = "PASS" if r['success'] else "FAIL"
        print(f"  {method}: {status}")
