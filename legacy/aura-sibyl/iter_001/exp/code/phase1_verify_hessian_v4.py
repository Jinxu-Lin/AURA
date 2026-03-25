"""
Phase 1: Verify full-model Hessian feasibility with limited GPU memory (~17GB free).
Uses small batch sizes for per-sample gradient computation.
"""
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json, time, gc, torch, torch.nn as nn
import torchvision, torchvision.transforms as transforms
from torchvision.models import resnet18
from pathlib import Path

PROJECT_DIR = Path("/home/jinxulin/sibyl_system/projects/AURA")
RESULTS_DIR = PROJECT_DIR / "exp" / "results"
DATA_DIR = Path("/home/jinxulin/sibyl_system/shared/datasets/cifar10")

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Free VRAM: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB")

# Load model
model = resnet18(num_classes=10)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
ckpt = torch.load(str(PROJECT_DIR / "exp/checkpoints/resnet18_cifar10_seed42.pt"),
                   map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device).eval()
print(f"Model: {ckpt['test_acc']:.2f}% test acc")

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=False, transform=transform)

# Small batch sizes for per-sample gradient computation (OOM-safe)
train_sub = torch.utils.data.Subset(trainset, list(range(500)))  # 500 train
test_sub = torch.utils.data.Subset(testset, list(range(10)))     # 10 test
# Small batch size = 16 for per-sample gradients
train_loader = torch.utils.data.DataLoader(train_sub, batch_size=16, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_sub, batch_size=10, shuffle=False)

results = {}

# ==== TEST 1: dattri IFAttributorEKFAC ====
print("\n=== TEST 1: dattri IFAttributorEKFAC (full-model, batch=16) ===")
try:
    from dattri.algorithm.influence_function import IFAttributorEKFAC
    from dattri.task import AttributionTask

    def f_target(params, data_target_pair):
        x, y = data_target_pair
        x, y = x.to(device), y.to(device)
        from torch.func import functional_call
        output = functional_call(model, params, (x,))
        return nn.functional.cross_entropy(output, y)

    task = AttributionTask(loss_func=f_target, model=model, checkpoints=model.state_dict())

    t0 = time.time()
    attributor = IFAttributorEKFAC(task=task, device='cuda', damping=0.1)
    attributor.cache(train_loader)
    fit_time = time.time() - t0
    print(f"  EK-FAC cache: {fit_time:.1f}s")

    t1 = time.time()
    attrs = attributor.attribute(train_loader, test_loader)
    attr_time = time.time() - t1
    print(f"  Attributions: shape={attrs.shape}, time={attr_time:.1f}s")
    print(f"  Range: [{attrs.min().item():.6f}, {attrs.max().item():.6f}], std={attrs.std().item():.6f}")
    print(f"  GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # Verify non-trivial: check top-10 rankings per test point
    for i in range(min(3, attrs.shape[0])):
        top10 = attrs[i].argsort(descending=True)[:10]
        print(f"  Test point {i}: top-10 train indices = {top10.tolist()}")

    results['dattri_ekfac'] = {
        'success': True, 'fit_time_sec': round(fit_time, 1),
        'attr_time_sec': round(attr_time, 1), 'shape': list(attrs.shape),
        'range': [attrs.min().item(), attrs.max().item()],
        'std': attrs.std().item(),
        'gpu_mem_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    print("  PASS")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    results['dattri_ekfac'] = {'success': False, 'error': str(e)}

# Clear memory
del attributor, task, attrs
gc.collect(); torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# ==== TEST 2: TRAK (small batch) ====
print("\n=== TEST 2: TRAK (batch=16, proj_dim=256) ===")
try:
    from trak import TRAKer
    import shutil

    save_dir = str(RESULTS_DIR / "trak_test")
    shutil.rmtree(save_dir, ignore_errors=True)

    traker = TRAKer(
        model=model,
        task='image_classification',
        train_set_size=len(train_sub),
        proj_dim=256,
        save_dir=save_dir,
        device=device,
    )

    traker.load_checkpoint(model.state_dict(), model_id=0)
    for batch in train_loader:
        b = (batch[0].to(device), batch[1].to(device))
        traker.featurize(batch=b, num_samples=b[0].shape[0])
    traker.finalize_features(model_ids=[0])

    traker.start_scoring_checkpoint(
        exp_name='test', checkpoint=model.state_dict(),
        model_id=0, num_targets=len(test_sub),
    )
    for batch in test_loader:
        b = (batch[0].to(device), batch[1].to(device))
        traker.score(batch=b, num_samples=b[0].shape[0])
    scores = traker.finalize_scores(exp_name='test')
    print(f"  TRAK: shape={scores.shape}, range=[{scores.min():.6f}, {scores.max():.6f}]")
    print(f"  GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    results['trak'] = {'success': True, 'shape': list(scores.shape),
                       'gpu_mem_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    shutil.rmtree(save_dir, ignore_errors=True)
    print("  PASS")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    results['trak'] = {'success': False, 'error': str(e)}

gc.collect(); torch.cuda.empty_cache()

# ==== TEST 3: RepSim ====
print("\n=== TEST 3: RepSim ===")
try:
    activation = {}
    def hook(m, inp, out): activation['f'] = out.detach()
    h = model.avgpool.register_forward_hook(hook)
    ft, fq = [], []
    with torch.no_grad():
        for b in train_loader:
            model(b[0].to(device)); ft.append(activation['f'].squeeze())
        for b in test_loader:
            model(b[0].to(device)); fq.append(activation['f'].squeeze())
    h.remove()
    ft = torch.cat(ft); fq = torch.cat(fq)
    sim = (fq / fq.norm(dim=1, keepdim=True)) @ (ft / ft.norm(dim=1, keepdim=True)).T
    print(f"  RepSim: shape={sim.shape}, range=[{sim.min().item():.4f}, {sim.max().item():.4f}]")
    results['repsim'] = {'success': True, 'shape': list(sim.shape)}
    print("  PASS")
except Exception as e:
    print(f"  FAILED: {e}"); results['repsim'] = {'success': False, 'error': str(e)}

# ==== Summary ====
print(f"\nPeak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
all_ok = all(v.get('success', True) for v in results.values() if isinstance(v, dict))
results['overall_pass'] = all_ok
results['pyDVL_note'] = "pyDVL EkfacInfluence does NOT support Conv2d. Use dattri for ResNet-18."
results['memory_note'] = "GPU shared with another process (~7GB). Full verification uses ~17GB total."

out = RESULTS_DIR / "phase1_hessian_verification.json"
out.write_text(json.dumps(results, indent=2, default=str))

print("\n" + "="*50)
for k, v in results.items():
    if isinstance(v, dict) and 'success' in v:
        print(f"  {k}: {'PASS' if v['success'] else 'FAIL'}")
print(f"Overall: {'PASS' if all_ok else 'PARTIAL'}")
