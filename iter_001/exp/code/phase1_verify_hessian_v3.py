"""
Phase 1: Verify full-model Hessian feasibility. Minimal, correct version.
"""
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import json, time, torch, torch.nn as nn
import torchvision, torchvision.transforms as transforms
from torchvision.models import resnet18
from pathlib import Path

PROJECT_DIR = Path("/home/jinxulin/sibyl_system/projects/AURA")
RESULTS_DIR = PROJECT_DIR / "exp" / "results"
DATA_DIR = Path("/home/jinxulin/sibyl_system/shared/datasets/cifar10")

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}")

# Load model
model = resnet18(num_classes=10)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
ckpt = torch.load(str(PROJECT_DIR / "exp/checkpoints/resnet18_cifar10_seed42.pt"),
                   map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device).eval()
print(f"Model loaded: {ckpt['test_acc']:.2f}%")

# Data (small subsets)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=False, transform=transform)
train_sub = torch.utils.data.Subset(trainset, list(range(2000)))
test_sub = torch.utils.data.Subset(testset, list(range(10)))
train_loader = torch.utils.data.DataLoader(train_sub, batch_size=128, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_sub, batch_size=10, shuffle=False)

results = {}

# ==== TEST 1: dattri IFAttributorEKFAC (full-model) ====
print("\n=== TEST 1: dattri IFAttributorEKFAC ===")
try:
    from dattri.algorithm.influence_function import IFAttributorEKFAC
    from dattri.task import AttributionTask

    def f_target(params, data_target_pair):
        x, y = data_target_pair
        x, y = x.to(device), y.to(device)
        from torch.func import functional_call
        output = functional_call(model, params, (x,))
        return nn.functional.cross_entropy(output, y)

    task = AttributionTask(
        loss_func=f_target,
        model=model,
        checkpoints=model.state_dict(),
    )

    t0 = time.time()
    attributor = IFAttributorEKFAC(
        task=task,
        device='cuda',
        damping=0.1,
    )
    attributor.cache(train_loader)
    fit_time = time.time() - t0
    print(f"  EK-FAC cache: {fit_time:.1f}s")

    t1 = time.time()
    attrs = attributor.attribute(train_loader, test_loader)
    attr_time = time.time() - t1
    print(f"  Attributions: shape={attrs.shape}, time={attr_time:.1f}s")
    print(f"  Range: [{attrs.min().item():.6f}, {attrs.max().item():.6f}], std={attrs.std().item():.6f}")
    results['dattri_ekfac'] = {
        'success': True, 'fit_time_sec': round(fit_time, 1),
        'attribution_time_sec': round(attr_time, 1),
        'shape': list(attrs.shape),
    }
    print("  PASS")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    results['dattri_ekfac'] = {'success': False, 'error': str(e)}

# Clear GPU memory
torch.cuda.empty_cache()
import gc; gc.collect()

# ==== TEST 2: TRAK ====
print("\n=== TEST 2: TRAK ===")
try:
    from trak import TRAKer
    import shutil

    save_dir = str(RESULTS_DIR / "trak_test")
    shutil.rmtree(save_dir, ignore_errors=True)

    traker = TRAKer(
        model=model,
        task='image_classification',
        train_set_size=len(train_sub),
        proj_dim=512,
        save_dir=save_dir,
        device=device,
    )

    # Need to move data to device in the batch
    traker.load_checkpoint(model.state_dict(), model_id=0)
    for batch in train_loader:
        batch_gpu = (batch[0].to(device), batch[1].to(device))
        traker.featurize(batch=batch_gpu, num_samples=batch_gpu[0].shape[0])
    traker.finalize_features(model_ids=[0])

    traker.start_scoring_checkpoint(
        exp_name='test',
        checkpoint=model.state_dict(),
        model_id=0,
        num_targets=len(test_sub),
    )
    for batch in test_loader:
        batch_gpu = (batch[0].to(device), batch[1].to(device))
        traker.score(batch=batch_gpu, num_samples=batch_gpu[0].shape[0])
    scores = traker.finalize_scores(exp_name='test')
    print(f"  TRAK scores: shape={scores.shape}")
    print(f"  Range: [{scores.min():.6f}, {scores.max():.6f}]")
    results['trak'] = {'success': True, 'shape': list(scores.shape)}
    shutil.rmtree(save_dir, ignore_errors=True)
    print("  PASS")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    results['trak'] = {'success': False, 'error': str(e)}

# ==== TEST 3: RepSim ====
print("\n=== TEST 3: RepSim ===")
try:
    activation = {}
    def hook(model_mod, inp, out):
        activation['feat'] = out.detach()
    handle = model.avgpool.register_forward_hook(hook)

    feats_train, feats_test = [], []
    with torch.no_grad():
        for batch in train_loader:
            model(batch[0].to(device))
            feats_train.append(activation['feat'].squeeze())
        for batch in test_loader:
            model(batch[0].to(device))
            feats_test.append(activation['feat'].squeeze())
    handle.remove()

    ft = torch.cat(feats_train); fq = torch.cat(feats_test)
    ft_n = ft / ft.norm(dim=1, keepdim=True)
    fq_n = fq / fq.norm(dim=1, keepdim=True)
    sim = fq_n @ ft_n.T
    print(f"  RepSim: shape={sim.shape}, range=[{sim.min().item():.4f}, {sim.max().item():.4f}]")
    results['repsim'] = {'success': True, 'shape': list(sim.shape)}
    print("  PASS")
except Exception as e:
    print(f"  FAILED: {e}")
    results['repsim'] = {'success': False, 'error': str(e)}

# ==== Summary ====
print(f"\nMax GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
results['gpu_max_mem_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

print("\n" + "="*50)
all_pass = True
for k, v in results.items():
    if isinstance(v, dict) and 'success' in v:
        s = "PASS" if v['success'] else "FAIL"
        if not v['success']: all_pass = False
        print(f"  {k}: {s}")

results['overall_pass'] = all_pass

# Note about pyDVL
results['pyDVL_note'] = "pyDVL EkfacInfluence only supports Linear layers. Use dattri for Conv2d models."

out = RESULTS_DIR / "phase1_hessian_verification.json"
out.write_text(json.dumps(results, indent=2, default=str))
print(f"\nOverall: {'PASS' if all_pass else 'PARTIAL'}")
print(f"Results: {out}")
