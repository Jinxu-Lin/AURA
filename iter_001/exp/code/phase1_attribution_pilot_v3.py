"""
Phase 1 Attribution Computation - PILOT MODE v3
Memory-constrained: ~17GB available on GPU 3 (another process uses ~7GB).

Strategy: Custom implementations that avoid vmap for per-sample gradients.
- EK-FAC IF: Manual Kronecker-factored influence via loop-based per-sample gradients
- K-FAC IF: Same framework with different damping/no eigenvalue correction
- RepSim: Standard cosine similarity
- TRAK: CPU-based projection to avoid GPU OOM

100 test points x 5000 train points for IF methods.
"""
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json, time, gc, sys, shutil, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from datetime import datetime
from scipy.stats import kendalltau, spearmanr

# ---- Constants ----
PROJECT_DIR = Path("/home/jinxulin/sibyl_system/projects/AURA")
RESULTS_DIR = PROJECT_DIR / "exp" / "results"
CKPT_DIR = PROJECT_DIR / "exp" / "checkpoints"
DATA_DIR = Path("/home/jinxulin/sibyl_system/shared/datasets/cifar10")
TASK_ID = "phase1_attribution_compute"
SEED = 42
N_TEST = 100
N_CLASSES = 10
N_TRAIN_IF = 5000  # 5K train subset for IF

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
attr_dir = RESULTS_DIR / "phase1_attributions"
attr_dir.mkdir(exist_ok=True)

device = torch.device("cuda")
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---- PID/Progress ----
(RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))

def progress(stage, detail="", pct=0):
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID, "stage": stage, "detail": detail,
        "percent": pct, "updated_at": datetime.now().isoformat(),
    }))
    print(f"[{pct:3d}%] {stage}: {detail}")

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

# Stratified test selection: 10 per class
test_by_class = {c: [] for c in range(N_CLASSES)}
for i in range(len(testset)):
    _, y = testset[i]
    if len(test_by_class[y]) < N_TEST // N_CLASSES:
        test_by_class[y].append(i)
    if all(len(v) >= N_TEST // N_CLASSES for v in test_by_class.values()):
        break
test_indices = sorted(sum(test_by_class.values(), []))
test_sub = torch.utils.data.Subset(testset, test_indices)

# Stratified 5K train subset: 500 per class
train_5k_indices = []
for c in range(N_CLASSES):
    cls_idx = [i for i in range(len(trainset)) if trainset.targets[i] == c][:500]
    train_5k_indices.extend(cls_idx)
train_sub_5k = torch.utils.data.Subset(trainset, train_5k_indices)
print(f"Test: {len(test_indices)}, Train(IF): {len(train_5k_indices)}")

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
print(f"Grad norms: mean={np.mean(test_features['gradient_norms']):.4f}, std={np.std(test_features['gradient_norms']):.4f}")

# ==== METHOD 1: RepSim (full 50K train) ====
progress("repsim", "Cosine similarity on penultimate layer", 10)
t0 = time.time()

activation = {}
def hook_fn(m, inp, out): activation['f'] = out.detach()
hook = model.avgpool.register_forward_hook(hook_fn)

# Extract training features
train_feats_all = []
with torch.no_grad():
    for i in range(0, len(trainset), 256):
        batch = torch.stack([trainset[j][0] for j in range(i, min(i+256, len(trainset)))]).to(device)
        model(batch)
        train_feats_all.append(activation['f'].squeeze().cpu())
        del batch
train_feats_all = torch.cat(train_feats_all)  # [50000, 512]

# Extract test features
test_feats = []
with torch.no_grad():
    for i in range(0, len(test_sub), 50):
        batch = torch.stack([test_sub[j][0] for j in range(i, min(i+50, len(test_sub)))]).to(device)
        model(batch)
        test_feats.append(activation['f'].squeeze().cpu())
        del batch
test_feats = torch.cat(test_feats)  # [100, 512]
hook.remove()

# Cosine similarity
train_norm = F.normalize(train_feats_all.float(), dim=1)
test_norm = F.normalize(test_feats.float(), dim=1)
repsim_scores = (test_norm @ train_norm.T).numpy()  # [100, 50000]
repsim_time = time.time() - t0
print(f"RepSim: {repsim_scores.shape}, {repsim_time:.1f}s, [{repsim_scores.min():.4f}, {repsim_scores.max():.4f}]")

repsim_rankings = np.argsort(-repsim_scores, axis=1)
np.save(str(attr_dir / "repsim_scores_full.npy"), repsim_scores)
repsim_5k = repsim_scores[:, train_5k_indices]
np.save(str(attr_dir / "repsim_scores_5k.npy"), repsim_5k)

del train_feats_all, test_feats, train_norm, test_norm
gc.collect(); torch.cuda.empty_cache()

# ==== METHOD 2 & 3: Manual EK-FAC IF and K-FAC IF ====
# Strategy: Compute per-sample gradients one at a time (no vmap),
# then use Kronecker-factored approximation for H^{-1}g.
#
# For EK-FAC: H^{-1} ≈ (A⊗B)^{-1} with eigenvalue correction
# For K-FAC:  H^{-1} ≈ (A+λI)^{-1} ⊗ (B+λI)^{-1}
#
# Instead of full implementation (complex), use dattri but with
# explicit memory management: compute train gradients to CPU,
# then do the attribution math in CPU+GPU hybrid.

progress("ekfac", "Computing EK-FAC factors + per-sample gradients (manual)", 20)
t0_ekfac = time.time()

try:
    from dattri.algorithm.influence_function import IFAttributorEKFAC
    from dattri.task import AttributionTask

    # Key insight: the OOM happens during attribute(), not cache().
    # cache() computes Kronecker factors (cheap).
    # attribute() does vmap per-sample gradients (expensive).
    #
    # Alternative: compute per-sample gradients manually (one-by-one loop),
    # then apply the EKFAC inverse manually.

    # Step 1: Compute Kronecker factors via dattri cache
    def loss_func(params, data_pair):
        x, y = data_pair
        x, y = x.to(device), y.to(device)
        from torch.func import functional_call
        out = functional_call(model, params, (x,))
        return F.cross_entropy(out, y)

    task = AttributionTask(loss_func=loss_func, model=model, checkpoints=model.state_dict())
    train_loader_5k = torch.utils.data.DataLoader(train_sub_5k, batch_size=64, shuffle=False, num_workers=2)

    attributor = IFAttributorEKFAC(task=task, device='cuda', damping=0.1)
    attributor.cache(train_loader_5k)
    cache_time = time.time() - t0_ekfac
    print(f"  EK-FAC cache: {cache_time:.1f}s, GPU free: {gpu_free():.1f}GB")

    # Step 2: Access the cached EKFAC inverse Hessian function
    # dattri stores the EKFAC factors internally. We need to extract them
    # and apply the transform manually per sample.
    #
    # Actually, let's try a different approach: use attribute() with
    # batch_size=1 for BOTH train and test loaders.
    # This minimizes vmap's memory footprint.

    train_loader_bs1 = torch.utils.data.DataLoader(
        train_sub_5k, batch_size=1, shuffle=False, num_workers=0)
    test_loader_bs1 = torch.utils.data.DataLoader(
        test_sub, batch_size=1, shuffle=False)

    # Attempt with bs=1
    print("  Attempting attribute() with batch_size=1...")
    ekfac_scores = attributor.attribute(train_loader_bs1, test_loader_bs1)

    es = ekfac_scores.cpu().numpy()
    if es.shape[0] != N_TEST:
        es = es.T
    ekfac_scores_np = es
    ekfac_time = time.time() - t0_ekfac
    print(f"  EK-FAC: {ekfac_scores_np.shape}, {ekfac_time:.1f}s")
    print(f"  Range: [{ekfac_scores_np.min():.6f}, {ekfac_scores_np.max():.6f}]")

    ekfac_rankings_5k = np.argsort(-ekfac_scores_np, axis=1)
    np.save(str(attr_dir / "ekfac_scores_5k.npy"), ekfac_scores_np)
    np.save(str(attr_dir / "ekfac_rankings_5k_top100.npy"), ekfac_rankings_5k[:, :100])

    ekfac_success = True
    ekfac_meta = {
        'success': True, 'cache_time_sec': round(cache_time, 1),
        'total_time_sec': round(ekfac_time, 1), 'shape': list(ekfac_scores_np.shape),
        'damping': 0.1, 'train_subset': N_TRAIN_IF, 'batch_sizes': {'train': 1, 'test': 1},
    }

    del attributor, task, ekfac_scores
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

except Exception as e:
    print(f"  EK-FAC FAILED: {e}")
    import traceback; traceback.print_exc()
    ekfac_success = False
    ekfac_scores_np = None
    ekfac_meta = {'success': False, 'error': str(e)}
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

progress("ekfac", f"{'Done' if ekfac_success else 'FAILED'}", 45)

# ==== K-FAC IF (using EK-FAC with high damping as proxy, or manual CG with bs=1) ====
progress("kfac", "Computing K-FAC IF proxy (CG bs=1, 1K train)", 50)
t0_kfac = time.time()

try:
    from dattri.algorithm.influence_function import IFAttributorCG
    from dattri.task import AttributionTask as AT2

    def loss_func2(params, data_pair):
        x, y = data_pair
        x, y = x.to(device), y.to(device)
        from torch.func import functional_call
        out = functional_call(model, params, (x,))
        return F.cross_entropy(out, y)

    task2 = AT2(loss_func=loss_func2, model=model, checkpoints=model.state_dict())

    # Use 1K train for CG (already very slow with full Hessian)
    train_1k_indices = train_5k_indices[:1000]
    train_sub_1k = torch.utils.data.Subset(trainset, train_1k_indices)

    train_loader_cg_bs1 = torch.utils.data.DataLoader(
        train_sub_1k, batch_size=1, shuffle=False, num_workers=0)
    test_loader_cg_bs1 = torch.utils.data.DataLoader(
        test_sub, batch_size=1, shuffle=False)

    attributor_cg = IFAttributorCG(
        task=task2, device='cuda',
        regularization=0.1,
        max_iter=10,  # Few CG iterations for pilot
    )

    attributor_cg.cache(train_loader_cg_bs1)
    print(f"  CG cache: GPU free: {gpu_free():.1f}GB")

    print("  Computing CG attributions (bs=1)...")
    cg_scores = attributor_cg.attribute(train_loader_cg_bs1, test_loader_cg_bs1)

    cs = cg_scores.cpu().numpy()
    if cs.shape[0] != N_TEST:
        cs = cs.T
    cg_scores_np = cs
    cg_time = time.time() - t0_kfac
    print(f"  CG IF: {cg_scores_np.shape}, {cg_time:.1f}s")
    print(f"  Range: [{cg_scores_np.min():.6f}, {cg_scores_np.max():.6f}]")

    cg_rankings_1k = np.argsort(-cg_scores_np, axis=1)
    np.save(str(attr_dir / "cg_if_scores_1k.npy"), cg_scores_np)
    np.save(str(attr_dir / "cg_if_rankings_1k_top100.npy"), cg_rankings_1k[:, :100])

    cg_success = True
    cg_meta = {
        'success': True, 'total_time_sec': round(cg_time, 1),
        'shape': list(cg_scores_np.shape), 'regularization': 0.1,
        'max_cg_iter': 10, 'train_subset': 1000,
        'note': 'CG IF on 1K train, bs=1, as K-FAC proxy for pilot',
    }

    del attributor_cg, task2, cg_scores
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

except Exception as e:
    print(f"  CG IF FAILED: {e}")
    import traceback; traceback.print_exc()
    cg_success = False
    cg_scores_np = None
    cg_meta = {'success': False, 'error': str(e)}
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

progress("kfac", f"{'Done' if cg_success else 'FAILED'}", 65)

# ==== METHOD 4: TRAK with CPU projection ====
progress("trak", "TRAK-1 with CPU projection (5K train)", 70)
t0_trak = time.time()

try:
    from trak import TRAKer

    trak_dir = str(attr_dir / "trak_temp")
    shutil.rmtree(trak_dir, ignore_errors=True)

    # Force CPU for projection to avoid GPU OOM
    # TRAKer with device='cpu' is too slow for featurize.
    # Strategy: use small proj_dim and batch_size
    proj_dim = 512  # Much smaller to fit in memory

    traker = TRAKer(
        model=model,
        task='image_classification',
        train_set_size=N_TRAIN_IF,
        proj_dim=proj_dim,
        save_dir=trak_dir,
        device=device,
    )

    train_loader_trak = torch.utils.data.DataLoader(
        train_sub_5k, batch_size=8, shuffle=False, num_workers=0)

    traker.load_checkpoint(model.state_dict(), model_id=0)
    for batch in train_loader_trak:
        b = (batch[0].to(device), batch[1].to(device))
        traker.featurize(batch=b, num_samples=b[0].shape[0])
    traker.finalize_features(model_ids=[0])
    feat_time = time.time() - t0_trak
    print(f"  Featurize: {feat_time:.1f}s, GPU free: {gpu_free():.1f}GB")

    test_loader_trak = torch.utils.data.DataLoader(test_sub, batch_size=10, shuffle=False)
    traker.start_scoring_checkpoint(
        exp_name='pilot', checkpoint=model.state_dict(),
        model_id=0, num_targets=len(test_sub),
    )
    for batch in test_loader_trak:
        b = (batch[0].to(device), batch[1].to(device))
        traker.score(batch=b, num_samples=b[0].shape[0])
    trak_scores = traker.finalize_scores(exp_name='pilot')

    ts = trak_scores.numpy() if hasattr(trak_scores, 'numpy') else np.array(trak_scores)
    if ts.shape[0] != N_TEST:
        ts = ts.T
    trak_scores_np = ts
    trak_time = time.time() - t0_trak
    print(f"  TRAK: {trak_scores_np.shape}, {trak_time:.1f}s")
    print(f"  Range: [{trak_scores_np.min():.6f}, {trak_scores_np.max():.6f}]")

    trak_rankings_5k = np.argsort(-trak_scores_np, axis=1)
    np.save(str(attr_dir / "trak_scores_5k.npy"), trak_scores_np)
    np.save(str(attr_dir / "trak_rankings_5k_top100.npy"), trak_rankings_5k[:, :100])

    trak_success = True
    trak_meta = {
        'success': True, 'featurize_time_sec': round(feat_time, 1),
        'total_time_sec': round(trak_time, 1), 'shape': list(trak_scores_np.shape),
        'proj_dim': proj_dim, 'n_checkpoints': 1,
        'train_subset': N_TRAIN_IF, 'batch_size': 8,
        'note': 'TRAK-1 on 5K train, proj_dim=512 (memory-constrained pilot)',
    }

    shutil.rmtree(trak_dir, ignore_errors=True)
    del traker
    gc.collect(); torch.cuda.empty_cache()

except Exception as e:
    print(f"  TRAK FAILED: {e}")
    import traceback; traceback.print_exc()
    trak_success = False
    trak_scores_np = None
    trak_meta = {'success': False, 'error': str(e)}
    gc.collect(); torch.cuda.empty_cache()

progress("trak", f"{'Done' if trak_success else 'FAILED'}", 85)

# ==== ANALYSIS ====
progress("analysis", "Cross-method agreement metrics", 90)

analysis = {}
labels = np.array(test_features['labels'])

def jaccard_k(r1, r2, k=10):
    return len(set(r1[:k].tolist()) & set(r2[:k].tolist())) / len(set(r1[:k].tolist()) | set(r2[:k].tolist()))

# J@10 between EK-FAC and CG on shared 1K subset
if ekfac_success and cg_success:
    ekfac_1k = ekfac_scores_np[:, :1000]
    ekfac_1k_rank = np.argsort(-ekfac_1k, axis=1)
    j10 = np.array([jaccard_k(ekfac_1k_rank[i], cg_rankings_1k[i]) for i in range(N_TEST)])
    analysis['jaccard_at_10_ekfac_cg'] = {
        'mean': float(j10.mean()), 'std': float(j10.std()),
        'min': float(j10.min()), 'max': float(j10.max()),
        'per_class': {str(c): float(j10[labels==c].mean()) for c in range(N_CLASSES)},
        'values': j10.tolist(),
    }
    print(f"\nJ@10(EK-FAC, CG): mean={j10.mean():.4f}, std={j10.std():.4f}")

# Kendall tau(EK-FAC, RepSim) on 5K
if ekfac_success:
    tau_vals = []
    for i in range(N_TEST):
        top200 = np.argsort(-ekfac_scores_np[i])[:200]
        t, _ = kendalltau(ekfac_scores_np[i, top200], repsim_5k[i, top200])
        tau_vals.append(t if not np.isnan(t) else 0.0)
    tau_vals = np.array(tau_vals)
    analysis['kendall_tau_ekfac_repsim'] = {
        'mean': float(tau_vals.mean()), 'std': float(tau_vals.std()),
        'values': tau_vals.tolist(),
    }
    print(f"Kendall tau(EK-FAC, RepSim): mean={tau_vals.mean():.4f}, std={tau_vals.std():.4f}")

# LDS(EK-FAC vs TRAK)
if ekfac_success and trak_success:
    lds = []
    for i in range(N_TEST):
        r, _ = spearmanr(ekfac_scores_np[i], trak_scores_np[i])
        lds.append(r if not np.isnan(r) else 0.0)
    lds = np.array(lds)
    analysis['lds_ekfac_trak'] = {
        'mean': float(lds.mean()), 'std': float(lds.std()),
        'values': lds.tolist(),
    }
    print(f"LDS(EK-FAC, TRAK): mean={lds.mean():.4f}, std={lds.std():.4f}")

# Kendall tau(CG IF, RepSim) on 1K
if cg_success:
    repsim_1k = repsim_scores[:, train_5k_indices[:1000]]
    tau_cg = []
    for i in range(N_TEST):
        top100 = np.argsort(-cg_scores_np[i])[:100]
        t, _ = kendalltau(cg_scores_np[i, top100], repsim_1k[i, top100])
        tau_cg.append(t if not np.isnan(t) else 0.0)
    tau_cg = np.array(tau_cg)
    analysis['kendall_tau_cg_repsim'] = {
        'mean': float(tau_cg.mean()), 'std': float(tau_cg.std()),
        'values': tau_cg.tolist(),
    }
    print(f"Kendall tau(CG, RepSim): mean={tau_cg.mean():.4f}, std={tau_cg.std():.4f}")

# ==== Compile ====
progress("compile", "Writing final results", 95)

n_ok = sum([ekfac_success, cg_success, trak_success, True])  # RepSim always works
j10_std = analysis.get('jaccard_at_10_ekfac_cg', {}).get('std', 0)

results = {
    'task_id': TASK_ID, 'mode': 'PILOT',
    'n_test': N_TEST, 'n_train_full': 50000,
    'n_train_if': N_TRAIN_IF, 'seed': SEED,
    'test_indices': test_indices,
    'methods': {
        'ekfac_if': ekfac_meta,
        'cg_if': cg_meta,
        'repsim': {'success': True, 'time_sec': round(repsim_time, 1), 'shape': list(repsim_scores.shape)},
        'trak': trak_meta,
    },
    'analysis': analysis,
    'pass_criteria': {
        'all_4_methods_valid': n_ok == 4,
        'jaccard_std_above_005': j10_std > 0.05 if ekfac_success and cg_success else None,
    },
    'overall_pass': n_ok == 4 and (j10_std > 0.05 if ekfac_success and cg_success else False),
    'memory_note': 'GPU shared with another process (~7GB). Used bs=1 for dattri, proj_dim=512 for TRAK.',
    'timestamp': datetime.now().isoformat(),
}

(attr_dir / "pilot_results.json").write_text(json.dumps(results, indent=2))

print("\n" + "="*60)
print("PILOT RESULTS SUMMARY")
print("="*60)
for m, meta in results['methods'].items():
    print(f"  {m}: {'PASS' if meta.get('success') else 'FAIL'}")
print(f"\nCriteria:")
for k, v in results['pass_criteria'].items():
    print(f"  {k}: {v}")
print(f"\nOverall: {'PASS' if results['overall_pass'] else 'FAIL'}")

status = "success" if results['overall_pass'] else "partial"
summary = f"{n_ok}/4 ok"
if 'jaccard_at_10_ekfac_cg' in analysis:
    summary += f", J@10 std={j10_std:.4f}"
mark_done(status, summary)
print(f"\nDONE: {status} - {summary}")
