"""
Phase 1 Attribution Computation - PILOT MODE v2
100 test points (10 per class), full 50K training set
4 methods: EK-FAC IF, CG IF, RepSim, TRAK-1

Memory-optimized: only ~17GB available (another process uses ~7GB on the GPU).
Strategy: very small batch sizes for per-sample gradient methods.
"""
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json, time, gc, sys, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from datetime import datetime

# ---- Constants ----
PROJECT_DIR = Path("/home/jinxulin/sibyl_system/projects/AURA")
RESULTS_DIR = PROJECT_DIR / "exp" / "results"
CKPT_DIR = PROJECT_DIR / "exp" / "checkpoints"
DATA_DIR = Path("/home/jinxulin/sibyl_system/shared/datasets/cifar10")
TASK_ID = "phase1_attribution_compute"
SEED = 42
N_TEST = 100  # 10 per class
N_CLASSES = 10

# Aggressive memory management
EKFAC_TRAIN_BS = 8     # Very small for per-sample gradients via vmap
EKFAC_TEST_BS = 2       # Process 2 test points at a time
CG_TRAIN_BS = 8
CG_TEST_BS = 1          # CG is very memory-hungry
TRAK_BS = 32            # Smaller batch for TRAK featurize
TRAK_PROJ_DIM = 2048    # Smaller projection dimension

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "phase1_attributions").mkdir(exist_ok=True)

device = torch.device("cuda")
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---- PID file ----
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))

def report_progress(stage, detail="", pct=0):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID, "stage": stage, "detail": detail,
        "percent": pct, "updated_at": datetime.now().isoformat(),
    }))
    print(f"[PROGRESS] {stage}: {detail} ({pct}%)")

def mark_done(status="success", summary=""):
    pf = RESULTS_DIR / f"{TASK_ID}.pid"
    if pf.exists(): pf.unlink()
    prog = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    fp = {}
    if prog.exists():
        try: fp = json.loads(prog.read_text())
        except: pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": fp, "timestamp": datetime.now().isoformat(),
    }))

def gpu_mem():
    return torch.cuda.memory_allocated() / 1e9, torch.cuda.max_memory_allocated() / 1e9

report_progress("init", "Loading model and data")

# ---- Model ----
from torchvision.models import resnet18

def make_model():
    m = resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m

model = make_model()
ckpt = torch.load(str(CKPT_DIR / "resnet18_cifar10_seed42.pt"),
                  map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device).eval()
print(f"Model loaded: {ckpt['test_acc']:.2f}% test acc, GPU mem: {gpu_mem()[0]:.2f}GB")

# ---- Data ----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=False, transform=transform)

# Stratified test selection
test_indices_by_class = {c: [] for c in range(N_CLASSES)}
for i in range(len(testset)):
    _, label = testset[i]
    if len(test_indices_by_class[label]) < N_TEST // N_CLASSES:
        test_indices_by_class[label].append(i)
    if all(len(v) >= N_TEST // N_CLASSES for v in test_indices_by_class.values()):
        break

test_indices = []
for c in range(N_CLASSES):
    test_indices.extend(test_indices_by_class[c])
test_indices = sorted(test_indices)
test_sub = torch.utils.data.Subset(testset, test_indices)
test_labels = torch.tensor([testset[i][1] for i in test_indices])
print(f"Selected {len(test_indices)} test points")

# Use 5K train subset for IF methods (stratified 500/class) to keep memory manageable
train_5k_indices = []
for c in range(N_CLASSES):
    class_indices = [i for i in range(len(trainset)) if trainset.targets[i] == c][:500]
    train_5k_indices.extend(class_indices)
train_sub_5k = torch.utils.data.Subset(trainset, train_5k_indices)
print(f"Using {len(train_5k_indices)} train points for IF methods (5K subset)")

# ---- Per-test-point features ----
report_progress("features", "Computing gradient norms, confidence, entropy", 5)

test_features = {
    'indices': test_indices, 'labels': test_labels.tolist(),
    'gradient_norms': [], 'confidences': [], 'entropies': [], 'predictions': [],
}

for idx in test_indices:
    x, y = testset[idx]
    x = x.unsqueeze(0).to(device)
    y_t = torch.tensor([y]).to(device)
    model.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits, y_t)
    loss.backward()
    gn = sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
    probs = F.softmax(logits, dim=1).detach()
    test_features['gradient_norms'].append(gn)
    test_features['confidences'].append(probs[0, y].item())
    test_features['entropies'].append(-(probs * probs.clamp(min=1e-8).log()).sum().item())
    test_features['predictions'].append(probs.argmax(dim=1).item())

model.zero_grad(); torch.cuda.empty_cache(); gc.collect()
print(f"Grad norms: mean={np.mean(test_features['gradient_norms']):.4f}, std={np.std(test_features['gradient_norms']):.4f}")

feat_path = RESULTS_DIR / "phase1_attributions" / "test_features.json"
feat_path.write_text(json.dumps(test_features, indent=2))

# ==== METHOD 1: RepSim ====
report_progress("repsim", "Computing RepSim attributions", 10)
t0 = time.time()

activation = {}
def hook_fn(m, inp, out): activation['feat'] = out.detach()
hook = model.avgpool.register_forward_hook(hook_fn)

# Full 50K train features
train_feats = []
with torch.no_grad():
    for i in range(0, len(trainset), 256):
        batch_indices = list(range(i, min(i+256, len(trainset))))
        batch_x = torch.stack([trainset[j][0] for j in batch_indices]).to(device)
        model(batch_x)
        train_feats.append(activation['feat'].squeeze().cpu())
        del batch_x
train_feats = torch.cat(train_feats, dim=0)
train_feats_norm = F.normalize(train_feats.float(), dim=1)

test_feats = []
with torch.no_grad():
    for i in range(0, len(test_sub), 50):
        batch_indices = list(range(i, min(i+50, len(test_sub))))
        batch_x = torch.stack([test_sub[j][0] for j in batch_indices]).to(device)
        model(batch_x)
        test_feats.append(activation['feat'].squeeze().cpu())
        del batch_x
test_feats = torch.cat(test_feats, dim=0)
test_feats_norm = F.normalize(test_feats.float(), dim=1)
hook.remove()

repsim_scores = (test_feats_norm @ train_feats_norm.T).numpy()  # [100, 50000]
repsim_time = time.time() - t0
print(f"RepSim: {repsim_scores.shape}, time={repsim_time:.1f}s, range=[{repsim_scores.min():.4f}, {repsim_scores.max():.4f}]")

# Save
repsim_rankings = np.argsort(-repsim_scores, axis=1)[:, :100]
np.save(str(RESULTS_DIR / "phase1_attributions" / "repsim_scores_full.npy"), repsim_scores)
np.save(str(RESULTS_DIR / "phase1_attributions" / "repsim_rankings_top100.npy"), repsim_rankings)

# Also save 5K subset scores for IF comparison
repsim_5k = repsim_scores[:, train_5k_indices]  # [100, 5000]
np.save(str(RESULTS_DIR / "phase1_attributions" / "repsim_scores_5k.npy"), repsim_5k)

del train_feats, test_feats, train_feats_norm, test_feats_norm
gc.collect(); torch.cuda.empty_cache()
report_progress("repsim", f"Done in {repsim_time:.0f}s", 15)

# ==== METHOD 2: EK-FAC IF on 5K train subset ====
report_progress("ekfac", "Computing EK-FAC IF (5K train, batch=8/2)", 20)
t0 = time.time()

try:
    from dattri.algorithm.influence_function import IFAttributorEKFAC
    from dattri.task import AttributionTask

    def loss_func(params, data_target_pair):
        x, y = data_target_pair
        x, y = x.to(device), y.to(device)
        from torch.func import functional_call
        output = functional_call(model, params, (x,))
        return F.cross_entropy(output, y)

    task = AttributionTask(loss_func=loss_func, model=model, checkpoints=model.state_dict())

    train_loader_ekfac = torch.utils.data.DataLoader(
        train_sub_5k, batch_size=EKFAC_TRAIN_BS, shuffle=False, num_workers=2)
    test_loader_ekfac = torch.utils.data.DataLoader(
        test_sub, batch_size=EKFAC_TEST_BS, shuffle=False)

    attributor = IFAttributorEKFAC(task=task, device='cuda', damping=0.1)

    print("  Caching EK-FAC factors on 5K train...")
    attributor.cache(train_loader_ekfac)
    cache_time = time.time() - t0
    print(f"  Cache: {cache_time:.1f}s, GPU: {gpu_mem()}")

    print("  Computing attributions...")
    t1 = time.time()
    ekfac_scores = attributor.attribute(train_loader_ekfac, test_loader_ekfac)
    attr_time = time.time() - t1

    # Ensure shape is [N_TEST, 5000]
    es = ekfac_scores.cpu().numpy()
    if es.shape[0] != N_TEST:
        es = es.T
    ekfac_scores_np = es

    total_time = time.time() - t0
    print(f"  EK-FAC: {ekfac_scores_np.shape}, time={total_time:.1f}s, range=[{ekfac_scores_np.min():.6f}, {ekfac_scores_np.max():.6f}]")
    print(f"  GPU peak: {gpu_mem()[1]:.2f}GB")

    ekfac_rankings = np.argsort(-ekfac_scores_np, axis=1)[:, :100]
    np.save(str(RESULTS_DIR / "phase1_attributions" / "ekfac_scores_5k.npy"), ekfac_scores_np)
    np.save(str(RESULTS_DIR / "phase1_attributions" / "ekfac_rankings_top100.npy"), ekfac_rankings)

    ekfac_success = True
    ekfac_meta = {
        'success': True, 'cache_time_sec': round(cache_time, 1),
        'attr_time_sec': round(attr_time, 1), 'total_time_sec': round(total_time, 1),
        'shape': list(ekfac_scores_np.shape), 'damping': 0.1,
        'gpu_peak_gb': round(gpu_mem()[1], 2),
        'train_subset_size': len(train_5k_indices),
        'batch_sizes': {'train': EKFAC_TRAIN_BS, 'test': EKFAC_TEST_BS},
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

report_progress("ekfac", f"{'Done' if ekfac_success else 'FAILED'}", 45)

# ==== METHOD 3: CG IF on 5K train ====
report_progress("cg_if", "Computing CG IF (5K train, batch=8/1)", 50)
t0 = time.time()

try:
    from dattri.algorithm.influence_function import IFAttributorCG
    from dattri.task import AttributionTask as AT2

    def loss_func2(params, data_target_pair):
        x, y = data_target_pair
        x, y = x.to(device), y.to(device)
        from torch.func import functional_call
        output = functional_call(model, params, (x,))
        return F.cross_entropy(output, y)

    task2 = AT2(loss_func=loss_func2, model=model, checkpoints=model.state_dict())

    # CG on very small subset: 1000 train to make it tractable
    train_1k_indices = train_5k_indices[:1000]
    train_sub_1k = torch.utils.data.Subset(trainset, train_1k_indices)

    train_loader_cg = torch.utils.data.DataLoader(
        train_sub_1k, batch_size=CG_TRAIN_BS, shuffle=False, num_workers=2)
    test_loader_cg = torch.utils.data.DataLoader(
        test_sub, batch_size=CG_TEST_BS, shuffle=False)

    attributor_cg = IFAttributorCG(
        task=task2, device='cuda',
        regularization=0.1,
        max_iter=20,  # Fewer CG iterations for pilot
    )

    print("  Caching CG (1K train)...")
    attributor_cg.cache(train_loader_cg)
    cache_time = time.time() - t0
    print(f"  Cache: {cache_time:.1f}s, GPU: {gpu_mem()}")

    print("  Computing CG attributions...")
    t1 = time.time()
    cg_scores = attributor_cg.attribute(train_loader_cg, test_loader_cg)
    attr_time = time.time() - t1

    cs = cg_scores.cpu().numpy()
    if cs.shape[0] != N_TEST:
        cs = cs.T
    cg_scores_np = cs

    total_time = time.time() - t0
    print(f"  CG IF: {cg_scores_np.shape}, time={total_time:.1f}s, range=[{cg_scores_np.min():.6f}, {cg_scores_np.max():.6f}]")
    print(f"  GPU peak: {gpu_mem()[1]:.2f}GB")

    cg_rankings = np.argsort(-cg_scores_np, axis=1)[:, :100]
    np.save(str(RESULTS_DIR / "phase1_attributions" / "cg_if_scores_1k.npy"), cg_scores_np)
    np.save(str(RESULTS_DIR / "phase1_attributions" / "cg_if_rankings_top100.npy"), cg_rankings)

    cg_success = True
    cg_meta = {
        'success': True, 'cache_time_sec': round(cache_time, 1),
        'attr_time_sec': round(attr_time, 1), 'total_time_sec': round(total_time, 1),
        'shape': list(cg_scores_np.shape), 'regularization': 0.1, 'max_cg_iter': 20,
        'gpu_peak_gb': round(gpu_mem()[1], 2),
        'train_subset_size': 1000,
        'batch_sizes': {'train': CG_TRAIN_BS, 'test': CG_TEST_BS},
        'note': 'CG IF on 1K train subset for pilot feasibility',
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

report_progress("cg_if", f"{'Done' if cg_success else 'FAILED'}", 65)

# ==== METHOD 4: TRAK (single checkpoint, smaller proj_dim) ====
report_progress("trak", "Computing TRAK (1 ckpt, proj_dim=2048, batch=32)", 70)
t0 = time.time()

try:
    from trak import TRAKer

    trak_save_dir = str(RESULTS_DIR / "trak_pilot")
    shutil.rmtree(trak_save_dir, ignore_errors=True)

    # Use 5K train subset for TRAK too (consistent with IF methods)
    traker = TRAKer(
        model=model,
        task='image_classification',
        train_set_size=len(train_sub_5k),
        proj_dim=TRAK_PROJ_DIM,
        save_dir=trak_save_dir,
        device=device,
    )

    train_loader_trak = torch.utils.data.DataLoader(
        train_sub_5k, batch_size=TRAK_BS, shuffle=False, num_workers=2)

    traker.load_checkpoint(model.state_dict(), model_id=0)
    for batch in train_loader_trak:
        b = (batch[0].to(device), batch[1].to(device))
        traker.featurize(batch=b, num_samples=b[0].shape[0])
    traker.finalize_features(model_ids=[0])
    feat_time = time.time() - t0
    print(f"  Featurize: {feat_time:.1f}s, GPU: {gpu_mem()}")

    test_loader_trak = torch.utils.data.DataLoader(test_sub, batch_size=50, shuffle=False)
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

    total_time = time.time() - t0
    print(f"  TRAK: {trak_scores_np.shape}, time={total_time:.1f}s, range=[{trak_scores_np.min():.6f}, {trak_scores_np.max():.6f}]")
    print(f"  GPU peak: {gpu_mem()[1]:.2f}GB")

    trak_rankings = np.argsort(-trak_scores_np, axis=1)[:, :100]
    np.save(str(RESULTS_DIR / "phase1_attributions" / "trak_scores_5k.npy"), trak_scores_np)
    np.save(str(RESULTS_DIR / "phase1_attributions" / "trak_rankings_top100.npy"), trak_rankings)

    trak_success = True
    trak_meta = {
        'success': True, 'featurize_time_sec': round(feat_time, 1),
        'total_time_sec': round(total_time, 1), 'shape': list(trak_scores_np.shape),
        'proj_dim': TRAK_PROJ_DIM, 'n_checkpoints': 1,
        'gpu_peak_gb': round(gpu_mem()[1], 2),
        'train_subset_size': len(train_5k_indices),
        'batch_size': TRAK_BS,
        'note': 'TRAK-1 on 5K train, pilot proxy for TRAK-50',
    }

    shutil.rmtree(trak_save_dir, ignore_errors=True)
    del traker
    gc.collect(); torch.cuda.empty_cache()

except Exception as e:
    print(f"  TRAK FAILED: {e}")
    import traceback; traceback.print_exc()
    trak_success = False
    trak_scores_np = None
    trak_meta = {'success': False, 'error': str(e)}
    gc.collect(); torch.cuda.empty_cache()

report_progress("trak", f"{'Done' if trak_success else 'FAILED'}", 85)

# ==== ANALYSIS ====
report_progress("analysis", "Computing cross-method agreement", 90)

analysis = {}

def jaccard_at_k(r1, r2, k=10):
    s1 = set(r1[:k].tolist())
    s2 = set(r2[:k].tolist())
    return len(s1 & s2) / len(s1 | s2)

# Jaccard@10(EK-FAC, CG)
# CG is on 1K subset, EK-FAC on 5K. Compare on the shared 1K.
if ekfac_success and cg_success:
    # Map CG 1K indices to EK-FAC 5K indices
    # CG uses train_1k_indices = train_5k_indices[:1000]
    # So CG rank i corresponds to EK-FAC column i (first 1000 of 5K)
    ekfac_1k = ekfac_scores_np[:, :1000]  # First 1000 of 5K = same as CG 1K
    ekfac_1k_rankings = np.argsort(-ekfac_1k, axis=1)

    j10_values = np.array([jaccard_at_k(ekfac_1k_rankings[i], cg_rankings[i], k=10) for i in range(N_TEST)])

    analysis['jaccard_at_10_ekfac_cg'] = {
        'mean': float(j10_values.mean()), 'std': float(j10_values.std()),
        'min': float(j10_values.min()), 'max': float(j10_values.max()),
        'per_class_mean': {str(c): float(j10_values[np.array(test_features['labels'])==c].mean()) for c in range(N_CLASSES)},
        'values': j10_values.tolist(),
        'note': 'Computed on shared 1K train subset',
    }
    print(f"\nJ@10(EK-FAC, CG IF): mean={j10_values.mean():.4f}, std={j10_values.std():.4f}")
    print(f"  Pass (std>0.05): {'PASS' if j10_values.std() > 0.05 else 'FAIL'}")

# Kendall tau(IF, RepSim) on 5K subset
if ekfac_success:
    from scipy.stats import kendalltau, spearmanr
    repsim_5k_scores = repsim_5k  # Already saved above

    tau_values = []
    for i in range(N_TEST):
        # Use top-200 by EK-FAC for efficiency
        top200 = np.argsort(-ekfac_scores_np[i])[:200]
        tau, _ = kendalltau(ekfac_scores_np[i][top200], repsim_5k_scores[i][top200])
        tau_values.append(tau if not np.isnan(tau) else 0.0)
    tau_values = np.array(tau_values)

    analysis['kendall_tau_ekfac_repsim'] = {
        'mean': float(tau_values.mean()), 'std': float(tau_values.std()),
        'values': tau_values.tolist(),
        'note': 'Kendall tau on top-200 by EK-FAC, 5K train subset',
    }
    print(f"Kendall tau(EK-FAC, RepSim): mean={tau_values.mean():.4f}, std={tau_values.std():.4f}")

# LDS(EK-FAC vs TRAK)
if ekfac_success and trak_success:
    lds_values = []
    for i in range(N_TEST):
        rho, _ = spearmanr(ekfac_scores_np[i], trak_scores_np[i])
        lds_values.append(rho if not np.isnan(rho) else 0.0)
    lds_values = np.array(lds_values)

    analysis['lds_ekfac_trak'] = {
        'mean': float(lds_values.mean()), 'std': float(lds_values.std()),
        'values': lds_values.tolist(),
        'note': 'LDS on 5K shared subset, TRAK-1 proxy',
    }
    print(f"LDS(EK-FAC vs TRAK-1): mean={lds_values.mean():.4f}, std={lds_values.std():.4f}")

# ==== Compile ====
report_progress("compile", "Final results", 95)

results = {
    'task_id': TASK_ID, 'mode': 'PILOT',
    'n_test': N_TEST, 'n_train_full': len(trainset),
    'n_train_if': len(train_5k_indices), 'n_train_cg': 1000,
    'seed': SEED, 'test_indices': test_indices,
    'methods': {
        'ekfac_if': ekfac_meta,
        'cg_if': cg_meta,
        'repsim': {'success': True, 'time_sec': round(repsim_time, 1), 'shape': list(repsim_scores.shape)},
        'trak': trak_meta,
    },
    'analysis': analysis,
    'pass_criteria': {
        'all_4_methods_valid': ekfac_success and cg_success and trak_success,
        'jaccard_std_above_005': analysis.get('jaccard_at_10_ekfac_cg', {}).get('std', 0) > 0.05,
    },
    'overall_pass': (
        ekfac_success and cg_success and trak_success and
        analysis.get('jaccard_at_10_ekfac_cg', {}).get('std', 0) > 0.05
    ),
    'memory_note': 'Another process uses ~7GB on GPU; all methods run with reduced batch sizes and train subsets',
    'timestamp': datetime.now().isoformat(),
    'gpu_info': {'device': torch.cuda.get_device_name(0)},
}

out_path = RESULTS_DIR / "phase1_attributions" / "pilot_results.json"
out_path.write_text(json.dumps(results, indent=2))

print("\n" + "=" * 60)
print("PILOT RESULTS SUMMARY")
print("=" * 60)
for m, meta in results['methods'].items():
    print(f"  {m}: {'PASS' if meta.get('success') else 'FAIL'}")
print(f"\nPass criteria:")
for k, v in results['pass_criteria'].items():
    print(f"  {k}: {'PASS' if v else 'FAIL'}")
print(f"\nOverall: {'PASS' if results['overall_pass'] else 'FAIL'}")

status = "success" if results['overall_pass'] else "partial"
summary = f"{sum(1 for m in results['methods'].values() if m.get('success'))}/4 methods ok"
if 'jaccard_at_10_ekfac_cg' in analysis:
    summary += f", J@10 std={analysis['jaccard_at_10_ekfac_cg']['std']:.4f}"
mark_done(status=status, summary=summary)
print(f"\nDONE: {status}")
