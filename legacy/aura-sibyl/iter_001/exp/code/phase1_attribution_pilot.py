"""
Phase 1 Attribution Computation - PILOT MODE
100 test points (10 per class), full 50K training set
4 methods: EK-FAC IF, CG IF (K-FAC substitute), RepSim, TRAK-1

Task: phase1_attribution_compute
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
N_TEST = 100  # PILOT: 10 per class
N_CLASSES = 10

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
        "task_id": TASK_ID,
        "stage": stage,
        "detail": detail,
        "percent": pct,
        "updated_at": datetime.now().isoformat(),
    }))
    print(f"[PROGRESS] {stage}: {detail} ({pct}%)")

def mark_done(status="success", summary=""):
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except:
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))

report_progress("init", "Loading model and data")

# ---- Load model ----
from torchvision.models import resnet18

def make_model():
    model = resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

model = make_model()
ckpt = torch.load(str(CKPT_DIR / "resnet18_cifar10_seed42.pt"),
                  map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device).eval()
print(f"Model loaded: {ckpt['test_acc']:.2f}% test acc")

# ---- Load data ----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=False, transform=transform)

# ---- Stratified test point selection: 10 per class ----
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
print(f"Selected {len(test_indices)} test points: {[len(test_indices_by_class[c]) for c in range(N_CLASSES)]} per class")

test_sub = torch.utils.data.Subset(testset, test_indices)
test_labels = torch.tensor([testset[i][1] for i in test_indices])

# ---- Compute per-test-point features: gradient norm, confidence, entropy ----
report_progress("features", "Computing per-test-point gradient norms, confidence, entropy", 5)

test_features = {
    'indices': test_indices,
    'labels': test_labels.tolist(),
    'gradient_norms': [],
    'confidences': [],
    'entropies': [],
    'predictions': [],
}

for i, idx in enumerate(test_indices):
    x, y = testset[idx]
    x = x.unsqueeze(0).to(device)
    y_tensor = torch.tensor([y]).to(device)

    model.zero_grad()
    x.requires_grad_(False)
    logits = model(x)
    loss = F.cross_entropy(logits, y_tensor)
    loss.backward()

    # Gradient norm (all parameters)
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5

    # Confidence and entropy
    probs = F.softmax(logits, dim=1).detach()
    confidence = probs[0, y].item()
    entropy = -(probs * probs.log()).sum().item()
    pred = probs.argmax(dim=1).item()

    test_features['gradient_norms'].append(grad_norm)
    test_features['confidences'].append(confidence)
    test_features['entropies'].append(entropy)
    test_features['predictions'].append(pred)

model.zero_grad()
torch.cuda.empty_cache()
gc.collect()

print(f"Gradient norms: mean={np.mean(test_features['gradient_norms']):.4f}, std={np.std(test_features['gradient_norms']):.4f}")
print(f"Confidence: mean={np.mean(test_features['confidences']):.4f}")
print(f"Entropy: mean={np.mean(test_features['entropies']):.4f}")

# Save features
feat_path = RESULTS_DIR / "phase1_attributions" / "test_features.json"
feat_path.write_text(json.dumps(test_features, indent=2))

# ==== METHOD 1: RepSim (cheapest, do first) ====
report_progress("repsim", "Computing RepSim attributions", 10)
t0 = time.time()

activation = {}
def hook_fn(m, inp, out):
    activation['feat'] = out.detach()

hook = model.avgpool.register_forward_hook(hook_fn)

# Extract all training features in chunks
train_feats = []
train_loader_full = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=False, num_workers=4)
with torch.no_grad():
    for batch_x, batch_y in train_loader_full:
        model(batch_x.to(device))
        train_feats.append(activation['feat'].squeeze().cpu())
train_feats = torch.cat(train_feats, dim=0)  # [50000, 512]
train_feats_norm = F.normalize(train_feats, dim=1)
print(f"Train features: {train_feats.shape}")

# Extract test features
test_feats = []
test_loader = torch.utils.data.DataLoader(test_sub, batch_size=50, shuffle=False)
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        model(batch_x.to(device))
        test_feats.append(activation['feat'].squeeze().cpu())
test_feats = torch.cat(test_feats, dim=0)  # [100, 512]
test_feats_norm = F.normalize(test_feats, dim=1)

hook.remove()

# Compute cosine similarity: [100, 50000]
repsim_scores = (test_feats_norm @ train_feats_norm.T).numpy()
repsim_time = time.time() - t0
print(f"RepSim: shape={repsim_scores.shape}, time={repsim_time:.1f}s")
print(f"RepSim range: [{repsim_scores.min():.4f}, {repsim_scores.max():.4f}]")

# Save RepSim rankings (top-100 per test point to save space)
repsim_rankings = np.argsort(-repsim_scores, axis=1)[:, :100]
np.save(str(RESULTS_DIR / "phase1_attributions" / "repsim_scores_top100.npy"), repsim_scores[np.arange(100)[:, None], repsim_rankings])
np.save(str(RESULTS_DIR / "phase1_attributions" / "repsim_rankings_top100.npy"), repsim_rankings)
# Also save full scores for LDS computation later
np.save(str(RESULTS_DIR / "phase1_attributions" / "repsim_scores_full.npy"), repsim_scores)

del train_feats, test_feats, train_feats_norm, test_feats_norm
gc.collect(); torch.cuda.empty_cache()
report_progress("repsim", f"Done in {repsim_time:.0f}s", 20)

# ==== METHOD 2: EK-FAC IF (dattri) ====
report_progress("ekfac", "Computing EK-FAC IF attributions", 25)
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

    # Use small batch for per-sample gradient computation to avoid OOM
    # Train loader with small batch for caching
    train_loader_ekfac = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=4)
    test_loader_ekfac = torch.utils.data.DataLoader(test_sub, batch_size=10, shuffle=False)

    attributor = IFAttributorEKFAC(task=task, device='cuda', damping=0.1)

    # Cache Kronecker factors on full training set
    print("  Caching EK-FAC factors on full training set...")
    attributor.cache(train_loader_ekfac)
    cache_time = time.time() - t0
    print(f"  EK-FAC cache: {cache_time:.1f}s, GPU: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

    # Compute attributions in chunks (100 test x 50000 train is large)
    # dattri returns [n_train, n_test] attribution matrix
    print("  Computing attributions (chunked)...")
    t1 = time.time()
    ekfac_scores = attributor.attribute(train_loader_ekfac, test_loader_ekfac)
    attr_time = time.time() - t1

    # ekfac_scores shape: [n_train, n_test] -> transpose to [n_test, n_train]
    if ekfac_scores.shape[0] != N_TEST:
        ekfac_scores = ekfac_scores.T
    ekfac_scores_np = ekfac_scores.cpu().numpy()

    total_ekfac_time = time.time() - t0
    print(f"  EK-FAC: shape={ekfac_scores_np.shape}, time={total_ekfac_time:.1f}s")
    print(f"  EK-FAC range: [{ekfac_scores_np.min():.6f}, {ekfac_scores_np.max():.6f}], std={ekfac_scores_np.std():.6f}")
    print(f"  GPU peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

    # Save
    ekfac_rankings = np.argsort(-ekfac_scores_np, axis=1)[:, :100]
    np.save(str(RESULTS_DIR / "phase1_attributions" / "ekfac_scores_full.npy"), ekfac_scores_np)
    np.save(str(RESULTS_DIR / "phase1_attributions" / "ekfac_rankings_top100.npy"), ekfac_rankings)

    ekfac_success = True
    ekfac_meta = {
        'success': True, 'cache_time_sec': round(cache_time, 1),
        'attr_time_sec': round(attr_time, 1), 'total_time_sec': round(total_ekfac_time, 1),
        'shape': list(ekfac_scores_np.shape),
        'gpu_peak_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2),
        'damping': 0.1,
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

report_progress("ekfac", f"{'Done' if ekfac_success else 'FAILED'}", 45)

# ==== METHOD 3: CG IF (as K-FAC substitute) ====
report_progress("cg_if", "Computing CG IF attributions (K-FAC substitute)", 50)
t0 = time.time()

try:
    from dattri.algorithm.influence_function import IFAttributorCG
    from dattri.task import AttributionTask

    def loss_func2(params, data_target_pair):
        x, y = data_target_pair
        x, y = x.to(device), y.to(device)
        from torch.func import functional_call
        output = functional_call(model, params, (x,))
        return F.cross_entropy(output, y)

    task2 = AttributionTask(loss_func=loss_func2, model=model, checkpoints=model.state_dict())

    # CG uses conjugate gradient to solve H^-1 * v, no Kronecker factorization
    # More accurate but slower. Use smaller train subset for pilot feasibility.
    # For pilot: use 5000 train samples (10% subset) to make CG tractable
    train_sub_5k_indices = []
    for c in range(N_CLASSES):
        class_indices = [i for i in range(len(trainset)) if trainset.targets[i] == c]
        train_sub_5k_indices.extend(class_indices[:500])
    train_sub_5k = torch.utils.data.Subset(trainset, train_sub_5k_indices)

    train_loader_cg = torch.utils.data.DataLoader(train_sub_5k, batch_size=64, shuffle=False, num_workers=4)
    test_loader_cg = torch.utils.data.DataLoader(test_sub, batch_size=10, shuffle=False)

    attributor_cg = IFAttributorCG(
        task=task2, device='cuda',
        regularization=0.1,  # damping
        max_iter=50,  # CG iterations
    )

    print("  Caching CG Hessian info...")
    attributor_cg.cache(train_loader_cg)
    cache_time = time.time() - t0
    print(f"  CG cache: {cache_time:.1f}s, GPU: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

    print("  Computing CG attributions...")
    t1 = time.time()
    cg_scores = attributor_cg.attribute(train_loader_cg, test_loader_cg)
    attr_time = time.time() - t1

    if cg_scores.shape[0] != N_TEST:
        cg_scores = cg_scores.T
    cg_scores_np = cg_scores.cpu().numpy()

    total_cg_time = time.time() - t0
    print(f"  CG IF: shape={cg_scores_np.shape}, time={total_cg_time:.1f}s")
    print(f"  CG IF range: [{cg_scores_np.min():.6f}, {cg_scores_np.max():.6f}], std={cg_scores_np.std():.6f}")
    print(f"  GPU peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

    # Save
    cg_rankings = np.argsort(-cg_scores_np, axis=1)[:, :100]
    np.save(str(RESULTS_DIR / "phase1_attributions" / "cg_if_scores_full.npy"), cg_scores_np)
    np.save(str(RESULTS_DIR / "phase1_attributions" / "cg_if_rankings_top100.npy"), cg_rankings)

    cg_success = True
    cg_meta = {
        'success': True, 'cache_time_sec': round(cache_time, 1),
        'attr_time_sec': round(attr_time, 1), 'total_time_sec': round(total_cg_time, 1),
        'shape': list(cg_scores_np.shape),
        'gpu_peak_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2),
        'regularization': 0.1, 'max_cg_iter': 50,
        'train_subset_size': len(train_sub_5k_indices),
        'note': 'CG IF on 5K train subset (K-FAC substitute for pilot)',
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

report_progress("cg_if", f"{'Done' if cg_success else 'FAILED'}", 65)

# ==== METHOD 4: TRAK (single checkpoint for pilot) ====
report_progress("trak", "Computing TRAK attributions (single checkpoint)", 70)
t0 = time.time()

try:
    from trak import TRAKer

    trak_save_dir = str(RESULTS_DIR / "trak_pilot")
    shutil.rmtree(trak_save_dir, ignore_errors=True)

    traker = TRAKer(
        model=model,
        task='image_classification',
        train_set_size=len(trainset),
        proj_dim=4096,
        save_dir=trak_save_dir,
        device=device,
    )

    # Featurize training set with single checkpoint
    train_loader_trak = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=4)

    traker.load_checkpoint(model.state_dict(), model_id=0)
    for batch in train_loader_trak:
        b = (batch[0].to(device), batch[1].to(device))
        traker.featurize(batch=b, num_samples=b[0].shape[0])
    traker.finalize_features(model_ids=[0])
    feat_time = time.time() - t0
    print(f"  TRAK featurize: {feat_time:.1f}s, GPU: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

    # Score test points
    test_loader_trak = torch.utils.data.DataLoader(test_sub, batch_size=50, shuffle=False)
    traker.start_scoring_checkpoint(
        exp_name='pilot',
        checkpoint=model.state_dict(),
        model_id=0,
        num_targets=len(test_sub),
    )
    for batch in test_loader_trak:
        b = (batch[0].to(device), batch[1].to(device))
        traker.score(batch=b, num_samples=b[0].shape[0])
    trak_scores = traker.finalize_scores(exp_name='pilot')

    # trak_scores shape: [n_test, n_train]
    trak_scores_np = trak_scores.numpy() if hasattr(trak_scores, 'numpy') else np.array(trak_scores)
    if trak_scores_np.shape[0] != N_TEST:
        trak_scores_np = trak_scores_np.T

    total_trak_time = time.time() - t0
    print(f"  TRAK: shape={trak_scores_np.shape}, time={total_trak_time:.1f}s")
    print(f"  TRAK range: [{trak_scores_np.min():.6f}, {trak_scores_np.max():.6f}], std={trak_scores_np.std():.6f}")
    print(f"  GPU peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

    # Save
    trak_rankings = np.argsort(-trak_scores_np, axis=1)[:, :100]
    np.save(str(RESULTS_DIR / "phase1_attributions" / "trak_scores_full.npy"), trak_scores_np)
    np.save(str(RESULTS_DIR / "phase1_attributions" / "trak_rankings_top100.npy"), trak_rankings)

    trak_success = True
    trak_meta = {
        'success': True, 'featurize_time_sec': round(feat_time, 1),
        'total_time_sec': round(total_trak_time, 1),
        'shape': list(trak_scores_np.shape), 'proj_dim': 4096,
        'n_checkpoints': 1,
        'gpu_peak_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2),
        'note': 'TRAK-1 (single checkpoint) as proxy for TRAK-50 in pilot',
    }

    # Clean up TRAK temp files
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

# ==== ANALYSIS: Cross-method agreement ====
report_progress("analysis", "Computing cross-method agreement metrics", 90)

analysis = {}

def jaccard_at_k(ranking1, ranking2, k=10):
    """Jaccard similarity of top-k elements."""
    set1 = set(ranking1[:k].tolist())
    set2 = set(ranking2[:k].tolist())
    return len(set1 & set2) / len(set1 | set2)

def kendall_tau_top_n(scores1, scores2, n=100):
    """Kendall tau of top-n elements by scores1."""
    from scipy.stats import kendalltau
    top_n = np.argsort(-scores1)[:n]
    r1 = np.argsort(np.argsort(-scores1[top_n]))
    r2 = np.argsort(np.argsort(-scores2[top_n]))
    tau, p = kendalltau(r1, r2)
    return tau

# Jaccard@10 between EK-FAC and CG IF
if ekfac_success and cg_success:
    # CG IF is on 5K subset, so we need to compare on the same train indices
    # For Jaccard@10, compare rankings on the 5K shared subset
    ekfac_5k = ekfac_scores_np[:, train_sub_5k_indices]  # [100, 5000]
    ekfac_5k_rankings = np.argsort(-ekfac_5k, axis=1)

    j10_values = []
    for i in range(N_TEST):
        j10 = jaccard_at_k(ekfac_5k_rankings[i], cg_rankings[i], k=10)
        j10_values.append(j10)
    j10_values = np.array(j10_values)

    analysis['jaccard_at_10_ekfac_cg'] = {
        'mean': float(j10_values.mean()),
        'std': float(j10_values.std()),
        'min': float(j10_values.min()),
        'max': float(j10_values.max()),
        'per_class_mean': {},
        'values': j10_values.tolist(),
    }
    for c in range(N_CLASSES):
        mask = np.array(test_features['labels']) == c
        if mask.sum() > 0:
            analysis['jaccard_at_10_ekfac_cg']['per_class_mean'][str(c)] = float(j10_values[mask].mean())

    print(f"\nJaccard@10(EK-FAC, CG IF): mean={j10_values.mean():.4f}, std={j10_values.std():.4f}")
    print(f"  PASS criteria (std > 0.05): {'PASS' if j10_values.std() > 0.05 else 'FAIL'}")

# Kendall tau between IF (EK-FAC) and RepSim
if ekfac_success:
    from scipy.stats import kendalltau
    tau_values = []
    for i in range(N_TEST):
        # Compare on full 50K train set
        tau, _ = kendalltau(
            np.argsort(np.argsort(-ekfac_scores_np[i]))[:1000],
            np.argsort(np.argsort(-repsim_scores[i]))[:1000]
        )
        tau_values.append(tau if not np.isnan(tau) else 0.0)
    tau_values = np.array(tau_values)

    analysis['kendall_tau_ekfac_repsim'] = {
        'mean': float(tau_values.mean()),
        'std': float(tau_values.std()),
        'values': tau_values.tolist(),
    }
    print(f"Kendall tau(EK-FAC, RepSim) top-1000: mean={tau_values.mean():.4f}, std={tau_values.std():.4f}")

# Per-point LDS (EK-FAC vs TRAK-1 proxy)
if ekfac_success and trak_success:
    from scipy.stats import spearmanr
    lds_values = []
    for i in range(N_TEST):
        rho, _ = spearmanr(ekfac_scores_np[i], trak_scores_np[i])
        lds_values.append(rho if not np.isnan(rho) else 0.0)
    lds_values = np.array(lds_values)

    analysis['lds_ekfac_trak'] = {
        'mean': float(lds_values.mean()),
        'std': float(lds_values.std()),
        'values': lds_values.tolist(),
        'note': 'LDS computed against TRAK-1 (single checkpoint, pilot proxy)',
    }
    print(f"LDS(EK-FAC vs TRAK-1): mean={lds_values.mean():.4f}, std={lds_values.std():.4f}")

# ==== Compile results ====
report_progress("compile", "Compiling final results", 95)

results = {
    'task_id': TASK_ID,
    'mode': 'PILOT',
    'n_test': N_TEST,
    'n_train': len(trainset),
    'seed': SEED,
    'test_indices': test_indices,
    'methods': {
        'ekfac_if': ekfac_meta,
        'cg_if': cg_meta,
        'repsim': {
            'success': True,
            'time_sec': round(repsim_time, 1),
            'shape': list(repsim_scores.shape),
        },
        'trak': trak_meta,
    },
    'analysis': analysis,
    'pass_criteria': {
        'all_4_methods_valid': ekfac_success and cg_success and trak_success,
        'jaccard_std_above_005': analysis.get('jaccard_at_10_ekfac_cg', {}).get('std', 0) > 0.05 if ekfac_success and cg_success else False,
    },
    'overall_pass': (
        ekfac_success and cg_success and trak_success and
        analysis.get('jaccard_at_10_ekfac_cg', {}).get('std', 0) > 0.05
    ),
    'timestamp': datetime.now().isoformat(),
    'gpu_info': {
        'device': torch.cuda.get_device_name(0),
        'vram_total_mb': torch.cuda.get_device_properties(0).total_memory // (1024*1024),
    },
}

# Save results
out_path = RESULTS_DIR / "phase1_attributions" / "pilot_results.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"\nResults saved to {out_path}")

# Print summary
print("\n" + "=" * 60)
print("PILOT RESULTS SUMMARY")
print("=" * 60)
for method, meta in results['methods'].items():
    status = "PASS" if meta.get('success', False) else "FAIL"
    print(f"  {method}: {status}")
print(f"\nPass criteria:")
for k, v in results['pass_criteria'].items():
    print(f"  {k}: {'PASS' if v else 'FAIL'}")
print(f"\nOverall: {'PASS' if results['overall_pass'] else 'FAIL'}")

# Mark done
status = "success" if results['overall_pass'] else "partial"
summary = f"Pilot attribution compute: {sum(1 for m in results['methods'].values() if m.get('success'))} / 4 methods succeeded"
if 'jaccard_at_10_ekfac_cg' in analysis:
    summary += f", J@10 std={analysis['jaccard_at_10_ekfac_cg']['std']:.4f}"
mark_done(status=status, summary=summary)
print(f"\nDONE marker written: {status}")
