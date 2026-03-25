"""
Phase 3 LOO Validation Pilot: Leave-One-Out on CIFAR-10 subset.

Strategy:
  1. Exact LOO on tiny subset (50 train, 10 test) for methodology validation
  2. Approximate LOO via influence functions on larger subset (500 train, 10 test)
  3. Compare LOO ground truth against TRAK/EKFAC/RepSim from Phase 1

GPU: Single GPU (CUDA_VISIBLE_DEVICES set externally).
"""
import os
import sys
import json
import time
import gc
from pathlib import Path
from datetime import datetime

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
from scipy.stats import spearmanr

# ===== Configuration =====
TASK_ID = "phase3_loo_validation"
SEED = 42
PROJECT_DIR = Path("/home/jinxulin/sibyl_system/projects/AURA")
RESULTS_DIR = PROJECT_DIR / "exp" / "results"
ATTR_DIR = PROJECT_DIR / "exp" / "results" / "phase1_attributions"
DATA_DIR = Path("/home/jinxulin/sibyl_system/shared/datasets/cifar10")

# Pilot parameters
N_TRAIN_TINY = 50         # Exact LOO: 50 train (5/class)
N_TRAIN_SMALL = 500       # Approx LOO: 500 train (50/class)
N_TEST = 10               # 10 test points (1/class)
EXACT_LOO_EPOCHS = 15     # Short training for exact LOO
BASE_EPOCHS = 30          # Base model training

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===== PID and progress =====
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))
print(f"[LOO] PID {os.getpid()} written to {pid_file}")

def report_progress(phase, step, total, metric=None, extra=""):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": step, "total_epochs": total,
        "step": step, "total_steps": total,
        "loss": None, "metric": metric or {},
        "extra": f"{phase}: {extra}",
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status="success", summary=""):
    p = RESULTS_DIR / f"{TASK_ID}.pid"
    if p.exists(): p.unlink()
    pf = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    fp = {}
    if pf.exists():
        try: fp = json.loads(pf.read_text())
        except: pass
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": fp, "timestamp": datetime.now().isoformat(),
    }))

# ===== Model =====
def create_model():
    model = resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

# ===== Data =====
def get_cifar10():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=False, transform=transform_test)
    # Also load without augmentation for deterministic evaluation
    trainset_noaug = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=False, transform=transform_test)
    return trainset, testset, trainset_noaug

def stratified_indices(targets, n_per_class, seed=42):
    rng = np.random.RandomState(seed)
    targets = np.array(targets)
    indices = []
    for c in range(10):
        ci = np.where(targets == c)[0]
        chosen = rng.choice(ci, min(n_per_class, len(ci)), replace=False)
        indices.extend(sorted(chosen.tolist()))
    return indices

# ===== Training =====
def train_fast(model, data_x, data_y, device, epochs, lr=0.1):
    """Train on preloaded tensors (no data loader overhead)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    n = data_x.size(0)
    model.train()
    for ep in range(epochs):
        # Single batch or split into mini-batches
        if n <= 256:
            optimizer.zero_grad()
            out = model(data_x)
            loss = criterion(out, data_y)
            loss.backward()
            optimizer.step()
        else:
            perm = torch.randperm(n, device=device)
            for i in range(0, n, 256):
                idx = perm[i:i+256]
                optimizer.zero_grad()
                out = model(data_x[idx])
                loss = criterion(out, data_y[idx])
                loss.backward()
                optimizer.step()
    return model

def eval_loss(model, x, y, device):
    """Evaluate cross-entropy loss on single point."""
    model.eval()
    with torch.no_grad():
        out = model(x.unsqueeze(0))
        loss = nn.CrossEntropyLoss()(out, y.unsqueeze(0))
    return loss.item()

# ===== Main =====
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[LOO] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

    trainset, testset, trainset_noaug = get_cifar10()
    train_targets = np.array(trainset.targets)
    test_targets = np.array(testset.targets)

    # Get indices
    tiny_indices = stratified_indices(train_targets, 5, seed=SEED)  # 50 train
    small_indices = stratified_indices(train_targets, 50, seed=SEED)  # 500 train
    test_indices = stratified_indices(test_targets, 1, seed=SEED)[:N_TEST]  # 10 test

    print(f"[LOO] Tiny train: {len(tiny_indices)}, Small train: {len(small_indices)}, Test: {len(test_indices)}")
    print(f"[LOO] Test labels: {[testset.targets[i] for i in test_indices]}")

    # Preload test data to GPU
    test_x = torch.stack([testset[i][0] for i in test_indices]).to(device)
    test_y = torch.tensor([testset.targets[i] for i in test_indices]).to(device)

    # ================================================================
    # PART 1: Exact LOO on tiny subset (50 train, 10 test)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"[LOO] PART 1: Exact LOO on {len(tiny_indices)} training samples")
    print(f"{'='*60}")

    # Preload tiny training data (no augmentation for deterministic LOO)
    tiny_x = torch.stack([trainset_noaug[i][0] for i in tiny_indices]).to(device)
    tiny_y = torch.tensor([train_targets[i] for i in tiny_indices]).to(device)

    # Train base model
    t0 = time.time()
    base_model = create_model().to(device)
    base_model = train_fast(base_model, tiny_x, tiny_y, device, epochs=EXACT_LOO_EPOCHS, lr=0.05)
    base_train_time = time.time() - t0
    print(f"[LOO] Base model trained in {base_train_time:.1f}s")

    # Base accuracy on test points
    base_model.eval()
    with torch.no_grad():
        preds = base_model(test_x).argmax(1)
        base_acc = (preds == test_y).float().mean().item() * 100
    print(f"[LOO] Base accuracy: {base_acc:.0f}% on {N_TEST} points")

    # Base losses
    base_losses = [eval_loss(base_model, test_x[i], test_y[i], device) for i in range(N_TEST)]
    del base_model
    torch.cuda.empty_cache()

    # Exact LOO: retrain 50 times per test point
    print(f"\n[LOO] Starting exact LOO: {N_TEST} test x {len(tiny_indices)} removals = {N_TEST * len(tiny_indices)} retrainings")
    t_loo_start = time.time()

    exact_loo = {}  # test_idx -> list of influence values
    for ti in range(N_TEST):
        test_idx = test_indices[ti]
        influences = []

        for ri in range(len(tiny_indices)):
            # Remove one training sample
            mask = torch.ones(len(tiny_indices), dtype=torch.bool)
            mask[ri] = False
            reduced_x = tiny_x[mask]
            reduced_y = tiny_y[mask]

            # Retrain
            torch.manual_seed(SEED)
            model = create_model().to(device)
            model = train_fast(model, reduced_x, reduced_y, device, epochs=EXACT_LOO_EPOCHS, lr=0.05)

            # Evaluate
            loss_without = eval_loss(model, test_x[ti], test_y[ti], device)
            influence = loss_without - base_losses[ti]
            influences.append(influence)

            del model
            if ri % 20 == 0:
                torch.cuda.empty_cache()

        exact_loo[test_idx] = {
            "influences": influences,
            "train_indices": tiny_indices,
            "test_label": int(test_y[ti].item()),
        }

        elapsed = time.time() - t_loo_start
        avg_per_removal = elapsed / ((ti + 1) * len(tiny_indices))
        remaining = avg_per_removal * (N_TEST - ti - 1) * len(tiny_indices)
        print(f"  Test {ti+1}/{N_TEST} (idx={test_idx}): mean_infl={np.mean(influences):.4f}, "
              f"std={np.std(influences):.4f}, elapsed={elapsed:.0f}s, remaining~{remaining:.0f}s")

        report_progress("exact_loo", ti+1, N_TEST,
                        metric={"mean_influence": float(np.mean(influences))},
                        extra=f"test {ti+1}/{N_TEST}, {elapsed:.0f}s elapsed")

    exact_loo_time = time.time() - t_loo_start
    print(f"\n[LOO] Exact LOO done in {exact_loo_time:.0f}s ({exact_loo_time/60:.1f}min)")

    # ================================================================
    # PART 2: Approximate LOO via influence functions on 500 samples
    # ================================================================
    print(f"\n{'='*60}")
    print(f"[LOO] PART 2: Approximate LOO on {len(small_indices)} training samples")
    print(f"{'='*60}")

    # Preload small training data
    small_x = torch.stack([trainset_noaug[i][0] for i in small_indices]).to(device)
    small_y = torch.tensor([train_targets[i] for i in small_indices]).to(device)

    # Train base model on 500 samples
    t0 = time.time()
    torch.manual_seed(SEED)
    base500 = create_model().to(device)
    base500 = train_fast(base500, small_x, small_y, device, epochs=BASE_EPOCHS, lr=0.05)
    print(f"[LOO] Base model (500) trained in {time.time()-t0:.1f}s")

    base500.eval()
    with torch.no_grad():
        preds500 = base500(test_x).argmax(1)
        acc500 = (preds500 == test_y).float().mean().item() * 100
    print(f"[LOO] Base accuracy (500): {acc500:.0f}%")

    base_losses_500 = [eval_loss(base500, test_x[i], test_y[i], device) for i in range(N_TEST)]

    # Compute per-sample gradients for influence function approximation
    # Using the simple dot-product approximation: IF(z_test, z_train) ~ -grad_test . H^{-1} grad_train
    # Approximate H^{-1} with identity (gradient dot product) as cheapest proxy
    print("[LOO] Computing per-sample gradients for approximate IF...")

    criterion = nn.CrossEntropyLoss()

    # Test gradients (keep on CPU to save GPU memory)
    test_grads = []
    for i in range(N_TEST):
        base500.zero_grad()
        out = base500(test_x[i:i+1])
        loss = criterion(out, test_y[i:i+1])
        loss.backward()
        g = torch.cat([p.grad.flatten().cpu() for p in base500.parameters() if p.grad is not None])
        test_grads.append(g)
    test_grads = torch.stack(test_grads)  # (N_TEST, D) on CPU
    print(f"[LOO] Test gradient shape: {test_grads.shape}")

    # Compute dot-product influence incrementally to avoid OOM
    # approx_if[i,j] = -test_grads[i] . train_grads[j] / N
    approx_if = np.zeros((N_TEST, len(small_indices)), dtype=np.float32)
    batch_sz = 50
    for start in range(0, len(small_indices), batch_sz):
        end = min(start + batch_sz, len(small_indices))
        batch_grads = []
        for j in range(start, end):
            base500.zero_grad()
            out = base500(small_x[j:j+1])
            loss = criterion(out, small_y[j:j+1])
            loss.backward()
            g = torch.cat([p.grad.flatten().cpu() for p in base500.parameters() if p.grad is not None])
            batch_grads.append(g)
        batch_grads = torch.stack(batch_grads)  # (batch, D) on CPU
        # Compute dot products on CPU
        dots = torch.mm(test_grads, batch_grads.t()).numpy()  # (N_TEST, batch)
        approx_if[:, start:end] = -dots / len(small_indices)
        del batch_grads
        if start % 200 == 0:
            print(f"  Train gradients: {min(start+batch_sz, len(small_indices))}/{len(small_indices)}")
    print(f"[LOO] Approximate IF shape: {approx_if.shape}")

    del test_grads
    torch.cuda.empty_cache()
    gc.collect()

    # ===== RepSim-based attribution on 500 samples =====
    print("[LOO] Computing RepSim attributions on 500 samples...")
    activation = {}
    def hook_fn(module, inp, out):
        activation['feat'] = out.squeeze()
    handle = base500.avgpool.register_forward_hook(hook_fn)

    base500.eval()
    with torch.no_grad():
        test_feats = []
        for i in range(N_TEST):
            _ = base500(test_x[i:i+1])
            test_feats.append(activation['feat'].cpu().numpy().copy())
        test_feats = np.array(test_feats)

        train_feats = []
        for j in range(len(small_indices)):
            _ = base500(small_x[j:j+1])
            train_feats.append(activation['feat'].cpu().numpy().copy())
        train_feats = np.array(train_feats)
    handle.remove()

    # Cosine similarity
    from numpy.linalg import norm
    repsim = np.array([
        [np.dot(test_feats[i], train_feats[j]) / (norm(test_feats[i]) * norm(train_feats[j]) + 1e-8)
         for j in range(len(small_indices))]
        for i in range(N_TEST)
    ])
    print(f"[LOO] RepSim shape: {repsim.shape}")

    del base500
    torch.cuda.empty_cache()
    gc.collect()

    # ===== Exact LOO on small sample of 500-train removals =====
    # Do exact LOO for first 100 removals per test point (subset validation)
    N_EXACT_FROM_500 = 100
    print(f"\n[LOO] Exact LOO on {N_EXACT_FROM_500} removals from 500-train set...")

    t_exact500_start = time.time()
    exact_loo_500_partial = {}

    for ti in range(N_TEST):
        test_idx = test_indices[ti]
        influences = []

        for ri in range(N_EXACT_FROM_500):
            mask = torch.ones(len(small_indices), dtype=torch.bool, device=device)
            mask[ri] = False
            reduced_x = small_x[mask]
            reduced_y = small_y[mask]

            torch.manual_seed(SEED)
            model = create_model().to(device)
            model = train_fast(model, reduced_x, reduced_y, device, epochs=20, lr=0.05)

            loss_without = eval_loss(model, test_x[ti], test_y[ti], device)
            influence = loss_without - base_losses_500[ti]
            influences.append(influence)

            del model
            if ri % 25 == 0:
                torch.cuda.empty_cache()

        exact_loo_500_partial[test_idx] = influences

        elapsed = time.time() - t_exact500_start
        print(f"  Test {ti+1}/{N_TEST}: mean_infl={np.mean(influences):.4f}, "
              f"elapsed={elapsed:.0f}s")

        report_progress("exact_loo_500", ti+1, N_TEST,
                        metric={"mean_influence": float(np.mean(influences))},
                        extra=f"partial exact on 500-set, test {ti+1}/{N_TEST}")

    exact500_time = time.time() - t_exact500_start
    print(f"[LOO] Partial exact LOO (500-train) done in {exact500_time:.0f}s")

    # ================================================================
    # PART 3: Correlations
    # ================================================================
    print(f"\n{'='*60}")
    print(f"[LOO] PART 3: Computing correlations")
    print(f"{'='*60}")

    # 3a: Exact LOO (50-train) structural validation
    exact_structural = []
    for ti, test_idx in enumerate(test_indices):
        infl = np.array(exact_loo[test_idx]["influences"])
        train_labels = np.array([train_targets[i] for i in tiny_indices])
        test_label = exact_loo[test_idx]["test_label"]

        same_mask = train_labels == test_label
        same_mean = float(np.mean(infl[same_mask])) if same_mask.any() else 0
        diff_mean = float(np.mean(infl[~same_mask])) if (~same_mask).any() else 0
        top10_idx = np.argsort(infl)[-10:]
        top10_same = int(same_mask[top10_idx].sum())

        exact_structural.append({
            "test_idx": int(test_idx),
            "test_label": test_label,
            "same_class_mean": same_mean,
            "diff_class_mean": diff_mean,
            "influence_gap": same_mean - diff_mean,
            "top10_same_class": top10_same,
            "influence_std": float(np.std(infl)),
        })

    # 3b: Approx IF vs Exact LOO (50-train) correlation
    # Map tiny_indices to positions in small_indices
    tiny_to_small = {}
    for tpos, tidx in enumerate(tiny_indices):
        if tidx in small_indices:
            spos = small_indices.index(tidx)
            tiny_to_small[tpos] = spos

    approx_vs_exact_corrs = []
    for ti, test_idx in enumerate(test_indices):
        exact_infl = np.array(exact_loo[test_idx]["influences"])
        # Get approx IF for the same training samples
        if len(tiny_to_small) >= 30:
            exact_vals = [exact_infl[tp] for tp in sorted(tiny_to_small.keys())]
            approx_vals = [approx_if[ti, tiny_to_small[tp]] for tp in sorted(tiny_to_small.keys())]
            rho, pval = spearmanr(exact_vals, approx_vals)
            approx_vs_exact_corrs.append({"test_idx": int(test_idx), "rho": float(rho), "p": float(pval), "n": len(exact_vals)})
            print(f"  Approx IF vs Exact LOO (test {test_idx}): rho={rho:.4f} (n={len(exact_vals)})")

    # 3c: Exact LOO (500-train partial) vs Approx IF
    exact500_vs_approx_corrs = []
    for ti, test_idx in enumerate(test_indices):
        exact_infl = np.array(exact_loo_500_partial[test_idx])
        approx_vals = approx_if[ti, :N_EXACT_FROM_500]
        rho, pval = spearmanr(exact_infl, approx_vals)
        exact500_vs_approx_corrs.append({"test_idx": int(test_idx), "rho": float(rho), "p": float(pval)})
        print(f"  Exact LOO(500) vs Approx IF (test {test_idx}): rho={rho:.4f}")

    # 3d: RepSim vs Exact LOO (50-train)
    repsim_vs_exact_corrs = []
    for ti, test_idx in enumerate(test_indices):
        exact_infl = np.array(exact_loo[test_idx]["influences"])
        if len(tiny_to_small) >= 30:
            exact_vals = [exact_infl[tp] for tp in sorted(tiny_to_small.keys())]
            repsim_vals = [repsim[ti, tiny_to_small[tp]] for tp in sorted(tiny_to_small.keys())]
            rho, pval = spearmanr(exact_vals, repsim_vals)
            repsim_vs_exact_corrs.append({"test_idx": int(test_idx), "rho": float(rho), "p": float(pval)})
            print(f"  RepSim vs Exact LOO (test {test_idx}): rho={rho:.4f}")

    # 3e: RepSim vs Exact LOO (500-train partial)
    repsim_vs_exact500_corrs = []
    for ti, test_idx in enumerate(test_indices):
        exact_infl = np.array(exact_loo_500_partial[test_idx])
        repsim_vals = repsim[ti, :N_EXACT_FROM_500]
        rho, pval = spearmanr(exact_infl, repsim_vals)
        repsim_vs_exact500_corrs.append({"test_idx": int(test_idx), "rho": float(rho), "p": float(pval)})
        print(f"  RepSim vs Exact LOO(500) (test {test_idx}): rho={rho:.4f}")

    # 3f: Load Phase 1 TRAK attributions if available
    trak_vs_loo_corrs = []
    ekfac_vs_loo_corrs = []
    trak_available = (ATTR_DIR / "trak_scores_5k.npy").exists()
    ekfac_available = (ATTR_DIR / "ekfac_scores_5k.npy").exists()
    phase1_pilot_path = ATTR_DIR / "pilot_results.json"

    if phase1_pilot_path.exists():
        phase1_data = json.loads(phase1_pilot_path.read_text())
        phase1_test_indices = phase1_data.get("test_indices", [])

        if trak_available:
            trak_scores = np.load(str(ATTR_DIR / "trak_scores_5k.npy"))
            print(f"\n[LOO] TRAK scores loaded: {trak_scores.shape}")

            # Phase 1 used 5K stratified samples. Map our 500 to those.
            phase1_5k = stratified_indices(train_targets, 500, seed=SEED)

            for ti, test_idx in enumerate(test_indices):
                if test_idx in phase1_test_indices:
                    p1_pos = phase1_test_indices.index(test_idx)
                    trak_row = trak_scores[p1_pos]

                    # Map our small_indices (500) to phase1_5k positions
                    overlap = {}
                    for spos, sidx in enumerate(small_indices):
                        if sidx in phase1_5k:
                            p1_train_pos = phase1_5k.index(sidx)
                            if p1_train_pos < trak_row.shape[0]:
                                overlap[spos] = p1_train_pos

                    if len(overlap) >= 50:
                        # Use exact LOO partial (100 removals) for those that overlap
                        loo_vals = []
                        trak_vals = []
                        for spos in sorted(overlap.keys()):
                            if spos < N_EXACT_FROM_500:
                                loo_vals.append(exact_loo_500_partial[test_idx][spos])
                                trak_vals.append(trak_row[overlap[spos]])

                        if len(loo_vals) >= 20:
                            rho, pval = spearmanr(loo_vals, trak_vals)
                            trak_vs_loo_corrs.append({"test_idx": int(test_idx), "rho": float(rho), "p": float(pval), "n": len(loo_vals)})
                            print(f"  TRAK vs Exact LOO(500) (test {test_idx}): rho={rho:.4f} (n={len(loo_vals)})")

        if ekfac_available:
            ekfac_scores = np.load(str(ATTR_DIR / "ekfac_scores_5k.npy"))
            print(f"[LOO] EKFAC scores loaded: {ekfac_scores.shape}")

            phase1_5k = stratified_indices(train_targets, 500, seed=SEED)

            for ti, test_idx in enumerate(test_indices):
                if test_idx in phase1_test_indices:
                    p1_pos = phase1_test_indices.index(test_idx)
                    ekfac_row = ekfac_scores[p1_pos]

                    overlap = {}
                    for spos, sidx in enumerate(small_indices):
                        if sidx in phase1_5k:
                            p1_train_pos = phase1_5k.index(sidx)
                            if p1_train_pos < ekfac_row.shape[0]:
                                overlap[spos] = p1_train_pos

                    if len(overlap) >= 50:
                        loo_vals = []
                        ekfac_vals = []
                        for spos in sorted(overlap.keys()):
                            if spos < N_EXACT_FROM_500:
                                loo_vals.append(exact_loo_500_partial[test_idx][spos])
                                ekfac_vals.append(ekfac_row[overlap[spos]])

                        if len(loo_vals) >= 20:
                            rho, pval = spearmanr(loo_vals, ekfac_vals)
                            ekfac_vs_loo_corrs.append({"test_idx": int(test_idx), "rho": float(rho), "p": float(pval), "n": len(loo_vals)})
                            print(f"  EKFAC vs Exact LOO(500) (test {test_idx}): rho={rho:.4f} (n={len(loo_vals)})")

    # ================================================================
    # PART 4: Summary and results
    # ================================================================
    print(f"\n{'='*60}")
    print(f"[LOO] PART 4: Summary")
    print(f"{'='*60}")

    total_time = time.time() - float(pid_file.stat().st_mtime) if pid_file.exists() else time.time() - t_loo_start

    def safe_mean(lst, key="rho"):
        vals = [x[key] for x in lst if not np.isnan(x[key])]
        return float(np.mean(vals)) if vals else None

    summary = {
        "structural": {
            "mean_influence_gap": float(np.mean([s["influence_gap"] for s in exact_structural])),
            "mean_top10_same_class": float(np.mean([s["top10_same_class"] for s in exact_structural])),
            "mean_influence_std": float(np.mean([s["influence_std"] for s in exact_structural])),
            "all_positive_gap": all(s["influence_gap"] > 0 for s in exact_structural),
        },
        "approx_if_vs_exact_loo": safe_mean(approx_vs_exact_corrs),
        "exact500_vs_approx_if": safe_mean(exact500_vs_approx_corrs),
        "repsim_vs_exact_loo": safe_mean(repsim_vs_exact_corrs),
        "repsim_vs_exact500": safe_mean(repsim_vs_exact500_corrs),
        "trak_vs_loo": safe_mean(trak_vs_loo_corrs) if trak_vs_loo_corrs else None,
        "ekfac_vs_loo": safe_mean(ekfac_vs_loo_corrs) if ekfac_vs_loo_corrs else None,
    }

    # Determine best correlation for pass/fail
    all_corrs = {
        "approx_IF_vs_exact_LOO_50": summary["approx_if_vs_exact_loo"],
        "exact500_vs_approx_IF": summary["exact500_vs_approx_if"],
        "RepSim_vs_exact_LOO_50": summary["repsim_vs_exact_loo"],
        "RepSim_vs_exact_LOO_500": summary["repsim_vs_exact500"],
        "TRAK_vs_LOO": summary["trak_vs_loo"],
        "EKFAC_vs_LOO": summary["ekfac_vs_loo"],
    }
    valid_corrs = {k: v for k, v in all_corrs.items() if v is not None}
    best_method = max(valid_corrs, key=valid_corrs.get) if valid_corrs else "none"
    best_corr = valid_corrs.get(best_method, 0)
    passes = best_corr > 0.3

    for name, val in sorted(all_corrs.items()):
        print(f"  {name}: {val:.4f}" if val is not None else f"  {name}: N/A")
    print(f"\n  Best: {best_method} = {best_corr:.4f}")
    print(f"  Pass (>0.3): {passes}")
    print(f"  Structural: gap={summary['structural']['mean_influence_gap']:.4f}, "
          f"top10_same={summary['structural']['mean_top10_same_class']:.1f}/10")

    results = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "seed": SEED,
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "n_train_tiny": N_TRAIN_TINY,
            "n_train_small": N_TRAIN_SMALL,
            "n_test": N_TEST,
            "exact_loo_epochs": EXACT_LOO_EPOCHS,
            "base_epochs": BASE_EPOCHS,
            "n_exact_from_500": N_EXACT_FROM_500,
        },
        "timing": {
            "exact_loo_50_sec": round(exact_loo_time),
            "exact_loo_500_partial_sec": round(exact500_time),
            "total_sec": round(time.time() - t_loo_start),
        },
        "base_model_accuracy": {
            "tiny_50": base_acc,
            "small_500": acc500,
        },
        "structural_validation": summary["structural"],
        "structural_per_point": exact_structural,
        "correlations": {
            "approx_if_vs_exact_loo_50": {
                "mean_rho": summary["approx_if_vs_exact_loo"],
                "per_point": approx_vs_exact_corrs,
            },
            "exact_loo_500_vs_approx_if": {
                "mean_rho": summary["exact500_vs_approx_if"],
                "per_point": exact500_vs_approx_corrs,
            },
            "repsim_vs_exact_loo_50": {
                "mean_rho": summary["repsim_vs_exact_loo"],
                "per_point": repsim_vs_exact_corrs,
            },
            "repsim_vs_exact_loo_500": {
                "mean_rho": summary["repsim_vs_exact500"],
                "per_point": repsim_vs_exact500_corrs,
            },
            "trak_vs_exact_loo_500": {
                "available": bool(trak_vs_loo_corrs),
                "mean_rho": summary["trak_vs_loo"],
                "per_point": trak_vs_loo_corrs,
            },
            "ekfac_vs_exact_loo_500": {
                "available": bool(ekfac_vs_loo_corrs),
                "mean_rho": summary["ekfac_vs_loo"],
                "per_point": ekfac_vs_loo_corrs,
            },
        },
        "pass_criteria": {
            "criterion": "LOO-TDA correlation > 0.3",
            "best_method": best_method,
            "best_correlation": best_corr,
            "passes": passes,
        },
        "gpu_info": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "vram_mb": torch.cuda.get_device_properties(0).total_memory // (1024**2) if torch.cuda.is_available() else 0,
        },
    }

    results_path = RESULTS_DIR / "phase3_loo_validation.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[LOO] Results saved to {results_path}")

    # Save raw LOO data
    loo_raw = {
        "exact_loo_50": {str(k): v for k, v in exact_loo.items()},
        "exact_loo_500_partial": {str(k): v for k, v in exact_loo_500_partial.items()},
        "approx_if": approx_if.tolist(),
        "repsim": repsim.tolist(),
        "train_indices_tiny": tiny_indices,
        "train_indices_small": small_indices,
        "test_indices": test_indices,
    }
    (RESULTS_DIR / "phase3_loo_raw.json").write_text(json.dumps(loo_raw, default=str))

    mark_done(
        status="success" if passes else "completed_below_threshold",
        summary=f"LOO pilot: best_corr={best_corr:.4f} ({best_method}), "
                f"passes={'YES' if passes else 'NO'}, "
                f"structural_gap={summary['structural']['mean_influence_gap']:.4f}"
    )

    return results


if __name__ == "__main__":
    results = main()
    print(f"\n[LOO] Phase 3 LOO validation pilot complete.")
    passed = results["pass_criteria"]["passes"]
    print(f"[LOO] GO/NO-GO: {'GO' if passed else 'NO-GO'}")
    sys.exit(0 if passed else 1)
