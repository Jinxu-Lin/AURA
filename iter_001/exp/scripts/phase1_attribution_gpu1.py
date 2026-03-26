#!/usr/bin/env python3
"""
Phase 1 Attribution — GPU 1: Full-model K-FAC/EK-FAC IF (layer4 + fc)

Process ONE LAYER AT A TIME to fit in GPU memory:
For each of 6 layers:
  1. Compute test IHVPs for this layer (500, dG, dA) on GPU — max 4.7 GB
  2. One pass over 50K training data, compute bilinear IF scores for this layer
  3. Accumulate into total IF scores (CPU)
  4. Free GPU memory for next layer

Total: 6 passes over 50K training data, each ~1 min → ~6-10 min.
"""
import os, sys, json, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from datetime import datetime

TASK_ID = "phase1_attribution_compute"
PROJECT_DIR = Path("/home/jinxulin/sibyl_system/projects/AURA")
RESULTS_DIR = PROJECT_DIR / "exp" / "results"
ATTR_DIR    = PROJECT_DIR / "exp" / "results" / "phase1_attributions"
CKPT_PATH   = PROJECT_DIR / "exp" / "checkpoints" / "resnet18_cifar10_seed42.pt"

SEED = 42
N_PER_CLASS = 50
DAMPING_EKFAC = 0.01
DAMPING_KFAC  = 0.1
SELECTED_LAYERS = {'layer4.0.conv1', 'layer4.0.conv2', 'layer4.0.downsample.0',
                   'layer4.1.conv1', 'layer4.1.conv2', 'fc'}

def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s)

def get_resnet18(nc=10):
    m = torchvision.models.resnet18(weights=None, num_classes=nc)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m

def select_stratified(testset, n_per_class, seed):
    rng = np.random.RandomState(seed)
    targets = np.array(testset.targets)
    idx = []
    for c in range(10):
        ci = np.where(targets == c)[0]
        idx.extend(sorted(rng.choice(ci, n_per_class, replace=False)))
    return [int(x) for x in sorted(idx)]

def report(stage, detail, pct):
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID, "stage": stage, "detail": detail,
        "percent": pct, "updated_at": datetime.now().isoformat()}))

def compute_test_features(model, test_subset, test_indices, device):
    model.eval()
    feats = []
    loader = torch.utils.data.DataLoader(test_subset, batch_size=1, shuffle=False, num_workers=2)
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        out = model(x)
        probs = F.softmax(out, dim=1)
        F.cross_entropy(out, y).backward()
        gn = sum(p.grad.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
        feats.append({
            "test_idx": int(test_indices[i]),
            "true_label": y.item(),
            "pred_label": out.argmax(1).item(),
            "correct": out.argmax(1).item() == y.item(),
            "confidence": probs.max().item(),
            "entropy": -(probs * torch.log(probs + 1e-10)).sum().item(),
            "grad_norm": gn,
            "log_grad_norm": float(np.log(gn + 1e-10)),
        })
    return feats


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    set_seed(SEED)
    ATTR_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))
    start_time = time.time()

    # ── Data ──
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='/home/jinxulin/sibyl_system/shared/datasets/cifar10',
        train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root='/home/jinxulin/sibyl_system/shared/datasets/cifar10',
        train=False, download=True, transform=transform)

    test_indices = select_stratified(testset, N_PER_CLASS, SEED)
    test_subset = torch.utils.data.Subset(testset, test_indices)
    n_test, n_train = len(test_indices), len(trainset)
    print(f"Test: {n_test}, Train: {n_train}")
    (ATTR_DIR / "test_indices_500.json").write_text(json.dumps(test_indices))

    # ── Model ──
    model = get_resnet18()
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(sd)
    model = model.to(device).eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # ── Layers ──
    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, (nn.Conv2d, nn.Linear)) and n in SELECTED_LAYERS]
    print(f"Using {len(layers)} layers: {[n for n,_ in layers]}")

    # ── Hook infrastructure ──
    act_buf, grad_buf = {}, {}
    active_hooks = []

    def attach_hooks(target_layers):
        for h in active_hooks:
            h.remove()
        active_hooks.clear()

        for name, mod in target_layers:
            def make_fwd(nm):
                def fn(m, inp, out):
                    if isinstance(m, nn.Linear):
                        a = inp[0].detach()
                        if m.bias is not None:
                            a = torch.cat([a, torch.ones(a.shape[0],1,device=a.device)],1)
                        act_buf[nm] = a
                    elif isinstance(m, nn.Conv2d):
                        act_buf[nm] = inp[0].detach()
                return fn
            def make_bwd(nm):
                def fn(m, gi, go):
                    grad_buf[nm] = go[0].detach()
                return fn
            active_hooks.append(mod.register_forward_hook(make_fwd(name)))
            active_hooks.append(mod.register_full_backward_hook(make_bwd(name)))

    # ══════════ Step 1: K-FAC factors (one pass, all layers) ══════════
    report("kfac_factors", "Computing K-FAC factors", 5)
    print("Step 1: K-FAC factors...")
    t0 = time.time()

    attach_hooks(layers)

    factor_A, factor_G = {}, {}
    n_batches = 0
    tl = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False,
                                      num_workers=4, pin_memory=True)
    for bi, (x, y) in enumerate(tl):
        x, y = x.to(device), y.to(device)
        bs = x.size(0)
        model.zero_grad()
        F.cross_entropy(model(x), y).backward()

        for name, mod in layers:
            if name not in act_buf:
                continue
            if isinstance(mod, nn.Linear):
                a, g = act_buf[name], grad_buf[name]
            elif isinstance(mod, nn.Conv2d):
                a_raw = act_buf[name]
                a = F.unfold(a_raw, mod.kernel_size, padding=mod.padding,
                             stride=mod.stride).mean(2)
                if mod.bias is not None:
                    a = torch.cat([a, torch.ones(bs,1,device=device)],1)
                g = grad_buf[name].mean(dim=[2,3])

            Ab = (a.T @ a) / bs
            Gb = (g.T @ g) / bs
            if name not in factor_A:
                factor_A[name] = Ab; factor_G[name] = Gb
            else:
                factor_A[name] += Ab; factor_G[name] += Gb

        act_buf.clear(); grad_buf.clear()
        n_batches += 1

    for name in factor_A:
        factor_A[name] /= n_batches; factor_G[name] /= n_batches
    print(f"  Factors: {(time.time()-t0)/60:.1f} min")

    # ══════════ Step 2: Inverses (compute and store on CPU) ══════════
    report("inverses", "K-FAC/EK-FAC inverses", 10)
    d_kfac = DAMPING_KFAC ** 0.5

    layer_inv = {}
    for name in factor_A:
        A = factor_A[name].float()
        G = factor_G[name].float()
        dA, dG = A.shape[0], G.shape[0]

        invA = torch.linalg.inv(A + d_kfac * torch.eye(dA, device=device)).cpu()
        invG = torch.linalg.inv(G + d_kfac * torch.eye(dG, device=device)).cpu()
        eA, vA = torch.linalg.eigh(A)
        eG, vG = torch.linalg.eigh(G)
        inv_lam = (1.0 / (eG.unsqueeze(1) * eA.unsqueeze(0) + DAMPING_EKFAC)).cpu()

        layer_inv[name] = {
            'invA': invA, 'invG': invG,
            'vA': vA.cpu(), 'vG': vG.cpu(), 'inv_lam': inv_lam,
            'dA': dA, 'dG': dG,
        }
        print(f"  {name}: dA={dA}, dG={dG}, IHVP_size={n_test*dG*dA*4/1e9:.2f} GB")

    del factor_A, factor_G
    torch.cuda.empty_cache()

    # ══════════ Step 3: Layer-by-layer IF computation ══════════
    # For each layer:
    #   a) Compute test IHVPs (on GPU, ~2.3 GB for largest layer)
    #   b) One pass over training data, bilinear scores
    #   c) Accumulate, free GPU memory

    ekfac_scores = torch.zeros(n_test, n_train, dtype=torch.float32)  # CPU
    kfac_scores = torch.zeros(n_test, n_train, dtype=torch.float32)

    n_layers = len(layers)
    for li, (lname, lmod) in enumerate(layers):
        if lname not in layer_inv:
            continue
        inv = layer_inv[lname]
        dG, dA = inv['dG'], inv['dA']

        print(f"\n=== Layer {li+1}/{n_layers}: {lname} (dG={dG}, dA={dA}) ===")
        report("layer_if", f"Layer {li+1}/{n_layers}: {lname}", 15 + 70 * li / n_layers)

        # ── 3a: Test IHVPs for this layer (on GPU) ──
        # Split test points into sub-batches if needed
        ihvp_mem = n_test * dG * dA * 4 * 2  # kfac + ekfac
        print(f"  IHVP GPU memory: {ihvp_mem / 1e9:.2f} GB")

        # Determine test sub-batch size to fit in ~8 GB
        max_test_sub = max(10, min(n_test, int(8e9 / (dG * dA * 4 * 2))))
        n_test_subs = (n_test + max_test_sub - 1) // max_test_sub
        print(f"  Test sub-batches: {n_test_subs} x {max_test_sub}")

        # ── For each test sub-batch, compute IHVPs then score ALL training data ──
        for tsi in range(n_test_subs):
            ts_start = tsi * max_test_sub
            ts_end = min(ts_start + max_test_sub, n_test)
            ts_size = ts_end - ts_start

            # Compute IHVPs
            ihvp_k = torch.zeros(ts_size, dG, dA, device=device, dtype=torch.float32)
            ihvp_e = torch.zeros(ts_size, dG, dA, device=device, dtype=torch.float32)

            # Attach hook for this layer only
            attach_hooks([(lname, lmod)])

            test_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(test_subset, list(range(ts_start, ts_end))),
                batch_size=1, shuffle=False, num_workers=0)

            invA_d = inv['invA'].to(device)
            invG_d = inv['invG'].to(device)
            vA_d = inv['vA'].to(device)
            vG_d = inv['vG'].to(device)
            inv_lam_d = inv['inv_lam'].to(device)

            for ti, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                model.zero_grad()
                F.cross_entropy(model(x), y).backward()

                if isinstance(lmod, nn.Linear):
                    gw = lmod.weight.grad.detach().float()
                    if lmod.bias is not None:
                        gw = torch.cat([gw, lmod.bias.grad.detach().float().unsqueeze(1)],1)
                elif isinstance(lmod, nn.Conv2d):
                    gw = lmod.weight.grad.detach().float().reshape(lmod.weight.shape[0], -1)
                    if lmod.bias is not None:
                        gw = torch.cat([gw, lmod.bias.grad.detach().float().unsqueeze(1)],1)

                ihvp_k[ti] = invG_d @ gw @ invA_d
                proj = vG_d.T @ gw @ vA_d
                ihvp_e[ti] = vG_d @ (inv_lam_d * proj) @ vA_d.T

                act_buf.clear(); grad_buf.clear()

            del invA_d, invG_d, vA_d, vG_d, inv_lam_d
            torch.cuda.empty_cache()

            # ── 3b: One pass over training data, bilinear scores ──
            attach_hooks([(lname, lmod)])

            BATCH_SIZE = 128
            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=4, pin_memory=True)

            t0 = time.time()
            for bi, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                bs = x.size(0)
                batch_start = bi * BATCH_SIZE

                model.zero_grad()
                F.cross_entropy(model(x), y).backward()

                # Get per-sample activations and output gradients
                if isinstance(lmod, nn.Linear):
                    a = act_buf[lname]
                    g = grad_buf[lname]
                elif isinstance(lmod, nn.Conv2d):
                    a_raw = act_buf[lname]
                    a = F.unfold(a_raw, lmod.kernel_size, padding=lmod.padding,
                                 stride=lmod.stride).mean(2)
                    if lmod.bias is not None:
                        a = torch.cat([a, torch.ones(bs,1,device=device)],1)
                    g = grad_buf[lname].mean(dim=[2,3])

                # Bilinear: score(t, b) = a_b^T @ ihvp[t]^T @ g_b
                # = (g @ ihvp) element-wise * a, summed over dA
                # Vectorized: einsum('bG, sGA, bA -> sb', g, ihvp, a)

                # Two-step for memory efficiency:
                # tmp = g @ ihvp  →  einsum('bG, sGA -> bsA', g, ihvp)
                # This is (bs, dG) @ (ts, dG, dA) → need to transpose ihvp
                # Actually: tmp[b,s,A] = sum_G g[b,G] * ihvp[s,G,A]
                # = (bs, ts, dA)
                # Final: sum_A tmp[b,s,A] * a[b,A] → (bs, ts) → transpose → (ts, bs)

                # K-FAC
                tmp = torch.einsum('bG, sGA -> bsA', g.float(), ihvp_k)  # (bs, ts, dA)
                partial_k = torch.einsum('bsA, bA -> sb', tmp, a.float())  # (ts, bs)
                kfac_scores[ts_start:ts_end, batch_start:batch_start+bs] += partial_k.cpu()

                # EK-FAC
                tmp = torch.einsum('bG, sGA -> bsA', g.float(), ihvp_e)
                partial_e = torch.einsum('bsA, bA -> sb', tmp, a.float())
                ekfac_scores[ts_start:ts_end, batch_start:batch_start+bs] += partial_e.cpu()

                act_buf.clear(); grad_buf.clear()

                if bi == 0:
                    first_batch_time = time.time() - t0
                    est_total = first_batch_time * len(train_loader)
                    print(f"  First batch: {first_batch_time:.2f}s, est total: {est_total:.0f}s")

            elapsed_train = time.time() - t0
            print(f"  Test sub {tsi+1}/{n_test_subs}: {elapsed_train:.1f}s "
                  f"({len(train_loader)/elapsed_train:.1f} batch/s)")

            # Free GPU memory
            del ihvp_k, ihvp_e
            torch.cuda.empty_cache()
            gc.collect()

    # Detach remaining hooks
    for h in active_hooks: h.remove()
    active_hooks.clear()

    # ── Save ──
    print("\nSaving results...")
    report("save", "Saving", 88)

    ekfac_np = ekfac_scores.numpy()
    kfac_np = kfac_scores.numpy()
    np.save(ATTR_DIR / "ekfac_scores_fullmodel.npy", ekfac_np)
    np.save(ATTR_DIR / "kfac_scores_fullmodel.npy", kfac_np)

    ekfac_rank = np.argsort(-ekfac_np, axis=1)[:, :100]
    kfac_rank  = np.argsort(-kfac_np, axis=1)[:, :100]
    np.save(ATTR_DIR / "ekfac_rankings_fullmodel_top100.npy", ekfac_rank)
    np.save(ATTR_DIR / "kfac_rankings_fullmodel_top100.npy", kfac_rank)

    # ── Analysis ──
    j10 = []
    for i in range(n_test):
        s1 = set(ekfac_rank[i,:10].tolist())
        s2 = set(kfac_rank[i,:10].tolist())
        j10.append(len(s1 & s2) / len(s1 | s2))
    j10 = np.array(j10)
    print(f"\nJ@10(EK-FAC, K-FAC): mean={j10.mean():.4f}, std={j10.std():.4f}")

    labels = np.array([testset.targets[idx] for idx in test_indices])
    per_class_j10 = {str(c): float(j10[labels==c].mean()) for c in range(10)}

    j10_data = [{"test_idx": int(test_indices[i]), "jaccard_at_10": float(j10[i])}
                for i in range(n_test)]
    (ATTR_DIR / "jaccard_at_10_ekfac_kfac.json").write_text(json.dumps(j10_data, indent=2))

    # ── Test features ──
    report("features", "Test features", 93)
    features = compute_test_features(model, test_subset, test_indices, device)
    (ATTR_DIR / "test_features_500.json").write_text(json.dumps(features, indent=2))

    elapsed = time.time() - start_time

    result = {
        "task_id": TASK_ID, "mode": "FULL", "n_test": n_test, "n_train": n_train,
        "method_note": "bilinear_kfac_ekfac_layer4_fc_layerwise",
        "selected_layers": sorted(SELECTED_LAYERS),
        "n_layers": n_layers,
        "ekfac": {"success": True, "shape": list(ekfac_np.shape),
                  "damping": DAMPING_EKFAC, "full_model": True},
        "kfac": {"success": True, "shape": list(kfac_np.shape),
                 "damping": DAMPING_KFAC, "full_model": True},
        "analysis": {
            "jaccard_at_10_ekfac_kfac": {
                "mean": float(j10.mean()), "std": float(j10.std()),
                "min": float(j10.min()), "max": float(j10.max()),
                "per_class": per_class_j10,
            }
        },
        "elapsed_minutes": elapsed / 60,
        "timestamp": datetime.now().isoformat(),
    }
    (ATTR_DIR / "if_results.json").write_text(json.dumps(result, indent=2))

    pid_file.unlink(missing_ok=True)
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID,
        "status": "success",
        "summary": f"layer4+fc K-FAC/EK-FAC IF ({n_layers} layers). "
                   f"J@10 mean={j10.mean():.4f} std={j10.std():.4f}. {elapsed/60:.1f}min.",
        "timestamp": datetime.now().isoformat(),
    }))

    report("done", f"J@10 std={j10.std():.4f}", 100)
    print(f"\nTotal: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
