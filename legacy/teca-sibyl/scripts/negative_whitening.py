#!/usr/bin/env python3
"""
Negative Path H6: Whitening Decomposition Analysis

Tests whether ROME's C^{-1} whitening is the cause of the TECS alignment gap.

ROME computes: delta_W = lambda * (C^{-1} k)^T
The C^{-1} whitening rotates the key direction. If this rotation causes misalignment
with TDA gradients, then:
  - TECS_unwhitened (undo C^{-1} by multiplying delta_W by C) should be higher
  - TECS_whitened (apply C^{-1} to g_M) should also be higher

We compute three variants:
  1. TECS_raw = cos(delta_W, g_M) — from pilot results
  2. TECS_unwhitened = cos(C @ delta_W, g_M) — undo whitening on delta_W
  3. TECS_whitened = cos(delta_W, C^{-1} @ g_M) — apply whitening to g_M

Since delta_W is rank-1 (delta_u outer delta_v):
  - C @ delta_W = (C @ delta_u) outer delta_v  → efficient
  - C^{-1} @ g_M requires pseudo-inverse via eigendecomposition
"""

import json
import os
import sys
import time
import gc
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F

# ── Config ──────────────────────────────────────────────────────────────
SEED = 42
PROJECT_DIR = Path("/home/jinxulin/sibyl_system/projects/TECA")
RESULTS_DIR = PROJECT_DIR / "results"
ROME_DIR = RESULTS_DIR / "rome_deltas_slim"
TDA_DIR = RESULTS_DIR / "tda_gradients"
PILOT_FILE = RESULTS_DIR / "pilot_tecs_results.json"
OUTPUT_FILE = RESULTS_DIR / "negative_whitening_results.json"
COV_CACHE = RESULTS_DIR / "cov_matrix_layer17.pt"

EDIT_LAYER = 17
MODEL_NAME = "gpt2-xl"
N_COV_SAMPLES = 100  # forward passes for covariance estimation
MAX_SEQ_LEN = 256
TOP_K_EIGEN = 512    # for pseudo-inverse approximation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_pilot_results():
    """Load pilot TECS results and extract per-fact raw cosine values."""
    with open(PILOT_FILE) as f:
        pilot = json.load(f)
    case_map = {}
    for r in pilot["per_fact_results"]:
        case_map[r["case_id"]] = r["tecs_real"]
    return case_map


def load_delta(case_id):
    """Load rank-1 delta_W decomposition: delta_u (6400,), delta_v (1600,)."""
    path = ROME_DIR / f"delta_case{case_id}.pt"
    d = torch.load(path, map_location="cpu", weights_only=True)
    return d["delta_u"].float(), d["delta_v"].float()


def load_gM(case_id):
    """Load TDA gradient g_M (6400, 1600)."""
    path = TDA_DIR / f"g_M_{case_id}.pt"
    return torch.load(path, map_location="cpu", weights_only=True).float()


def compute_covariance_matrix():
    """
    Compute C = E[k k^T] where k is the MLP input activation at layer 17.

    For GPT-2-XL, the MLP input at layer `l` is the output of LayerNorm
    before c_fc, which has dimension d_ff_in = d_model = 1600.

    Wait -- the delta_u has shape 6400 (= 4 * 1600), which is the d_ff dimension.
    In GPT-2, c_fc maps 1600 → 6400. The weight is (6400, 1600).
    ROME edits c_fc.weight which is (out_features, in_features) = (6400, 1600).

    Actually, ROME's C is the covariance of the INPUT to c_fc, which has dim 1600.
    But delta_u has dim 6400... Let me re-examine.

    ROME formula: delta_W = (v_new - v_old) @ (C^{-1} k_star)^T / (k_star^T C^{-1} k_star)
    where k_star has dim = input dim of the MLP weight = 1600 for c_fc.

    But delta_u is 6400. Looking at the rank-1 decomposition stored:
    delta_W (6400, 1600) = delta_u (6400,) ⊗ delta_v (1600,)

    So delta_v corresponds to the key direction (1600-dim), and delta_u is the value direction.
    In ROME: delta_v = C^{-1} k_star / (k_star^T C^{-1} k_star), and delta_u = v_new - v_old.

    Therefore C is (1600, 1600) -- covariance of 1600-dim key vectors (MLP input).

    The whitening acts on the COLUMN space (1600-dim) of delta_W:
    - TECS_unwhitened: undo C^{-1} on delta_v → replace delta_v with C @ delta_v
      So new delta_W = delta_u ⊗ (C @ delta_v), shape still (6400, 1600)
    - TECS_whitened: apply C^{-1} to g_M's column space → g_M @ C^{-1}
      (C^{-1} acts on the 1600-dim key axis which is the column axis of g_M)
    """
    if COV_CACHE.exists():
        print(f"Loading cached covariance matrix from {COV_CACHE}")
        return torch.load(COV_CACHE, map_location="cpu", weights_only=True).float()

    print(f"Computing covariance matrix from {N_COV_SAMPLES} Wikipedia forward passes...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    ).to(DEVICE).eval()

    # Hook to capture MLP input at layer 17
    activations = []

    def hook_fn(module, input, output):
        # input[0] shape: (batch, seq_len, 1600)
        # Take mean over sequence positions
        inp = input[0].detach().float()  # (batch, seq, 1600)
        # Collect all token activations (subsample to control memory)
        for b in range(inp.shape[0]):
            # Take every 4th token to limit memory
            acts = inp[b, ::4, :]  # (seq/4, 1600)
            activations.append(acts.cpu())

    # Register hook on the c_fc layer (MLP first linear)
    target_layer = model.transformer.h[EDIT_LAYER].mlp.c_fc
    handle = target_layer.register_forward_hook(hook_fn)

    # Load Wikipedia data
    print("Loading Wikipedia dataset...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)

    texts = []
    for item in ds:
        t = item["text"].strip()
        if len(t) > 100:
            texts.append(t)
        if len(texts) >= N_COV_SAMPLES:
            break

    print(f"Running {len(texts)} forward passes...")
    with torch.no_grad():
        for i, text in enumerate(texts):
            tokens = tokenizer(
                text, return_tensors="pt", max_length=MAX_SEQ_LEN,
                truncation=True, padding=False
            ).to(DEVICE)
            model(**tokens)
            if (i + 1) % 20 == 0:
                print(f"  Forward pass {i+1}/{len(texts)}")

    handle.remove()
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Stack all activations and compute covariance
    print("Computing covariance from collected activations...")
    all_acts = torch.cat(activations, dim=0)  # (N_tokens, 1600)
    print(f"  Total tokens collected: {all_acts.shape[0]}")

    # Center
    mean_act = all_acts.mean(dim=0)
    all_acts_centered = all_acts - mean_act

    # C = (1/N) X^T X, shape (1600, 1600)
    N = all_acts_centered.shape[0]
    C = (all_acts_centered.T @ all_acts_centered) / N

    print(f"  C shape: {C.shape}, dtype: {C.dtype}")
    print(f"  C trace: {C.trace().item():.4f}, C[0,0]: {C[0,0].item():.6f}")

    # Save cache
    torch.save(C, COV_CACHE)
    print(f"  Saved covariance to {COV_CACHE}")

    del all_acts, all_acts_centered, activations
    gc.collect()

    return C


def compute_pseudo_inverse_components(C, top_k=TOP_K_EIGEN):
    """
    Compute top-k eigendecomposition of C for pseudo-inverse.
    C^{-1} ≈ V @ diag(1/eigenvalues) @ V^T using top-k eigenvectors.

    Returns eigenvalues and eigenvectors for efficient C^{-1} @ x computation.
    """
    print(f"Computing top-{top_k} eigendecomposition of C ({C.shape})...")
    t0 = time.time()

    # Use symmetric eigendecomposition (C is symmetric PSD)
    eigenvalues, eigenvectors = torch.linalg.eigh(C)

    # eigh returns eigenvalues in ascending order; take top-k (largest)
    eigenvalues = eigenvalues[-top_k:]
    eigenvectors = eigenvectors[:, -top_k:]

    # Filter out near-zero eigenvalues
    threshold = eigenvalues.max() * 1e-6
    valid = eigenvalues > threshold
    eigenvalues = eigenvalues[valid]
    eigenvectors = eigenvectors[:, valid]

    print(f"  Eigendecomposition done in {time.time()-t0:.1f}s")
    print(f"  Used {eigenvalues.shape[0]} eigenvalues (range: {eigenvalues.min():.6e} to {eigenvalues.max():.6e})")
    print(f"  Condition number (approx): {eigenvalues.max()/eigenvalues.min():.2e}")

    return eigenvalues, eigenvectors


def apply_C(C, v):
    """Compute C @ v where v is (1600,)."""
    return C @ v


def apply_C_inv(eigenvalues, eigenvectors, x):
    """
    Compute C^{-1} @ x using eigendecomposition.
    C^{-1} @ x = V @ diag(1/lambda) @ V^T @ x
    where x can be (1600,) or (1600, K).
    """
    # Project into eigenspace
    proj = eigenvectors.T @ x  # (k, ...)
    # Scale by inverse eigenvalues
    if proj.dim() == 1:
        scaled = proj / eigenvalues
    else:
        scaled = proj / eigenvalues.unsqueeze(1)
    # Project back
    return eigenvectors @ scaled


def cosine_sim_matrices(A, B):
    """Compute cosine similarity between two flattened matrices."""
    a = A.reshape(-1).float()
    b = B.reshape(-1).float()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    t_start = time.time()
    print("=" * 60)
    print("Negative Path H6: Whitening Decomposition Analysis")
    print("=" * 60)

    # 1. Load pilot results
    print("\n[1/5] Loading pilot TECS results...")
    case_tecs_raw = load_pilot_results()
    case_ids = sorted(case_tecs_raw.keys())
    print(f"  Loaded {len(case_ids)} cases")

    # 2. Compute or load covariance matrix
    print("\n[2/5] Computing covariance matrix C...")
    C = compute_covariance_matrix()
    print(f"  C shape: {C.shape}")

    # 3. Compute eigendecomposition for pseudo-inverse
    print("\n[3/5] Computing eigendecomposition for C^{{-1}}...")
    eigenvalues, eigenvectors = compute_pseudo_inverse_components(C)

    # 4. For each fact, compute three TECS variants
    print(f"\n[4/5] Computing TECS variants for {len(case_ids)} facts...")

    results = []
    tecs_raw_list = []
    tecs_unwhitened_list = []
    tecs_whitened_list = []

    for i, cid in enumerate(case_ids):
        # Load delta components
        delta_u, delta_v = load_delta(cid)  # (6400,), (1600,)
        g_M = load_gM(cid)  # (6400, 1600)

        # Reconstruct delta_W = delta_u ⊗ delta_v → (6400, 1600)
        delta_W = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)  # (6400, 1) @ (1, 1600) = (6400, 1600)

        # a. TECS_raw = cos(delta_W, g_M) — from pilot
        tecs_raw = case_tecs_raw[cid]

        # Verify raw computation matches pilot
        tecs_raw_computed = cosine_sim_matrices(delta_W, g_M)

        # b. TECS_unwhitened = cos(C @ delta_W, g_M)
        # delta_W = delta_u ⊗ delta_v
        # C @ delta_W: C acts on the column space (1600-dim)
        # Since delta_W[:, j] = delta_u * delta_v[j], and C is (1600, 1600):
        # Actually C @ delta_W doesn't make sense dimensionally if C is (1600, 1600) and delta_W is (6400, 1600).
        #
        # Let me reconsider. ROME's delta_W = lambda * k_hat^T where k_hat = C^{-1} k / (k^T C^{-1} k).
        # delta_W shape is (d_out, d_in) = (6400, 1600).
        # To "undo" whitening, we want to replace k_hat with just k (up to scaling).
        # delta_v ~ C^{-1} k, so C @ delta_v ~ k (the original un-whitened key).
        #
        # TECS_unwhitened: replace delta_v with C @ delta_v
        C_delta_v = apply_C(C, delta_v)  # (1600,)
        delta_W_unwhitened = delta_u.unsqueeze(1) @ C_delta_v.unsqueeze(0)  # (6400, 1600)
        tecs_unwhitened = cosine_sim_matrices(delta_W_unwhitened, g_M)

        # c. TECS_whitened = cos(delta_W, C^{-1} @ g_M)
        # Apply C^{-1} to the column (1600-dim) axis of g_M
        # g_M is (6400, 1600). C^{-1} is (1600, 1600).
        # g_M_whitened = g_M @ C^{-1}.T = (g_M.T → (1600, 6400), then C^{-1} @ g_M.T → (1600, 6400), transpose → (6400, 1600))
        # Actually: we want to whiten the key-axis of g_M.
        # g_M_whitened[i, :] = C^{-1} @ g_M[i, :] for each row i? No, that's wrong dimensionally.
        # g_M is (d_out=6400, d_in=1600). The key axis is d_in=1600 (columns).
        # To apply C^{-1} to the key axis: g_M_whitened = g_M @ C^{-1}
        # (6400, 1600) @ (1600, 1600) = (6400, 1600) ✓
        g_M_T = g_M.T  # (1600, 6400)
        g_M_whitened_T = apply_C_inv(eigenvalues, eigenvectors, g_M_T)  # (1600, 6400)
        g_M_whitened = g_M_whitened_T.T  # (6400, 1600)
        tecs_whitened = cosine_sim_matrices(delta_W, g_M_whitened)

        tecs_raw_list.append(tecs_raw)
        tecs_unwhitened_list.append(tecs_unwhitened)
        tecs_whitened_list.append(tecs_whitened)

        results.append({
            "case_id": cid,
            "tecs_raw": tecs_raw,
            "tecs_raw_computed": tecs_raw_computed,
            "tecs_unwhitened": tecs_unwhitened,
            "tecs_whitened": tecs_whitened,
        })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(case_ids)} facts")

        # Free memory
        del delta_W, delta_W_unwhitened, g_M, g_M_whitened, g_M_T, g_M_whitened_T

    tecs_raw_arr = np.array(tecs_raw_list)
    tecs_unwhitened_arr = np.array(tecs_unwhitened_list)
    tecs_whitened_arr = np.array(tecs_whitened_list)

    # 5. Statistical tests
    print("\n[5/5] Running statistical tests...")
    from scipy import stats

    def paired_test(a, b, name):
        """Paired t-test and Cohen's d."""
        diff = a - b
        t_stat, p_val = stats.ttest_rel(a, b)
        d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0
        ci_95 = stats.t.interval(0.95, len(diff)-1, loc=diff.mean(), scale=stats.sem(diff))
        return {
            "name": name,
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)),
            "cohens_d": float(d),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "ci_95_low": float(ci_95[0]),
            "ci_95_high": float(ci_95[1]),
            "n": len(diff),
        }

    test_unwhitened = paired_test(
        tecs_unwhitened_arr, tecs_raw_arr,
        "TECS_unwhitened vs TECS_raw (does undoing whitening increase alignment?)"
    )
    test_whitened = paired_test(
        tecs_whitened_arr, tecs_raw_arr,
        "TECS_whitened vs TECS_raw (does whitening g_M increase alignment?)"
    )

    # Also test absolute values (alignment magnitude)
    test_unwhitened_abs = paired_test(
        np.abs(tecs_unwhitened_arr), np.abs(tecs_raw_arr),
        "|TECS_unwhitened| vs |TECS_raw| (absolute alignment magnitude)"
    )
    test_whitened_abs = paired_test(
        np.abs(tecs_whitened_arr), np.abs(tecs_raw_arr),
        "|TECS_whitened| vs |TECS_raw| (absolute alignment magnitude)"
    )

    # Cross comparison
    test_unwhitened_vs_whitened = paired_test(
        tecs_unwhitened_arr, tecs_whitened_arr,
        "TECS_unwhitened vs TECS_whitened (which correction helps more?)"
    )

    # Correlation analysis
    corr_raw_unwhitened = float(np.corrcoef(tecs_raw_arr, tecs_unwhitened_arr)[0, 1])
    corr_raw_whitened = float(np.corrcoef(tecs_raw_arr, tecs_whitened_arr)[0, 1])

    # Summary statistics
    summary = {
        "tecs_raw": {
            "mean": float(tecs_raw_arr.mean()),
            "std": float(tecs_raw_arr.std()),
            "median": float(np.median(tecs_raw_arr)),
            "abs_mean": float(np.abs(tecs_raw_arr).mean()),
        },
        "tecs_unwhitened": {
            "mean": float(tecs_unwhitened_arr.mean()),
            "std": float(tecs_unwhitened_arr.std()),
            "median": float(np.median(tecs_unwhitened_arr)),
            "abs_mean": float(np.abs(tecs_unwhitened_arr).mean()),
        },
        "tecs_whitened": {
            "mean": float(tecs_whitened_arr.mean()),
            "std": float(tecs_whitened_arr.std()),
            "median": float(np.median(tecs_whitened_arr)),
            "abs_mean": float(np.abs(tecs_whitened_arr).mean()),
        },
    }

    # Decision
    # If whitening is the cause, we expect significant improvement in at least one variant
    h6_supported = (
        (test_unwhitened["p_value"] < 0.05 and test_unwhitened["cohens_d"] > 0.2) or
        (test_whitened["p_value"] < 0.05 and test_whitened["cohens_d"] > 0.2)
    )

    elapsed = time.time() - t_start

    output = {
        "task_id": "negative_whitening",
        "mode": "negative_path_h6",
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": elapsed,
        "config": {
            "n_facts": len(case_ids),
            "seed": SEED,
            "model": MODEL_NAME,
            "edit_layer": EDIT_LAYER,
            "n_cov_samples": N_COV_SAMPLES,
            "max_seq_len": MAX_SEQ_LEN,
            "top_k_eigen": TOP_K_EIGEN,
            "n_eigen_used": int(eigenvalues.shape[0]),
            "cov_condition_number": float(eigenvalues.max() / eigenvalues.min()),
        },
        "summary": summary,
        "statistical_tests": {
            "unwhitened_vs_raw": test_unwhitened,
            "whitened_vs_raw": test_whitened,
            "unwhitened_abs_vs_raw_abs": test_unwhitened_abs,
            "whitened_abs_vs_raw_abs": test_whitened_abs,
            "unwhitened_vs_whitened": test_unwhitened_vs_whitened,
        },
        "correlations": {
            "raw_vs_unwhitened": corr_raw_unwhitened,
            "raw_vs_whitened": corr_raw_whitened,
        },
        "decision": {
            "h6_whitening_explains_gap": h6_supported,
            "interpretation": (
                "SUPPORTED: Whitening correction significantly improves alignment"
                if h6_supported else
                "NOT SUPPORTED: Whitening correction does not significantly improve alignment"
            ),
            "unwhitened_d": test_unwhitened["cohens_d"],
            "whitened_d": test_whitened["cohens_d"],
            "unwhitened_p": test_unwhitened["p_value"],
            "whitened_p": test_whitened["p_value"],
        },
        "per_fact_results": results,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nTECS distributions:")
    for k, v in summary.items():
        print(f"  {k}: mean={v['mean']:.6f}, std={v['std']:.6f}, |mean|={v['abs_mean']:.6f}")

    print(f"\nStatistical tests:")
    print(f"  Unwhitened vs Raw: d={test_unwhitened['cohens_d']:.4f}, p={test_unwhitened['p_value']:.4f}")
    print(f"  Whitened vs Raw:   d={test_whitened['cohens_d']:.4f}, p={test_whitened['p_value']:.4f}")
    print(f"  |Unwhitened| vs |Raw|: d={test_unwhitened_abs['cohens_d']:.4f}, p={test_unwhitened_abs['p_value']:.4f}")
    print(f"  |Whitened| vs |Raw|:   d={test_whitened_abs['cohens_d']:.4f}, p={test_whitened_abs['p_value']:.4f}")

    print(f"\nCorrelations:")
    print(f"  raw vs unwhitened: r={corr_raw_unwhitened:.4f}")
    print(f"  raw vs whitened:   r={corr_raw_whitened:.4f}")

    print(f"\nDecision: H6 {'SUPPORTED' if h6_supported else 'NOT SUPPORTED'}")
    print(f"  (unwhitened d={test_unwhitened['cohens_d']:.4f}, whitened d={test_whitened['cohens_d']:.4f})")

    print(f"\nResults saved to {OUTPUT_FILE}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
