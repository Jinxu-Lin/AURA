#!/usr/bin/env python3
"""
Negative Path H7: Subspace Geometry — Principal Angle Analysis

Tests whether the misalignment between editing subspace (delta_W) and
attribution subspace (g_M) is structured or random.

Key optimization: Since D and G each have 100 rows in R^{10M}, both live in
at most a 100-dim subspace. We first project into a joint subspace of dim <= 200
(union of row-spaces of D and G), then compute all principal angles and null
distributions in this reduced space. This makes the null distribution computation
(100 random trials) take seconds instead of hours.

Steps:
1. Load 100 delta_W and 100 g_M tensors, form D (100x10M) and G (100x10M)
2. Project into joint subspace of dim ~200 via QR on [D; G]^T
3. Compute SVD in reduced space for top-k principal components
4. Compute principal angles between editing and attribution subspaces
5. Compare with null: random k-dim subspaces in the joint ~200-dim space
6. Report Grassmann distance, effective dimensionality, explained variance
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
OUTPUT_FILE = RESULTS_DIR / "negative_subspace_results.json"

TASK_ID = "negative_subspace_geometry"
K_VALUES = [10, 20, 50]
N_RANDOM_TRIALS = 1000   # cheap in reduced space, so use more trials
DEVICE = "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── PID file ────────────────────────────────────────────────────────────
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))


def report_progress(task_id, results_dir, epoch, total_epochs, step=0,
                    total_steps=0, loss=None, metric=None):
    progress = Path(results_dir) / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_task_done(task_id, results_dir, status="success", summary=""):
    pid_f = Path(results_dir) / f"{task_id}.pid"
    if pid_f.exists():
        pid_f.unlink()
    progress_file = Path(results_dir) / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    marker = Path(results_dir) / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


def find_common_case_ids():
    rome_ids = set()
    for f in os.listdir(ROME_DIR):
        if f.startswith('delta_case') and f.endswith('.pt'):
            cid = f.replace('delta_case', '').replace('.pt', '')
            rome_ids.add(cid)
    tda_ids = set()
    for f in os.listdir(TDA_DIR):
        if f.startswith('g_M_') and f.endswith('.pt'):
            cid = f.replace('g_M_', '').replace('.pt', '')
            tda_ids.add(cid)
    common = sorted(rome_ids & tda_ids, key=int)
    return common


def load_matrices(case_ids):
    """Load and stack delta_W (rank-1 outer products) and g_M tensors."""
    n = len(case_ids)
    dim = 6400 * 1600
    print(f"Loading {n} tensor pairs (dim={dim})...")

    D_rows = []
    G_rows = []

    for i, cid in enumerate(case_ids):
        delta_data = torch.load(
            ROME_DIR / f"delta_case{cid}.pt",
            map_location='cpu', weights_only=False
        )
        delta_u = delta_data['delta_u'].float()
        delta_v = delta_data['delta_v'].float()
        delta_flat = torch.kron(delta_u, delta_v)
        D_rows.append(delta_flat)

        g_M = torch.load(
            TDA_DIR / f"g_M_{cid}.pt",
            map_location='cpu', weights_only=False
        ).float().flatten()
        G_rows.append(g_M)

        if (i + 1) % 20 == 0:
            print(f"  Loaded {i+1}/{n} pairs", flush=True)

    print("Stacking into matrices...", flush=True)
    D = torch.stack(D_rows, dim=0)
    G = torch.stack(G_rows, dim=0)
    del D_rows, G_rows
    gc.collect()

    print(f"  D shape: {D.shape}, G shape: {G.shape}")
    print(f"  Memory: {(D.element_size() * D.nelement() + G.element_size() * G.nelement()) / 1e9:.2f} GB")
    return D, G


def project_to_joint_subspace(D, G):
    """
    Project D and G into their joint row-space.

    The joint row-space of D and G has dimension <= 200 (100 rows each).
    We compute an orthonormal basis Q for this space, then represent
    D and G as D_proj = D @ Q and G_proj = G @ Q.

    All subspace geometry (principal angles, etc.) is preserved exactly
    because we're only discarding the null space that neither D nor G occupy.

    Implementation: Stack [D; G] as (200, dim), compute SVD to get Q.
    """
    print("\nProjecting to joint subspace...", flush=True)
    t0 = time.time()

    n_d, dim = D.shape
    n_g = G.shape[0]

    # Stack D and G: (200, dim)
    DG = torch.cat([D, G], dim=0)  # (200, dim)

    # Center
    mean_vec = DG.mean(dim=0, keepdim=True)
    DG_centered = DG - mean_vec

    # SVD to find the joint subspace
    # DG = U @ S @ V^T, V columns span the row-space
    # Use svd_lowrank for efficiency
    max_rank = min(n_d + n_g, 200)
    U_joint, S_joint, V_joint = torch.svd_lowrank(DG_centered, q=max_rank, niter=5)

    # Determine effective rank (drop near-zero singular values)
    threshold = S_joint[0] * 1e-10
    effective_rank = (S_joint > threshold).sum().item()
    V_basis = V_joint[:, :effective_rank]  # (dim, effective_rank)

    print(f"  Joint subspace rank: {effective_rank} (out of {max_rank})")
    print(f"  Top-5 singular values: {S_joint[:5].tolist()}")

    # Project D and G into this subspace
    D_proj = (D - mean_vec) @ V_basis  # (100, effective_rank)
    G_proj = (G - mean_vec) @ V_basis  # (100, effective_rank)

    elapsed = time.time() - t0
    print(f"  Projection done in {elapsed:.1f}s")
    print(f"  D_proj shape: {D_proj.shape}, G_proj shape: {G_proj.shape}")

    return D_proj, G_proj, V_basis, S_joint, effective_rank


def compute_principal_angles(V1, V2):
    """Principal angles between subspaces spanned by columns of V1 and V2."""
    M = V1.T @ V2
    _, sigmas, _ = torch.linalg.svd(M)
    sigmas = torch.clamp(sigmas, 0.0, 1.0)
    return torch.acos(sigmas)


def compute_grassmann_distance(angles):
    return torch.sqrt(torch.sum(angles ** 2)).item()


def compute_effective_dimensionality(S):
    s2 = S ** 2
    s4 = S ** 4
    return (s2.sum() ** 2 / s4.sum()).item()


def compute_explained_variance_ratio(S):
    s2 = S ** 2
    total = s2.sum()
    return (s2 / total).tolist()


def get_subspace_basis(M_proj, k):
    """
    Get top-k right singular vectors of M_proj (n x reduced_dim).
    These span the k-dim subspace of M in the reduced coordinate system.
    """
    U, S, V = torch.linalg.svd(M_proj, full_matrices=False)
    return V[:k, :].T, S  # V_k: (reduced_dim, k), S: singular values


def random_subspace_in_reduced(reduced_dim, k, rng):
    """
    Generate a uniformly random k-dim subspace in R^{reduced_dim}.
    Returns orthonormal basis Q: (reduced_dim, k).
    """
    R = rng.standard_normal((reduced_dim, k)).astype(np.float32)
    Q, _ = np.linalg.qr(R)
    return torch.from_numpy(Q[:, :k])


def main():
    start_time = time.time()
    print("=" * 60)
    print("Negative Path H7: Subspace Geometry Analysis")
    print("=" * 60, flush=True)

    report_progress(TASK_ID, RESULTS_DIR, 0, 8, step=0, total_steps=8,
                    metric={"phase": "loading_data"})

    # Step 1: Load data
    case_ids = find_common_case_ids()
    n_facts = len(case_ids)
    print(f"\nFound {n_facts} common case IDs")
    D, G = load_matrices(case_ids)
    dim = D.shape[1]

    report_progress(TASK_ID, RESULTS_DIR, 1, 8, step=1, total_steps=8,
                    metric={"phase": "data_loaded", "n_facts": n_facts, "dim": dim})

    # Step 2: Project to joint subspace
    D_proj, G_proj, V_basis, S_joint, effective_rank = project_to_joint_subspace(D, G)

    # Free original large matrices
    del D, G
    gc.collect()

    report_progress(TASK_ID, RESULTS_DIR, 2, 8, step=2, total_steps=8,
                    metric={"phase": "projected", "effective_rank": effective_rank})

    # Step 3: Compute SVD of D_proj and G_proj in reduced space
    max_k = max(K_VALUES)
    print(f"\nStep 3: SVD of projected matrices (reduced dim={effective_rank})...", flush=True)

    V_D_full, S_D = get_subspace_basis(D_proj, min(max_k, D_proj.shape[0]))
    V_G_full, S_G = get_subspace_basis(G_proj, min(max_k, G_proj.shape[0]))

    print(f"  D singular values (top-5): {S_D[:5].tolist()}")
    print(f"  G singular values (top-5): {S_G[:5].tolist()}")

    report_progress(TASK_ID, RESULTS_DIR, 3, 8, step=3, total_steps=8,
                    metric={"phase": "svd_complete"})

    # Step 4: Subspace properties
    print("\nStep 4: Computing subspace properties...", flush=True)

    evr_D = compute_explained_variance_ratio(S_D[:max_k])
    evr_G = compute_explained_variance_ratio(S_G[:max_k])
    eff_dim_D = compute_effective_dimensionality(S_D[:max_k])
    eff_dim_G = compute_effective_dimensionality(S_G[:max_k])

    print(f"  D: effective dim = {eff_dim_D:.1f}")
    print(f"     cumulative var top-10: {sum(evr_D[:10]):.4f}")
    print(f"     cumulative var top-20: {sum(evr_D[:20]):.4f}")
    print(f"     cumulative var top-50: {sum(evr_D[:50]):.4f}")
    print(f"  G: effective dim = {eff_dim_G:.1f}")
    print(f"     cumulative var top-10: {sum(evr_G[:10]):.4f}")
    print(f"     cumulative var top-20: {sum(evr_G[:20]):.4f}")
    print(f"     cumulative var top-50: {sum(evr_G[:50]):.4f}")

    report_progress(TASK_ID, RESULTS_DIR, 4, 8, step=4, total_steps=8,
                    metric={"phase": "properties_computed",
                            "eff_dim_D": eff_dim_D, "eff_dim_G": eff_dim_G})

    # Step 5: Principal angles for each k
    print("\nStep 5: Computing principal angles...", flush=True)

    principal_angle_results = {}
    for k in K_VALUES:
        V_D_k = V_D_full[:, :k]  # (reduced_dim, k)
        V_G_k = V_G_full[:, :k]

        angles = compute_principal_angles(V_D_k, V_G_k)
        angles_deg = torch.rad2deg(angles).tolist()
        grassmann_dist = compute_grassmann_distance(angles)

        print(f"\n  k = {k}:")
        print(f"    Min angle: {min(angles_deg):.2f} deg")
        print(f"    Max angle: {max(angles_deg):.2f} deg")
        print(f"    Mean angle: {np.mean(angles_deg):.2f} deg")
        print(f"    Median angle: {np.median(angles_deg):.2f} deg")
        print(f"    Grassmann distance: {grassmann_dist:.4f}")
        print(f"    Angles < 45 deg: {sum(1 for a in angles_deg if a < 45)}/{k}")
        print(f"    Angles < 30 deg: {sum(1 for a in angles_deg if a < 30)}/{k}")

        principal_angle_results[f"k{k}"] = {
            "k": k,
            "angles_degrees": angles_deg,
            "min_angle": min(angles_deg),
            "max_angle": max(angles_deg),
            "mean_angle": float(np.mean(angles_deg)),
            "median_angle": float(np.median(angles_deg)),
            "std_angle": float(np.std(angles_deg)),
            "grassmann_distance": grassmann_dist,
            "n_angles_below_45": sum(1 for a in angles_deg if a < 45),
            "n_angles_below_30": sum(1 for a in angles_deg if a < 30),
            "n_angles_below_15": sum(1 for a in angles_deg if a < 15),
        }

    report_progress(TASK_ID, RESULTS_DIR, 5, 8, step=5, total_steps=8,
                    metric={"phase": "principal_angles_computed"})

    # Step 6: Null distribution in reduced space
    # Generate random k-dim subspaces in R^{effective_rank} and compare with V_D_k
    print(f"\nStep 6: Null distribution ({N_RANDOM_TRIALS} trials in reduced dim={effective_rank})...", flush=True)

    rng = np.random.RandomState(SEED)
    null_results = {}

    for k in K_VALUES:
        print(f"\n  k = {k}:", flush=True)
        V_D_k = V_D_full[:, :k]

        random_min_angles = []
        random_mean_angles = []
        random_grassmann_dists = []

        t0 = time.time()
        for trial in range(N_RANDOM_TRIALS):
            V_rand = random_subspace_in_reduced(effective_rank, k, rng)
            angles = compute_principal_angles(V_D_k, V_rand)
            angles_deg = torch.rad2deg(angles)

            random_min_angles.append(angles_deg.min().item())
            random_mean_angles.append(angles_deg.mean().item())
            random_grassmann_dists.append(compute_grassmann_distance(angles))

        elapsed_null = time.time() - t0
        print(f"    {N_RANDOM_TRIALS} trials in {elapsed_null:.1f}s")

        real_min = principal_angle_results[f"k{k}"]["min_angle"]
        real_mean = principal_angle_results[f"k{k}"]["mean_angle"]
        real_grassmann = principal_angle_results[f"k{k}"]["grassmann_distance"]

        p_value_min = sum(1 for a in random_min_angles if a <= real_min) / N_RANDOM_TRIALS
        p_value_mean = sum(1 for a in random_mean_angles if a <= real_mean) / N_RANDOM_TRIALS
        p_value_grassmann = sum(1 for d in random_grassmann_dists if d <= real_grassmann) / N_RANDOM_TRIALS

        print(f"    Real min angle: {real_min:.2f} deg")
        print(f"    Null mean min:  {np.mean(random_min_angles):.2f} +/- {np.std(random_min_angles):.2f} deg")
        print(f"    p-value (min):  {p_value_min:.4f}")
        print(f"    Real mean angle: {real_mean:.2f} deg")
        print(f"    Null mean:       {np.mean(random_mean_angles):.2f} +/- {np.std(random_mean_angles):.2f} deg")
        print(f"    p-value (mean):  {p_value_mean:.4f}")
        print(f"    Real Grassmann:  {real_grassmann:.4f}")
        print(f"    Null Grassmann:  {np.mean(random_grassmann_dists):.4f} +/- {np.std(random_grassmann_dists):.4f}")
        print(f"    p-value (Grass): {p_value_grassmann:.4f}")

        null_results[f"k{k}"] = {
            "k": k,
            "n_trials": N_RANDOM_TRIALS,
            "reduced_dim": effective_rank,
            "random_min_angles": {
                "mean": float(np.mean(random_min_angles)),
                "std": float(np.std(random_min_angles)),
                "min": float(np.min(random_min_angles)),
                "max": float(np.max(random_min_angles)),
                "q05": float(np.percentile(random_min_angles, 5)),
                "q95": float(np.percentile(random_min_angles, 95)),
            },
            "random_mean_angles": {
                "mean": float(np.mean(random_mean_angles)),
                "std": float(np.std(random_mean_angles)),
            },
            "random_grassmann_dists": {
                "mean": float(np.mean(random_grassmann_dists)),
                "std": float(np.std(random_grassmann_dists)),
            },
            "p_value_min_angle": p_value_min,
            "p_value_mean_angle": p_value_mean,
            "p_value_grassmann": p_value_grassmann,
            "structured_misalignment": p_value_min < 0.01,
        }

    report_progress(TASK_ID, RESULTS_DIR, 6, 8, step=6, total_steps=8,
                    metric={"phase": "null_distribution_computed"})

    # Step 7: Cross-projection diagnostics
    print("\nStep 7: Cross-projection diagnostics...", flush=True)

    cross_projection = {}
    for k in K_VALUES:
        V_D_k = V_D_full[:, :k]
        V_G_k = V_G_full[:, :k]

        # Variance of D_proj in G's subspace
        D_proj_G = D_proj @ V_G_k  # (100, k)
        var_D_in_G = (D_proj_G ** 2).sum().item()
        var_D_total = (D_proj ** 2).sum().item()

        G_proj_D = G_proj @ V_D_k
        var_G_in_D = (G_proj_D ** 2).sum().item()
        var_G_total = (G_proj ** 2).sum().item()

        D_self = D_proj @ V_D_k
        G_self = G_proj @ V_G_k

        cross_projection[f"k{k}"] = {
            "k": k,
            "D_variance_in_G_subspace": var_D_in_G / var_D_total,
            "G_variance_in_D_subspace": var_G_in_D / var_G_total,
            "D_self_variance_ratio": (D_self ** 2).sum().item() / var_D_total,
            "G_self_variance_ratio": (G_self ** 2).sum().item() / var_G_total,
        }

        print(f"  k={k}:")
        print(f"    D variance in G subspace: {cross_projection[f'k{k}']['D_variance_in_G_subspace']:.6f}")
        print(f"    G variance in D subspace: {cross_projection[f'k{k}']['G_variance_in_D_subspace']:.6f}")
        print(f"    D self-capture: {cross_projection[f'k{k}']['D_self_variance_ratio']:.6f}")
        print(f"    G self-capture: {cross_projection[f'k{k}']['G_self_variance_ratio']:.6f}")

    # Singular value spectra
    singular_values = {
        "D": S_D[:max_k].tolist(),
        "G": S_G[:max_k].tolist(),
        "D_condition_number": (S_D[0] / S_D[min(max_k, len(S_D))-1]).item() if S_D[min(max_k, len(S_D))-1] > 0 else float('inf'),
        "G_condition_number": (S_G[0] / S_G[min(max_k, len(S_G))-1]).item() if S_G[min(max_k, len(S_G))-1] > 0 else float('inf'),
    }

    # Elbow criterion
    cum_var_D = np.cumsum(evr_D)
    cum_var_G = np.cumsum(evr_G)
    elbow_D_90 = int(np.searchsorted(cum_var_D, 0.90)) + 1
    elbow_G_90 = int(np.searchsorted(cum_var_G, 0.90)) + 1
    elbow_D_80 = int(np.searchsorted(cum_var_D, 0.80)) + 1
    elbow_G_80 = int(np.searchsorted(cum_var_G, 0.80)) + 1

    print(f"\n  Elbow (90% var): D={elbow_D_90}, G={elbow_G_90}")
    print(f"  Elbow (80% var): D={elbow_D_80}, G={elbow_G_80}")

    report_progress(TASK_ID, RESULTS_DIR, 7, 8, step=7, total_steps=8,
                    metric={"phase": "diagnostics_complete"})

    # Step 8: Decision
    print("\nStep 8: Final decision...", flush=True)

    decisions = {}
    for k in K_VALUES:
        key = f"k{k}"
        pa = principal_angle_results[key]
        null = null_results[key]
        structured = null["structured_misalignment"]
        decisions[key] = {
            "k": k,
            "structured_misalignment": structured,
            "min_angle_deg": pa["min_angle"],
            "null_min_angle_mean": null["random_min_angles"]["mean"],
            "null_min_angle_std": null["random_min_angles"]["std"],
            "p_value": null["p_value_min_angle"],
            "interpretation": (
                f"STRUCTURED: Min principal angle ({pa['min_angle']:.1f} deg) significantly smaller "
                f"than random ({null['random_min_angles']['mean']:.1f} +/- {null['random_min_angles']['std']:.1f} deg), "
                f"p={null['p_value_min_angle']:.4f}"
            ) if structured else (
                f"RANDOM: Min principal angle ({pa['min_angle']:.1f} deg) not significantly different "
                f"from random ({null['random_min_angles']['mean']:.1f} +/- {null['random_min_angles']['std']:.1f} deg), "
                f"p={null['p_value_min_angle']:.4f}"
            )
        }
        print(f"  k={k}: {'STRUCTURED' if structured else 'RANDOM'} (p={null['p_value_min_angle']:.4f})")

    elapsed = time.time() - start_time

    results = {
        "task_id": TASK_ID,
        "mode": "negative_path_h7",
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": elapsed,
        "config": {
            "n_facts": n_facts,
            "original_dim": 10240000,
            "reduced_dim": effective_rank,
            "seed": SEED,
            "k_values": K_VALUES,
            "n_random_trials": N_RANDOM_TRIALS,
            "device": DEVICE,
            "note": "Null distribution computed in joint subspace (dim=reduced_dim) "
                    "which exactly preserves all subspace geometry since D and G "
                    "have at most 100 rows each."
        },
        "subspace_properties": {
            "D_editing": {
                "effective_dimensionality": eff_dim_D,
                "explained_variance_ratio": evr_D,
                "cumulative_variance": {
                    "top_10": sum(evr_D[:10]),
                    "top_20": sum(evr_D[:20]),
                    "top_50": sum(evr_D[:min(50, len(evr_D))]),
                },
                "elbow_90pct": elbow_D_90,
                "elbow_80pct": elbow_D_80,
            },
            "G_attribution": {
                "effective_dimensionality": eff_dim_G,
                "explained_variance_ratio": evr_G,
                "cumulative_variance": {
                    "top_10": sum(evr_G[:10]),
                    "top_20": sum(evr_G[:20]),
                    "top_50": sum(evr_G[:min(50, len(evr_G))]),
                },
                "elbow_90pct": elbow_G_90,
                "elbow_80pct": elbow_G_80,
            },
        },
        "singular_values": singular_values,
        "principal_angles": principal_angle_results,
        "null_distribution": null_results,
        "cross_projection": cross_projection,
        "decisions": decisions,
        "overall_decision": {
            "any_structured": any(d["structured_misalignment"] for d in decisions.values()),
            "all_structured": all(d["structured_misalignment"] for d in decisions.values()),
            "summary": (
                "STRUCTURED misalignment detected — editing and attribution subspaces "
                "share significantly more geometric overlap than random subspaces, "
                "suggesting the misalignment has a specific, non-random structure"
            ) if any(d["structured_misalignment"] for d in decisions.values())
            else (
                "RANDOM misalignment: editing and attribution subspaces are approximately "
                "as orthogonal as random subspaces — no structured overlap detected"
            ),
        },
    }

    OUTPUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {OUTPUT_FILE}")
    print(f"Total elapsed: {elapsed:.1f}s", flush=True)

    mark_task_done(TASK_ID, RESULTS_DIR, status="success",
                   summary=results["overall_decision"]["summary"])

    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        mark_task_done(TASK_ID, RESULTS_DIR, status="failed", summary=str(e))
        sys.exit(1)
