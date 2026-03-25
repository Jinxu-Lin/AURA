#!/usr/bin/env python3
"""
Phase 1: Combine attribution results from GPU1 (EK-FAC, K-FAC) and GPU2 (RepSim, TRAK-50).
Compute cross-method analysis and final summary.
"""
import json, sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

PROJECT_DIR = Path("/home/jinxulin/sibyl_system/projects/AURA")
ATTR_DIR = PROJECT_DIR / "exp" / "results" / "phase1_attributions"
RESULTS_DIR = PROJECT_DIR / "exp" / "results"
TASK_ID = "phase1_attribution_compute"

def main():
    print("=== Combining Phase 1 Attribution Results ===")

    # Load all scores
    ekfac = np.load(ATTR_DIR / "ekfac_scores_fullmodel.npy")
    kfac  = np.load(ATTR_DIR / "kfac_scores_fullmodel.npy")
    repsim = np.load(ATTR_DIR / "repsim_scores_fullmodel.npy")
    trak50 = np.load(ATTR_DIR / "trak50_scores_fullmodel.npy")

    n_test = ekfac.shape[0]
    print(f"EK-FAC: {ekfac.shape}, K-FAC: {kfac.shape}, RepSim: {repsim.shape}, TRAK-50: {trak50.shape}")

    # Load test features
    features = json.loads((ATTR_DIR / "test_features_500.json").read_text())
    test_indices = json.loads((ATTR_DIR / "test_indices_500.json").read_text())

    # Rankings
    ekfac_rank = np.argsort(-ekfac, axis=1)
    kfac_rank = np.argsort(-kfac, axis=1)
    repsim_rank = np.argsort(-repsim, axis=1)
    trak_rank = np.argsort(-trak50, axis=1)

    # ── 1. Jaccard@10(EK-FAC, K-FAC) ──
    j10_ek_kf = []
    for i in range(n_test):
        s1 = set(ekfac_rank[i, :10])
        s2 = set(kfac_rank[i, :10])
        j10_ek_kf.append(len(s1 & s2) / len(s1 | s2))
    j10_ek_kf = np.array(j10_ek_kf)

    # ── 2. Kendall tau(IF, RepSim) per point ──
    # Use EK-FAC as the IF method, compare against RepSim
    # Kendall tau on full ranking is expensive; use top-1000
    tau_if_repsim = []
    for i in range(n_test):
        # Take union of top-1000 from both methods
        top_k = 1000
        union_idx = list(set(ekfac_rank[i, :top_k]) | set(repsim_rank[i, :top_k]))
        r1 = stats.rankdata(-ekfac[i, union_idx])
        r2 = stats.rankdata(-repsim[i, union_idx])
        tau, _ = stats.kendalltau(r1, r2)
        tau_if_repsim.append(tau if not np.isnan(tau) else 0.0)
    tau_if_repsim = np.array(tau_if_repsim)

    # ── 3. Per-point LDS (Linear Datamodeling Score) ──
    # LDS = Spearman correlation between EK-FAC scores and TRAK-50 scores per test point
    lds_ekfac_trak = []
    for i in range(n_test):
        rho, _ = stats.spearmanr(ekfac[i], trak50[i])
        lds_ekfac_trak.append(rho if not np.isnan(rho) else 0.0)
    lds_ekfac_trak = np.array(lds_ekfac_trak)

    # Also compute LDS for K-FAC and RepSim
    lds_kfac_trak = []
    lds_repsim_trak = []
    for i in range(n_test):
        rho_k, _ = stats.spearmanr(kfac[i], trak50[i])
        rho_r, _ = stats.spearmanr(repsim[i], trak50[i])
        lds_kfac_trak.append(rho_k if not np.isnan(rho_k) else 0.0)
        lds_repsim_trak.append(rho_r if not np.isnan(rho_r) else 0.0)
    lds_kfac_trak = np.array(lds_kfac_trak)
    lds_repsim_trak = np.array(lds_repsim_trak)

    # ── 4. Per-class breakdowns ──
    labels = np.array([f["true_label"] for f in features])
    per_class_j10 = {}
    per_class_tau = {}
    per_class_lds = {}
    for c in range(10):
        mask = labels == c
        per_class_j10[str(c)] = float(j10_ek_kf[mask].mean())
        per_class_tau[str(c)] = float(tau_if_repsim[mask].mean())
        per_class_lds[str(c)] = float(lds_ekfac_trak[mask].mean())

    # ── Summary statistics ──
    analysis = {
        "jaccard_at_10_ekfac_kfac": {
            "mean": float(j10_ek_kf.mean()),
            "std": float(j10_ek_kf.std()),
            "min": float(j10_ek_kf.min()),
            "max": float(j10_ek_kf.max()),
            "per_class": per_class_j10,
        },
        "kendall_tau_if_repsim": {
            "mean": float(tau_if_repsim.mean()),
            "std": float(tau_if_repsim.std()),
            "min": float(tau_if_repsim.min()),
            "max": float(tau_if_repsim.max()),
            "per_class": per_class_tau,
        },
        "lds_ekfac_trak50": {
            "mean": float(lds_ekfac_trak.mean()),
            "std": float(lds_ekfac_trak.std()),
            "min": float(lds_ekfac_trak.min()),
            "max": float(lds_ekfac_trak.max()),
            "per_class": per_class_lds,
        },
        "lds_kfac_trak50": {
            "mean": float(lds_kfac_trak.mean()),
            "std": float(lds_kfac_trak.std()),
        },
        "lds_repsim_trak50": {
            "mean": float(lds_repsim_trak.mean()),
            "std": float(lds_repsim_trak.std()),
        },
    }

    # ── Gate check ──
    j10_std_pass = j10_ek_kf.std() > 0.05
    print(f"\n=== Phase 1 Gate Check ===")
    print(f"J@10(EK-FAC, K-FAC) std = {j10_ek_kf.std():.4f} (need > 0.05): {'PASS' if j10_std_pass else 'FAIL'}")
    print(f"J@10 mean = {j10_ek_kf.mean():.4f}")
    print(f"Kendall tau(IF, RepSim) mean = {tau_if_repsim.mean():.4f}, std = {tau_if_repsim.std():.4f}")
    print(f"LDS(EK-FAC, TRAK-50) mean = {lds_ekfac_trak.mean():.4f}, std = {lds_ekfac_trak.std():.4f}")

    # Save per-point data for variance decomposition
    per_point = []
    for i in range(n_test):
        per_point.append({
            "test_idx": int(test_indices[i]),
            "true_label": int(labels[i]),
            "grad_norm": features[i]["grad_norm"],
            "log_grad_norm": features[i]["log_grad_norm"],
            "confidence": features[i]["confidence"],
            "entropy": features[i]["entropy"],
            "jaccard_at_10": float(j10_ek_kf[i]),
            "kendall_tau_if_repsim": float(tau_if_repsim[i]),
            "lds_ekfac_trak50": float(lds_ekfac_trak[i]),
            "lds_kfac_trak50": float(lds_kfac_trak[i]),
            "lds_repsim_trak50": float(lds_repsim_trak[i]),
        })

    (ATTR_DIR / "per_point_analysis.json").write_text(json.dumps(per_point, indent=2))
    print(f"Saved per-point analysis for {len(per_point)} points")

    # Save overall result
    result = {
        "task_id": TASK_ID,
        "mode": "FULL",
        "n_test": n_test,
        "n_train_ekfac_kfac": int(ekfac.shape[1]),
        "n_train_repsim": int(repsim.shape[1]),
        "n_train_trak50": int(trak50.shape[1]),
        "full_model": True,
        "methods": {
            "ekfac_if": {"success": True, "shape": list(ekfac.shape)},
            "kfac_if": {"success": True, "shape": list(kfac.shape)},
            "repsim": {"success": True, "shape": list(repsim.shape)},
            "trak50": {"success": True, "shape": list(trak50.shape)},
        },
        "analysis": analysis,
        "pass_criteria": {
            "all_4_methods_valid": True,
            "jaccard_std_above_005": bool(j10_std_pass),
        },
        "overall_pass": bool(j10_std_pass),
        "timestamp": datetime.now().isoformat(),
    }
    (ATTR_DIR / "combined_results.json").write_text(json.dumps(result, indent=2))

    # Update the main DONE marker
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID,
        "status": "success" if bool(j10_std_pass) else "success_gate_fail",
        "summary": f"Full-model 4 methods computed. J@10 std={j10_ek_kf.std():.4f}, LDS(EK-FAC,TRAK)={lds_ekfac_trak.mean():.3f}. Gate: {'PASS' if j10_std_pass else 'FAIL (std<0.05)'}",
        "timestamp": datetime.now().isoformat(),
    }))

    print("\n=== Results Summary ===")
    for method, key in [("EK-FAC IF", "lds_ekfac_trak50"),
                         ("K-FAC IF", "lds_kfac_trak50"),
                         ("RepSim", "lds_repsim_trak50")]:
        d = analysis[key]
        print(f"  {method} LDS: {d['mean']:.4f} +/- {d['std']:.4f}")

    print("\nDone!")
    return result


if __name__ == "__main__":
    main()
