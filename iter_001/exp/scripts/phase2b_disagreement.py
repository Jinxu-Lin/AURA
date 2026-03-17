#!/usr/bin/env python3
"""
Phase 2b: Cross-Method Disagreement Analysis (H-D4)

Using Phase 1 attribution data (no new GPU compute):
1. Label points as IF-better or RepSim-better by LDS
2. Compute per-point Kendall tau(IF, RepSim)
3. Compute global AUROC of tau as predictor
4. Compute class-stratified AUROC (mean across 10 classes)

Gate: global AUROC > 0.60 AND class-stratified > 0.55
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score

# Task metadata
TASK_ID = "phase2b_disagreement"
PROJECT_DIR = Path("/home/jinxulin/sibyl_system/projects/AURA")
RESULTS_DIR = PROJECT_DIR / "exp" / "results"
ATTR_DIR = RESULTS_DIR / "phase1_attributions"

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
    pid_file = Path(results_dir) / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()
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

def main():
    # Write PID file
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

    print("=" * 60)
    print("Phase 2b: Cross-Method Disagreement Analysis (H-D4)")
    print("=" * 60)

    report_progress(TASK_ID, RESULTS_DIR, 0, 4, metric={"stage": "loading_data"})

    # ==========================================
    # Step 1: Load Phase 1 attribution data
    # ==========================================
    print("\n[Step 1] Loading Phase 1 attribution data...")

    # Load attribution scores (100 test x 5000 train)
    ekfac_scores = np.load(ATTR_DIR / "ekfac_scores_5k.npy")  # (100, 5000)
    kfac_scores = np.load(ATTR_DIR / "kfac_scores_5k.npy")    # (100, 5000)
    trak_scores = np.load(ATTR_DIR / "trak_scores_5k.npy")    # (100, 5000)

    # RepSim scores are against full 50K train but we need 5K subset for comparison
    repsim_scores_5k = np.load(ATTR_DIR / "repsim_scores_5k.npy")  # (100, 5000)

    # Load test features
    with open(ATTR_DIR / "test_features.json") as f:
        features = json.load(f)

    labels = np.array(features["labels"])
    grad_norms = np.array(features["gradient_norms"])
    confidences = np.array(features["confidences"])
    entropies = np.array(features["entropies"])

    n_test = len(labels)
    n_train = ekfac_scores.shape[1]

    print(f"  Loaded {n_test} test points, {n_train} training points")
    print(f"  EK-FAC scores shape: {ekfac_scores.shape}")
    print(f"  RepSim scores shape: {repsim_scores_5k.shape}")
    print(f"  TRAK scores shape: {trak_scores.shape}")
    print(f"  Classes: {np.unique(labels)} (counts: {np.bincount(labels)})")

    report_progress(TASK_ID, RESULTS_DIR, 1, 4, metric={"stage": "computing_lds"})

    # ==========================================
    # Step 2: Compute per-point LDS for IF and RepSim
    # ==========================================
    print("\n[Step 2] Computing per-point LDS (Spearman correlation with TRAK ground truth)...")

    # LDS = Spearman rank correlation between method's attributions and TRAK attributions
    # per test point (across training points)
    lds_if = np.zeros(n_test)      # LDS for EK-FAC IF
    lds_repsim = np.zeros(n_test)  # LDS for RepSim
    lds_kfac = np.zeros(n_test)    # LDS for K-FAC IF

    for i in range(n_test):
        # EK-FAC IF vs TRAK
        rho_if, _ = spearmanr(ekfac_scores[i], trak_scores[i])
        lds_if[i] = rho_if if not np.isnan(rho_if) else 0.0

        # RepSim vs TRAK
        rho_rep, _ = spearmanr(repsim_scores_5k[i], trak_scores[i])
        lds_repsim[i] = rho_rep if not np.isnan(rho_rep) else 0.0

        # K-FAC IF vs TRAK
        rho_kfac, _ = spearmanr(kfac_scores[i], trak_scores[i])
        lds_kfac[i] = rho_kfac if not np.isnan(rho_kfac) else 0.0

    print(f"  LDS(EK-FAC IF, TRAK): mean={lds_if.mean():.4f}, std={lds_if.std():.4f}")
    print(f"  LDS(RepSim, TRAK):    mean={lds_repsim.mean():.4f}, std={lds_repsim.std():.4f}")
    print(f"  LDS(K-FAC IF, TRAK):  mean={lds_kfac.mean():.4f}, std={lds_kfac.std():.4f}")

    # ==========================================
    # Step 3: Label points as IF-better or RepSim-better
    # ==========================================
    print("\n[Step 3] Labeling points as IF-better vs RepSim-better...")

    lds_diff = lds_if - lds_repsim  # positive = IF better
    if_better = (lds_diff > 0).astype(int)  # 1 = IF better, 0 = RepSim better

    n_if_better = if_better.sum()
    n_repsim_better = n_test - n_if_better
    print(f"  IF-better: {n_if_better} ({100*n_if_better/n_test:.1f}%)")
    print(f"  RepSim-better: {n_repsim_better} ({100*n_repsim_better/n_test:.1f}%)")
    print(f"  LDS diff: mean={lds_diff.mean():.4f}, std={lds_diff.std():.4f}")

    # Per-class breakdown
    print("\n  Per-class IF-better fraction:")
    per_class_if_better = {}
    for c in range(10):
        mask = labels == c
        if mask.sum() > 0:
            frac = if_better[mask].mean()
            per_class_if_better[str(c)] = float(frac)
            print(f"    Class {c}: {frac:.2f} ({if_better[mask].sum()}/{mask.sum()})")

    report_progress(TASK_ID, RESULTS_DIR, 2, 4, metric={"stage": "computing_kendall_tau"})

    # ==========================================
    # Step 4: Compute per-point Kendall tau(IF, RepSim)
    # ==========================================
    print("\n[Step 4] Computing per-point Kendall tau(IF rankings, RepSim rankings)...")

    # For efficiency, use top-K rankings instead of all 5000 train points
    # Kendall tau on 5000 pairs is still feasible
    tau_values = np.zeros(n_test)
    tau_pvalues = np.zeros(n_test)

    for i in range(n_test):
        # Kendall tau between IF and RepSim rankings for this test point
        tau, pval = kendalltau(ekfac_scores[i], repsim_scores_5k[i])
        tau_values[i] = tau if not np.isnan(tau) else 0.0
        tau_pvalues[i] = pval if not np.isnan(pval) else 1.0

    print(f"  Kendall tau(IF, RepSim): mean={tau_values.mean():.4f}, std={tau_values.std():.4f}")
    print(f"  Range: [{tau_values.min():.4f}, {tau_values.max():.4f}]")
    print(f"  Significant (p<0.05): {(tau_pvalues < 0.05).sum()}/{n_test}")

    # Also compute per-point Kendall tau(K-FAC IF, RepSim) for comparison
    tau_kfac_repsim = np.zeros(n_test)
    for i in range(n_test):
        tau, _ = kendalltau(kfac_scores[i], repsim_scores_5k[i])
        tau_kfac_repsim[i] = tau if not np.isnan(tau) else 0.0

    report_progress(TASK_ID, RESULTS_DIR, 3, 4, metric={"stage": "computing_auroc"})

    # ==========================================
    # Step 5: Compute AUROC - tau as predictor of IF-better
    # ==========================================
    print("\n[Step 5] Computing AUROC of tau as predictor of IF-better...")

    # Global AUROC: can tau predict which test points have IF > RepSim in LDS?
    # Higher tau => IF and RepSim agree more => may predict IF reliability
    # Or lower tau => more disagreement => one method might be failing

    # Check if we have both classes
    if n_if_better == 0 or n_repsim_better == 0:
        print("  WARNING: All points in same class, AUROC undefined")
        global_auroc = 0.5
        global_auroc_note = "degenerate: all points in same class"
    else:
        # Try both directions: tau predicting IF-better, and -tau predicting IF-better
        auroc_pos = roc_auc_score(if_better, tau_values)
        auroc_neg = roc_auc_score(if_better, -tau_values)

        # Also try absolute tau as predictor
        auroc_abs = roc_auc_score(if_better, np.abs(tau_values))
        auroc_abs_neg = roc_auc_score(if_better, -np.abs(tau_values))

        global_auroc = max(auroc_pos, auroc_neg, auroc_abs, auroc_abs_neg)
        best_direction = ["tau", "-tau", "|tau|", "-|tau|"][
            np.argmax([auroc_pos, auroc_neg, auroc_abs, auroc_abs_neg])]

        print(f"  AUROC(tau -> IF-better):    {auroc_pos:.4f}")
        print(f"  AUROC(-tau -> IF-better):   {auroc_neg:.4f}")
        print(f"  AUROC(|tau| -> IF-better):  {auroc_abs:.4f}")
        print(f"  AUROC(-|tau| -> IF-better): {auroc_abs_neg:.4f}")
        print(f"  Best: {best_direction} with AUROC = {global_auroc:.4f}")
        global_auroc_note = f"best_direction={best_direction}"

    # ==========================================
    # Step 6: Class-stratified AUROC
    # ==========================================
    print("\n[Step 6] Computing class-stratified AUROC...")

    class_aurocs = {}
    class_auroc_valid = 0
    class_auroc_sum = 0.0

    for c in range(10):
        mask = labels == c
        n_c = mask.sum()
        y_c = if_better[mask]
        tau_c = tau_values[mask]

        if n_c < 3 or y_c.sum() == 0 or y_c.sum() == n_c:
            # Can't compute AUROC with single class
            class_aurocs[str(c)] = {
                "auroc": None,
                "n_points": int(n_c),
                "n_if_better": int(y_c.sum()),
                "note": "insufficient_variation"
            }
            print(f"  Class {c}: N={n_c}, IF-better={y_c.sum()} - AUROC undefined (single class)")
        else:
            auroc_pos_c = roc_auc_score(y_c, tau_c)
            auroc_neg_c = roc_auc_score(y_c, -tau_c)
            auroc_abs_c = roc_auc_score(y_c, np.abs(tau_c))
            auroc_abs_neg_c = roc_auc_score(y_c, -np.abs(tau_c))
            best_c = max(auroc_pos_c, auroc_neg_c, auroc_abs_c, auroc_abs_neg_c)
            best_dir_c = ["tau", "-tau", "|tau|", "-|tau|"][
                np.argmax([auroc_pos_c, auroc_neg_c, auroc_abs_c, auroc_abs_neg_c])]

            class_aurocs[str(c)] = {
                "auroc": float(best_c),
                "auroc_tau": float(auroc_pos_c),
                "auroc_neg_tau": float(auroc_neg_c),
                "n_points": int(n_c),
                "n_if_better": int(y_c.sum()),
                "best_direction": best_dir_c
            }
            class_auroc_valid += 1
            class_auroc_sum += best_c
            print(f"  Class {c}: N={n_c}, IF-better={y_c.sum()}, AUROC={best_c:.4f} ({best_dir_c})")

    if class_auroc_valid > 0:
        class_stratified_auroc = class_auroc_sum / class_auroc_valid
    else:
        class_stratified_auroc = 0.5

    print(f"\n  Class-stratified AUROC (mean of {class_auroc_valid} valid classes): {class_stratified_auroc:.4f}")

    # ==========================================
    # Step 7: Additional analysis - feature correlations with LDS_diff
    # ==========================================
    print("\n[Step 7] Additional analysis: feature correlations with LDS_diff...")

    # What features predict IF-better vs RepSim-better?
    feature_correlations = {}
    for name, vals in [("tau", tau_values), ("abs_tau", np.abs(tau_values)),
                       ("log_grad_norm", np.log1p(grad_norms)),
                       ("confidence", confidences), ("entropy", entropies)]:
        rho, pval = spearmanr(vals, lds_diff)
        feature_correlations[name] = {
            "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
            "p_value": float(pval) if not np.isnan(pval) else 1.0
        }
        print(f"  Spearman({name}, LDS_diff): rho={rho:.4f}, p={pval:.4e}")

    # Multi-feature AUROC using logistic regression
    print("\n  Multi-feature logistic regression AUROC...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X_features = np.column_stack([
        tau_values,
        np.log1p(grad_norms),
        confidences,
        entropies
    ])

    if n_if_better > 0 and n_repsim_better > 0:
        # 5-fold CV AUROC
        lr = LogisticRegression(max_iter=1000, solver='lbfgs')
        cv_aurocs = cross_val_score(lr, X_features, if_better, cv=min(5, min(n_if_better, n_repsim_better)), scoring='roc_auc')
        multi_feature_auroc = float(cv_aurocs.mean())
        multi_feature_auroc_std = float(cv_aurocs.std())
        print(f"  Multi-feature LR AUROC (5-fold CV): {multi_feature_auroc:.4f} +/- {multi_feature_auroc_std:.4f}")
    else:
        multi_feature_auroc = 0.5
        multi_feature_auroc_std = 0.0
        print(f"  Multi-feature LR AUROC: N/A (degenerate)")

    # ==========================================
    # Step 8: Gate evaluation
    # ==========================================
    print("\n" + "=" * 60)
    print("Gate Evaluation (H-D4)")
    print("=" * 60)

    gate_global = global_auroc > 0.60
    gate_stratified = class_stratified_auroc > 0.55
    gate_pass = gate_global and gate_stratified

    print(f"  Global AUROC:          {global_auroc:.4f} (threshold: > 0.60) → {'PASS' if gate_global else 'FAIL'}")
    print(f"  Class-stratified AUROC: {class_stratified_auroc:.4f} (threshold: > 0.55) → {'PASS' if gate_stratified else 'FAIL'}")
    print(f"  Overall gate:          {'PASS' if gate_pass else 'FAIL'}")

    # ==========================================
    # Step 9: Compile results
    # ==========================================
    results = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "n_test": int(n_test),
        "n_train": int(n_train),
        "seed": 42,
        "timestamp": datetime.now().isoformat(),

        "lds_comparison": {
            "lds_if_mean": float(lds_if.mean()),
            "lds_if_std": float(lds_if.std()),
            "lds_repsim_mean": float(lds_repsim.mean()),
            "lds_repsim_std": float(lds_repsim.std()),
            "lds_kfac_mean": float(lds_kfac.mean()),
            "lds_kfac_std": float(lds_kfac.std()),
            "lds_diff_mean": float(lds_diff.mean()),
            "lds_diff_std": float(lds_diff.std()),
            "n_if_better": int(n_if_better),
            "n_repsim_better": int(n_repsim_better),
            "per_class_if_better_fraction": per_class_if_better
        },

        "kendall_tau_if_repsim": {
            "mean": float(tau_values.mean()),
            "std": float(tau_values.std()),
            "min": float(tau_values.min()),
            "max": float(tau_values.max()),
            "median": float(np.median(tau_values)),
            "n_significant_p005": int((tau_pvalues < 0.05).sum()),
            "per_point_values": tau_values.tolist()
        },

        "global_auroc": {
            "value": float(global_auroc),
            "note": global_auroc_note,
            "all_directions": {
                "tau": float(auroc_pos) if 'auroc_pos' in dir() else None,
                "neg_tau": float(auroc_neg) if 'auroc_neg' in dir() else None,
                "abs_tau": float(auroc_abs) if 'auroc_abs' in dir() else None,
                "neg_abs_tau": float(auroc_abs_neg) if 'auroc_abs_neg' in dir() else None
            }
        },

        "class_stratified_auroc": {
            "mean": float(class_stratified_auroc),
            "n_valid_classes": int(class_auroc_valid),
            "per_class": class_aurocs
        },

        "feature_correlations_with_lds_diff": feature_correlations,

        "multi_feature_auroc": {
            "mean": float(multi_feature_auroc),
            "std": float(multi_feature_auroc_std),
            "features_used": ["tau", "log_grad_norm", "confidence", "entropy"],
            "method": "5-fold CV LogisticRegression"
        },

        "gate_evaluation": {
            "criterion_global": "AUROC > 0.60",
            "criterion_stratified": "class-stratified AUROC > 0.55",
            "global_auroc": float(global_auroc),
            "class_stratified_auroc": float(class_stratified_auroc),
            "global_pass": gate_global,
            "stratified_pass": gate_stratified,
            "overall_pass": gate_pass,
            "decision": "PASS" if gate_pass else "FAIL"
        },

        "descriptive_statistics": {
            "per_point_lds_if": {
                "values": lds_if.tolist()
            },
            "per_point_lds_repsim": {
                "values": lds_repsim.tolist()
            },
            "per_point_lds_diff": {
                "values": lds_diff.tolist()
            }
        },

        "key_observations": [],
        "limitations": [
            "Pilot uses 100 test points (not 500 as planned for full experiment)",
            "IF methods use layer4+fc only (not full model)",
            "TRAK uses 1 checkpoint (not TRAK-50)",
            "Train subset is 5K (not full 50K)",
            "Small per-class sample sizes (10 points/class) limit class-stratified AUROC reliability"
        ]
    }

    # Add key observations based on results
    obs = results["key_observations"]
    if lds_diff.mean() > 0:
        obs.append(f"IF is overall better than RepSim (mean LDS diff = {lds_diff.mean():.4f})")
    else:
        obs.append(f"RepSim is overall better than IF (mean LDS diff = {lds_diff.mean():.4f})")

    if global_auroc > 0.60:
        obs.append(f"Cross-method disagreement (tau) is a useful predictor of IF reliability (AUROC={global_auroc:.4f})")
    else:
        obs.append(f"Cross-method disagreement (tau) is a weak predictor of IF reliability (AUROC={global_auroc:.4f})")

    if class_stratified_auroc > 0.55:
        obs.append(f"Disagreement signal survives class stratification (mean AUROC={class_stratified_auroc:.4f})")
    else:
        obs.append(f"Disagreement signal does NOT survive class stratification (mean AUROC={class_stratified_auroc:.4f})")

    if multi_feature_auroc > global_auroc + 0.05:
        obs.append(f"Multi-feature selector improves over tau alone ({multi_feature_auroc:.4f} vs {global_auroc:.4f})")

    # Save results
    output_path = RESULTS_DIR / "phase2b_disagreement.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save summary
    summary_path = RESULTS_DIR / "phase2b_disagreement_summary.md"
    with open(summary_path, "w") as f:
        f.write("# Phase 2b: Cross-Method Disagreement Analysis (H-D4) — Pilot\n\n")
        f.write(f"**Date**: {datetime.now().isoformat()}\n")
        f.write(f"**Mode**: PILOT (100 test points, 5K train)\n\n")

        f.write("## LDS Comparison\n\n")
        f.write(f"| Method | Mean LDS | Std LDS |\n")
        f.write(f"|--------|----------|--------|\n")
        f.write(f"| EK-FAC IF | {lds_if.mean():.4f} | {lds_if.std():.4f} |\n")
        f.write(f"| K-FAC IF | {lds_kfac.mean():.4f} | {lds_kfac.std():.4f} |\n")
        f.write(f"| RepSim | {lds_repsim.mean():.4f} | {lds_repsim.std():.4f} |\n\n")

        f.write(f"IF-better: {n_if_better}/{n_test} ({100*n_if_better/n_test:.1f}%)\n")
        f.write(f"RepSim-better: {n_repsim_better}/{n_test} ({100*n_repsim_better/n_test:.1f}%)\n\n")

        f.write("## Kendall Tau (IF, RepSim)\n\n")
        f.write(f"- Mean: {tau_values.mean():.4f}, Std: {tau_values.std():.4f}\n")
        f.write(f"- Range: [{tau_values.min():.4f}, {tau_values.max():.4f}]\n")
        f.write(f"- Significant (p<0.05): {(tau_pvalues < 0.05).sum()}/{n_test}\n\n")

        f.write("## AUROC Results\n\n")
        f.write(f"- **Global AUROC**: {global_auroc:.4f} (threshold: >0.60) → {'PASS' if gate_global else 'FAIL'}\n")
        f.write(f"- **Class-stratified AUROC**: {class_stratified_auroc:.4f} (threshold: >0.55) → {'PASS' if gate_stratified else 'FAIL'}\n")
        f.write(f"- **Multi-feature LR AUROC**: {multi_feature_auroc:.4f} ± {multi_feature_auroc_std:.4f}\n\n")

        f.write("## Class-Stratified AUROC\n\n")
        f.write("| Class | N | IF-better | AUROC |\n")
        f.write("|-------|---|-----------|-------|\n")
        for c in range(10):
            info = class_aurocs[str(c)]
            auroc_str = f"{info['auroc']:.4f}" if info['auroc'] is not None else "N/A"
            f.write(f"| {c} | {info['n_points']} | {info['n_if_better']} | {auroc_str} |\n")

        f.write(f"\n## Gate Decision: **{'PASS' if gate_pass else 'FAIL'}**\n\n")

        f.write("## Key Observations\n\n")
        for o in obs:
            f.write(f"- {o}\n")

        f.write("\n## Limitations\n\n")
        for lim in results["limitations"]:
            f.write(f"- {lim}\n")

    print(f"Summary saved to {summary_path}")

    # Mark done
    mark_task_done(TASK_ID, RESULTS_DIR, status="success",
                   summary=f"Phase 2b pilot complete. Global AUROC={global_auroc:.4f}, "
                           f"Class-stratified AUROC={class_stratified_auroc:.4f}. "
                           f"Gate: {'PASS' if gate_pass else 'FAIL'}")

    report_progress(TASK_ID, RESULTS_DIR, 4, 4, metric={
        "stage": "complete",
        "global_auroc": float(global_auroc),
        "class_stratified_auroc": float(class_stratified_auroc),
        "gate": "PASS" if gate_pass else "FAIL"
    })

    print(f"\n{'='*60}")
    print(f"DONE. Gate: {'PASS' if gate_pass else 'FAIL'}")
    print(f"{'='*60}")

    return 0 if gate_pass else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        mark_task_done(TASK_ID, RESULTS_DIR, status="failed", summary=str(e))
        raise
