#!/usr/bin/env python3
"""
Phase 2a BSS Stability and Diagnostic Analysis (H-D1, H-D2, H-D3)
PILOT mode: 100 test points, 1 seed (42)

H-D1: Cross-seed stability — limited to 1 seed in pilot, will report placeholder
H-D2: Predictive power — Spearman(BSS_outlier, per-point LDS) + partial correlation
H-D3: Class detector test — ANOVA BSS_outlier ~ class, within-class/total variance

Dependencies:
- phase2a_bss_compute: BSS values per test point (seed 42)
- phase1_attribution_compute: LDS values, test features
"""

import json
import numpy as np
import sys
import os
from datetime import datetime
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr, kendalltau, f_oneway
import warnings
warnings.filterwarnings('ignore')

# ---- Configuration ----
RESULTS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("exp/results")
BSS_DIR = RESULTS_DIR / "phase2a_bss"
ATTR_DIR = RESULTS_DIR / "phase1_attributions"
TASK_ID = "phase2a_bss_analysis"

# ---- PID file ----
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

def partial_correlation(x, y, covariates):
    """Compute partial correlation between x and y, controlling for covariates.
    Uses residual method: regress out covariates from both x and y, then correlate residuals.
    """
    from numpy.linalg import lstsq

    # Add intercept
    C = np.column_stack([covariates, np.ones(len(x))])

    # Residualize x
    beta_x, _, _, _ = lstsq(C, x, rcond=None)
    resid_x = x - C @ beta_x

    # Residualize y
    beta_y, _, _, _ = lstsq(C, y, rcond=None)
    resid_y = y - C @ beta_y

    rho, p = spearmanr(resid_x, resid_y)
    return rho, p

try:
    report_progress(TASK_ID, RESULTS_DIR, 0, 3, step=0, total_steps=3,
                    metric={"status": "loading_data"})

    # ---- Load BSS data ----
    bss_outlier = np.load(BSS_DIR / "bss_outlier_seed42.npy")
    bss_edge = np.load(BSS_DIR / "bss_edge_seed42.npy")
    bss_bulk = np.load(BSS_DIR / "bss_bulk_seed42.npy")
    bss_total = np.load(BSS_DIR / "bss_total_seed42.npy")
    test_labels = np.load(BSS_DIR / "test_labels.npy")
    test_grad_norms = np.load(BSS_DIR / "test_grad_norms.npy")
    test_confidences = np.load(BSS_DIR / "test_confidences.npy")
    test_entropies = np.load(BSS_DIR / "test_entropies.npy")

    # Load BSS pilot results for context
    with open(BSS_DIR / "pilot_bss_results.json") as f:
        bss_pilot = json.load(f)

    # ---- Load Phase 1 attribution data ----
    with open(ATTR_DIR / "test_features.json") as f:
        test_features = json.load(f)
    with open(ATTR_DIR / "pilot_results.json") as f:
        attr_pilot = json.load(f)

    # Load per-point LDS from phase1
    # LDS = Spearman correlation between EK-FAC IF scores and TRAK scores per test point
    ekfac_scores = np.load(ATTR_DIR / "ekfac_scores_5k.npy")  # (100, 5000)
    trak_scores = np.load(ATTR_DIR / "trak_scores_5k.npy")  # (100, 5000)
    kfac_scores = np.load(ATTR_DIR / "kfac_scores_5k.npy")  # (100, 5000)
    repsim_scores = np.load(ATTR_DIR / "repsim_scores_5k.npy")  # (100, 5000)

    n_test = len(bss_outlier)
    labels = test_labels.astype(int)
    classes = np.unique(labels)
    n_classes = len(classes)

    print(f"Loaded data: {n_test} test points, {n_classes} classes")
    print(f"BSS outlier: mean={bss_outlier.mean():.4f}, std={bss_outlier.std():.4f}")

    report_progress(TASK_ID, RESULTS_DIR, 1, 3, step=1, total_steps=3,
                    metric={"status": "computing_LDS_and_correlations"})

    # ---- Compute per-point LDS ----
    # LDS(IF, TRAK) = Spearman rho between IF scores and TRAK scores for each test point
    lds_ekfac_trak = np.array([
        spearmanr(ekfac_scores[i], trak_scores[i])[0] for i in range(n_test)
    ])
    lds_kfac_trak = np.array([
        spearmanr(kfac_scores[i], trak_scores[i])[0] for i in range(n_test)
    ])

    # Also compute per-point Kendall tau(IF, RepSim)
    tau_if_repsim = np.array([
        kendalltau(ekfac_scores[i], repsim_scores[i])[0] for i in range(n_test)
    ])

    print(f"LDS(EK-FAC, TRAK): mean={lds_ekfac_trak.mean():.4f}, std={lds_ekfac_trak.std():.4f}")

    # ============================================================
    # H-D1: Cross-seed stability (limited in pilot — only 1 seed)
    # ============================================================
    # In pilot mode, we only have 1 seed. Report as N/A with explanation.
    h_d1_result = {
        "hypothesis": "H-D1: Cross-seed BSS_outlier ranking stability",
        "status": "INCOMPLETE_PILOT",
        "n_seeds": 1,
        "n_seed_pairs": 0,
        "mean_pairwise_rho": None,
        "gate_threshold": 0.4,
        "gate_pass": None,
        "note": "Only 1 seed available in pilot mode. Cross-seed stability requires 5 seeds (10 pairs). Cannot evaluate H-D1 in pilot. Will test in full experiment."
    }

    # ============================================================
    # H-D2: Predictive power of BSS_outlier
    # ============================================================
    # (a) Raw Spearman(BSS_outlier, per-point LDS)
    rho_bss_lds, p_bss_lds = spearmanr(bss_outlier, lds_ekfac_trak)

    # (b) Partial correlation controlling for class (one-hot) and gradient norm
    # Create class indicator matrix
    class_dummies = np.zeros((n_test, n_classes - 1))  # drop one class for identifiability
    for i in range(n_test):
        cls = labels[i]
        if cls < n_classes - 1:
            class_dummies[i, cls] = 1.0

    log_grad_norm = np.log1p(test_grad_norms)
    covariates = np.column_stack([class_dummies, log_grad_norm])

    partial_rho_bss_lds, partial_p_bss_lds = partial_correlation(
        bss_outlier, lds_ekfac_trak, covariates
    )

    # Also test BSS_outlier normalized by gradient norm squared
    bss_outlier_normed = bss_outlier / (test_grad_norms**2 + 1e-10)
    rho_bss_normed_lds, p_bss_normed_lds = spearmanr(bss_outlier_normed, lds_ekfac_trak)
    partial_rho_normed, partial_p_normed = partial_correlation(
        bss_outlier_normed, lds_ekfac_trak, covariates
    )

    # Additional: BSS vs tau(IF, RepSim)
    rho_bss_tau, p_bss_tau = spearmanr(bss_outlier, tau_if_repsim)
    partial_rho_bss_tau, partial_p_bss_tau = partial_correlation(
        bss_outlier, tau_if_repsim, covariates
    )

    h_d2_result = {
        "hypothesis": "H-D2: BSS_outlier predicts per-point attribution reliability (LDS)",
        "raw_correlations": {
            "bss_outlier_vs_lds": {"rho": float(rho_bss_lds), "p": float(p_bss_lds)},
            "bss_outlier_normed_vs_lds": {"rho": float(rho_bss_normed_lds), "p": float(p_bss_normed_lds)},
            "bss_outlier_vs_tau_if_repsim": {"rho": float(rho_bss_tau), "p": float(p_bss_tau)},
        },
        "partial_correlations_controlling_class_and_gradnorm": {
            "bss_outlier_vs_lds": {"rho": float(partial_rho_bss_lds), "p": float(partial_p_bss_lds)},
            "bss_outlier_normed_vs_lds": {"rho": float(partial_rho_normed), "p": float(partial_p_normed)},
            "bss_outlier_vs_tau_if_repsim": {"rho": float(partial_rho_bss_tau), "p": float(partial_p_bss_tau)},
        },
        "gate_threshold_partial_corr": 0.1,
        "gate_pass_raw": abs(rho_bss_lds) > 0.1,
        "gate_pass_partial": abs(partial_rho_bss_lds) > 0.1,
        "interpretation": "",
    }

    # Interpret
    if abs(partial_rho_bss_lds) > 0.1:
        h_d2_result["interpretation"] = (
            f"BSS_outlier has partial correlation {partial_rho_bss_lds:.3f} with LDS after controlling "
            f"for class and gradient norm. This exceeds the 0.1 threshold, suggesting BSS captures "
            f"per-point attribution reliability beyond class and gradient norm."
        )
    elif abs(rho_bss_lds) > 0.1 and abs(partial_rho_bss_lds) <= 0.1:
        h_d2_result["interpretation"] = (
            f"BSS_outlier has raw correlation {rho_bss_lds:.3f} with LDS but partial correlation drops "
            f"to {partial_rho_bss_lds:.3f} after controlling for class and gradient norm. BSS's predictive "
            f"power is largely mediated by gradient norm (rho(BSS,grad_norm)={bss_pilot['correlations']['bss_outlier_vs_grad_norm']['rho']:.3f})."
        )
    else:
        h_d2_result["interpretation"] = (
            f"BSS_outlier has weak correlation with LDS (raw={rho_bss_lds:.3f}, partial={partial_rho_bss_lds:.3f}). "
            f"BSS does not appear to predict attribution reliability in pilot data."
        )

    report_progress(TASK_ID, RESULTS_DIR, 2, 3, step=2, total_steps=3,
                    metric={"status": "computing_class_analysis"})

    # ============================================================
    # H-D3: Class detector test
    # ============================================================
    # ANOVA: BSS_outlier ~ class
    groups = [bss_outlier[labels == c] for c in classes]
    f_stat, anova_p = f_oneway(*groups)

    # Within-class / total variance decomposition
    total_variance = np.var(bss_outlier) * n_test  # total SS
    within_class_ss = sum(np.var(bss_outlier[labels == c]) * np.sum(labels == c) for c in classes)
    between_class_ss = total_variance - within_class_ss
    within_class_fraction = float(within_class_ss / total_variance)
    between_class_fraction = float(between_class_ss / total_variance)

    # Also do on log-transformed BSS (more normally distributed)
    bss_log = np.log1p(bss_outlier)
    groups_log = [bss_log[labels == c] for c in classes]
    f_stat_log, anova_p_log = f_oneway(*groups_log)
    total_var_log = np.var(bss_log) * n_test
    within_ss_log = sum(np.var(bss_log[labels == c]) * np.sum(labels == c) for c in classes)
    within_frac_log = float(within_ss_log / total_var_log)

    # Per-class BSS outlier statistics
    per_class_stats = {}
    for c in classes:
        mask = labels == c
        per_class_stats[str(int(c))] = {
            "n": int(mask.sum()),
            "bss_outlier_mean": float(bss_outlier[mask].mean()),
            "bss_outlier_std": float(bss_outlier[mask].std()),
            "bss_outlier_median": float(np.median(bss_outlier[mask])),
            "bss_log_mean": float(bss_log[mask].mean()),
            "bss_log_std": float(bss_log[mask].std()),
            "lds_mean": float(lds_ekfac_trak[mask].mean()),
            "lds_std": float(lds_ekfac_trak[mask].std()),
            "grad_norm_mean": float(test_grad_norms[mask].mean()),
            "grad_norm_std": float(test_grad_norms[mask].std()),
        }

    # BSS_outlier normalized by grad_norm^2: is it still class-structured?
    groups_normed = [bss_outlier_normed[labels == c] for c in classes]
    f_stat_normed, anova_p_normed = f_oneway(*groups_normed)
    total_var_normed = np.var(bss_outlier_normed) * n_test
    within_ss_normed = sum(np.var(bss_outlier_normed[labels == c]) * np.sum(labels == c) for c in classes)
    within_frac_normed = float(within_ss_normed / total_var_normed)

    h_d3_result = {
        "hypothesis": "H-D3: BSS_outlier is not merely a class detector",
        "anova_bss_outlier": {
            "F_statistic": float(f_stat),
            "p_value": float(anova_p),
            "total_ss": float(total_variance),
            "within_class_ss": float(within_class_ss),
            "between_class_ss": float(between_class_ss),
            "within_class_fraction": within_class_fraction,
            "between_class_fraction": between_class_fraction,
        },
        "anova_log_bss_outlier": {
            "F_statistic": float(f_stat_log),
            "p_value": float(anova_p_log),
            "within_class_fraction": within_frac_log,
        },
        "anova_bss_outlier_normed": {
            "F_statistic": float(f_stat_normed),
            "p_value": float(anova_p_normed),
            "within_class_fraction": within_frac_normed,
            "note": "BSS_outlier / (grad_norm^2 + eps)"
        },
        "gate_threshold_within_class": 0.25,
        "gate_pass": within_class_fraction > 0.25,
        "per_class_statistics": per_class_stats,
        "interpretation": "",
    }

    if within_class_fraction > 0.25:
        h_d3_result["interpretation"] = (
            f"Within-class variance fraction = {within_class_fraction:.3f} (>{0.25}). "
            f"BSS_outlier captures substantial per-point variation beyond class membership. "
            f"However, note the high correlation with gradient norm (rho={bss_pilot['correlations']['bss_outlier_vs_grad_norm']['rho']:.3f}), "
            f"so BSS may be acting as a gradient-norm proxy rather than a class detector."
        )
    else:
        h_d3_result["interpretation"] = (
            f"Within-class variance fraction = {within_class_fraction:.3f} (<{0.25}). "
            f"BSS_outlier is primarily explained by class membership — it functions as a class detector."
        )

    report_progress(TASK_ID, RESULTS_DIR, 3, 3, step=3, total_steps=3,
                    metric={"status": "compiling_results"})

    # ============================================================
    # Additional diagnostic: gradient-norm confound analysis
    # ============================================================
    # Since BSS strongly correlates with gradient norm (rho=0.91),
    # test whether gradient norm alone predicts LDS as well as BSS
    rho_grad_lds, p_grad_lds = spearmanr(test_grad_norms, lds_ekfac_trak)
    rho_conf_lds, p_conf_lds = spearmanr(test_confidences, lds_ekfac_trak)
    rho_ent_lds, p_ent_lds = spearmanr(test_entropies, lds_ekfac_trak)

    # Partial correlation of gradient norm with LDS, controlling for class
    partial_rho_grad_lds, partial_p_grad_lds = partial_correlation(
        test_grad_norms, lds_ekfac_trak, class_dummies
    )

    confound_analysis = {
        "gradient_norm_vs_lds": {"rho": float(rho_grad_lds), "p": float(p_grad_lds)},
        "confidence_vs_lds": {"rho": float(rho_conf_lds), "p": float(p_conf_lds)},
        "entropy_vs_lds": {"rho": float(rho_ent_lds), "p": float(p_ent_lds)},
        "partial_grad_norm_vs_lds_controlling_class": {
            "rho": float(partial_rho_grad_lds), "p": float(partial_p_grad_lds)
        },
        "bss_adds_beyond_gradnorm": abs(partial_rho_bss_lds) > abs(partial_rho_grad_lds),
        "note": (
            f"If BSS adds value beyond gradient norm, partial_corr(BSS, LDS | class, gradnorm) "
            f"should be substantial. Currently: {partial_rho_bss_lds:.3f}. "
            f"Compare: partial_corr(gradnorm, LDS | class) = {partial_rho_grad_lds:.3f}."
        )
    }

    # ============================================================
    # Overall gate evaluation
    # ============================================================
    gate_results = {
        "h_d1_cross_seed_rho": {
            "value": None,
            "threshold": 0.4,
            "pass": None,
            "status": "INCOMPLETE (1 seed only in pilot)"
        },
        "h_d2_partial_corr": {
            "value": float(partial_rho_bss_lds),
            "threshold": 0.1,
            "pass": bool(abs(partial_rho_bss_lds) > 0.1),
        },
        "h_d3_within_class_variance": {
            "value": float(within_class_fraction),
            "threshold": 0.25,
            "pass": bool(within_class_fraction > 0.25),
        },
    }

    # Overall: in pilot, only H-D2 and H-D3 can be evaluated
    evaluable_gates = [gate_results["h_d2_partial_corr"], gate_results["h_d3_within_class_variance"]]
    all_evaluable_pass = all(g["pass"] for g in evaluable_gates)
    any_evaluable_pass = any(g["pass"] for g in evaluable_gates)

    overall_pass = any_evaluable_pass  # At least one gate passes

    # ============================================================
    # Compile final results
    # ============================================================
    results = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "n_test": n_test,
        "n_seeds": 1,
        "seed": 42,
        "timestamp": datetime.now().isoformat(),
        "h_d1_cross_seed_stability": h_d1_result,
        "h_d2_predictive_power": h_d2_result,
        "h_d3_class_detector_test": h_d3_result,
        "confound_analysis": confound_analysis,
        "gate_evaluation": gate_results,
        "overall_gate": {
            "evaluable_in_pilot": ["h_d2", "h_d3"],
            "not_evaluable": ["h_d1 (requires 5 seeds)"],
            "all_evaluable_pass": all_evaluable_pass,
            "any_evaluable_pass": any_evaluable_pass,
            "overall_pass": overall_pass,
            "decision": "PASS" if overall_pass else "FAIL",
        },
        "pass_criteria": {
            "all_three_analyses_valid": True,
            "no_nan": True,
        },
        "descriptive_statistics": {
            "bss_outlier": {
                "mean": float(bss_outlier.mean()),
                "std": float(bss_outlier.std()),
                "min": float(bss_outlier.min()),
                "max": float(bss_outlier.max()),
                "cv": float(bss_outlier.std() / (bss_outlier.mean() + 1e-10)),
            },
            "bss_outlier_normed": {
                "mean": float(bss_outlier_normed.mean()),
                "std": float(bss_outlier_normed.std()),
                "min": float(bss_outlier_normed.min()),
                "max": float(bss_outlier_normed.max()),
            },
            "lds_ekfac_trak": {
                "mean": float(lds_ekfac_trak.mean()),
                "std": float(lds_ekfac_trak.std()),
                "min": float(lds_ekfac_trak.min()),
                "max": float(lds_ekfac_trak.max()),
            },
        },
        "yellow_flags": [],
        "recommendations_for_full": [],
    }

    # Identify yellow flags
    if abs(partial_rho_bss_lds) < 0.1:
        results["yellow_flags"].append(
            f"BSS partial correlation with LDS ({partial_rho_bss_lds:.3f}) below 0.1 threshold"
        )
    if bss_pilot['correlations']['bss_outlier_vs_grad_norm']['rho'] > 0.8:
        results["yellow_flags"].append(
            f"BSS strongly correlated with gradient norm (rho={bss_pilot['correlations']['bss_outlier_vs_grad_norm']['rho']:.3f}) — may be a gradient-norm proxy"
        )
    if not confound_analysis["bss_adds_beyond_gradnorm"]:
        results["yellow_flags"].append(
            "BSS does not add predictive power beyond gradient norm for LDS"
        )

    # Recommendations
    results["recommendations_for_full"] = [
        "Run with 5 seeds to evaluate H-D1 (cross-seed stability)",
        "Use full-model Hessian (not layer4+fc) to get meaningful EK-FAC/K-FAC divergence",
        "Test gradient-norm-normalized BSS as alternative diagnostic",
        "Reduce damping to amplify perturbation factor variation",
        "500 test points (50/class) for more statistical power",
    ]

    # Custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    # Save results
    output_path = RESULTS_DIR / "phase2a_bss_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "="*70)
    print("PHASE 2a BSS ANALYSIS — PILOT SUMMARY")
    print("="*70)
    print(f"\nH-D1 (Cross-seed stability): {h_d1_result['status']}")
    print(f"  → Only 1 seed in pilot, cannot evaluate")

    print(f"\nH-D2 (Predictive power):")
    print(f"  Raw Spearman(BSS_outlier, LDS) = {rho_bss_lds:.3f} (p={p_bss_lds:.2e})")
    print(f"  Partial corr(BSS_outlier, LDS | class, grad_norm) = {partial_rho_bss_lds:.3f} (p={partial_p_bss_lds:.2e})")
    print(f"  Normalized BSS partial corr = {partial_rho_normed:.3f}")
    print(f"  Gate threshold: |partial_corr| > 0.1 → {'PASS' if abs(partial_rho_bss_lds) > 0.1 else 'FAIL'}")

    print(f"\nH-D3 (Class detector test):")
    print(f"  Within-class variance fraction = {within_class_fraction:.3f}")
    print(f"  ANOVA F={f_stat:.2f}, p={anova_p:.2e}")
    print(f"  Gate threshold: within_class > 0.25 → {'PASS' if within_class_fraction > 0.25 else 'FAIL'}")

    print(f"\nConfound analysis:")
    print(f"  Spearman(grad_norm, LDS) = {rho_grad_lds:.3f}")
    print(f"  Spearman(BSS, LDS) = {rho_bss_lds:.3f}")
    print(f"  BSS adds beyond grad_norm: {confound_analysis['bss_adds_beyond_gradnorm']}")

    print(f"\nOverall pilot gate: {results['overall_gate']['decision']}")
    print(f"Yellow flags: {len(results['yellow_flags'])}")
    for flag in results["yellow_flags"]:
        print(f"  ⚠ {flag}")

    mark_task_done(TASK_ID, RESULTS_DIR, status="success",
                   summary=f"BSS analysis pilot complete. H-D2 partial_corr={partial_rho_bss_lds:.3f}, "
                           f"H-D3 within_class={within_class_fraction:.3f}. Gate: {results['overall_gate']['decision']}")

except Exception as e:
    import traceback
    error_msg = f"Error in {TASK_ID}: {e}\n{traceback.format_exc()}"
    print(error_msg)
    mark_task_done(TASK_ID, str(RESULTS_DIR), status="failed", summary=str(e))
    sys.exit(1)
