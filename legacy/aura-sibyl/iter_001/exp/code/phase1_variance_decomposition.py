"""
Phase 1 Variance Decomposition (H-G1) - PILOT
Two-way ANOVA on 100 test points.
Response variables:
  1. J10 = Jaccard@10(EK-FAC, K-FAC)
  2. tau = Kendall tau(IF vs RepSim) per point
  3. LDS = per-point Spearman(EK-FAC vs TRAK) on 5K training subset
Predictors: class (10 levels), log(gradient_norm)
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Setup paths ───────────────────────────────────────────────────────────────
TASK_ID = "phase1_variance_decomposition"
PROJECT_DIR = Path("/home/jinxulin/sibyl_system/projects/AURA")
ATTR_DIR = PROJECT_DIR / "exp" / "results" / "phase1_attributions"
RESULTS_DIR = PROJECT_DIR / "exp" / "results"

# Write PID
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

try:
    report_progress(TASK_ID, RESULTS_DIR, 0, 5, step=0, total_steps=5,
                    metric={"phase": "loading_data"})

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("[1/5] Loading attribution data...")

    # Load test features
    with open(ATTR_DIR / "test_features.json") as f:
        features = json.load(f)

    labels = np.array(features["labels"])
    gradient_norms = np.array(features["gradient_norms"])
    confidences = np.array(features["confidences"])
    entropies = np.array(features["entropies"])
    n_test = len(labels)

    print(f"  n_test = {n_test}")
    print(f"  classes = {np.unique(labels).tolist()}")
    print(f"  grad_norm range = [{gradient_norms.min():.4f}, {gradient_norms.max():.4f}]")

    # Load attribution scores  (100 test x 5000 train for IF/TRAK)
    ekfac_scores = np.load(ATTR_DIR / "ekfac_scores_5k.npy")
    kfac_scores = np.load(ATTR_DIR / "kfac_scores_5k.npy")
    trak_scores = np.load(ATTR_DIR / "trak_scores_5k.npy")

    # RepSim: could be full (100 x 50000) or 5k subset - use 5k for consistency
    repsim_scores_5k = np.load(ATTR_DIR / "repsim_scores_5k.npy")

    # Also load top-100 rankings for Jaccard computation
    ekfac_top100 = np.load(ATTR_DIR / "ekfac_rankings_5k_top100.npy")
    kfac_top100 = np.load(ATTR_DIR / "kfac_rankings_5k_top100.npy")

    print(f"  ekfac_scores shape = {ekfac_scores.shape}")
    print(f"  kfac_scores shape = {kfac_scores.shape}")
    print(f"  trak_scores shape = {trak_scores.shape}")
    print(f"  repsim_scores_5k shape = {repsim_scores_5k.shape}")
    print(f"  ekfac_top100 shape = {ekfac_top100.shape}")

    report_progress(TASK_ID, RESULTS_DIR, 1, 5, step=1, total_steps=5,
                    metric={"phase": "computing_metrics"})

    # ── 2. Compute per-point metrics ─────────────────────────────────────────
    print("\n[2/5] Computing per-point metrics...")

    from scipy.stats import kendalltau, spearmanr

    # 2a. J10 = Jaccard@10(EK-FAC, K-FAC)
    j10_values = np.zeros(n_test)
    for i in range(n_test):
        ekfac_top10 = set(ekfac_top100[i, :10].tolist())
        kfac_top10 = set(kfac_top100[i, :10].tolist())
        j10_values[i] = len(ekfac_top10 & kfac_top10) / len(ekfac_top10 | kfac_top10)

    print(f"  J10 mean={j10_values.mean():.4f}, std={j10_values.std():.4f}, "
          f"min={j10_values.min():.4f}, max={j10_values.max():.4f}")

    # 2b. tau = Kendall tau(EK-FAC IF, RepSim) per point (on 5K subset)
    tau_values = np.zeros(n_test)
    # Kendall tau on full 5K is slow; use top-500 rankings for efficiency
    TOP_K_TAU = 500
    for i in range(n_test):
        # Get top-500 indices by EK-FAC
        ekfac_topk_idx = np.argsort(-ekfac_scores[i])[:TOP_K_TAU]
        # Compute Kendall tau between EK-FAC and RepSim rankings for these indices
        ekfac_ranks_subset = np.argsort(np.argsort(-ekfac_scores[i, ekfac_topk_idx]))
        repsim_ranks_subset = np.argsort(np.argsort(-repsim_scores_5k[i, ekfac_topk_idx]))
        tau_val, _ = kendalltau(ekfac_ranks_subset, repsim_ranks_subset)
        tau_values[i] = tau_val if not np.isnan(tau_val) else 0.0

    print(f"  tau(IF,RepSim) mean={tau_values.mean():.4f}, std={tau_values.std():.4f}, "
          f"min={tau_values.min():.4f}, max={tau_values.max():.4f}")

    # 2c. LDS = per-point Spearman(EK-FAC, TRAK) on 5K subset
    lds_values = np.zeros(n_test)
    for i in range(n_test):
        rho, _ = spearmanr(ekfac_scores[i], trak_scores[i])
        lds_values[i] = rho if not np.isnan(rho) else 0.0

    print(f"  LDS(EK-FAC,TRAK) mean={lds_values.mean():.4f}, std={lds_values.std():.4f}, "
          f"min={lds_values.min():.4f}, max={lds_values.max():.4f}")

    report_progress(TASK_ID, RESULTS_DIR, 2, 5, step=2, total_steps=5,
                    metric={"phase": "running_anova"})

    # ── 3. Two-way ANOVA ─────────────────────────────────────────────────────
    print("\n[3/5] Running two-way ANOVA...")

    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    import pandas as pd

    # Build dataframe
    log_grad_norm = np.log1p(gradient_norms)  # log(1 + grad_norm) to handle zeros

    df = pd.DataFrame({
        "class_label": pd.Categorical(labels),
        "log_grad_norm": log_grad_norm,
        "J10": j10_values,
        "tau": tau_values,
        "LDS": lds_values,
        "confidence": confidences,
        "entropy": entropies,
    })

    # Type I (sequential) ANOVA with class entered first
    response_vars = ["J10", "tau", "LDS"]
    anova_results = {}

    for resp in response_vars:
        print(f"\n  --- ANOVA for {resp} ---")

        # Check for constant response
        resp_std = df[resp].std()
        if resp_std < 1e-10:
            print(f"    WARNING: {resp} has near-zero variance (std={resp_std:.2e}), skipping ANOVA")
            anova_results[resp] = {
                "warning": f"Near-zero variance (std={resp_std:.2e})",
                "class_R2": 0.0,
                "grad_norm_R2": 0.0,
                "interaction_R2": 0.0,
                "residual_R2": 1.0,
                "total_ss": float(resp_std**2 * (n_test - 1)),
            }
            continue

        # OLS model: response ~ C(class_label) + log_grad_norm + C(class_label):log_grad_norm
        formula = f"{resp} ~ C(class_label) + log_grad_norm + C(class_label):log_grad_norm"
        try:
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=1)

            print(anova_table.to_string())

            total_ss = anova_table["sum_sq"].sum()

            # Extract partial R-squared (proportion of total SS)
            class_ss = anova_table.loc["C(class_label)", "sum_sq"]
            grad_ss = anova_table.loc["log_grad_norm", "sum_sq"]
            interaction_ss = anova_table.loc["C(class_label):log_grad_norm", "sum_sq"]
            residual_ss = anova_table.loc["Residual", "sum_sq"]

            class_r2 = class_ss / total_ss
            grad_r2 = grad_ss / total_ss
            interaction_r2 = interaction_ss / total_ss
            residual_r2 = residual_ss / total_ss

            print(f"\n    Class R2 = {class_r2:.4f}")
            print(f"    GradNorm R2 = {grad_r2:.4f}")
            print(f"    Interaction R2 = {interaction_r2:.4f}")
            print(f"    Residual R2 = {residual_r2:.4f}")
            print(f"    Total SS = {total_ss:.6f}")

            # Also compute Type III for comparison
            anova_type3 = sm.stats.anova_lm(model, typ=3)

            anova_results[resp] = {
                "class_R2": float(class_r2),
                "grad_norm_R2": float(grad_r2),
                "interaction_R2": float(interaction_r2),
                "residual_R2": float(residual_r2),
                "total_ss": float(total_ss),
                "class_ss": float(class_ss),
                "grad_norm_ss": float(grad_ss),
                "interaction_ss": float(interaction_ss),
                "residual_ss": float(residual_ss),
                "class_F": float(anova_table.loc["C(class_label)", "F"]) if not np.isnan(anova_table.loc["C(class_label)", "F"]) else None,
                "class_p": float(anova_table.loc["C(class_label)", "PR(>F)"]) if not np.isnan(anova_table.loc["C(class_label)", "PR(>F)"]) else None,
                "grad_norm_F": float(anova_table.loc["log_grad_norm", "F"]) if not np.isnan(anova_table.loc["log_grad_norm", "F"]) else None,
                "grad_norm_p": float(anova_table.loc["log_grad_norm", "PR(>F)"]) if not np.isnan(anova_table.loc["log_grad_norm", "PR(>F)"]) else None,
                "interaction_F": float(anova_table.loc["C(class_label):log_grad_norm", "F"]) if not np.isnan(anova_table.loc["C(class_label):log_grad_norm", "F"]) else None,
                "interaction_p": float(anova_table.loc["C(class_label):log_grad_norm", "PR(>F)"]) if not np.isnan(anova_table.loc["C(class_label):log_grad_norm", "PR(>F)"]) else None,
                "model_R2": float(model.rsquared),
                "model_adj_R2": float(model.rsquared_adj),
                "n_obs": int(model.nobs),
                "df_model": int(model.df_model),
                "df_resid": int(model.df_resid),
            }
        except Exception as e:
            print(f"    ANOVA failed for {resp}: {e}")
            anova_results[resp] = {
                "error": str(e),
                "class_R2": None,
                "grad_norm_R2": None,
                "interaction_R2": None,
                "residual_R2": None,
            }

    report_progress(TASK_ID, RESULTS_DIR, 3, 5, step=3, total_steps=5,
                    metric={"phase": "gate_evaluation"})

    # ── 4. Gate evaluation ────────────────────────────────────────────────────
    print("\n[4/5] Gate evaluation...")

    gate_pass = False
    residuals = {}
    for resp in response_vars:
        r = anova_results[resp]
        res_r2 = r.get("residual_R2")
        if res_r2 is not None:
            residuals[resp] = res_r2
            print(f"  {resp}: residual R2 = {res_r2:.4f} ({'PASS (>0.30)' if res_r2 > 0.30 else 'FAIL (<0.30)'})")
            if res_r2 > 0.30:
                gate_pass = True

    print(f"\n  GATE DECISION: {'PASS' if gate_pass else 'FAIL'}")
    print(f"  (Criteria: residual > 30% on at least 1 metric)")

    report_progress(TASK_ID, RESULTS_DIR, 4, 5, step=4, total_steps=5,
                    metric={"phase": "saving_results"})

    # ── 5. Additional diagnostics ─────────────────────────────────────────────
    print("\n[5/5] Additional diagnostics...")

    # Per-class summary statistics
    per_class_stats = {}
    for c in range(10):
        mask = labels == c
        n_c = mask.sum()
        if n_c == 0:
            continue
        per_class_stats[str(c)] = {
            "n": int(n_c),
            "J10_mean": float(j10_values[mask].mean()),
            "J10_std": float(j10_values[mask].std()),
            "tau_mean": float(tau_values[mask].mean()),
            "tau_std": float(tau_values[mask].std()),
            "LDS_mean": float(lds_values[mask].mean()),
            "LDS_std": float(lds_values[mask].std()),
            "grad_norm_mean": float(gradient_norms[mask].mean()),
            "grad_norm_std": float(gradient_norms[mask].std()),
        }

    # Correlation matrix among metrics
    metrics_matrix = np.column_stack([j10_values, tau_values, lds_values,
                                       log_grad_norm, confidences, entropies])
    metric_names = ["J10", "tau", "LDS", "log_grad_norm", "confidence", "entropy"]
    corr_matrix = {}
    for i, ni in enumerate(metric_names):
        for j, nj in enumerate(metric_names):
            rho, p = spearmanr(metrics_matrix[:, i], metrics_matrix[:, j])
            corr_matrix[f"{ni}_vs_{nj}"] = {
                "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
                "p_value": float(p) if not np.isnan(p) else 1.0,
            }

    # Descriptive stats for response variables
    descriptive = {}
    for name, vals in [("J10", j10_values), ("tau", tau_values), ("LDS", lds_values)]:
        descriptive[name] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "median": float(np.median(vals)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
        }

    # ── 6. Assemble and save results ──────────────────────────────────────────
    results = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "n_test": n_test,
        "seed": 42,
        "timestamp": datetime.now().isoformat(),
        "anova_type": "Type I (sequential), class entered first",
        "predictors": ["class (10 levels)", "log(1 + gradient_norm)"],
        "response_variables": response_vars,
        "variance_decomposition": anova_results,
        "gate_evaluation": {
            "criterion": "residual > 30% on at least 1 metric",
            "residual_fractions": residuals,
            "pass": gate_pass,
            "decision": "PASS" if gate_pass else "FAIL",
        },
        "descriptive_statistics": descriptive,
        "per_class_statistics": per_class_stats,
        "correlation_matrix": corr_matrix,
        "key_observations": [],
        "limitations": [
            "Pilot uses 100 test points (not 500 as planned for full experiment)",
            "IF methods use layer4+fc only (not full model) - J10 has very low variance",
            "TRAK uses 1 checkpoint, JL dim=512 (pilot settings, not TRAK-50)",
            "Train subset is 5K (not full 50K)",
            "Class distribution may not be perfectly balanced (non-stratified 100 points)",
        ],
    }

    # Add key observations based on results
    obs = results["key_observations"]
    for resp in response_vars:
        r = anova_results[resp]
        if r.get("residual_R2") is not None:
            if r["residual_R2"] > 0.50:
                obs.append(f"{resp}: Residual dominates ({r['residual_R2']:.1%}) - strong per-point signal")
            elif r["residual_R2"] > 0.30:
                obs.append(f"{resp}: Substantial residual ({r['residual_R2']:.1%}) - meaningful per-point variation")
            else:
                obs.append(f"{resp}: Low residual ({r['residual_R2']:.1%}) - class/grad_norm explain most variance")

    if j10_values.std() < 0.05:
        obs.append("J10 has very low variance (std < 0.05) - likely due to layer4+fc only setting")

    # Save results
    output_path = RESULTS_DIR / "phase1_variance_decomposition.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Also save a human-readable summary
    summary_path = RESULTS_DIR / "phase1_variance_decomposition_summary.md"
    with open(summary_path, "w") as f:
        f.write("# Phase 1 Variance Decomposition - PILOT Results\n\n")
        f.write(f"**N test points**: {n_test}\n")
        f.write(f"**ANOVA type**: Type I sequential, class entered first\n\n")

        f.write("## Variance Decomposition (Partial R-squared)\n\n")
        f.write("| Response | Class R² | GradNorm R² | Interaction R² | Residual R² | Gate |\n")
        f.write("|----------|----------|-------------|----------------|-------------|------|\n")
        for resp in response_vars:
            r = anova_results[resp]
            if r.get("residual_R2") is not None:
                gate_str = "PASS" if r["residual_R2"] > 0.30 else "FAIL"
                f.write(f"| {resp} | {r['class_R2']:.4f} | {r['grad_norm_R2']:.4f} | "
                        f"{r['interaction_R2']:.4f} | {r['residual_R2']:.4f} | {gate_str} |\n")
            else:
                f.write(f"| {resp} | ERROR | | | | |\n")

        f.write(f"\n## Gate Decision: **{'PASS' if gate_pass else 'FAIL'}**\n\n")
        f.write(f"Criterion: residual > 30% on at least 1 metric\n\n")

        f.write("## Descriptive Statistics\n\n")
        f.write("| Metric | Mean | Std | Min | Max | Median |\n")
        f.write("|--------|------|-----|-----|-----|--------|\n")
        for name in response_vars:
            d = descriptive[name]
            f.write(f"| {name} | {d['mean']:.4f} | {d['std']:.4f} | {d['min']:.4f} | "
                    f"{d['max']:.4f} | {d['median']:.4f} |\n")

        f.write("\n## Key Observations\n\n")
        for o in obs:
            f.write(f"- {o}\n")

        f.write("\n## Limitations\n\n")
        for lim in results["limitations"]:
            f.write(f"- {lim}\n")

    print(f"Summary saved to {summary_path}")

    # Print final summary
    print("\n" + "="*70)
    print("VARIANCE DECOMPOSITION PILOT COMPLETE")
    print("="*70)
    print(f"\nGate decision: {'PASS' if gate_pass else 'FAIL'}")
    for resp in response_vars:
        r = anova_results[resp]
        if r.get("residual_R2") is not None:
            print(f"  {resp}: class={r['class_R2']:.4f}, grad_norm={r['grad_norm_R2']:.4f}, "
                  f"interaction={r['interaction_R2']:.4f}, residual={r['residual_R2']:.4f}")

    mark_task_done(TASK_ID, RESULTS_DIR, status="success",
                   summary=f"Gate={'PASS' if gate_pass else 'FAIL'}, "
                   f"residuals={json.dumps({k: round(v,4) for k,v in residuals.items()})}")

except Exception as e:
    import traceback
    error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    print(f"\nFATAL ERROR: {error_msg}", file=sys.stderr)
    mark_task_done(TASK_ID, str(RESULTS_DIR), status="failed", summary=error_msg[:500])
    sys.exit(1)
