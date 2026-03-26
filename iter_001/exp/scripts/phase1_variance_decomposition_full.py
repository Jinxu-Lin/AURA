#!/usr/bin/env python3
"""
Phase 1 Variance Decomposition (FULL mode, 500 test points)

Two-way ANOVA on 500 test points:
- Response variables: J10 (Jaccard@10 EK-FAC vs K-FAC), tau (Kendall tau IF vs RepSim), LDS (per-point LDS EK-FAC vs TRAK-50)
- Predictors: class (10 levels), log(gradient_norm)
- Type I sequential SS with class entered first
- Report partial R-squared for class, grad_norm, interaction, residual
- Gate: residual > 30% on >=1 metric = PASS
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = WORKSPACE / "exp" / "results"
INPUT_FILE = RESULTS_DIR / "phase1_attributions" / "per_point_analysis.json"
OUTPUT_FILE = RESULTS_DIR / "phase1_variance_decomposition.json"
SUMMARY_FILE = RESULTS_DIR / "phase1_variance_decomposition_summary.md"

def load_data():
    with open(INPUT_FILE) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} test points")
    return data

def compute_anova_type1(y, class_labels, log_grad_norm):
    """
    Two-way ANOVA Type I (sequential SS) with class entered first.
    Returns partial R-squared for class, grad_norm, interaction, residual.
    Uses statsmodels OLS for proper Type I SS computation.
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    import pandas as pd

    df = pd.DataFrame({
        'y': y,
        'class_label': pd.Categorical(class_labels),
        'log_grad_norm': log_grad_norm
    })

    # Type I ANOVA: class entered first
    model = ols('y ~ C(class_label) + log_grad_norm + C(class_label):log_grad_norm', data=df).fit()

    # Type I (sequential) SS
    anova_table = sm.stats.anova_lm(model, typ=1)

    total_ss = np.sum((y - np.mean(y))**2)

    # Extract SS from ANOVA table
    class_ss = anova_table.loc['C(class_label)', 'sum_sq']
    grad_norm_ss = anova_table.loc['log_grad_norm', 'sum_sq']
    interaction_ss = anova_table.loc['C(class_label):log_grad_norm', 'sum_sq']
    residual_ss = anova_table.loc['Residual', 'sum_sq']

    # R-squared fractions (proportion of total SS)
    class_R2 = class_ss / total_ss
    grad_norm_R2 = grad_norm_ss / total_ss
    interaction_R2 = interaction_ss / total_ss
    residual_R2 = residual_ss / total_ss

    # F-statistics and p-values
    class_F = anova_table.loc['C(class_label)', 'F']
    class_p = anova_table.loc['C(class_label)', 'PR(>F)']
    grad_norm_F = anova_table.loc['log_grad_norm', 'F']
    grad_norm_p = anova_table.loc['log_grad_norm', 'PR(>F)']
    interaction_F = anova_table.loc['C(class_label):log_grad_norm', 'F']
    interaction_p = anova_table.loc['C(class_label):log_grad_norm', 'PR(>F)']

    n_obs = len(y)
    df_model = int(anova_table['df'].sum() - anova_table.loc['Residual', 'df'])
    df_resid = int(anova_table.loc['Residual', 'df'])

    return {
        'class_R2': float(class_R2),
        'grad_norm_R2': float(grad_norm_R2),
        'interaction_R2': float(interaction_R2),
        'residual_R2': float(residual_R2),
        'total_ss': float(total_ss),
        'class_ss': float(class_ss),
        'grad_norm_ss': float(grad_norm_ss),
        'interaction_ss': float(interaction_ss),
        'residual_ss': float(residual_ss),
        'class_F': float(class_F),
        'class_p': float(class_p),
        'grad_norm_F': float(grad_norm_F),
        'grad_norm_p': float(grad_norm_p),
        'interaction_F': float(interaction_F),
        'interaction_p': float(interaction_p),
        'model_R2': float(model.rsquared),
        'model_adj_R2': float(model.rsquared_adj),
        'n_obs': n_obs,
        'df_model': df_model,
        'df_resid': df_resid,
    }

def compute_descriptive_stats(values, name):
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75)),
    }

def compute_per_class_stats(data):
    from collections import defaultdict
    classes = defaultdict(list)
    for d in data:
        classes[d['true_label']].append(d)

    result = {}
    for cls in sorted(classes.keys()):
        pts = classes[cls]
        result[str(cls)] = {
            'n': len(pts),
            'J10_mean': float(np.mean([p['jaccard_at_10'] for p in pts])),
            'J10_std': float(np.std([p['jaccard_at_10'] for p in pts], ddof=1)),
            'tau_mean': float(np.mean([p['kendall_tau_if_repsim'] for p in pts])),
            'tau_std': float(np.std([p['kendall_tau_if_repsim'] for p in pts], ddof=1)),
            'LDS_mean': float(np.mean([p['lds_ekfac_trak50'] for p in pts])),
            'LDS_std': float(np.std([p['lds_ekfac_trak50'] for p in pts], ddof=1)),
            'grad_norm_mean': float(np.mean([p['grad_norm'] for p in pts])),
            'grad_norm_std': float(np.std([p['grad_norm'] for p in pts], ddof=1)),
        }
    return result

def compute_correlation_matrix(data):
    """Compute Spearman correlations between all key variables."""
    from scipy.stats import spearmanr

    variables = {
        'J10': [d['jaccard_at_10'] for d in data],
        'tau': [d['kendall_tau_if_repsim'] for d in data],
        'LDS': [d['lds_ekfac_trak50'] for d in data],
        'log_grad_norm': [d['log_grad_norm'] for d in data],
        'confidence': [d['confidence'] for d in data],
        'entropy': [d['entropy'] for d in data],
    }

    result = {}
    for name1, vals1 in variables.items():
        for name2, vals2 in variables.items():
            rho, p = spearmanr(vals1, vals2)
            result[f'{name1}_vs_{name2}'] = {
                'spearman_rho': float(rho),
                'p_value': float(p),
            }
    return result

def generate_summary(results):
    """Generate markdown summary."""
    vd = results['variance_decomposition']
    gate = results['gate_evaluation']

    lines = [
        "# Phase 1 Variance Decomposition (FULL, 500 test points)",
        "",
        "## Task",
        "Two-way ANOVA on 500 CIFAR-10 test points (50/class, stratified).",
        "Predictors: class (10 levels), log(gradient_norm). Type I sequential SS.",
        "",
        "## Variance Decomposition Results",
        "",
        "| Response | Class R² | GradNorm R² | Interaction R² | Residual R² |",
        "|----------|----------|-------------|----------------|-------------|",
    ]

    for metric in ['J10', 'tau', 'LDS']:
        d = vd[metric]
        lines.append(
            f"| {metric} | {d['class_R2']:.3f} | {d['grad_norm_R2']:.3f} | "
            f"{d['interaction_R2']:.3f} | {d['residual_R2']:.3f} |"
        )

    lines += [
        "",
        "## Statistical Significance",
        "",
        "| Response | Class F | Class p | GradNorm F | GradNorm p | Interact F | Interact p |",
        "|----------|---------|---------|------------|------------|------------|------------|",
    ]

    for metric in ['J10', 'tau', 'LDS']:
        d = vd[metric]
        lines.append(
            f"| {metric} | {d['class_F']:.2f} | {d['class_p']:.2e} | "
            f"{d['grad_norm_F']:.2f} | {d['grad_norm_p']:.2e} | "
            f"{d['interaction_F']:.2f} | {d['interaction_p']:.2e} |"
        )

    lines += [
        "",
        "## Descriptive Statistics",
        "",
        "| Metric | Mean | Std | Min | Max | Median |",
        "|--------|------|-----|-----|-----|--------|",
    ]

    for metric in ['J10', 'tau', 'LDS']:
        d = results['descriptive_statistics'][metric]
        lines.append(
            f"| {metric} | {d['mean']:.4f} | {d['std']:.4f} | "
            f"{d['min']:.4f} | {d['max']:.4f} | {d['median']:.4f} |"
        )

    lines += [
        "",
        "## Gate Evaluation",
        "",
        f"**Criterion**: {gate['criterion']}",
        "",
        "| Metric | Residual Fraction | Pass? |",
        "|--------|-------------------|-------|",
    ]

    for metric in ['J10', 'tau', 'LDS']:
        frac = gate['residual_fractions'][metric]
        passed = frac > 0.30
        lines.append(f"| {metric} | {frac:.3f} | {'YES' if passed else 'NO'} |")

    lines += [
        "",
        f"**Overall Decision**: **{gate['decision']}**",
        "",
        "## Key Observations",
        "",
    ]
    for obs in results['key_observations']:
        lines.append(f"- {obs}")

    if results.get('limitations'):
        lines += ["", "## Limitations", ""]
        for lim in results['limitations']:
            lines.append(f"- {lim}")

    return "\n".join(lines)


def main():
    data = load_data()

    # Extract arrays
    class_labels = [d['true_label'] for d in data]
    log_grad_norm = [d['log_grad_norm'] for d in data]

    J10 = [d['jaccard_at_10'] for d in data]
    tau = [d['kendall_tau_if_repsim'] for d in data]
    LDS = [d['lds_ekfac_trak50'] for d in data]

    # Variance decomposition for each response variable
    print("Computing ANOVA for J10 (Jaccard@10 EK-FAC vs K-FAC)...")
    vd_J10 = compute_anova_type1(np.array(J10), class_labels, np.array(log_grad_norm))

    print("Computing ANOVA for tau (Kendall tau IF vs RepSim)...")
    vd_tau = compute_anova_type1(np.array(tau), class_labels, np.array(log_grad_norm))

    print("Computing ANOVA for LDS (per-point LDS EK-FAC vs TRAK-50)...")
    vd_LDS = compute_anova_type1(np.array(LDS), class_labels, np.array(log_grad_norm))

    # Gate evaluation
    residual_fractions = {
        'J10': vd_J10['residual_R2'],
        'tau': vd_tau['residual_R2'],
        'LDS': vd_LDS['residual_R2'],
    }
    gate_pass = any(r > 0.30 for r in residual_fractions.values())
    gate_all_below_20 = all(r < 0.20 for r in residual_fractions.values())

    if gate_pass:
        decision = "PASS"
    elif gate_all_below_20:
        decision = "STOP"
    else:
        decision = "BORDERLINE"

    # Key observations
    observations = []
    for metric, frac in residual_fractions.items():
        if frac > 0.50:
            observations.append(f"{metric}: Residual dominates ({frac*100:.1f}%) - strong per-point signal")
        elif frac > 0.30:
            observations.append(f"{metric}: Moderate residual ({frac*100:.1f}%) - meaningful per-point signal")
        else:
            observations.append(f"{metric}: Low residual ({frac*100:.1f}%) - class/grad_norm explain most variance")

    # Check J10 variance
    j10_std = np.std(J10, ddof=1)
    if j10_std < 0.05:
        observations.append(f"J10 has very low variance (std={j10_std:.4f}) - ceiling effect likely")

    # Check if class or grad_norm is dominant
    for metric, vd_data in [('J10', vd_J10), ('tau', vd_tau), ('LDS', vd_LDS)]:
        if vd_data['class_R2'] > 0.40:
            observations.append(f"{metric}: Class explains {vd_data['class_R2']*100:.1f}% of variance - significant class effect")
        if vd_data['grad_norm_R2'] > 0.40:
            observations.append(f"{metric}: Grad norm explains {vd_data['grad_norm_R2']*100:.1f}% of variance - significant gradient effect")

    # Correlation matrix
    print("Computing correlation matrix...")
    corr_matrix = compute_correlation_matrix(data)

    # Descriptive statistics
    desc_stats = {
        'J10': compute_descriptive_stats(J10, 'J10'),
        'tau': compute_descriptive_stats(tau, 'tau'),
        'LDS': compute_descriptive_stats(LDS, 'LDS'),
    }

    # Per-class statistics
    per_class = compute_per_class_stats(data)

    # Additional: also compute LDS for K-FAC and RepSim for richer analysis
    lds_kfac = [d['lds_kfac_trak50'] for d in data]
    lds_repsim = [d['lds_repsim_trak50'] for d in data]

    additional_lds = {
        'lds_kfac_trak50': compute_descriptive_stats(lds_kfac, 'LDS_kfac'),
        'lds_repsim_trak50': compute_descriptive_stats(lds_repsim, 'LDS_repsim'),
    }

    # Compile results
    results = {
        'task_id': 'phase1_variance_decomposition',
        'mode': 'FULL',
        'n_test': 500,
        'seed': 42,
        'timestamp': datetime.now().isoformat(),
        'anova_type': 'Type I (sequential), class entered first',
        'predictors': ['class (10 levels)', 'log(1 + gradient_norm)'],
        'response_variables': ['J10', 'tau', 'LDS'],
        'variance_decomposition': {
            'J10': vd_J10,
            'tau': vd_tau,
            'LDS': vd_LDS,
        },
        'gate_evaluation': {
            'criterion': 'residual > 30% on at least 1 metric',
            'residual_fractions': residual_fractions,
            'pass': gate_pass,
            'decision': decision,
        },
        'descriptive_statistics': desc_stats,
        'additional_lds_statistics': additional_lds,
        'per_class_statistics': per_class,
        'correlation_matrix': corr_matrix,
        'key_observations': observations,
        'limitations': [],  # Full experiment, no limitations beyond design choices
    }

    # Write results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {OUTPUT_FILE}")

    # Write summary
    summary = generate_summary(results)
    with open(SUMMARY_FILE, 'w') as f:
        f.write(summary)
    print(f"Summary written to {SUMMARY_FILE}")

    # Print key results
    print("\n" + "="*60)
    print("VARIANCE DECOMPOSITION RESULTS (FULL, N=500)")
    print("="*60)
    for metric in ['J10', 'tau', 'LDS']:
        d = results['variance_decomposition'][metric]
        print(f"\n{metric}:")
        print(f"  Class R²:       {d['class_R2']:.4f} (F={d['class_F']:.2f}, p={d['class_p']:.2e})")
        print(f"  GradNorm R²:    {d['grad_norm_R2']:.4f} (F={d['grad_norm_F']:.2f}, p={d['grad_norm_p']:.2e})")
        print(f"  Interaction R²: {d['interaction_R2']:.4f} (F={d['interaction_F']:.2f}, p={d['interaction_p']:.2e})")
        print(f"  Residual R²:    {d['residual_R2']:.4f}")

    print(f"\nGate: {decision}")
    print(f"  J10 residual: {residual_fractions['J10']:.3f}")
    print(f"  tau residual: {residual_fractions['tau']:.3f}")
    print(f"  LDS residual: {residual_fractions['LDS']:.3f}")

    return results

if __name__ == '__main__':
    main()
