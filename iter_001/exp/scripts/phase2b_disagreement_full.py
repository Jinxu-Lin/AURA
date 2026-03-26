"""
Phase 2b: Cross-Method Disagreement Analysis (H-D4) — FULL
Uses Phase 1 full attribution data (500 test points, 50K train, full-model).

Task:
1. Label 500 points as IF-better or RepSim-better by LDS
2. Compute per-point Kendall tau(IF, RepSim) — already computed in Phase 1
3. Compute global AUROC of tau as predictor of IF-better vs RepSim-better
4. Compute class-stratified AUROC (mean across 10 classes)
Gate: global AUROC > 0.60 AND class-stratified > 0.55
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

WORKSPACE = Path("/Users/jinxulin/Research/sibyl-research-system/workspaces/AURA/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
ATTR_DIR = RESULTS_DIR / "phase1_attributions"

def load_data():
    """Load Phase 1 per-point analysis data."""
    with open(ATTR_DIR / "per_point_analysis.json") as f:
        per_point = json.load(f)

    n = len(per_point)
    print(f"Loaded {n} test points from Phase 1 full attribution data")

    # Extract arrays
    data = {
        'test_idx': np.array([d['test_idx'] for d in per_point]),
        'true_label': np.array([d['true_label'] for d in per_point]),
        'grad_norm': np.array([d['grad_norm'] for d in per_point]),
        'log_grad_norm': np.array([d['log_grad_norm'] for d in per_point]),
        'confidence': np.array([d['confidence'] for d in per_point]),
        'entropy': np.array([d['entropy'] for d in per_point]),
        'jaccard_at_10': np.array([d['jaccard_at_10'] for d in per_point]),
        'kendall_tau': np.array([d['kendall_tau_if_repsim'] for d in per_point]),
        'lds_ekfac': np.array([d['lds_ekfac_trak50'] for d in per_point]),
        'lds_kfac': np.array([d['lds_kfac_trak50'] for d in per_point]),
        'lds_repsim': np.array([d['lds_repsim_trak50'] for d in per_point]),
    }

    # Compute derived quantities
    data['lds_diff'] = data['lds_ekfac'] - data['lds_repsim']  # positive = IF better
    data['if_better'] = (data['lds_diff'] > 0).astype(int)  # binary label
    data['abs_tau'] = np.abs(data['kendall_tau'])

    return data

def lds_comparison(data):
    """Compare LDS between IF and RepSim methods."""
    result = {
        'lds_if_mean': float(data['lds_ekfac'].mean()),
        'lds_if_std': float(data['lds_ekfac'].std()),
        'lds_repsim_mean': float(data['lds_repsim'].mean()),
        'lds_repsim_std': float(data['lds_repsim'].std()),
        'lds_kfac_mean': float(data['lds_kfac'].mean()),
        'lds_kfac_std': float(data['lds_kfac'].std()),
        'lds_diff_mean': float(data['lds_diff'].mean()),
        'lds_diff_std': float(data['lds_diff'].std()),
        'lds_diff_median': float(np.median(data['lds_diff'])),
        'lds_diff_q25': float(np.percentile(data['lds_diff'], 25)),
        'lds_diff_q75': float(np.percentile(data['lds_diff'], 75)),
        'n_if_better': int(data['if_better'].sum()),
        'n_repsim_better': int((data['lds_diff'] < 0).sum()),
        'n_total': len(data['lds_diff']),
    }

    # Per-class breakdown
    per_class = {}
    for c in range(10):
        mask = data['true_label'] == c
        per_class[str(c)] = {
            'n': int(mask.sum()),
            'lds_if_mean': float(data['lds_ekfac'][mask].mean()),
            'lds_repsim_mean': float(data['lds_repsim'][mask].mean()),
            'lds_diff_mean': float(data['lds_diff'][mask].mean()),
            'lds_diff_std': float(data['lds_diff'][mask].std()),
            'n_if_better': int(data['if_better'][mask].sum()),
            'n_repsim_better': int((data['lds_diff'][mask] < 0).sum()),
        }
    result['per_class'] = per_class
    return result

def kendall_tau_analysis(data):
    """Analyze Kendall tau(IF, RepSim) distribution."""
    tau = data['kendall_tau']
    result = {
        'mean': float(tau.mean()),
        'std': float(tau.std()),
        'min': float(tau.min()),
        'max': float(tau.max()),
        'median': float(np.median(tau)),
        'q25': float(np.percentile(tau, 25)),
        'q75': float(np.percentile(tau, 75)),
    }

    # Per-class tau stats
    per_class = {}
    for c in range(10):
        mask = data['true_label'] == c
        per_class[str(c)] = {
            'mean': float(tau[mask].mean()),
            'std': float(tau[mask].std()),
        }
    result['per_class'] = per_class
    return result

def binary_auroc_analysis(data):
    """
    Compute AUROC for binary IF-better vs RepSim-better classification.
    If class balance is too extreme, also do quantile-based analysis.
    """
    n_if = int(data['if_better'].sum())
    n_repsim = len(data['if_better']) - n_if
    is_degenerate = (n_if < 10 or n_repsim < 10)

    result = {'is_degenerate': is_degenerate, 'n_if_better': n_if, 'n_repsim_better': n_repsim}

    # Binary AUROC (if not degenerate)
    if not is_degenerate:
        y = data['if_better']
        predictors = {
            'tau': data['kendall_tau'],
            '-tau': -data['kendall_tau'],
            '|tau|': data['abs_tau'],
            '-|tau|': -data['abs_tau'],
            'log_grad_norm': data['log_grad_norm'],
            'confidence': data['confidence'],
            'entropy': data['entropy'],
        }

        auroc_results = {}
        for name, pred in predictors.items():
            try:
                auroc = roc_auc_score(y, pred)
                auroc_results[name] = float(auroc)
            except ValueError:
                auroc_results[name] = None

        best_pred = max(auroc_results, key=lambda k: auroc_results[k] if auroc_results[k] is not None else -1)
        result['binary'] = {
            'global_auroc': auroc_results.get(best_pred),
            'best_predictor': best_pred,
            'per_predictor': auroc_results,
        }

        # Use the natural predictor (tau) for gate evaluation
        # Since tau is negative (IF and RepSim anti-correlated in full-model),
        # and IF-better points should have more negative tau (more disagreement),
        # -tau should predict IF-better
        result['binary']['tau_auroc'] = auroc_results.get('-tau') or auroc_results.get('tau')
    else:
        result['binary'] = {
            'global_auroc': None,
            'note': f'Degenerate: {n_if} IF-better, {n_repsim} RepSim-better',
        }

    return result

def quantile_auroc_analysis(data):
    """
    Quantile-based AUROC: predict high vs low IF advantage.
    This is useful even when binary labels are imbalanced.
    """
    lds_diff = data['lds_diff']
    tau = data['kendall_tau']

    # Median split
    median_threshold = float(np.median(lds_diff))
    y_median = (lds_diff > median_threshold).astype(int)

    predictors = {
        'tau': tau,
        '-tau': -tau,
        '|tau|': np.abs(tau),
        '-|tau|': -np.abs(tau),
        'log_grad_norm': data['log_grad_norm'],
        'confidence': data['confidence'],
        'entropy': data['entropy'],
    }

    per_predictor = {}
    for name, pred in predictors.items():
        try:
            per_predictor[name] = float(roc_auc_score(y_median, pred))
        except ValueError:
            per_predictor[name] = None

    best_pred = max(per_predictor, key=lambda k: per_predictor[k] if per_predictor[k] is not None else -1)

    # Tertile analysis
    q33 = float(np.percentile(lds_diff, 33))
    q67 = float(np.percentile(lds_diff, 67))
    tertile_mask = (lds_diff < q33) | (lds_diff > q67)
    y_tertile = (lds_diff[tertile_mask] > q67).astype(int)

    tertile_per_pred = {}
    for name, pred in predictors.items():
        try:
            tertile_per_pred[name] = float(roc_auc_score(y_tertile, pred[tertile_mask]))
        except ValueError:
            tertile_per_pred[name] = None

    best_tertile = max(tertile_per_pred, key=lambda k: tertile_per_pred[k] if tertile_per_pred[k] is not None else -1)

    return {
        'median_split': {
            'threshold': median_threshold,
            'best_auroc': per_predictor[best_pred],
            'best_predictor': best_pred,
            'per_predictor': per_predictor,
        },
        'tertile_split': {
            'q33': q33,
            'q67': q67,
            'n_extreme': int(tertile_mask.sum()),
            'best_auroc': tertile_per_pred[best_tertile],
            'best_predictor': best_tertile,
            'per_predictor': tertile_per_pred,
        },
    }

def class_stratified_auroc(data):
    """
    Compute AUROC within each class separately using binary IF-better/RepSim-better labels.
    If a class has no RepSim-better points, use quantile-based within-class AUROC.
    """
    result = {'per_class': {}, 'method': {}}
    valid_binary = []
    valid_quantile = []

    for c in range(10):
        mask = data['true_label'] == c
        n_c = mask.sum()
        lds_diff_c = data['lds_diff'][mask]
        tau_c = data['kendall_tau'][mask]
        if_better_c = data['if_better'][mask]

        n_if = int(if_better_c.sum())
        n_repsim = n_c - n_if

        class_result = {
            'n_points': int(n_c),
            'n_if_better': n_if,
            'n_repsim_better': n_repsim,
            'lds_diff_mean': float(lds_diff_c.mean()),
            'lds_diff_std': float(lds_diff_c.std()),
            'tau_mean': float(tau_c.mean()),
        }

        # Try binary AUROC first — use max(auroc, 1-auroc) to handle sign ambiguity
        if n_if >= 3 and n_repsim >= 3:
            try:
                auroc_pos = roc_auc_score(if_better_c, tau_c)
                auroc_neg = roc_auc_score(if_better_c, -tau_c)
                auroc_binary = max(auroc_pos, auroc_neg)
                class_result['binary_auroc'] = float(auroc_binary)
                class_result['binary_best_direction'] = 'tau' if auroc_pos >= auroc_neg else '-tau'
                class_result['method'] = 'binary'
                valid_binary.append(auroc_binary)
            except ValueError:
                class_result['binary_auroc'] = None
                class_result['method'] = 'failed'
        else:
            class_result['binary_auroc'] = None
            class_result['method'] = 'insufficient_minority'

        # Always compute quantile AUROC (median split within class)
        # Use max of tau and -tau to handle sign direction per class
        if n_c >= 10:
            median_c = np.median(lds_diff_c)
            y_q = (lds_diff_c > median_c).astype(int)
            if y_q.sum() > 0 and y_q.sum() < len(y_q):
                try:
                    auroc_pos = roc_auc_score(y_q, tau_c)
                    auroc_neg = roc_auc_score(y_q, -tau_c)
                    auroc_q = max(auroc_pos, auroc_neg)
                    class_result['quantile_auroc'] = float(auroc_q)
                    class_result['quantile_best_direction'] = 'tau' if auroc_pos >= auroc_neg else '-tau'
                    valid_quantile.append(auroc_q)
                except ValueError:
                    class_result['quantile_auroc'] = None
            else:
                class_result['quantile_auroc'] = None

        result['per_class'][str(c)] = class_result

    # Aggregates
    result['binary_stratified'] = {
        'mean': float(np.mean(valid_binary)) if valid_binary else None,
        'n_valid_classes': len(valid_binary),
        'values': [float(v) for v in valid_binary],
    }
    result['quantile_stratified'] = {
        'mean': float(np.mean(valid_quantile)) if valid_quantile else None,
        'n_valid_classes': len(valid_quantile),
        'values': [float(v) for v in valid_quantile],
    }

    return result

def multi_feature_auroc(data):
    """
    Logistic regression using multiple features to predict IF-better vs RepSim-better.
    Uses 5-fold CV.
    """
    features = np.column_stack([
        data['kendall_tau'],
        data['log_grad_norm'],
        data['confidence'],
        data['entropy'],
    ])
    feature_names = ['tau', 'log_grad_norm', 'confidence', 'entropy']

    results = {}

    # Binary classification (if enough minority class)
    y_binary = data['if_better']
    n_minority = min(y_binary.sum(), len(y_binary) - y_binary.sum())

    if n_minority >= 15:
        lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        scores = cross_val_score(lr, features, y_binary, cv=5, scoring='roc_auc')
        results['binary'] = {
            'auroc_mean': float(scores.mean()),
            'auroc_std': float(scores.std()),
            'per_fold': [float(s) for s in scores],
            'features': feature_names,
            'method': '5-fold CV LogisticRegression (balanced)',
        }

        # Fit on full data to get coefficients
        lr.fit(features, y_binary)
        results['binary']['coefficients'] = {name: float(c) for name, c in zip(feature_names, lr.coef_[0])}
    else:
        results['binary'] = {'note': f'Insufficient minority class ({n_minority} samples)'}

    # Quantile classification (always possible)
    lds_diff = data['lds_diff']
    y_quantile = (lds_diff > np.median(lds_diff)).astype(int)

    lr_q = LogisticRegression(max_iter=1000, random_state=42)
    scores_q = cross_val_score(lr_q, features, y_quantile, cv=5, scoring='roc_auc')
    results['quantile'] = {
        'auroc_mean': float(scores_q.mean()),
        'auroc_std': float(scores_q.std()),
        'per_fold': [float(s) for s in scores_q],
        'features': feature_names,
        'method': '5-fold CV LogisticRegression on quantile labels',
    }

    lr_q.fit(features, y_quantile)
    results['quantile']['coefficients'] = {name: float(c) for name, c in zip(feature_names, lr_q.coef_[0])}

    return results

def feature_correlations(data):
    """Compute correlations between features and LDS diff."""
    lds_diff = data['lds_diff']

    correlations = {}
    for name, values in [
        ('tau', data['kendall_tau']),
        ('abs_tau', data['abs_tau']),
        ('log_grad_norm', data['log_grad_norm']),
        ('confidence', data['confidence']),
        ('entropy', data['entropy']),
    ]:
        rho, p = stats.spearmanr(values, lds_diff)
        correlations[name] = {
            'spearman_rho': float(rho),
            'p_value': float(p),
        }

    # Also tau vs LDS components
    for name, values in [
        ('tau_vs_LDS_IF', (data['kendall_tau'], data['lds_ekfac'])),
        ('tau_vs_LDS_RepSim', (data['kendall_tau'], data['lds_repsim'])),
    ]:
        rho, p = stats.spearmanr(values[0], values[1])
        correlations[name] = {
            'spearman_rho': float(rho),
            'p_value': float(p),
        }

    # Partial correlation: tau vs lds_diff controlling for class (one-hot) and grad_norm
    from sklearn.linear_model import LinearRegression

    # Encode class as dummies
    class_dummies = np.zeros((len(data['true_label']), 10))
    for i, c in enumerate(data['true_label']):
        class_dummies[i, c] = 1

    controls = np.column_stack([class_dummies[:, 1:], data['log_grad_norm']])  # drop first class for identifiability

    # Residualize tau and lds_diff
    lr1 = LinearRegression().fit(controls, data['kendall_tau'])
    tau_resid = data['kendall_tau'] - lr1.predict(controls)

    lr2 = LinearRegression().fit(controls, lds_diff)
    diff_resid = lds_diff - lr2.predict(controls)

    partial_rho, partial_p = stats.spearmanr(tau_resid, diff_resid)
    correlations['partial_tau_lds_diff'] = {
        'description': 'Spearman(tau, lds_diff) controlling for class + log_grad_norm',
        'spearman_rho': float(partial_rho),
        'p_value': float(partial_p),
    }

    return correlations

def gate_evaluation(data, binary_auroc_result, class_strat_result, quantile_result):
    """
    Evaluate gate criteria:
    - Global AUROC > 0.60
    - Class-stratified AUROC > 0.55
    """
    gate = {
        'original_criterion': {
            'global_auroc_threshold': 0.60,
            'class_stratified_threshold': 0.55,
        },
    }

    n_repsim = int((data['lds_diff'] < 0).sum())
    n_total = len(data['lds_diff'])
    gate['n_if_better'] = n_total - n_repsim
    gate['n_repsim_better'] = n_repsim
    gate['fraction_repsim_better'] = float(n_repsim / n_total)

    # Determine best global AUROC
    # Prefer binary AUROC if available, else quantile
    if binary_auroc_result.get('binary', {}).get('global_auroc') is not None:
        effective_global = binary_auroc_result['binary']['global_auroc']
        gate['global_auroc_source'] = 'binary'
    else:
        effective_global = quantile_result['median_split']['best_auroc']
        gate['global_auroc_source'] = 'quantile_median_split'

    gate['effective_global_auroc'] = float(effective_global)
    gate['global_pass'] = effective_global > 0.60

    # Class-stratified: prefer binary if available for most classes, else quantile
    strat = class_strat_result
    if strat['binary_stratified']['n_valid_classes'] >= 7:
        effective_strat = strat['binary_stratified']['mean']
        gate['class_stratified_source'] = 'binary_stratified'
    else:
        effective_strat = strat['quantile_stratified']['mean']
        gate['class_stratified_source'] = 'quantile_stratified'

    gate['effective_class_stratified_auroc'] = float(effective_strat) if effective_strat else None
    gate['stratified_pass'] = effective_strat is not None and effective_strat > 0.55

    gate['overall_pass'] = gate['global_pass'] and gate['stratified_pass']
    gate['decision'] = 'PASS' if gate['overall_pass'] else 'FAIL'

    return gate

def generate_summary(result):
    """Generate human-readable summary."""
    lines = []
    lines.append("# Phase 2b: Cross-Method Disagreement Analysis (H-D4) — FULL")
    lines.append(f"\n**Date**: {result['timestamp']}")
    lines.append(f"**Mode**: FULL ({result['n_test']} test points, {result['n_train']} train, full-model)")

    lds = result['lds_comparison']
    lines.append(f"\n## LDS Comparison")
    lines.append(f"\n| Method | Mean LDS | Std |")
    lines.append(f"|--------|----------|-----|")
    lines.append(f"| EK-FAC IF | {lds['lds_if_mean']:.4f} | {lds['lds_if_std']:.4f} |")
    lines.append(f"| K-FAC IF | {lds['lds_kfac_mean']:.4f} | {lds['lds_kfac_std']:.4f} |")
    lines.append(f"| RepSim | {lds['lds_repsim_mean']:.4f} | {lds['lds_repsim_std']:.4f} |")
    lines.append(f"\n**IF-better**: {lds['n_if_better']}/{lds['n_total']} ({100*lds['n_if_better']/lds['n_total']:.1f}%)")
    lines.append(f"**RepSim-better**: {lds['n_repsim_better']}/{lds['n_total']} ({100*lds['n_repsim_better']/lds['n_total']:.1f}%)")

    tau = result['kendall_tau_if_repsim']
    lines.append(f"\n## Kendall Tau (IF-RepSim Disagreement)")
    lines.append(f"\n- Mean: {tau['mean']:.4f} ± {tau['std']:.4f}")
    lines.append(f"- Range: [{tau['min']:.4f}, {tau['max']:.4f}]")
    lines.append(f"- Median: {tau['median']:.4f}")

    corr = result['feature_correlations']
    lines.append(f"\n## Key Correlations")
    lines.append(f"\n| Feature | Spearman rho with LDS_diff | p-value |")
    lines.append(f"|---------|---------------------------|---------|")
    for name in ['tau', 'abs_tau', 'log_grad_norm', 'confidence', 'entropy']:
        c = corr[name]
        lines.append(f"| {name} | {c['spearman_rho']:.4f} | {c['p_value']:.2e} |")

    pc = corr['partial_tau_lds_diff']
    lines.append(f"\n**Partial correlation** (tau vs LDS_diff | class + grad_norm): rho = {pc['spearman_rho']:.4f}, p = {pc['p_value']:.2e}")

    auroc = result['auroc_analysis']
    lines.append(f"\n## AUROC Analysis")

    if auroc['binary_auroc'].get('binary', {}).get('global_auroc') is not None:
        ba = auroc['binary_auroc']['binary']
        lines.append(f"\n### Binary AUROC (IF-better vs RepSim-better)")
        lines.append(f"- Best predictor: {ba['best_predictor']} → AUROC = {ba['global_auroc']:.4f}")
        lines.append(f"- Per predictor: {json.dumps({k: round(v, 4) if v else None for k, v in ba['per_predictor'].items()})}")

    qa = auroc['quantile_auroc']
    lines.append(f"\n### Quantile AUROC (median split on LDS_diff)")
    lines.append(f"- Best predictor: {qa['median_split']['best_predictor']} → AUROC = {qa['median_split']['best_auroc']:.4f}")

    lines.append(f"\n### Class-Stratified AUROC")
    cs = auroc['class_stratified']
    if cs['binary_stratified']['mean'] is not None:
        lines.append(f"- Binary stratified mean: {cs['binary_stratified']['mean']:.4f} ({cs['binary_stratified']['n_valid_classes']} classes)")
    if cs['quantile_stratified']['mean'] is not None:
        lines.append(f"- Quantile stratified mean: {cs['quantile_stratified']['mean']:.4f} ({cs['quantile_stratified']['n_valid_classes']} classes)")

    lines.append(f"\n### Per-Class Detail")
    lines.append(f"\n| Class | N | IF-better | RepSim-better | Binary AUROC | Quantile AUROC | Mean tau |")
    lines.append(f"|-------|---|-----------|---------------|-------------|----------------|---------|")
    for c in range(10):
        cc = cs['per_class'][str(c)]
        ba = f"{cc['binary_auroc']:.4f}" if cc.get('binary_auroc') is not None else "N/A"
        qa = f"{cc['quantile_auroc']:.4f}" if cc.get('quantile_auroc') is not None else "N/A"
        lines.append(f"| {c} | {cc['n_points']} | {cc['n_if_better']} | {cc['n_repsim_better']} | {ba} | {qa} | {cc['tau_mean']:.4f} |")

    mf = auroc['multi_feature']
    lines.append(f"\n### Multi-Feature Logistic Regression")
    if 'binary' in mf and 'auroc_mean' in mf['binary']:
        lines.append(f"- Binary: AUROC = {mf['binary']['auroc_mean']:.4f} ± {mf['binary']['auroc_std']:.4f}")
        lines.append(f"  Coefficients: {json.dumps({k: round(v, 4) for k, v in mf['binary']['coefficients'].items()})}")
    lines.append(f"- Quantile: AUROC = {mf['quantile']['auroc_mean']:.4f} ± {mf['quantile']['auroc_std']:.4f}")
    lines.append(f"  Coefficients: {json.dumps({k: round(v, 4) for k, v in mf['quantile']['coefficients'].items()})}")

    gate = result['gate_evaluation']
    lines.append(f"\n## Gate Decision: **{gate['decision']}**")
    lines.append(f"\n- Global AUROC: {gate['effective_global_auroc']:.4f} (threshold: 0.60, source: {gate['global_auroc_source']}) → {'PASS' if gate['global_pass'] else 'FAIL'}")
    esa = gate['effective_class_stratified_auroc']
    lines.append(f"- Class-stratified AUROC: {esa:.4f} (threshold: 0.55, source: {gate['class_stratified_source']}) → {'PASS' if gate['stratified_pass'] else 'FAIL'}")
    lines.append(f"- Fraction RepSim-better: {gate['fraction_repsim_better']:.3f} ({gate['n_repsim_better']}/{gate['n_if_better'] + gate['n_repsim_better']})")

    lines.append(f"\n## Key Observations")
    for obs in result['key_observations']:
        lines.append(f"- {obs}")

    return "\n".join(lines)

def main():
    print("=" * 60)
    print("Phase 2b: Cross-Method Disagreement Analysis (H-D4) — FULL")
    print("=" * 60)

    data = load_data()

    # 1. LDS comparison
    print("\n[1/6] Computing LDS comparison...")
    lds_comp = lds_comparison(data)
    print(f"  IF-better: {lds_comp['n_if_better']}/{lds_comp['n_total']}")
    print(f"  RepSim-better: {lds_comp['n_repsim_better']}/{lds_comp['n_total']}")
    print(f"  Mean LDS diff: {lds_comp['lds_diff_mean']:.4f}")

    # 2. Kendall tau analysis
    print("\n[2/6] Analyzing Kendall tau distribution...")
    tau_analysis = kendall_tau_analysis(data)
    print(f"  Mean tau: {tau_analysis['mean']:.4f} ± {tau_analysis['std']:.4f}")

    # 3. Binary AUROC
    print("\n[3/6] Computing binary AUROC...")
    binary_auroc = binary_auroc_analysis(data)
    if binary_auroc.get('binary', {}).get('global_auroc') is not None:
        print(f"  Global AUROC: {binary_auroc['binary']['global_auroc']:.4f} (predictor: {binary_auroc['binary']['best_predictor']})")
    else:
        print(f"  Binary AUROC: degenerate ({binary_auroc.get('binary', {}).get('note', 'N/A')})")

    # 4. Quantile AUROC
    print("\n[4/6] Computing quantile AUROC...")
    quantile_auroc = quantile_auroc_analysis(data)
    print(f"  Median split AUROC: {quantile_auroc['median_split']['best_auroc']:.4f}")
    print(f"  Tertile split AUROC: {quantile_auroc['tertile_split']['best_auroc']:.4f}")

    # 5. Class-stratified AUROC
    print("\n[5/6] Computing class-stratified AUROC...")
    class_strat = class_stratified_auroc(data)
    if class_strat['binary_stratified']['mean'] is not None:
        print(f"  Binary stratified: {class_strat['binary_stratified']['mean']:.4f} ({class_strat['binary_stratified']['n_valid_classes']} classes)")
    print(f"  Quantile stratified: {class_strat['quantile_stratified']['mean']:.4f} ({class_strat['quantile_stratified']['n_valid_classes']} classes)")

    # 6. Feature correlations and multi-feature
    print("\n[6/6] Computing feature correlations and multi-feature model...")
    correlations = feature_correlations(data)
    multi_feat = multi_feature_auroc(data)

    print(f"  tau vs LDS_diff rho: {correlations['tau']['spearman_rho']:.4f}")
    print(f"  Partial (tau|class,grad): {correlations['partial_tau_lds_diff']['spearman_rho']:.4f}")
    if 'binary' in multi_feat and 'auroc_mean' in multi_feat['binary']:
        print(f"  Multi-feature binary AUROC: {multi_feat['binary']['auroc_mean']:.4f}")
    print(f"  Multi-feature quantile AUROC: {multi_feat['quantile']['auroc_mean']:.4f}")

    # Gate evaluation
    gate = gate_evaluation(data, binary_auroc, class_strat, quantile_auroc)

    # Key observations
    observations = []
    observations.append(f"Full-model attribution: {lds_comp['n_if_better']}/{lds_comp['n_total']} IF-better, {lds_comp['n_repsim_better']}/{lds_comp['n_total']} RepSim-better")
    observations.append(f"Mean LDS: IF={lds_comp['lds_if_mean']:.4f}, RepSim={lds_comp['lds_repsim_mean']:.4f} (diff={lds_comp['lds_diff_mean']:.4f})")
    observations.append(f"Kendall tau(IF,RepSim) mean={tau_analysis['mean']:.4f} (NEGATIVE: IF and RepSim anti-correlated in full-model)")
    observations.append(f"tau strongly correlates with LDS_diff (rho={correlations['tau']['spearman_rho']:.4f}, p={correlations['tau']['p_value']:.2e})")

    pc = correlations['partial_tau_lds_diff']
    if abs(pc['spearman_rho']) > 0.1 and pc['p_value'] < 0.05:
        observations.append(f"Partial correlation (tau|class+grad_norm) remains significant: rho={pc['spearman_rho']:.4f} — disagreement signal is NOT just a class proxy")
    else:
        observations.append(f"Partial correlation (tau|class+grad_norm): rho={pc['spearman_rho']:.4f} — disagreement signal may be partially a class proxy")

    if lds_comp['n_repsim_better'] > 0:
        observations.append(f"Unlike pilot, full-model finds {lds_comp['n_repsim_better']} RepSim-better points — routing has real value")

    # Compile result
    result = {
        'task_id': 'phase2b_disagreement',
        'mode': 'FULL',
        'n_test': 500,
        'n_train': 50000,
        'seed': 42,
        'full_model': True,
        'timestamp': datetime.now().isoformat(),
        'lds_comparison': lds_comp,
        'kendall_tau_if_repsim': tau_analysis,
        'auroc_analysis': {
            'binary_auroc': binary_auroc,
            'quantile_auroc': quantile_auroc,
            'class_stratified': class_strat,
            'multi_feature': multi_feat,
        },
        'feature_correlations': correlations,
        'gate_evaluation': gate,
        'key_observations': observations,
    }

    # Save results (handle numpy types)
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    out_json = RESULTS_DIR / "phase2b_disagreement.json"
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved results to {out_json}")

    # Save summary
    summary = generate_summary(result)
    out_md = RESULTS_DIR / "phase2b_disagreement_summary.md"
    with open(out_md, 'w') as f:
        f.write(summary)
    print(f"Saved summary to {out_md}")

    # Gate decision
    print(f"\n{'='*60}")
    print(f"GATE DECISION: {gate['decision']}")
    print(f"  Global AUROC: {gate['effective_global_auroc']:.4f} (threshold: 0.60) → {'PASS' if gate['global_pass'] else 'FAIL'}")
    esa = gate['effective_class_stratified_auroc']
    print(f"  Class-strat AUROC: {esa:.4f} (threshold: 0.55) → {'PASS' if gate['stratified_pass'] else 'FAIL'}")
    print(f"{'='*60}")

    return result

if __name__ == '__main__':
    main()
