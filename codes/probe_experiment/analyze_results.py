"""
Analyze probe experiment results for the AURA project.

Produces:
1. TRV distribution analysis (falsification check: need >=3 levels with >10%)
2. Jaccard@k degradation curves (falsification check: adjacent levels avg <0.85)
3. SI-TRV correlation (H4 validation)
4. Cross-seed TRV stability (training variance confound: Spearman rho >0.6)
5. OOD confound check (TRV-OOD correlation < 0.8)
6. High vs Low confidence TRV comparison
7. Summary statistics and verdict
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_results(output_dir, seeds):
    results = {}
    for seed in seeds:
        path = os.path.join(output_dir, f'attribution_results_seed{seed}.json')
        if os.path.exists(path):
            with open(path) as f:
                results[seed] = json.load(f)
    return results


def analyze_trv_distribution(results, seed, output_dir):
    """Check falsification condition: >=3 TRV levels with >10% of test points."""
    n_test = results['n_test']
    trv = np.array(results['trv'][:n_test])  # Only in-distribution test points
    n_levels = len(results['level_names'])

    counts = np.bincount(trv, minlength=n_levels + 1)
    fractions = counts / n_test

    print(f"\n  TRV Distribution (seed {seed}, n={n_test}):")
    for i, name in enumerate(['TRV=0 (unstable)'] + [f'TRV={j+1} ({results["level_names"][j]})'
                                                       for j in range(n_levels)]):
        print(f"    {name}: {counts[i]} ({fractions[i]*100:.1f}%)")

    # Falsification check
    levels_above_10pct = np.sum(fractions > 0.10)
    pass_trv = levels_above_10pct >= 3
    print(f"\n  Levels with >10% of points: {levels_above_10pct}")
    print(f"  Falsification check (>=3 levels with >10%): {'PASS' if pass_trv else 'FAIL'}")

    return {
        'counts': counts.tolist(),
        'fractions': fractions.tolist(),
        'levels_above_10pct': int(levels_above_10pct),
        'pass_trv_distribution': pass_trv,
    }


def analyze_jaccard_degradation(results, seed, output_dir):
    """Check early signal falsification: adjacent levels avg Jaccard@10 < 0.85."""
    n_test = results['n_test']
    jaccard = np.array(results['jaccard_matrix'][:n_test])
    level_names = results['level_names']

    mean_jaccard = jaccard.mean(axis=0)
    std_jaccard = jaccard.std(axis=0)

    print(f"\n  Jaccard@{results['k']} Degradation (seed {seed}):")
    for i, name in enumerate(level_names):
        print(f"    {name}: {mean_jaccard[i]:.3f} ± {std_jaccard[i]:.3f}")

    # Adjacent level differences
    adj_jaccards = []
    for i in range(1, len(level_names)):
        adj_j = jaccard[:, i].mean()
        adj_jaccards.append(adj_j)

    # Falsification: adjacent levels should have avg Jaccard < 0.85
    max_adj_jaccard = max(adj_jaccards) if adj_jaccards else 1.0
    pass_degradation = max_adj_jaccard < 0.85
    print(f"\n  Max adjacent-level avg Jaccard: {max_adj_jaccard:.3f}")
    print(f"  Falsification check (adj Jaccard < 0.85): {'PASS' if pass_degradation else 'FAIL'}")

    # Per-point variance of Jaccard (should be std > 0.15 for meaningful variation)
    per_point_std = jaccard[:, 1:].std(axis=1)  # Exclude self-comparison (level 1)
    mean_per_point_std = per_point_std.mean()
    pass_variance = mean_per_point_std > 0.15
    print(f"  Mean per-point Jaccard std: {mean_per_point_std:.3f}")
    print(f"  Variance check (std > 0.15): {'PASS' if pass_variance else 'FAIL'}")

    return {
        'mean_jaccard': mean_jaccard.tolist(),
        'std_jaccard': std_jaccard.tolist(),
        'adj_jaccards': adj_jaccards,
        'max_adj_jaccard': max_adj_jaccard,
        'pass_degradation': pass_degradation,
        'mean_per_point_std': float(mean_per_point_std),
        'pass_variance': pass_variance,
    }


def analyze_si_trv_correlation(results, seed):
    """Check H4: SI-TRV correlation (Spearman)."""
    n_test = results['n_test']
    si = np.array(results['si_eval'][:n_test])
    trv = np.array(results['trv'][:n_test])

    # TRV should be negatively correlated with SI (high SI = high leverage = unstable = low TRV)
    rho_si_trv, p_si_trv = stats.spearmanr(si, trv)
    # Also check 1/SI vs TRV (should be positively correlated)
    rho_inv_si_trv, p_inv_si_trv = stats.spearmanr(1.0 / (si + 1e-10), trv)

    print(f"\n  SI-TRV Correlation (seed {seed}):")
    print(f"    Spearman(SI, TRV) = {rho_si_trv:.3f} (p={p_si_trv:.4f})")
    print(f"    Spearman(1/SI, TRV) = {rho_inv_si_trv:.3f} (p={p_inv_si_trv:.4f})")

    h4_valid = abs(rho_inv_si_trv) > 0.3 and p_inv_si_trv < 0.05
    print(f"    H4 (SI is TRV proxy): {'Supported' if h4_valid else 'Weak/Not supported'}")

    return {
        'rho_si_trv': float(rho_si_trv),
        'p_si_trv': float(p_si_trv),
        'rho_inv_si_trv': float(rho_inv_si_trv),
        'p_inv_si_trv': float(p_inv_si_trv),
        'h4_supported': h4_valid,
    }


def analyze_confidence_stratification(results, seed):
    """Compare TRV between high and low confidence test points."""
    n_test = results['n_test']
    trv = np.array(results['trv'][:n_test])
    conf = np.array(results['confidence_labels'])

    high_trv = trv[conf == 'high']
    low_trv = trv[conf == 'low']

    stat, p_value = stats.mannwhitneyu(high_trv, low_trv, alternative='two-sided')

    print(f"\n  Confidence Stratification (seed {seed}):")
    print(f"    High confidence TRV: mean={high_trv.mean():.2f}, median={np.median(high_trv):.1f}")
    print(f"    Low confidence TRV:  mean={low_trv.mean():.2f}, median={np.median(low_trv):.1f}")
    print(f"    Mann-Whitney U test: U={stat:.1f}, p={p_value:.4f}")

    return {
        'high_trv_mean': float(high_trv.mean()),
        'low_trv_mean': float(low_trv.mean()),
        'high_trv_median': float(np.median(high_trv)),
        'low_trv_median': float(np.median(low_trv)),
        'mannwhitney_u': float(stat),
        'mannwhitney_p': float(p_value),
        'significant': p_value < 0.05,
    }


def analyze_ood_confound(results, seed):
    """Check if TRV correlates too strongly with OOD-ness."""
    n_test = results['n_test']
    n_ood = results['n_ood']
    trv = np.array(results['trv'])

    id_trv = trv[:n_test]
    ood_trv = trv[n_test:n_test + n_ood]

    # Simple comparison: are OOD points' TRV systematically different?
    stat, p_value = stats.mannwhitneyu(id_trv, ood_trv, alternative='two-sided')

    # Create binary OOD label and compute point-biserial correlation
    ood_label = np.concatenate([np.zeros(n_test), np.ones(n_ood)])
    all_trv = np.concatenate([id_trv, ood_trv])
    rho, p_rho = stats.spearmanr(ood_label, all_trv)

    print(f"\n  OOD Confound Check (seed {seed}):")
    print(f"    ID TRV:  mean={id_trv.mean():.2f}, median={np.median(id_trv):.1f}")
    print(f"    OOD TRV: mean={ood_trv.mean():.2f}, median={np.median(ood_trv):.1f}")
    print(f"    Spearman(OOD_label, TRV) = {rho:.3f} (p={p_rho:.4f})")
    pass_ood = abs(rho) < 0.8
    print(f"    OOD confound check (|rho| < 0.8): {'PASS' if pass_ood else 'FAIL'}")

    return {
        'id_trv_mean': float(id_trv.mean()),
        'ood_trv_mean': float(ood_trv.mean()),
        'rho_ood_trv': float(rho),
        'p_ood_trv': float(p_rho),
        'pass_ood_confound': pass_ood,
    }


def analyze_cross_seed_stability(all_results, seeds, output_dir):
    """Check training variance confound: TRV rank stability across seeds."""
    if len(all_results) < 2:
        print("\n  Cross-seed stability: Not enough seeds to analyze.")
        return {'sufficient_seeds': False}

    seed_list = sorted(all_results.keys())
    n_test = all_results[seed_list[0]]['n_test']

    # Get TRV for each seed (only in-distribution test points)
    trv_per_seed = {}
    for seed in seed_list:
        trv_per_seed[seed] = np.array(all_results[seed]['trv'][:n_test])

    # Pairwise Spearman correlation
    rhos = []
    print(f"\n  Cross-seed TRV Stability:")
    for i in range(len(seed_list)):
        for j in range(i + 1, len(seed_list)):
            s1, s2 = seed_list[i], seed_list[j]
            rho, p = stats.spearmanr(trv_per_seed[s1], trv_per_seed[s2])
            rhos.append(rho)
            print(f"    Seeds {s1} vs {s2}: Spearman rho = {rho:.3f} (p={p:.4f})")

    mean_rho = np.mean(rhos)
    pass_stability = mean_rho > 0.6
    print(f"    Mean Spearman rho: {mean_rho:.3f}")
    print(f"    Stability check (rho > 0.6): {'PASS' if pass_stability else 'FAIL'}")

    return {
        'seed_pairs': [(seed_list[i], seed_list[j])
                       for i in range(len(seed_list)) for j in range(i+1, len(seed_list))],
        'rhos': [float(r) for r in rhos],
        'mean_rho': float(mean_rho),
        'pass_stability': pass_stability,
    }


def generate_plots(all_results, seeds, output_dir):
    """Generate visualization plots."""
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

    seed_list = sorted(all_results.keys())
    first_seed = seed_list[0]
    res = all_results[first_seed]
    n_test = res['n_test']
    level_names = res['level_names']

    # --- Plot 1: Jaccard Degradation Curves ---
    fig, axes = plt.subplots(1, len(seed_list), figsize=(5*len(seed_list), 4), squeeze=False)
    for idx, seed in enumerate(seed_list):
        ax = axes[0, idx]
        jaccard = np.array(all_results[seed]['jaccard_matrix'][:n_test])
        conf = np.array(all_results[seed]['confidence_labels'])

        # Mean curve
        mean_j = jaccard.mean(axis=0)
        std_j = jaccard.std(axis=0)
        x = range(len(level_names))
        ax.errorbar(x, mean_j, yerr=std_j, marker='o', label='All', color='black', linewidth=2)

        # By confidence
        high_mask = conf == 'high'
        low_mask = conf == 'low'
        ax.plot(x, jaccard[high_mask].mean(axis=0), marker='s', label='High conf', alpha=0.7)
        ax.plot(x, jaccard[low_mask].mean(axis=0), marker='^', label='Low conf', alpha=0.7)

        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='TRV threshold')
        ax.axhline(y=0.85, color='orange', linestyle='--', alpha=0.5, label='Falsification line')
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace('_', '\n') for n in level_names], fontsize=8)
        ax.set_ylabel('Jaccard@10')
        ax.set_title(f'Seed {seed}')
        ax.legend(fontsize=7)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Jaccard@10 Degradation Across Hessian Levels', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'jaccard_degradation.png'), dpi=150)
    plt.close()

    # --- Plot 2: TRV Distribution ---
    fig, axes = plt.subplots(1, len(seed_list), figsize=(5*len(seed_list), 4), squeeze=False)
    for idx, seed in enumerate(seed_list):
        ax = axes[0, idx]
        trv = np.array(all_results[seed]['trv'][:n_test])
        n_levels = len(level_names)
        counts = np.bincount(trv, minlength=n_levels + 1)
        bars = ax.bar(range(n_levels + 1), counts / n_test * 100)

        # Color bars above 10% threshold
        for bar, frac in zip(bars, counts / n_test):
            if frac > 0.10:
                bar.set_color('green')
                bar.set_alpha(0.7)
            else:
                bar.set_color('gray')
                bar.set_alpha(0.5)

        ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
        ax.set_xlabel('TRV Level')
        ax.set_ylabel('Percentage of test points')
        ax.set_title(f'Seed {seed}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('TRV Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'trv_distribution.png'), dpi=150)
    plt.close()

    # --- Plot 3: SI vs TRV scatter ---
    fig, axes = plt.subplots(1, len(seed_list), figsize=(5*len(seed_list), 4), squeeze=False)
    for idx, seed in enumerate(seed_list):
        ax = axes[0, idx]
        si = np.array(all_results[seed]['si_eval'][:n_test])
        trv = np.array(all_results[seed]['trv'][:n_test])
        conf = np.array(all_results[seed]['confidence_labels'])

        high_mask = conf == 'high'
        low_mask = conf == 'low'

        ax.scatter(si[high_mask], trv[high_mask] + np.random.randn(high_mask.sum())*0.1,
                   alpha=0.5, label='High conf', s=20)
        ax.scatter(si[low_mask], trv[low_mask] + np.random.randn(low_mask.sum())*0.1,
                   alpha=0.5, label='Low conf', s=20)
        ax.set_xlabel('Self-Influence (SI)')
        ax.set_ylabel('TRV (jittered)')
        ax.set_title(f'Seed {seed}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('SI vs TRV', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'si_vs_trv.png'), dpi=150)
    plt.close()

    # --- Plot 4: Per-point Jaccard heatmap (first seed) ---
    fig, ax = plt.subplots(figsize=(8, 10))
    jaccard = np.array(all_results[first_seed]['jaccard_matrix'][:n_test])
    # Sort by TRV for better visualization
    trv = np.array(all_results[first_seed]['trv'][:n_test])
    sort_idx = np.argsort(trv)
    im = ax.imshow(jaccard[sort_idx], aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(len(level_names)))
    ax.set_xticklabels([n.replace('_', '\n') for n in level_names], fontsize=8)
    ax.set_ylabel('Test points (sorted by TRV)')
    ax.set_title(f'Jaccard@10 Heatmap (seed {first_seed})')
    plt.colorbar(im, ax=ax, label='Jaccard@10')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'jaccard_heatmap.png'), dpi=150)
    plt.close()

    # --- Plot 5: OOD comparison ---
    fig, ax = plt.subplots(figsize=(6, 4))
    for seed in seed_list:
        n_t = all_results[seed]['n_test']
        n_o = all_results[seed]['n_ood']
        id_trv = np.array(all_results[seed]['trv'][:n_t])
        ood_trv = np.array(all_results[seed]['trv'][n_t:n_t+n_o])
        positions = [seed_list.index(seed)*3, seed_list.index(seed)*3 + 1]
        bp = ax.boxplot([id_trv, ood_trv], positions=positions, widths=0.8)

    ax.set_xticks([i*3 + 0.5 for i in range(len(seed_list))])
    ax.set_xticklabels([f'Seed {s}\n(ID / OOD)' for s in seed_list])
    ax.set_ylabel('TRV')
    ax.set_title('ID vs OOD TRV Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'ood_comparison.png'), dpi=150)
    plt.close()

    # --- Plot 6: Cross-seed TRV scatter (if multiple seeds) ---
    if len(seed_list) >= 2:
        n_pairs = len(seed_list) * (len(seed_list) - 1) // 2
        fig, axes = plt.subplots(1, n_pairs, figsize=(5*n_pairs, 4), squeeze=False)
        pair_idx = 0
        for i in range(len(seed_list)):
            for j in range(i+1, len(seed_list)):
                ax = axes[0, pair_idx]
                trv1 = np.array(all_results[seed_list[i]]['trv'][:n_test])
                trv2 = np.array(all_results[seed_list[j]]['trv'][:n_test])
                ax.scatter(trv1 + np.random.randn(n_test)*0.1,
                           trv2 + np.random.randn(n_test)*0.1, alpha=0.4, s=15)
                rho, _ = stats.spearmanr(trv1, trv2)
                ax.set_xlabel(f'TRV (seed {seed_list[i]})')
                ax.set_ylabel(f'TRV (seed {seed_list[j]})')
                ax.set_title(f'rho={rho:.3f}')
                ax.grid(True, alpha=0.3)
                pair_idx += 1

        fig.suptitle('Cross-seed TRV Stability', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'plots', 'cross_seed_trv.png'), dpi=150)
        plt.close()

    print(f"\n  Plots saved to {os.path.join(output_dir, 'plots')}/")


def generate_summary(all_results, seeds, output_dir):
    """Generate comprehensive summary report."""
    summary = {
        'experiment': 'AURA Phase 1 Probe - TRV Pilot',
        'dataset': 'CIFAR-10',
        'model': 'ResNet-18 (CIFAR-adapted)',
        'seeds': seeds,
        'n_test': 100,
        'n_ood': 20,
        'k': 10,
        'hessian_levels': ['full_ggn', 'kfac', 'diagonal', 'damped_identity', 'identity'],
        'per_seed_analysis': {},
        'cross_seed_analysis': None,
        'verdict': {},
    }

    for seed in sorted(all_results.keys()):
        res = all_results[seed]
        print(f"\n{'='*60}")
        print(f"Analysis for seed {seed}")
        print(f"{'='*60}")
        print(f"  Condition number κ = {res['kappa']:.2e}")

        seed_analysis = {
            'kappa': res['kappa'],
        }
        seed_analysis['trv_distribution'] = analyze_trv_distribution(res, seed, output_dir)
        seed_analysis['jaccard_degradation'] = analyze_jaccard_degradation(res, seed, output_dir)
        seed_analysis['si_trv_correlation'] = analyze_si_trv_correlation(res, seed)
        seed_analysis['confidence_stratification'] = analyze_confidence_stratification(res, seed)
        seed_analysis['ood_confound'] = analyze_ood_confound(res, seed)

        summary['per_seed_analysis'][seed] = seed_analysis

    # Cross-seed analysis
    summary['cross_seed_analysis'] = analyze_cross_seed_stability(all_results, seeds, output_dir)

    # Generate plots
    generate_plots(all_results, seeds, output_dir)

    # Overall verdict
    print(f"\n{'='*60}")
    print("OVERALL VERDICT")
    print(f"{'='*60}")

    # Aggregate checks across seeds
    all_pass_trv = all(
        summary['per_seed_analysis'][s]['trv_distribution']['pass_trv_distribution']
        for s in summary['per_seed_analysis']
    )
    all_pass_degradation = all(
        summary['per_seed_analysis'][s]['jaccard_degradation']['pass_degradation']
        for s in summary['per_seed_analysis']
    )
    all_pass_variance = all(
        summary['per_seed_analysis'][s]['jaccard_degradation']['pass_variance']
        for s in summary['per_seed_analysis']
    )
    all_pass_ood = all(
        summary['per_seed_analysis'][s]['ood_confound']['pass_ood_confound']
        for s in summary['per_seed_analysis']
    )
    pass_stability = summary['cross_seed_analysis'].get('pass_stability', False)

    verdict = {
        'H1_trv_variability': all_pass_trv,
        'early_signal_degradation': all_pass_degradation,
        'per_point_variance': all_pass_variance,
        'ood_confound_clear': all_pass_ood,
        'cross_seed_stable': pass_stability,
        'phase1_viable': all_pass_trv and all_pass_degradation,
    }

    summary['verdict'] = verdict

    for check, passed in verdict.items():
        status = 'PASS' if passed else 'FAIL'
        print(f"  {check}: {status}")

    if verdict['phase1_viable']:
        print("\n  ==> Phase 1 is VIABLE. Proceed to full-scale TRV computation.")
    else:
        print("\n  ==> Phase 1 viability UNCERTAIN. Review failed checks.")

    # Save summary (convert numpy/bool types for JSON)
    def convert_types(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_types(i) for i in obj]
        return obj

    summary = convert_types(summary)
    summary_path = os.path.join(output_dir, 'probe_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--output_dir', type=str, default='./outputs/attributions')
    args = parser.parse_args()

    all_results = load_results(args.output_dir, args.seeds)
    if not all_results:
        print("No results found. Run compute_attributions.py first.")
        return

    print(f"Loaded results for seeds: {sorted(all_results.keys())}")
    summary = generate_summary(all_results, args.seeds, args.output_dir)


if __name__ == '__main__':
    main()
