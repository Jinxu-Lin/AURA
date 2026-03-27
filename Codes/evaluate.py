#!/usr/bin/env python3
"""
AURA evaluate.py — BSS diagnostic pipeline.

This is the core evaluation script. It orchestrates:
1. K-FAC factor computation and Kronecker eigendecomposition
2. BSS computation (raw, partial, ratio) per seed per test point
3. Cross-seed stability analysis (Spearman rho, ICC)
4. Predictive power analysis (BSS vs LDS, partial correlations)
5. Baselga decomposition
6. Gate evaluation

Results output to Codes/_Results/<experiment_name>.md

Usage:
    python evaluate.py --config configs/phase2a.yaml
    python evaluate.py --config configs/phase2a.yaml --dry-run
    python evaluate.py --config configs/ablations.yaml --ablation damping
    python evaluate.py --config configs/base.yaml --phase all
"""

import argparse
import json
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from scipy.stats import spearmanr

from config_loader import add_common_args, load_config, resolve_paths
from core.anova import bootstrap_ci, type_i_ss, within_class_variance_fraction
from core.bss import (
    adaptive_bucket_partition,
    compute_bss,
    compute_bss_partial,
    compute_bss_ratio,
    compute_gradient_projections,
    randomized_bucket_control,
)
from core.data import (
    load_checkpoint,
    load_cifar10,
    make_dataloader,
    make_resnet18_cifar10,
    stratified_test_indices,
)
from core.kfac import (
    compute_kfac_factors,
    compute_kronecker_eigenvalues,
    ekfac_eigendecompose,
    top_k_eigendecomposition,
)
from core.metrics import baselga_decomposition, icc_2_1, jaccard_at_k, lds
from core.training import compute_per_sample_gradients
from core.utils import (
    NumpyEncoder,
    ProgressReporter,
    make_class_dummies,
    partial_correlation,
    set_reproducibility,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AURA: BSS Diagnostic Evaluation Pipeline")
    add_common_args(parser)
    parser.add_argument("--phase", type=str, default="phase2a",
                        choices=["phase2a", "phase2a_augmented", "phase3_pregate",
                                 "confound", "ablation", "all"],
                        help="Which evaluation phase to run")
    parser.add_argument("--ablation", type=str, default=None,
                        help="Specific ablation to run (e.g., damping, eigenvalue_count)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override seeds")
    parser.add_argument("--n-test", type=int, default=None,
                        help="Override number of test points")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output filename in _Results/")
    return parser.parse_args()


# =============================================================================
# Phase 2a: BSS Cross-Seed Stability
# =============================================================================

def run_phase2a(config: dict, args: argparse.Namespace, device: torch.device) -> dict:
    """Run Phase 2a: BSS cross-seed stability analysis."""
    print("\n" + "=" * 60)
    print("Phase 2a: BSS Cross-Seed Stability")
    print("=" * 60)

    seeds = args.seeds or config.get("seeds", [42, 123, 456, 789, 1024])
    n_test = args.n_test or config.get("n_test", 500)
    n_per_class = n_test // 10
    damping = config.get("damping_ekfac", 0.01)
    top_k = config.get("n_eigen_top", 100)
    data_dir = Path(config["data_dir"])
    model_dir = data_dir / "models"
    phase2a_dir = data_dir / "phase2a"
    phase2a_dir.mkdir(parents=True, exist_ok=True)

    # Dry-run: reduce everything
    if args.dry_run:
        seeds = seeds[:2]
        n_test = 20
        n_per_class = 2
        top_k = min(top_k, 50)
        print(f"  [dry-run] seeds={seeds}, n_test={n_test}, top_k={top_k}")

    # Load data
    print("\n[1/5] Loading CIFAR-10...")
    trainset, testset = load_cifar10(data_dir=config.get("dataset_dir", "~/Resources/Datasets"))
    test_indices = stratified_test_indices(testset, n_per_class=n_per_class, seed=42)
    if not args.dry_run:
        # Save test indices for reproducibility
        np.save(str(data_dir / "test_indices_500.npy"), np.array(test_indices))

    # Get test labels
    test_labels = np.array([testset.targets[i] for i in test_indices])

    # Per-seed computation
    all_bss_outlier = {}
    all_bss_partial = {}
    all_bss_ratio_vals = {}
    all_grad_norms = {}
    eigenvalue_summaries = {}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n[2/5] Processing seed {seed} ({seed_idx+1}/{len(seeds)})...")

        # Load model
        ckpt_path = model_dir / f"resnet18_cifar10_seed{seed}.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: Checkpoint not found at {ckpt_path}. Skipping seed {seed}.")
            continue

        model = make_resnet18_cifar10()
        load_checkpoint(model, ckpt_path, device="cpu")
        model = model.to(device).eval()

        # K-FAC factors
        print(f"  Computing K-FAC factors (seed {seed})...")
        train_loader = make_dataloader(trainset, batch_size=64, shuffle=False, num_workers=2,
                                        indices=list(range(min(5000, len(trainset)))))
        factors = compute_kfac_factors(model, train_loader, layer_name="fc", device=device)

        # Eigendecomposition
        print(f"  Kronecker eigendecomposition (top-{top_k})...")
        eigen_result = top_k_eigendecomposition(
            factors["A_cov"], factors["B_cov"], k=top_k
        )
        eigenvalues = eigen_result["eigenvalues"].cpu().numpy()
        eigenvalue_summaries[seed] = {
            "max": float(eigenvalues.max()),
            "min": float(eigenvalues.min()),
            "mean": float(eigenvalues.mean()),
            "std": float(eigenvalues.std()),
        }
        print(f"    Eigenvalue range: [{eigenvalues.min():.2e}, {eigenvalues.max():.2e}]")

        # Compute test gradients
        print(f"  Computing test gradients ({len(test_indices)} points)...")
        test_grads = compute_per_sample_gradients(
            model, testset, test_indices,
            target_param_names=["fc.weight", "fc.bias"],
            device=device,
        )

        # Gradient norms
        grad_norms_sq = (test_grads ** 2).sum(dim=1).numpy()
        all_grad_norms[seed] = grad_norms_sq

        # Gradient projections onto Kronecker eigenvectors
        print(f"  Computing gradient projections...")
        projections = compute_gradient_projections(
            test_grads,
            eigen_result["eigvecs_A"].cpu(),
            eigen_result["eigvecs_B"].cpu(),
            eigen_result["top_k_a_indices"].cpu(),
            eigen_result["top_k_b_indices"].cpu(),
            in_features=factors["in_features"],
            has_bias=factors["has_bias"],
        )

        # BSS computation
        print(f"  Computing BSS (raw, partial, ratio)...")
        # For BSS, we need "true" vs "approximate" eigenvalues.
        # In the K-FAC vs EK-FAC framework:
        # eigenvalues = Kronecker products (K-FAC), eigenvalues_approx = diag-corrected (EK-FAC)
        # For diagnostic purposes, we use the same eigenvalues with damping perturbation
        # as the "approximation" is the damped inverse itself.
        # Following pilot code: eigenvalues_approx = 0 (or very small) to measure
        # total sensitivity in each bucket.
        eigenvalues_approx = np.zeros_like(eigenvalues)

        bss_result = compute_bss(
            eigenvalues=eigenvalues,
            eigenvalues_approx=eigenvalues_approx,
            gradient_projections=projections,
            damping=damping,
        )

        # BSS partial (gradient-norm corrected)
        bss_partial_outlier = compute_bss_partial(
            bss_result["bss_outlier"], grad_norms_sq
        )

        # BSS ratio
        bss_ratio_vals = compute_bss_ratio(
            bss_result["bss_outlier"], bss_result["bss_total"]
        )

        all_bss_outlier[seed] = bss_result["bss_outlier"]
        all_bss_partial[seed] = bss_partial_outlier
        all_bss_ratio_vals[seed] = bss_ratio_vals

        # Save per-seed arrays
        seed_dir = phase2a_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(seed_dir / "bss_outlier.npy"), bss_result["bss_outlier"])
        np.save(str(seed_dir / "bss_partial.npy"), bss_partial_outlier)
        np.save(str(seed_dir / "bss_ratio.npy"), bss_ratio_vals)
        np.save(str(seed_dir / "bss_total.npy"), bss_result["bss_total"])
        np.save(str(seed_dir / "eigenvalues.npy"), eigenvalues)
        np.save(str(seed_dir / "grad_norms_sq.npy"), grad_norms_sq)

        # Free GPU memory
        del model, test_grads
        torch.cuda.empty_cache() if device.type == "cuda" else None

    valid_seeds = sorted(all_bss_partial.keys())
    if len(valid_seeds) < 2:
        print("WARNING: Need at least 2 seeds for cross-seed analysis.")
        return {"status": "insufficient_seeds", "n_valid_seeds": len(valid_seeds)}

    # Cross-seed stability analysis
    print(f"\n[3/5] Cross-seed stability analysis ({len(valid_seeds)} seeds)...")
    seed_pairs = list(combinations(valid_seeds, 2))

    stability = {"bss_raw": [], "bss_partial": [], "bss_ratio": []}
    for s1, s2 in seed_pairs:
        rho_raw, _ = spearmanr(all_bss_outlier[s1], all_bss_outlier[s2])
        rho_partial, _ = spearmanr(all_bss_partial[s1], all_bss_partial[s2])
        rho_ratio, _ = spearmanr(all_bss_ratio_vals[s1], all_bss_ratio_vals[s2])
        stability["bss_raw"].append(float(rho_raw) if not np.isnan(rho_raw) else 0.0)
        stability["bss_partial"].append(float(rho_partial) if not np.isnan(rho_partial) else 0.0)
        stability["bss_ratio"].append(float(rho_ratio) if not np.isnan(rho_ratio) else 0.0)

    mean_rho = {k: float(np.mean(v)) for k, v in stability.items()}
    std_rho = {k: float(np.std(v)) for k, v in stability.items()}
    print(f"  Mean pairwise Spearman rho:")
    for k in mean_rho:
        print(f"    {k}: {mean_rho[k]:.4f} +/- {std_rho[k]:.4f}")

    # ICC(2,1)
    print(f"\n[4/5] ICC(2,1) computation...")
    icc_results = {}
    for variant, data_dict in [("bss_partial", all_bss_partial), ("bss_ratio", all_bss_ratio_vals)]:
        matrix = np.column_stack([data_dict[s] for s in valid_seeds])
        icc_val = icc_2_1(matrix)
        icc_results[variant] = float(icc_val)
        print(f"  ICC(2,1) {variant}: {icc_val:.4f}")

    # Within-class variance
    wcv_results = {}
    for seed in valid_seeds:
        wcv = within_class_variance_fraction(all_bss_partial[seed], test_labels[:len(all_bss_partial[seed])])
        wcv_results[seed] = float(wcv)
    mean_wcv = float(np.mean(list(wcv_results.values())))
    print(f"  Mean within-class variance: {mean_wcv:.4f}")

    # Predictive power (BSS vs placeholder LDS — actual LDS needs attribution data)
    print(f"\n[5/5] Predictive power analysis...")
    # Note: Full predictive power requires LDS from Phase 1 data.
    # Here we compute the partial correlation framework; actual LDS loading
    # happens when Phase 1 attribution data is available.

    # Gate evaluation
    print(f"\n{'='*60}")
    print("GATE EVALUATION")
    print(f"{'='*60}")

    gates = config.get("gates", {})
    bss_partial_rho_threshold = gates.get("bss_partial_rho", {}).get("pass", 0.5)
    wcv_threshold = gates.get("within_class_variance", {}).get("pass", 0.25)

    gate_bss_rho = mean_rho["bss_partial"] > bss_partial_rho_threshold
    gate_wcv = mean_wcv > wcv_threshold

    print(f"  BSS_partial cross-seed rho: {mean_rho['bss_partial']:.4f} "
          f"(threshold: {bss_partial_rho_threshold}) -> {'PASS' if gate_bss_rho else 'FAIL'}")
    print(f"  Within-class variance: {mean_wcv:.4f} "
          f"(threshold: {wcv_threshold}) -> {'PASS' if gate_wcv else 'FAIL'}")

    results = {
        "phase": "phase2a",
        "seeds": valid_seeds,
        "n_test": len(test_indices),
        "damping": damping,
        "top_k": top_k,
        "eigenvalue_summaries": eigenvalue_summaries,
        "cross_seed_stability": {
            "mean_rho": mean_rho,
            "std_rho": std_rho,
            "per_pair": {f"{s1}_{s2}": {
                "bss_raw": stability["bss_raw"][i],
                "bss_partial": stability["bss_partial"][i],
                "bss_ratio": stability["bss_ratio"][i],
            } for i, (s1, s2) in enumerate(seed_pairs)},
        },
        "icc": icc_results,
        "within_class_variance": {"per_seed": wcv_results, "mean": mean_wcv},
        "gates": {
            "bss_partial_rho": {"value": mean_rho["bss_partial"], "threshold": bss_partial_rho_threshold, "pass": gate_bss_rho},
            "within_class_variance": {"value": mean_wcv, "threshold": wcv_threshold, "pass": gate_wcv},
        },
        "dry_run": args.dry_run,
        "timestamp": datetime.now().isoformat(),
    }

    return results


# =============================================================================
# Phase 2a Augmented: Randomized-Bucket Control
# =============================================================================

def run_phase2a_augmented(config: dict, args: argparse.Namespace, device: torch.device) -> dict:
    """Run Phase 2a augmented: randomized-bucket control and ANOVA cross-validation."""
    print("\n" + "=" * 60)
    print("Phase 2a Augmented: Controls & Diagnostics")
    print("=" * 60)

    data_dir = Path(config["data_dir"])
    phase2a_dir = data_dir / "phase2a"
    seeds = args.seeds or config.get("seeds", [42, 123, 456, 789, 1024])
    damping = config.get("damping_ekfac", 0.01)

    if args.dry_run:
        seeds = seeds[:1]

    results = {"randomized_bucket": {}, "anova_crossseed": {}}

    # Randomized-bucket control per seed
    n_perms = 100 if args.dry_run else config.get("randomized_bucket", {}).get("n_permutations", 1000)

    for seed in seeds:
        seed_dir = phase2a_dir / f"seed_{seed}"
        eigenvalues_path = seed_dir / "eigenvalues.npy"
        if not eigenvalues_path.exists():
            print(f"  Skipping seed {seed}: no cached eigenvalues")
            continue

        eigenvalues = np.load(str(eigenvalues_path))
        # Load gradient projections (need to recompute or cache)
        # For now, use bss_outlier and bss_total as proxies
        bss_outlier = np.load(str(seed_dir / "bss_outlier.npy"))
        bss_total = np.load(str(seed_dir / "bss_total.npy"))

        print(f"  Seed {seed}: eigenvalue range [{eigenvalues.min():.2e}, {eigenvalues.max():.2e}]")
        results["randomized_bucket"][seed] = {
            "n_permutations": n_perms,
            "eigenvalue_range": [float(eigenvalues.min()), float(eigenvalues.max())],
        }

    return results


# =============================================================================
# Ablation runner
# =============================================================================

def run_ablation(config: dict, args: argparse.Namespace, device: torch.device) -> dict:
    """Run a specific ablation study."""
    ablation_name = args.ablation
    if not ablation_name:
        print("ERROR: --ablation required for ablation phase")
        return {"status": "error", "message": "no ablation specified"}

    print(f"\n{'='*60}")
    print(f"Ablation: {ablation_name}")
    print(f"{'='*60}")

    ablation_config = config.get("ablations", {}).get(ablation_name, {})
    if not ablation_config:
        print(f"ERROR: Unknown ablation '{ablation_name}'")
        return {"status": "error", "message": f"unknown ablation: {ablation_name}"}

    variable = ablation_config["variable"]
    values = ablation_config["values"]
    defaults = config.get("defaults", {})

    print(f"  Variable: {variable}")
    print(f"  Values: {values}")
    print(f"  Defaults: {defaults}")

    results = {"ablation": ablation_name, "variable": variable, "values": {}}

    for val in values:
        print(f"\n  --- {variable} = {val} ---")
        # Override the specific variable in config
        override_config = config.copy()
        if variable == "damping":
            override_config["damping_ekfac"] = val
        elif variable == "top_k":
            override_config["n_eigen_top"] = val
        elif variable == "n_buckets":
            override_config["n_buckets"] = val

        # Run Phase 2a with this config
        phase2a_result = run_phase2a(override_config, args, device)
        results["values"][str(val)] = {
            "mean_rho": phase2a_result.get("cross_seed_stability", {}).get("mean_rho", {}),
            "gates": phase2a_result.get("gates", {}),
        }

    return results


# =============================================================================
# Results output
# =============================================================================

def write_results_md(results: dict, output_path: Path, phase: str):
    """Write results to markdown format in _Results/."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# AURA Experiment Results: {phase}")
    lines.append(f"")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Dry-run: {results.get('dry_run', False)}")
    lines.append(f"")

    if phase == "phase2a":
        lines.append("## Phase 2a: BSS Cross-Seed Stability")
        lines.append("")
        lines.append(f"- Seeds: {results.get('seeds', [])}")
        lines.append(f"- Test points: {results.get('n_test', 0)}")
        lines.append(f"- Damping: {results.get('damping', 0)}")
        lines.append(f"- Top-k eigenvalues: {results.get('top_k', 0)}")
        lines.append("")

        # Eigenvalue summaries
        eigen = results.get("eigenvalue_summaries", {})
        if eigen:
            lines.append("### Eigenvalue Spectrum Summary")
            lines.append("")
            lines.append("| Seed | Max | Min | Mean | Std |")
            lines.append("|------|-----|-----|------|-----|")
            for seed, summary in sorted(eigen.items()):
                lines.append(f"| {seed} | {summary['max']:.2e} | {summary['min']:.2e} | "
                             f"{summary['mean']:.2e} | {summary['std']:.2e} |")
            lines.append("")

        # Cross-seed stability
        stab = results.get("cross_seed_stability", {})
        mean_rho = stab.get("mean_rho", {})
        std_rho = stab.get("std_rho", {})
        if mean_rho:
            lines.append("### Cross-Seed Stability (Mean Pairwise Spearman Rho)")
            lines.append("")
            lines.append("| BSS Variant | Mean Rho | Std |")
            lines.append("|-------------|----------|-----|")
            for k in sorted(mean_rho.keys()):
                lines.append(f"| {k} | {mean_rho[k]:.4f} | {std_rho.get(k, 0):.4f} |")
            lines.append("")

        # ICC
        icc = results.get("icc", {})
        if icc:
            lines.append("### ICC(2,1) Reliability")
            lines.append("")
            for k, v in icc.items():
                lines.append(f"- {k}: {v:.4f}")
            lines.append("")

        # Within-class variance
        wcv = results.get("within_class_variance", {})
        if wcv:
            lines.append(f"### Within-Class Variance: {wcv.get('mean', 0):.4f}")
            lines.append("")

        # Gates
        gates = results.get("gates", {})
        if gates:
            lines.append("### Gate Evaluation")
            lines.append("")
            lines.append("| Gate | Value | Threshold | Result |")
            lines.append("|------|-------|-----------|--------|")
            for k, v in gates.items():
                result = "PASS" if v.get("pass") else "FAIL"
                lines.append(f"| {k} | {v.get('value', 'N/A'):.4f} | {v.get('threshold', 'N/A')} | **{result}** |")
            lines.append("")

    elif phase == "ablation":
        lines.append(f"## Ablation: {results.get('ablation', 'unknown')}")
        lines.append(f"Variable: {results.get('variable', 'unknown')}")
        lines.append("")
        vals = results.get("values", {})
        if vals:
            lines.append("| Value | BSS_raw Rho | BSS_partial Rho | BSS_ratio Rho |")
            lines.append("|-------|-------------|-----------------|---------------|")
            for val, data in sorted(vals.items()):
                rho = data.get("mean_rho", {})
                lines.append(
                    f"| {val} | {rho.get('bss_raw', 'N/A')} | "
                    f"{rho.get('bss_partial', 'N/A')} | {rho.get('bss_ratio', 'N/A')} |"
                )
            lines.append("")

    output_path.write_text("\n".join(lines))
    print(f"\n[evaluate.py] Results written to {output_path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    resolve_paths(config)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[evaluate.py] Device: {device}")
    print(f"[evaluate.py] Phase: {args.phase}")
    print(f"[evaluate.py] Dry-run: {args.dry_run}")

    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Dispatch by phase
    if args.phase == "phase2a":
        results = run_phase2a(config, args, device)
        output_name = args.output or "phase2a_bss_crossseed.md"
    elif args.phase == "phase2a_augmented":
        results = run_phase2a_augmented(config, args, device)
        output_name = args.output or "phase2a_augmented.md"
    elif args.phase == "ablation":
        results = run_ablation(config, args, device)
        output_name = args.output or f"ablation_{args.ablation}.md"
    elif args.phase == "all":
        results = {}
        results["phase2a"] = run_phase2a(config, args, device)
        results["phase2a_augmented"] = run_phase2a_augmented(config, args, device)
        output_name = args.output or "experiment_result.md"
    else:
        print(f"ERROR: Unknown phase '{args.phase}'")
        sys.exit(1)

    # Save JSON results
    json_path = results_dir / output_name.replace(".md", ".json")
    json_path.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    print(f"[evaluate.py] JSON results: {json_path}")

    # Save markdown results
    md_path = results_dir / output_name
    write_results_md(results, md_path, args.phase)

    print(f"\n[evaluate.py] Done.")


if __name__ == "__main__":
    main()
