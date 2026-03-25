#!/usr/bin/env python3
"""
Phase 3 Pilot: TECS Core Measurement + 5 Null Baselines on 100 facts.

DISK-ZERO: All computation in-memory. Only writes final JSON result (~50KB).
Uses rank-one identity: vec(u v^T) . vec(G) = u^T G v

GPU: 2 (CUDA_VISIBLE_DEVICES=2)
"""

import sys, os, json, time, random, gc, traceback, warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

TASK_ID = "phase3_tecs_core"
PROJECT_DIR = "/home/jinxulin/sibyl_system/projects/TECA"
EASYEDIT_DIR = f"{PROJECT_DIR}/EasyEdit"
MODEL_PATH = "/home/jinxulin/sibyl_system/shared/checkpoints/gpt2-xl"
COUNTERFACT_PATH = "/home/jinxulin/sibyl_system/shared/datasets/counterfact/counterfact.json"
RESULTS_DIR = f"{PROJECT_DIR}/results"
GRADIENTS_DIR = f"{RESULTS_DIR}/tda_gradients"
DELTAS_SLIM_DIR = f"{RESULTS_DIR}/rome_deltas_slim"
ROME_RESULTS_PATH = f"{RESULTS_DIR}/pilot_rome_results.json"

SEED = 42
NULL_B_SUBSET = 20
NULL_B_LAYERS = [12, 22, 27, 32, 37]
N_NULL_REPEATS = 10
BOOTSTRAP_N = 10000

sys.path.insert(0, EASYEDIT_DIR)
Path(DELTAS_SLIM_DIR).mkdir(parents=True, exist_ok=True)

# Minimal progress reporting -- only write small files
def write_progress(stage, detail=""):
    try:
        Path(RESULTS_DIR, f"{TASK_ID}_PROGRESS.json").write_text(
            json.dumps({"task_id": TASK_ID, "stage": stage, "detail": detail,
                        "updated_at": datetime.now().isoformat()}))
    except: pass

start_time = time.time()

try:
    import torch
    import numpy as np
    from scipy import stats as scipy_stats

    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    print(f"[TECS] PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)

    # === Helper functions ===
    def tecs_rank1(u, v, G, needs_t=False):
        """cos(vec(u v^T), vec(G)) = (u^T G v) / (||u||*||v||*||G||)"""
        u_f, v_f, G_f = u.float(), v.float(), G.float()
        if needs_t: G_f = G_f.T
        dot = u_f @ G_f @ v_f
        norm_uv = u_f.norm() * v_f.norm()
        norm_G = G_f.norm()
        if norm_uv < 1e-12 or norm_G < 1e-12: return 0.0
        return (dot / (norm_uv * norm_G)).item()

    def cohens_d(x, y):
        diff = x - y
        s = diff.std(ddof=1)
        return diff.mean() / s if s > 1e-12 else 0.0

    def bootstrap_ci(data, n_boot=BOOTSTRAP_N, ci=0.95):
        rng = np.random.RandomState(SEED)
        bm = np.array([np.mean(rng.choice(data, len(data), True)) for _ in range(n_boot)])
        a = (1 - ci) / 2
        return float(np.percentile(bm, a*100)), float(np.percentile(bm, (1-a)*100))

    def paired_test(real, null_m, name):
        n = min(len(real), len(null_m))
        r, nm = real[:n], null_m[:n]
        d = cohens_d(r, nm)
        t, p = scipy_stats.ttest_rel(r, nm)
        diff = r - nm
        ci = bootstrap_ci(diff)
        return {"name": name, "mean_real": float(r.mean()), "mean_null": float(nm.mean()),
                "mean_diff": float(diff.mean()), "cohens_d": float(d),
                "t_stat": float(t), "p_value": float(p),
                "ci_95_low": ci[0], "ci_95_high": ci[1], "n": int(n)}

    # === Load metadata ===
    # NOTE: ROME and TDA phases used different sampling, producing disjoint case ID sets.
    # We use TDA case_ids as the primary set (g_M tensors exist) and extract ROME deltas for them.
    write_progress("loading_metadata")
    with open(ROME_RESULTS_PATH) as f:
        rome_data = json.load(f)
    rome_facts = rome_data["per_fact_results"]
    rome_facts_by_id = {r["case_id"]: r for r in rome_facts}

    # Discover case_ids from g_M files (TDA phase output)
    import re
    gm_files = [f for f in os.listdir(GRADIENTS_DIR) if f.startswith("g_M_") and f.endswith(".pt")]
    case_ids = sorted([int(re.search(r"g_M_(\d+)\.pt", f).group(1)) for f in gm_files])
    print(f"[TECS] {len(case_ids)} facts with g_M tensors (from TDA phase)", flush=True)

    # These are our valid_gm_ids
    valid_gm_ids = case_ids

    # Load CounterFact for metadata and ROME extraction
    with open(COUNTERFACT_PATH) as f:
        cf_data = json.load(f)
    cf_lookup = {item["case_id"]: item for item in cf_data}

    # === Load existing slim deltas ===
    write_progress("loading_existing_deltas")
    delta_factors = {}  # cid -> (u, v, needs_transpose)
    for cid in case_ids:
        dp = Path(DELTAS_SLIM_DIR) / f"delta_case{cid}.pt"
        if dp.exists():
            try:
                d = torch.load(dp, map_location="cpu", weights_only=False)
                delta_factors[cid] = (d["delta_u"].float(), d["delta_v"].float(),
                                      d.get("needs_transpose", False))
            except: pass

    missing = [cid for cid in valid_gm_ids if cid not in delta_factors]
    print(f"[TECS] {len(delta_factors)} existing deltas, {len(missing)} to extract", flush=True)

    # === Extract missing deltas in-memory (no disk write for delta tensors) ===
    if missing:
        write_progress("extracting_deltas", f"{len(missing)} remaining")

        from easyeditor.models.rome import ROMEHyperParams, execute_rome
        from easyeditor.util import nethook
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hparams = ROMEHyperParams.from_hparams(f"{EASYEDIT_DIR}/hparams/ROME/gpt2-xl.yaml")
        hparams.model_name = MODEL_PATH
        hparams.stats_dir = f"{EASYEDIT_DIR}/data/stats"
        hparams.fp16 = True
        Path(hparams.stats_dir).mkdir(parents=True, exist_ok=True)

        # Suppress ROME's verbose output
        import logging
        logging.getLogger().setLevel(logging.WARNING)

        tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16).cuda()
        model.eval()
        print(f"[TECS] Model loaded, VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

        for idx, cid in enumerate(missing):
            cf = cf_lookup.get(cid)
            if cf is None: continue
            try:
                prompt_template = cf["requested_rewrite"]["prompt"].format(
                    cf["requested_rewrite"]["subject"])
                request = {
                    "prompt": prompt_template,
                    "subject": cf["requested_rewrite"]["subject"],
                    "target_new": cf["requested_rewrite"]["target_new"]["str"],
                    "target_true": cf["requested_rewrite"]["target_true"]["str"],
                }
                deltas = execute_rome(model, tok, request, hparams)
                for w_name, (delta_u, delta_v) in deltas.items():
                    w = nethook.get_parameter(model, w_name)
                    upd_test = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                    needs_t = (upd_test.shape != w.shape and upd_test.T.shape == w.shape)
                    delta_factors[cid] = (delta_u.cpu().float(), delta_v.cpu().float(), needs_t)

                    # Try to save slim delta (tiny ~16KB), but don't fail if quota hit
                    try:
                        dp = Path(DELTAS_SLIM_DIR) / f"delta_case{cid}.pt"
                        torch.save({"case_id": cid, "weight_name": w_name,
                                    "delta_u": delta_u.cpu().half(),
                                    "delta_v": delta_v.cpu().half(),
                                    "needs_transpose": needs_t, "layer": hparams.layers[0]}, dp)
                    except OSError:
                        pass  # Quota exceeded, keep in memory only

                if (idx + 1) % 10 == 0:
                    write_progress("extracting_deltas", f"{idx+1}/{len(missing)}")
                    print(f"[TECS] Delta [{idx+1}/{len(missing)}] case {cid}", flush=True)

            except Exception as e:
                print(f"[TECS] ERROR delta case {cid}: {type(e).__name__}: {str(e)[:100]}", flush=True)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Valid facts = have both delta and g_M
    valid_ids = [cid for cid in case_ids if cid in delta_factors
                 and (Path(GRADIENTS_DIR) / f"g_M_{cid}.pt").exists()]
    print(f"[TECS] {len(valid_ids)} valid facts for TECS computation", flush=True)

    if len(valid_ids) < 10:
        raise RuntimeError(f"Only {len(valid_ids)} valid, need >= 10")

    # ============================================================
    # TECS + Null-A, C, D, E (CPU, streaming g_M one at a time)
    # ============================================================
    write_progress("computing_tecs")
    print(f"[TECS] Computing TECS and null baselines...", flush=True)

    tecs_real = []
    null_a_means = []; null_c_means = []; null_d_means = []
    null_e_vals = []
    null_a_all = []; null_c_all = []; null_d_all = []
    per_fact = []

    for fi, cid in enumerate(valid_ids):
        u, v, needs_t = delta_factors[cid]
        gm = torch.load(Path(GRADIENTS_DIR) / f"g_M_{cid}.pt",
                         map_location="cpu", weights_only=False).float()

        # TECS_real
        tv = tecs_rank1(u, v, gm, needs_t)
        tecs_real.append(tv)

        # Get metadata: try rome_facts first, then TDA results, then CounterFact
        rome_fact = rome_facts_by_id.get(cid, {})
        cf_fact = cf_lookup.get(cid, {}) if 'cf_lookup' in dir() else {}
        rr = cf_fact.get("requested_rewrite", {})
        fi_info = {"case_id": cid, "tecs_real": tv,
                   "subject": rome_fact.get("subject", rr.get("subject", "")),
                   "edit_success": rome_fact.get("edit_success", ""),
                   "post_prob_new": rome_fact.get("post_prob_new", None)}

        # Null-A: random fact swap
        others = [x for x in valid_ids if x != cid]
        swaps = random.sample(others, min(N_NULL_REPEATS, len(others)))
        na = [tecs_rank1(*delta_factors[s][:2], gm, delta_factors[s][2]) for s in swaps]
        null_a_all.extend(na)
        null_a_means.append(np.mean(na))
        fi_info["null_a_mean"] = np.mean(na)

        # Null-C: shuffled gradient
        gm_flat = gm.reshape(-1)
        nc = []
        for _ in range(N_NULL_REPEATS):
            perm = torch.randperm(gm_flat.shape[0])
            gs = gm_flat[perm].reshape(gm.shape)
            nc.append(tecs_rank1(u, v, gs, needs_t))
        null_c_all.extend(nc)
        null_c_means.append(np.mean(nc))
        fi_info["null_c_mean"] = np.mean(nc)

        # Null-D: random direction
        nd = []
        for _ in range(N_NULL_REPEATS):
            rG = torch.randn_like(gm)
            nd.append(tecs_rank1(u, v, rG, needs_t))
        null_d_all.extend(nd)
        null_d_means.append(np.mean(nd))
        fi_info["null_d_mean"] = np.mean(nd)

        # Null-E: test prompt gradient
        gt_path = Path(GRADIENTS_DIR) / f"g_test_{cid}.pt"
        if gt_path.exists():
            gt = torch.load(gt_path, map_location="cpu", weights_only=False).float()
            ne = tecs_rank1(u, v, gt, needs_t)
            null_e_vals.append(ne)
            fi_info["null_e"] = ne
            del gt
        else:
            null_e_vals.append(float("nan"))
            fi_info["null_e"] = float("nan")

        per_fact.append(fi_info)
        del gm

        if (fi + 1) % 20 == 0:
            write_progress("computing_tecs", f"{fi+1}/{len(valid_ids)}")
            print(f"[TECS] [{fi+1}/{len(valid_ids)}] TECS={tv:.6f}", flush=True)

    tecs_arr = np.array(tecs_real)
    na_arr = np.array(null_a_means)
    nc_arr = np.array(null_c_means)
    nd_arr = np.array(null_d_means)
    ne_arr = np.array([x for x in null_e_vals if not np.isnan(x)])

    print(f"[TECS] TECS_real: mean={tecs_arr.mean():.6f} std={tecs_arr.std():.6f}", flush=True)
    print(f"[TECS] Null-A: {na_arr.mean():.6f}  Null-C: {nc_arr.mean():.6f}  Null-D: {nd_arr.mean():.6f}", flush=True)

    # ============================================================
    # Null-B: cross-layer placebo (needs GPU, 20 facts only)
    # ============================================================
    write_progress("null_b_cross_layer")
    print(f"[TECS] Computing Null-B (cross-layer, {NULL_B_SUBSET} facts)...", flush=True)

    nb_results = {l: [] for l in NULL_B_LAYERS}
    nb_computed = False

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16).cuda()
        model.eval()

        subset = valid_ids[:NULL_B_SUBSET]
        for fi, cid in enumerate(subset):
            cf = cf_lookup.get(cid)
            if cf is None:
                for l in NULL_B_LAYERS: nb_results[l].append(float("nan"))
                continue

            u, v, needs_t = delta_factors[cid]
            subj = cf["requested_rewrite"]["subject"]
            tgt = cf["requested_rewrite"]["target_true"]["str"]
            inputs = tok(f"{subj} is associated with {tgt}",
                         return_tensors="pt", truncation=True, max_length=128).to("cuda")

            for layer in NULL_B_LAYERS:
                try:
                    model.zero_grad()
                    for p in model.parameters(): p.requires_grad_(False)
                    w_name = f"transformer.h.{layer}.mlp.c_proj.weight"
                    tp = None
                    for nm, p in model.named_parameters():
                        if nm == w_name:
                            p.requires_grad_(True); tp = p; break
                    if tp is None:
                        nb_results[layer].append(float("nan")); continue

                    out = model(**inputs, labels=inputs["input_ids"])
                    out.loss.backward()
                    grad = tp.grad.detach().cpu().float()
                    nb_results[layer].append(tecs_rank1(u, v, grad, needs_t))
                    model.zero_grad()
                    for p in model.parameters(): p.requires_grad_(False)
                except Exception as e:
                    nb_results[layer].append(float("nan"))

            if (fi + 1) % 5 == 0:
                print(f"[TECS] Null-B [{fi+1}/{NULL_B_SUBSET}]", flush=True)
            gc.collect(); torch.cuda.empty_cache()

        del model; gc.collect(); torch.cuda.empty_cache()
        nb_computed = True

        nb_per_fact = []
        for fi in range(NULL_B_SUBSET):
            lv = [nb_results[l][fi] for l in NULL_B_LAYERS
                  if fi < len(nb_results[l]) and not np.isnan(nb_results[l][fi])]
            nb_per_fact.append(np.mean(lv) if lv else float("nan"))
        nb_all = [v for l in NULL_B_LAYERS for v in nb_results[l] if not np.isnan(v)]
        nb_all_arr = np.array(nb_all)

        print(f"[TECS] Null-B layers:", flush=True)
        for l in NULL_B_LAYERS:
            vs = [v for v in nb_results[l] if not np.isnan(v)]
            if vs: print(f"  L{l}: mean={np.mean(vs):.6f} std={np.std(vs):.6f}", flush=True)

    except Exception as e:
        print(f"[TECS] Null-B failed: {e}", flush=True)
        nb_all_arr = np.array([]); nb_per_fact = []

    # ============================================================
    # Statistical Analysis
    # ============================================================
    write_progress("statistics")
    print(f"\n[TECS] Statistical analysis...", flush=True)

    comparisons = {}
    comp_a = paired_test(tecs_arr, na_arr, "TECS vs Null-A (random fact)")
    comparisons["vs_null_a"] = comp_a
    print(f"  vs Null-A: d={comp_a['cohens_d']:.4f}, p={comp_a['p_value']:.2e}", flush=True)

    comp_c = paired_test(tecs_arr, nc_arr, "TECS vs Null-C (shuffled)")
    comparisons["vs_null_c"] = comp_c
    print(f"  vs Null-C: d={comp_c['cohens_d']:.4f}, p={comp_c['p_value']:.2e}", flush=True)

    comp_d = paired_test(tecs_arr, nd_arr, "TECS vs Null-D (random)")
    comparisons["vs_null_d"] = comp_d
    print(f"  vs Null-D: d={comp_d['cohens_d']:.4f}, p={comp_d['p_value']:.2e}", flush=True)

    # Null-E
    ve_mask = ~np.isnan(np.array(null_e_vals))
    if ve_mask.sum() > 10:
        te = tecs_arr[ve_mask]; ne_t = ne_arr
        de = cohens_d(te, ne_t); t_e, p_e = scipy_stats.ttest_rel(te, ne_t)
        ci_e = bootstrap_ci(te - ne_t)
        comparisons["vs_null_e"] = {
            "name": "TECS vs Null-E (test grad)", "mean_real": float(te.mean()),
            "mean_null": float(ne_t.mean()), "mean_diff": float((te-ne_t).mean()),
            "cohens_d": float(de), "t_stat": float(t_e), "p_value": float(p_e),
            "ci_95_low": ci_e[0], "ci_95_high": ci_e[1], "n": int(len(te))}
        print(f"  vs Null-E: d={de:.4f}, p={p_e:.2e}", flush=True)

    # Null-B
    if nb_computed and nb_per_fact:
        vb = ~np.isnan(np.array(nb_per_fact))
        if vb.sum() > 5:
            tb = tecs_arr[:NULL_B_SUBSET][vb]; nb_c = np.array(nb_per_fact)[vb]
            db = cohens_d(tb, nb_c); t_b, p_b = scipy_stats.ttest_rel(tb, nb_c)
            ci_b = bootstrap_ci(tb - nb_c)
            comparisons["vs_null_b"] = {
                "name": "TECS vs Null-B (cross-layer)", "mean_real": float(tb.mean()),
                "mean_null": float(nb_c.mean()), "mean_diff": float((tb-nb_c).mean()),
                "cohens_d": float(db), "t_stat": float(t_b), "p_value": float(p_b),
                "ci_95_low": ci_b[0], "ci_95_high": ci_b[1], "n": int(len(tb)),
                "layers": NULL_B_LAYERS,
                "per_layer_means": {str(l): float(np.nanmean(nb_results[l]))
                                    for l in NULL_B_LAYERS if nb_results.get(l)}}
            print(f"  vs Null-B: d={db:.4f}, p={p_b:.2e}", flush=True)

    tecs_ci = bootstrap_ci(tecs_arr)
    print(f"  TECS 95% CI: [{tecs_ci[0]:.6f}, {tecs_ci[1]:.6f}]", flush=True)

    # Subgroup
    pp = np.array([pf.get("post_prob_new", 0.5) or 0.5 for pf in per_fact])
    med = np.median(pp); hi = pp >= med; lo = pp < med
    subgroup = None
    if hi.sum() > 5 and lo.sum() > 5:
        th, tl = tecs_arr[hi], tecs_arr[lo]
        ts, ps = scipy_stats.ttest_ind(th, tl)
        subgroup = {"high_mean": float(th.mean()), "low_mean": float(tl.mean()),
                    "t_stat": float(ts), "p_value": float(ps),
                    "n_high": int(hi.sum()), "n_low": int(lo.sum())}

    # ============================================================
    # Decision gate + save
    # ============================================================
    primary_d = comp_a["cohens_d"]
    secondary_d = comparisons.get("vs_null_b", {}).get("cohens_d", None)
    decision = "POSITIVE" if primary_d > 0.2 else "NEGATIVE"

    bonf = 0.01 / max(len(comparisons), 1)
    elapsed = time.time() - start_time

    results = {
        "task_id": TASK_ID, "mode": "pilot", "timestamp": datetime.now().isoformat(),
        "elapsed_sec": elapsed,
        "config": {"n_facts": len(valid_ids), "seed": SEED, "model": "gpt2-xl",
                   "edit_layer": 17, "null_repeats": N_NULL_REPEATS,
                   "null_b_subset": NULL_B_SUBSET, "null_b_layers": NULL_B_LAYERS,
                   "bootstrap_n": BOOTSTRAP_N, "tecs_method": "rank1_identity"},
        "tecs_distribution": {
            "mean": float(tecs_arr.mean()), "std": float(tecs_arr.std()),
            "median": float(np.median(tecs_arr)),
            "min": float(tecs_arr.min()), "max": float(tecs_arr.max()),
            "q25": float(np.percentile(tecs_arr, 25)),
            "q75": float(np.percentile(tecs_arr, 75)),
            "bootstrap_ci_95": list(tecs_ci),
            "n_positive": int((tecs_arr > 0).sum()),
            "n_negative": int((tecs_arr < 0).sum()),
        },
        "null_distributions": {
            "null_a": {"mean": float(np.mean(null_a_all)), "std": float(np.std(null_a_all)),
                       "per_fact_mean": float(na_arr.mean())},
            "null_b": {"computed": nb_computed,
                       "mean": float(nb_all_arr.mean()) if len(nb_all_arr) else None,
                       "std": float(nb_all_arr.std()) if len(nb_all_arr) else None,
                       "per_layer": {str(l): {"mean": float(np.nanmean(nb_results[l])),
                                              "std": float(np.nanstd(nb_results[l]))}
                                     for l in NULL_B_LAYERS if nb_results.get(l)} if nb_computed else {}},
            "null_c": {"mean": float(np.mean(null_c_all)), "std": float(np.std(null_c_all)),
                       "per_fact_mean": float(nc_arr.mean())},
            "null_d": {"mean": float(np.mean(null_d_all)), "std": float(np.std(null_d_all)),
                       "per_fact_mean": float(nd_arr.mean())},
            "null_e": {"mean": float(ne_arr.mean()) if len(ne_arr) else None,
                       "std": float(ne_arr.std()) if len(ne_arr) else None,
                       "n_valid": int(len(ne_arr))},
        },
        "statistical_tests": comparisons,
        "bonferroni": {"alpha": bonf, "n_comp": len(comparisons),
                       "significant": {k: v["p_value"] < bonf for k, v in comparisons.items()}},
        "decision_gate": {"decision": decision,
                          "reason": f"d(vs Null-A)={primary_d:.4f} {'>' if primary_d > 0.2 else '<='} 0.2",
                          "primary_d": primary_d, "secondary_d_null_b": secondary_d, "threshold": 0.2},
        "subgroup_analysis": subgroup,
        "per_fact_results": per_fact,
        "pass_criteria": {
            "all_nulls": all(k in comparisons for k in ["vs_null_a", "vs_null_c", "vs_null_d"]),
            "not_degenerate": float(tecs_arr.std()) > 1e-6,
            "ci_computed": tecs_ci[0] != tecs_ci[1],
        },
    }
    results["all_pass"] = all(results["pass_criteria"].values())

    # Save (this is the only significant disk write)
    out_path = Path(RESULTS_DIR) / "pilot_tecs_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"  TECS CORE MEASUREMENT SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  N: {len(valid_ids)}", flush=True)
    print(f"  TECS_real: {tecs_arr.mean():.6f} +/- {tecs_arr.std():.6f}", flush=True)
    print(f"  95% CI: [{tecs_ci[0]:.6f}, {tecs_ci[1]:.6f}]", flush=True)
    print(f"  Null-A: {na_arr.mean():.6f} | Null-C: {nc_arr.mean():.6f} | Null-D: {nd_arr.mean():.6f}", flush=True)
    if len(ne_arr): print(f"  Null-E: {ne_arr.mean():.6f}", flush=True)
    if nb_computed and len(nb_all_arr): print(f"  Null-B: {nb_all_arr.mean():.6f}", flush=True)
    print(f"  Cohen's d (vs Null-A): {primary_d:.4f}", flush=True)
    print(f"  Decision: *** {decision} ***", flush=True)
    print(f"  Elapsed: {elapsed:.1f}s", flush=True)
    print(f"{'='*60}", flush=True)

    # Write DONE marker
    Path(RESULTS_DIR, f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID, "status": "success",
        "summary": f"TECS on {len(valid_ids)} facts. {decision}. d={primary_d:.4f}",
        "timestamp": datetime.now().isoformat()}))
    # Clean PID
    pid_f = Path(RESULTS_DIR) / f"{TASK_ID}.pid"
    if pid_f.exists(): pid_f.unlink()

except Exception as e:
    em = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    print(f"[TECS] FATAL: {em}", flush=True)
    try:
        Path(RESULTS_DIR, f"{TASK_ID}_DONE").write_text(json.dumps({
            "task_id": TASK_ID, "status": "failed", "summary": em[:500],
            "timestamp": datetime.now().isoformat()}))
        pid_f = Path(RESULTS_DIR) / f"{TASK_ID}.pid"
        if pid_f.exists(): pid_f.unlink()
    except: pass
    sys.exit(1)
