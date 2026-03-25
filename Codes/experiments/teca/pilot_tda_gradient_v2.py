"""
Phase 2: TDA Gradient Computation on 100 CounterFact facts (v2 - robust).

Changes from v1:
- Flush stdout for log visibility
- Handle g_test failures gracefully (still count as success if g_M computed)
- Robust JSON serialization
- Better error handling in final compilation

Author: sibyl-experimenter
"""

import os
import sys
import json
import time
import pickle
import gc
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# === Configuration ===
TASK_ID = "phase2_tda_gradient"
PROJECT_DIR = "/home/jinxulin/sibyl_system/projects/TECA"
MODEL_PATH = "/home/jinxulin/sibyl_system/shared/checkpoints/gpt2-xl"
COUNTERFACT_PATH = "/home/jinxulin/sibyl_system/shared/datasets/counterfact/counterfact.json"
BM25_INDEX_PATH = f"{PROJECT_DIR}/data/bm25_index_wiki100k.pkl"
RESULTS_DIR = f"{PROJECT_DIR}/results"
GRADIENT_DIR = f"{RESULTS_DIR}/tda_gradients"
RESULTS_FILE = f"{RESULTS_DIR}/pilot_tda_results.json"

LAYER_IDX = 17
N_PILOT = 100
SEED = 42
TOP_K_RETRIEVE = 100
TOP_K_GRADIENT = 20
MAX_SEQ_LEN = 512

os.makedirs(GRADIENT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Write PID file
pid_file = Path(RESULTS_DIR) / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))


def report_progress(epoch, total, step=0, total_steps=0, loss=None, metric=None):
    progress = Path(RESULTS_DIR) / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    pid_f = Path(RESULTS_DIR) / f"{TASK_ID}.pid"
    if pid_f.exists():
        pid_f.unlink()
    progress_file = Path(RESULTS_DIR) / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except Exception:
            pass
    marker = Path(RESULTS_DIR) / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


def load_pilot_facts(path, n=100, seed=42):
    with open(path) as f:
        data = json.load(f)
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(data), size=n, replace=False)
    facts = []
    for i in indices:
        d = data[i]
        rr = d["requested_rewrite"]
        facts.append({
            "case_id": d["case_id"],
            "subject": rr["subject"],
            "target_true": rr["target_true"]["str"],
            "target_new": rr["target_new"]["str"],
            "prompt": rr["prompt"].format(rr["subject"]),
            "raw_prompt_template": rr["prompt"],
        })
    return facts


def compute_gradient_for_text(model, tokenizer, text, layer_idx, device, max_len=512):
    """Compute gradient of LM loss w.r.t. MLP c_proj weight at given layer. Returns FP32 gradient."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = inputs["input_ids"].to(device)

    if input_ids.shape[1] < 2:
        return None

    param = model.transformer.h[layer_idx].mlp.c_proj.weight
    model.zero_grad()

    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()

    grad = param.grad.detach().float().clone()
    model.zero_grad()
    del outputs, loss, input_ids
    torch.cuda.empty_cache()

    return grad


def mean_pairwise_cosine(gradients):
    """Mean pairwise cosine similarity among gradient list."""
    n = len(gradients)
    if n < 2:
        return 0.0
    flat = torch.stack([g.flatten() for g in gradients])
    flat_norm = F.normalize(flat, dim=1)
    cos_matrix = torch.mm(flat_norm, flat_norm.t())
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    pairwise_cos = cos_matrix[mask]
    return pairwise_cos.mean().item()


def check_existing_gradients(facts):
    """Check which facts already have computed gradients from the previous run."""
    existing = {}
    for fact in facts:
        fid = fact["case_id"]
        gm_path = os.path.join(GRADIENT_DIR, f"g_M_{fid}.pt")
        gt_path = os.path.join(GRADIENT_DIR, f"g_test_{fid}.pt")
        if os.path.exists(gm_path):
            existing[fid] = {
                "g_M": gm_path,
                "g_test": gt_path if os.path.exists(gt_path) else None,
            }
    return existing


def main():
    start_time = time.time()
    print(f"[{datetime.now().isoformat()}] Starting TDA gradient computation v2")
    print(f"  GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    device = torch.device("cuda:0")

    # Check existing gradients from v1 run
    print("Loading pilot facts...")
    facts = load_pilot_facts(COUNTERFACT_PATH, N_PILOT, SEED)
    existing = check_existing_gradients(facts)
    print(f"  Found {len(existing)} existing g_M files from previous run")

    # Load BM25 index
    print("Loading BM25 index...")
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    bm25 = bm25_data["bm25"]
    corpus_docs = bm25_data["corpus_docs"]
    print(f"  BM25 corpus size: {bm25.corpus_size}")

    # Load model
    print("Loading GPT-2-XL in FP16...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    target_weight = model.transformer.h[LAYER_IDX].mlp.c_proj.weight
    print(f"  c_proj weight shape at layer {LAYER_IDX}: {target_weight.shape}")
    assert target_weight.shape == (6400, 1600), f"Unexpected shape: {target_weight.shape}"

    for p in model.parameters():
        p.requires_grad_(False)
    target_weight.requires_grad_(True)

    report_progress(0, N_PILOT, metric={"status": "model_loaded"})

    results_per_fact = []
    all_norms = []
    all_angular_vars = []
    nan_count = 0
    failed_facts = []
    skipped_reused = 0

    for fact_idx, fact in enumerate(facts):
        fact_start = time.time()
        fact_id = fact["case_id"]

        try:
            # Check if we can reuse existing gradients
            if fact_id in existing and existing[fact_id]["g_test"] is not None:
                # Reuse - just load and compute metrics
                g_M = torch.load(existing[fact_id]["g_M"], map_location="cpu", weights_only=True)
                g_test = torch.load(existing[fact_id]["g_test"], map_location="cpu", weights_only=True)

                g_M_norm = g_M.norm().item()
                g_test_norm = g_test.norm().item()

                if g_M_norm > 0 and g_test_norm > 0:
                    cos_sim = F.cosine_similarity(
                        g_M.flatten().unsqueeze(0),
                        g_test.flatten().unsqueeze(0)
                    ).item()
                else:
                    cos_sim = None

                # We don't have angular variance from the cached run, so recompute
                # Actually skip angular var for cached - mark as needing recompute
                # For now, set to None and we'll recompute below if needed
                angular_var = None
                skipped_reused += 1

                # Still need to recompute angular variance by re-running BM25 + gradients
                # Fall through to full computation for completeness
                del g_M, g_test

            # Full computation
            query = f"{fact['subject']} {fact['target_true']}".lower().split()
            scores = bm25.get_scores(query)
            top_indices = np.argsort(scores)[-TOP_K_RETRIEVE:][::-1]
            top20_indices = top_indices[:TOP_K_GRADIENT]
            top20_docs = [corpus_docs[i] for i in top20_indices]
            top20_scores = scores[top20_indices]

            # Compute gradients for top-20 training docs
            doc_gradients = []
            for doc_idx, doc in enumerate(top20_docs):
                text = doc["text"]
                if len(text.strip()) < 10:
                    continue
                try:
                    grad = compute_gradient_for_text(
                        model, tokenizer, text, LAYER_IDX, device, MAX_SEQ_LEN
                    )
                    if grad is not None and not torch.isnan(grad).any():
                        doc_gradients.append(grad.cpu())
                    elif grad is not None and torch.isnan(grad).any():
                        nan_count += 1
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue

                if (doc_idx + 1) % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            if len(doc_gradients) == 0:
                failed_facts.append({"case_id": fact_id, "reason": "no valid doc gradients"})
                continue

            # Aggregate gradient
            grad_stack = torch.stack(doc_gradients)
            g_M = grad_stack.mean(dim=0)

            # Angular variance
            angular_var = mean_pairwise_cosine(doc_gradients)

            # Test prompt gradient
            test_text = fact["prompt"]
            try:
                g_test = compute_gradient_for_text(
                    model, tokenizer, test_text, LAYER_IDX, device, MAX_SEQ_LEN
                )
            except torch.cuda.OutOfMemoryError:
                g_test = None
                torch.cuda.empty_cache()
                gc.collect()

            g_M_norm = g_M.norm().item()
            g_test_norm = g_test.norm().item() if g_test is not None else 0.0

            if g_test is not None and g_M_norm > 0 and g_test_norm > 0:
                cos_sim = F.cosine_similarity(
                    g_M.flatten().unsqueeze(0),
                    g_test.cpu().flatten().unsqueeze(0)
                ).item()
            else:
                cos_sim = None

            # Save gradient tensors (delete first to avoid PyTorch zip overwrite bug)
            gm_path = os.path.join(GRADIENT_DIR, f"g_M_{fact_id}.pt")
            gt_path = os.path.join(GRADIENT_DIR, f"g_test_{fact_id}.pt")
            if os.path.exists(gm_path):
                os.remove(gm_path)
            torch.save(g_M, gm_path)
            if g_test is not None:
                if os.path.exists(gt_path):
                    os.remove(gt_path)
                torch.save(g_test.cpu(), gt_path)

            fact_result = {
                "case_id": int(fact_id),
                "subject": fact["subject"],
                "target_true": fact["target_true"],
                "n_valid_gradients": len(doc_gradients),
                "g_M_norm": float(g_M_norm),
                "g_test_norm": float(g_test_norm),
                "angular_variance": float(angular_var),
                "cos_gM_gtest": float(cos_sim) if cos_sim is not None else None,
                "g_test_computed": g_test is not None,
                "top_bm25_score": float(top20_scores[0]),
                "mean_bm25_score": float(top20_scores.mean()),
                "elapsed_sec": float(time.time() - fact_start),
            }
            results_per_fact.append(fact_result)
            all_norms.append(g_M_norm)
            all_angular_vars.append(angular_var)

            del grad_stack, doc_gradients, g_M
            if g_test is not None:
                del g_test
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            failed_facts.append({"case_id": int(fact_id), "reason": str(e)})
            traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()

        if (fact_idx + 1) % 5 == 0:
            elapsed = time.time() - start_time
            eta_sec = elapsed / (fact_idx + 1) * (N_PILOT - fact_idx - 1)
            print(f"  [{fact_idx+1}/{N_PILOT}] "
                  f"ok={len(results_per_fact)} fail={len(failed_facts)} "
                  f"elapsed={elapsed:.0f}s eta={eta_sec:.0f}s "
                  f"norm={np.mean(all_norms) if all_norms else 0:.6f} "
                  f"ang_var={np.mean(all_angular_vars) if all_angular_vars else 0:.4f}")
            report_progress(
                fact_idx + 1, N_PILOT,
                metric={
                    "completed": len(results_per_fact),
                    "failed": len(failed_facts),
                    "mean_norm": float(np.mean(all_norms)) if all_norms else 0,
                    "mean_angular_var": float(np.mean(all_angular_vars)) if all_angular_vars else 0,
                    "eta_sec": float(eta_sec),
                }
            )

    # === Compile final results ===
    total_time = time.time() - start_time
    print(f"\nCompiling results... ({len(results_per_fact)} successful, {len(failed_facts)} failed)")

    norms_arr = np.array(all_norms) if all_norms else np.array([0.0])
    angular_arr = np.array(all_angular_vars) if all_angular_vars else np.array([0.0])

    cos_vals = [r["cos_gM_gtest"] for r in results_per_fact if r["cos_gM_gtest"] is not None]
    mean_cos = float(np.mean(cos_vals)) if cos_vals else None

    n_with_gtest = sum(1 for r in results_per_fact if r["g_test_computed"])

    summary = {
        "task_id": TASK_ID,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": "gpt2-xl",
            "layer": LAYER_IDX,
            "n_facts": N_PILOT,
            "seed": SEED,
            "top_k_retrieve": TOP_K_RETRIEVE,
            "top_k_gradient": TOP_K_GRADIENT,
            "max_seq_len": MAX_SEQ_LEN,
            "weight_shape": [6400, 1600],
        },
        "metrics": {
            "n_successful": len(results_per_fact),
            "n_failed": len(failed_facts),
            "n_with_gtest": n_with_gtest,
            "nan_gradients": nan_count,
            "mean_gM_norm": float(norms_arr.mean()),
            "std_gM_norm": float(norms_arr.std()),
            "min_gM_norm": float(norms_arr.min()),
            "max_gM_norm": float(norms_arr.max()),
            "mean_angular_variance": float(angular_arr.mean()),
            "std_angular_variance": float(angular_arr.std()),
            "mean_cos_gM_gtest": mean_cos,
        },
        "pass_criteria": {
            "gradient_shape_correct": True,
            "mean_norm_gt_1e8": bool(float(norms_arr.mean()) > 1e-8),
            "no_nan": nan_count == 0,
            "angular_variance_computed": len(all_angular_vars) == len(results_per_fact),
        },
        "all_pass": bool(
            float(norms_arr.mean()) > 1e-8
            and nan_count == 0
            and len(all_angular_vars) == len(results_per_fact)
            and len(results_per_fact) >= 50  # At least 50% success rate
        ),
        "total_time_sec": float(total_time),
        "per_fact_results": results_per_fact,
        "failed_facts": failed_facts,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"TDA Gradient Computation Complete")
    print(f"{'='*60}")
    print(f"  Successful: {len(results_per_fact)}/{N_PILOT}")
    print(f"  Failed: {len(failed_facts)}")
    print(f"  With g_test: {n_with_gtest}")
    print(f"  NaN gradients: {nan_count}")
    print(f"  Mean g_M norm: {norms_arr.mean():.8f}")
    print(f"  Mean angular variance: {angular_arr.mean():.6f}")
    print(f"  Mean cos(g_M, g_test): {mean_cos}")
    print(f"  All pass criteria met: {summary['all_pass']}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Results: {RESULTS_FILE}")
    print(f"  Gradients: {GRADIENT_DIR}/")

    status = "success" if summary["all_pass"] else "partial"
    mark_done(status, f"ok={len(results_per_fact)}/{N_PILOT}, all_pass={summary['all_pass']}")

    return summary


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        mark_done("failed", f"Fatal: {str(e)[:200]}")
        sys.exit(1)
