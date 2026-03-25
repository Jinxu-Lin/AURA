"""
Phase 2: TDA Gradient Computation on 100 CounterFact facts.

For each fact:
1. BM25 retrieve top-100 candidate training docs
2. For top-20 candidates, compute gradient of training sample loss w.r.t. MLP c_proj weight at layer 17
3. Aggregate gradient: g_M = mean(g_1, ..., g_20)
4. Compute angular variance (mean pairwise cosine among top-20 gradients)
5. Compute test prompt gradient: g_test = nabla_{W_17} L(theta; x_test)
6. Save all results

Author: sibyl-experimenter
"""

import os
import sys
import json
import time
import pickle
import gc
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        except (json.JSONDecodeError, ValueError):
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
    """Load same 100 facts as pilot_rome (seed=42)."""
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
    """
    Compute gradient of language modeling loss w.r.t. MLP c_proj weight at given layer.
    Returns gradient tensor in FP32.
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    if input_ids.shape[1] < 2:
        return None  # Too short for LM loss

    # Get the target parameter
    param = model.transformer.h[layer_idx].mlp.c_proj.weight

    # Zero all grads
    model.zero_grad()

    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss

    # Backward pass - compute gradient only for the target parameter
    loss.backward()

    # Extract gradient in FP32
    grad = param.grad.detach().float().clone()

    # Clean up
    model.zero_grad()
    del outputs, loss
    torch.cuda.empty_cache()

    return grad


def mean_pairwise_cosine(gradients):
    """Compute mean pairwise cosine similarity (angular variance proxy)."""
    n = len(gradients)
    if n < 2:
        return 0.0

    # Flatten and stack
    flat = torch.stack([g.flatten() for g in gradients])
    # Normalize
    flat_norm = F.normalize(flat, dim=1)
    # Cosine similarity matrix
    cos_matrix = torch.mm(flat_norm, flat_norm.t())

    # Extract upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    pairwise_cos = cos_matrix[mask]

    return pairwise_cos.mean().item()


def main():
    start_time = time.time()
    print(f"[{datetime.now().isoformat()}] Starting TDA gradient computation")
    print(f"  GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    device = torch.device("cuda:0")

    # Load BM25 index
    print("Loading BM25 index...")
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    bm25 = bm25_data["bm25"]
    corpus_docs = bm25_data["corpus_docs"]
    print(f"  BM25 corpus size: {bm25.corpus_size}")

    # Load facts
    print("Loading pilot facts...")
    facts = load_pilot_facts(COUNTERFACT_PATH, N_PILOT, SEED)
    print(f"  Loaded {len(facts)} facts")

    # Load model
    print("Loading GPT-2-XL in FP16...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()  # Set to eval but we need gradients for specific params

    # Verify weight shape
    target_weight = model.transformer.h[LAYER_IDX].mlp.c_proj.weight
    print(f"  c_proj weight shape at layer {LAYER_IDX}: {target_weight.shape}")
    # GPT-2 Conv1D: (in_features, out_features) = (6400, 1600)
    assert target_weight.shape == (6400, 1600), f"Unexpected shape: {target_weight.shape}"

    # Freeze all parameters, only enable grad for target
    for p in model.parameters():
        p.requires_grad_(False)
    target_weight.requires_grad_(True)

    report_progress(0, N_PILOT, metric={"status": "model_loaded"})

    # Process each fact
    results_per_fact = []
    all_norms = []
    all_angular_vars = []
    nan_count = 0
    failed_facts = []

    for fact_idx, fact in enumerate(facts):
        fact_start = time.time()
        fact_id = fact["case_id"]

        try:
            # Step 1: BM25 retrieval
            query = f"{fact['subject']} {fact['target_true']}".lower().split()
            scores = bm25.get_scores(query)
            top_indices = np.argsort(scores)[-TOP_K_RETRIEVE:][::-1]

            # Get top-20 for gradient computation
            top20_indices = top_indices[:TOP_K_GRADIENT]
            top20_docs = [corpus_docs[i] for i in top20_indices]
            top20_scores = scores[top20_indices]

            # Step 2: Compute gradients for top-20 training docs
            doc_gradients = []
            for doc_idx, doc in enumerate(top20_docs):
                text = doc["text"]
                grad = compute_gradient_for_text(
                    model, tokenizer, text, LAYER_IDX, device, MAX_SEQ_LEN
                )
                if grad is not None:
                    if torch.isnan(grad).any():
                        nan_count += 1
                        continue
                    doc_gradients.append(grad.cpu())

                # Aggressive memory cleanup every 5 docs
                if (doc_idx + 1) % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            if len(doc_gradients) == 0:
                failed_facts.append({"case_id": fact_id, "reason": "no valid gradients"})
                continue

            # Step 3: Aggregate gradient: g_M = mean
            grad_stack = torch.stack(doc_gradients)
            g_M = grad_stack.mean(dim=0)

            # Step 4: Angular variance (mean pairwise cosine)
            angular_var = mean_pairwise_cosine(doc_gradients)

            # Step 5: Compute test prompt gradient
            test_text = fact["prompt"]
            g_test = compute_gradient_for_text(
                model, tokenizer, test_text, LAYER_IDX, device, MAX_SEQ_LEN
            )

            # Compute norms and metrics
            g_M_norm = g_M.norm().item()
            g_test_norm = g_test.norm().item() if g_test is not None else 0.0

            # Cosine similarity between g_M and g_test
            if g_test is not None and g_M_norm > 0 and g_test_norm > 0:
                cos_sim = F.cosine_similarity(
                    g_M.flatten().unsqueeze(0),
                    g_test.cpu().flatten().unsqueeze(0)
                ).item()
            else:
                cos_sim = None

            # Save gradient tensors
            torch.save(g_M, os.path.join(GRADIENT_DIR, f"g_M_{fact_id}.pt"))
            if g_test is not None:
                torch.save(g_test.cpu(), os.path.join(GRADIENT_DIR, f"g_test_{fact_id}.pt"))

            fact_result = {
                "case_id": fact_id,
                "subject": fact["subject"],
                "target_true": fact["target_true"],
                "n_valid_gradients": len(doc_gradients),
                "g_M_norm": g_M_norm,
                "g_test_norm": g_test_norm,
                "angular_variance": angular_var,
                "cos_gM_gtest": cos_sim,
                "top_bm25_score": float(top20_scores[0]),
                "mean_bm25_score": float(top20_scores.mean()),
                "elapsed_sec": time.time() - fact_start,
            }
            results_per_fact.append(fact_result)
            all_norms.append(g_M_norm)
            all_angular_vars.append(angular_var)

            # Clean up
            del grad_stack, doc_gradients, g_M
            if g_test is not None:
                del g_test
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            failed_facts.append({"case_id": fact_id, "reason": str(e)})
            torch.cuda.empty_cache()
            gc.collect()

        # Progress report every 5 facts
        if (fact_idx + 1) % 5 == 0:
            elapsed = time.time() - start_time
            eta_sec = elapsed / (fact_idx + 1) * (N_PILOT - fact_idx - 1)
            print(f"  [{fact_idx+1}/{N_PILOT}] "
                  f"elapsed={elapsed:.0f}s eta={eta_sec:.0f}s "
                  f"norms_mean={np.mean(all_norms) if all_norms else 0:.6f} "
                  f"angular_var_mean={np.mean(all_angular_vars) if all_angular_vars else 0:.4f}")
            report_progress(
                fact_idx + 1, N_PILOT,
                metric={
                    "completed": fact_idx + 1,
                    "failed": len(failed_facts),
                    "mean_norm": float(np.mean(all_norms)) if all_norms else 0,
                    "mean_angular_var": float(np.mean(all_angular_vars)) if all_angular_vars else 0,
                    "eta_sec": eta_sec,
                }
            )

    # === Compile final results ===
    total_time = time.time() - start_time
    norms_arr = np.array(all_norms) if all_norms else np.array([0.0])
    angular_arr = np.array(all_angular_vars) if all_angular_vars else np.array([0.0])

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
            "nan_gradients": nan_count,
            "mean_gM_norm": float(norms_arr.mean()),
            "std_gM_norm": float(norms_arr.std()),
            "min_gM_norm": float(norms_arr.min()),
            "max_gM_norm": float(norms_arr.max()),
            "mean_angular_variance": float(angular_arr.mean()),
            "std_angular_variance": float(angular_arr.std()),
            "mean_cos_gM_gtest": float(np.mean([r["cos_gM_gtest"] for r in results_per_fact if r["cos_gM_gtest"] is not None])) if any(r["cos_gM_gtest"] is not None for r in results_per_fact) else None,
        },
        "pass_criteria": {
            "gradient_shape_correct": True,  # Verified by assertion
            "mean_norm_gt_1e8": float(norms_arr.mean()) > 1e-8,
            "no_nan": nan_count == 0,
            "angular_variance_computed": len(all_angular_vars) == len(results_per_fact),
        },
        "all_pass": (
            float(norms_arr.mean()) > 1e-8
            and nan_count == 0
            and len(all_angular_vars) == len(results_per_fact)
        ),
        "total_time_sec": total_time,
        "per_fact_results": results_per_fact,
        "failed_facts": failed_facts,
    }

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"TDA Gradient Computation Complete")
    print(f"{'='*60}")
    print(f"  Successful: {len(results_per_fact)}/{N_PILOT}")
    print(f"  Failed: {len(failed_facts)}")
    print(f"  NaN gradients: {nan_count}")
    print(f"  Mean g_M norm: {norms_arr.mean():.8f}")
    print(f"  Mean angular variance: {angular_arr.mean():.6f}")
    print(f"  All pass criteria met: {summary['all_pass']}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Results: {RESULTS_FILE}")
    print(f"  Gradients: {GRADIENT_DIR}/")

    status = "success" if summary["all_pass"] else "partial"
    mark_done(status, f"Completed {len(results_per_fact)}/{N_PILOT} facts, all_pass={summary['all_pass']}")

    return summary


if __name__ == "__main__":
    main()
