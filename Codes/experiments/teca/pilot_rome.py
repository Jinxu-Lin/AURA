#!/usr/bin/env python3
"""
Pilot ROME Editing Validation on 100 CounterFact facts using EasyEdit.

This script:
1. Loads GPT-2-XL from local checkpoint
2. Loads CounterFact dataset and samples 100 facts (seed=42)
3. For each fact, runs ROME editing via EasyEdit (with covariance matrix)
4. Records: edit_success (rewrite_acc), edit_layer, delta_weight tensor,
   pre/post probabilities
5. Computes editing efficacy rate (target: >75%)
6. Saves results JSON and delta_weight .pt files

Uses EasyEdit's proper ROME with C^{-1} covariance matrix computation.
"""

import sys
import os
import json
import time
import random
import traceback
from pathlib import Path
from datetime import datetime

# ---- PID file for system recovery ----
TASK_ID = "pilot_rome"
RESULTS_DIR = Path("/home/jinxulin/sibyl_system/projects/TECA/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))

# ---- Config ----
MODEL_PATH = "/home/jinxulin/sibyl_system/shared/checkpoints/gpt2-xl"
COUNTERFACT_PATH = "/home/jinxulin/sibyl_system/shared/datasets/counterfact/counterfact.json"
EASYEDIT_PATH = "/home/jinxulin/sibyl_system/projects/TECA/EasyEdit"
DELTAS_DIR = RESULTS_DIR / "rome_deltas"
DELTAS_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR = "/home/jinxulin/sibyl_system/projects/TECA/data/stats"
os.makedirs(STATS_DIR, exist_ok=True)

N_FACTS = 100
SEED = 42
DEVICE = 2  # CUDA_VISIBLE_DEVICES=2
EDIT_LAYER = 17

# ---- Progress reporting ----
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

# ---- Setup EasyEdit ----
sys.path.insert(0, EASYEDIT_PATH)
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE)

import torch
import numpy as np

print(f"[{datetime.now().isoformat()}] Starting pilot ROME experiment")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ---- Load dataset ----
print(f"\n[{datetime.now().isoformat()}] Loading CounterFact dataset...")
with open(COUNTERFACT_PATH, "r") as f:
    counterfact_data = json.load(f)
print(f"Total facts in dataset: {len(counterfact_data)}")

# Sample 100 facts
random.seed(SEED)
np.random.seed(SEED)
sampled_indices = random.sample(range(len(counterfact_data)), N_FACTS)
sampled_facts = [counterfact_data[i] for i in sampled_indices]
print(f"Sampled {N_FACTS} facts (seed={SEED})")

# ---- Load model ----
print(f"\n[{datetime.now().isoformat()}] Loading GPT-2-XL from {MODEL_PATH}...")
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
model = model.cuda()
model.eval()
print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
print(f"VRAM after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# ---- Import EasyEdit ROME ----
from easyeditor.models.rome.rome_main import execute_rome, apply_rome_to_model, upd_matrix_match_shape
from easyeditor.models.rome.rome_hparams import ROMEHyperParams
from easyeditor.util import nethook

# Build hparams manually to control model path and stats_dir
hparams = ROMEHyperParams.from_hparams(f"{EASYEDIT_PATH}/hparams/ROME/gpt2-xl.yaml")
hparams.model_name = MODEL_PATH
hparams.stats_dir = STATS_DIR
hparams.device = 0  # Since CUDA_VISIBLE_DEVICES maps device 2 -> logical 0

print(f"\n[{datetime.now().isoformat()}] ROME hparams loaded:")
print(f"  layers: {hparams.layers}")
print(f"  stats_dir: {hparams.stats_dir}")
print(f"  v_num_grad_steps: {hparams.v_num_grad_steps}")
print(f"  mom2_dataset: {hparams.mom2_dataset}")
print(f"  mom2_n_samples: {hparams.mom2_n_samples}")

# ---- Helper: compute pre/post probabilities ----
def compute_target_probability(model, tokenizer, prompt, target_str, device="cuda:0"):
    """Compute P(target | prompt) using the model."""
    full_text = prompt + " " + target_str
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    full_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)

    target_len = full_ids.shape[1] - prompt_ids.shape[1]
    if target_len <= 0:
        return 0.0

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits[0]  # (seq_len, vocab_size)

    # Get log probs for target tokens
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    target_token_ids = full_ids[0, prompt_ids.shape[1]:]
    token_log_probs = []
    for i, tid in enumerate(target_token_ids):
        pos = prompt_ids.shape[1] - 1 + i  # position in logits
        token_log_probs.append(log_probs[pos, tid].item())

    avg_log_prob = sum(token_log_probs) / len(token_log_probs)
    return float(np.exp(avg_log_prob))


# ---- Main editing loop ----
results = []
success_count = 0
failure_count = 0
error_count = 0
start_time = time.time()

print(f"\n[{datetime.now().isoformat()}] Starting ROME editing on {N_FACTS} facts...")
print("=" * 70)

# Note: The first edit will trigger covariance matrix computation from Wikipedia
# This is expected and takes extra time on the first run only.

for idx, fact in enumerate(sampled_facts):
    fact_start = time.time()
    case_id = fact["case_id"]
    rw = fact["requested_rewrite"]
    subject = rw["subject"]
    prompt_template = rw["prompt"]  # Contains {} for subject
    target_new = rw["target_new"]["str"]
    target_true = rw["target_true"]["str"]

    # Build the prompt text
    prompt_text = prompt_template.format(subject)

    print(f"\n[{idx+1}/{N_FACTS}] Case {case_id}: '{prompt_text}' -> '{target_new}' (was: '{target_true}')")

    result_entry = {
        "case_id": case_id,
        "idx_in_sample": idx,
        "subject": subject,
        "prompt": prompt_text,
        "prompt_template": prompt_template,
        "target_new": target_new,
        "target_true": target_true,
        "edit_layer": EDIT_LAYER,
    }

    try:
        # 1. Compute pre-edit probabilities
        pre_prob_new = compute_target_probability(model, tokenizer, prompt_text, target_new)
        pre_prob_true = compute_target_probability(model, tokenizer, prompt_text, target_true)
        result_entry["pre_prob_new"] = pre_prob_new
        result_entry["pre_prob_true"] = pre_prob_true

        # 2. Execute ROME edit
        request = {
            "prompt": prompt_template,
            "subject": subject,
            "target_new": target_new,
            "target_true": target_true,
        }

        # Save original weights before editing
        weight_name = f"transformer.h.{EDIT_LAYER}.mlp.c_proj.weight"
        orig_weight = nethook.get_parameter(model, weight_name).detach().clone()

        # Run ROME - computes deltas but RESTORES original weights internally
        deltas = execute_rome(model, tokenizer, request, hparams)

        # Extract the rank-1 delta components
        delta_u, delta_v = deltas[weight_name]
        # Compute full delta_W = u @ v^T
        delta_W = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
        delta_W_matched = upd_matrix_match_shape(delta_W, orig_weight.shape)

        # 3. Apply delta to model weights for post-edit evaluation
        # (execute_rome restores original weights, so we must apply manually)
        with torch.no_grad():
            w = nethook.get_parameter(model, weight_name)
            w[...] += delta_W_matched

        # Compute post-edit probabilities
        post_prob_new = compute_target_probability(model, tokenizer, prompt_text, target_new)
        post_prob_true = compute_target_probability(model, tokenizer, prompt_text, target_true)
        result_entry["post_prob_new"] = post_prob_new
        result_entry["post_prob_true"] = post_prob_true

        # 4. Determine edit success
        edit_success = post_prob_new > post_prob_true
        result_entry["edit_success"] = edit_success
        result_entry["rewrite_acc"] = 1.0 if edit_success else 0.0

        # Record delta weight stats
        result_entry["delta_u_norm"] = float(delta_u.norm().item())
        result_entry["delta_v_norm"] = float(delta_v.norm().item())
        result_entry["delta_W_norm"] = float(delta_W_matched.norm().item())
        result_entry["delta_W_shape"] = list(delta_W_matched.shape)

        # 5. Save delta_W tensor
        delta_save_path = DELTAS_DIR / f"delta_W_case_{case_id}.pt"
        torch.save({
            "delta_W": delta_W_matched.cpu(),
            "delta_u": delta_u.cpu(),
            "delta_v": delta_v.cpu(),
            "case_id": case_id,
            "layer": EDIT_LAYER,
        }, delta_save_path)
        result_entry["delta_saved"] = True

        if edit_success:
            success_count += 1
        else:
            failure_count += 1

        status_str = "SUCCESS" if edit_success else "FAIL"
        print(f"  {status_str}: P(new)={post_prob_new:.4f}, P(true)={post_prob_true:.4f} "
              f"(pre: P(new)={pre_prob_new:.4f}, P(true)={pre_prob_true:.4f})")

        # 6. Restore original weights for next edit (ROME modifies in place)
        with torch.no_grad():
            w = nethook.get_parameter(model, weight_name)
            w[...] = orig_weight

        result_entry["error"] = None

    except torch.cuda.OutOfMemoryError:
        print(f"  OOM ERROR at fact {idx}")
        torch.cuda.empty_cache()
        result_entry["edit_success"] = False
        result_entry["rewrite_acc"] = 0.0
        result_entry["error"] = "OOM"
        result_entry["delta_saved"] = False
        error_count += 1
        # Try to restore weights
        try:
            with torch.no_grad():
                w = nethook.get_parameter(model, weight_name)
                w[...] = orig_weight
        except:
            pass

    except Exception as e:
        print(f"  ERROR at fact {idx}: {type(e).__name__}: {e}")
        traceback.print_exc()
        result_entry["edit_success"] = False
        result_entry["rewrite_acc"] = 0.0
        result_entry["error"] = f"{type(e).__name__}: {str(e)}"
        result_entry["delta_saved"] = False
        error_count += 1
        # Try to restore weights
        try:
            with torch.no_grad():
                w = nethook.get_parameter(model, weight_name)
                w[...] = orig_weight
        except:
            pass

    fact_elapsed = time.time() - fact_start
    result_entry["elapsed_sec"] = round(fact_elapsed, 2)
    results.append(result_entry)

    # Report progress every 10 facts
    if (idx + 1) % 10 == 0:
        current_efficacy = success_count / (idx + 1)
        report_progress(TASK_ID, RESULTS_DIR, idx + 1, N_FACTS,
                        metric={"efficacy": round(current_efficacy, 4),
                                "success": success_count,
                                "fail": failure_count,
                                "error": error_count})
        elapsed = time.time() - start_time
        eta = elapsed / (idx + 1) * (N_FACTS - idx - 1)
        print(f"\n--- Progress: {idx+1}/{N_FACTS} | Efficacy: {current_efficacy:.1%} | "
              f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s ---\n")

# ---- Summary ----
total_time = time.time() - start_time
total_attempted = success_count + failure_count + error_count
efficacy = success_count / total_attempted if total_attempted > 0 else 0.0

summary = {
    "task_id": TASK_ID,
    "model": "gpt2-xl",
    "model_path": MODEL_PATH,
    "dataset": "CounterFact",
    "n_facts": N_FACTS,
    "seed": SEED,
    "edit_layer": EDIT_LAYER,
    "edit_method": "ROME (EasyEdit, with C^{-1} covariance)",
    "total_attempted": total_attempted,
    "success_count": success_count,
    "failure_count": failure_count,
    "error_count": error_count,
    "efficacy_rate": round(efficacy, 4),
    "pass_criteria": "efficacy > 0.75",
    "pass": efficacy > 0.75,
    "total_time_sec": round(total_time, 2),
    "avg_time_per_edit_sec": round(total_time / max(total_attempted, 1), 2),
    "timestamp": datetime.now().isoformat(),
}

# Compute per-metric stats
if results:
    pre_probs_new = [r.get("pre_prob_new", 0) for r in results if r.get("error") is None]
    post_probs_new = [r.get("post_prob_new", 0) for r in results if r.get("error") is None]
    pre_probs_true = [r.get("pre_prob_true", 0) for r in results if r.get("error") is None]
    post_probs_true = [r.get("post_prob_true", 0) for r in results if r.get("error") is None]
    delta_norms = [r.get("delta_W_norm", 0) for r in results if r.get("delta_saved")]

    summary["metrics"] = {
        "pre_prob_new_mean": round(float(np.mean(pre_probs_new)), 6) if pre_probs_new else None,
        "post_prob_new_mean": round(float(np.mean(post_probs_new)), 6) if post_probs_new else None,
        "pre_prob_true_mean": round(float(np.mean(pre_probs_true)), 6) if pre_probs_true else None,
        "post_prob_true_mean": round(float(np.mean(post_probs_true)), 6) if post_probs_true else None,
        "delta_W_norm_mean": round(float(np.mean(delta_norms)), 4) if delta_norms else None,
        "delta_W_norm_std": round(float(np.std(delta_norms)), 4) if delta_norms else None,
    }

output = {
    "summary": summary,
    "per_fact_results": results,
}

# Save results
output_path = RESULTS_DIR / "pilot_rome_results.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n{'=' * 70}")
print(f"Results saved to: {output_path}")

# Print summary
print(f"\n=== PILOT ROME VALIDATION SUMMARY ===")
print(f"Total facts: {N_FACTS}")
print(f"Editing efficacy: {efficacy:.1%} ({success_count}/{total_attempted})")
print(f"Errors: {error_count}")
print(f"Pass criteria (>75%): {'PASS' if summary['pass'] else 'FAIL'}")
print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
print(f"Avg time per edit: {summary['avg_time_per_edit_sec']:.1f}s")
if summary.get("metrics"):
    m = summary["metrics"]
    print(f"\nProbability stats:")
    print(f"  Pre-edit  P(new):  {m['pre_prob_new_mean']:.6f}")
    print(f"  Post-edit P(new):  {m['post_prob_new_mean']:.6f}")
    print(f"  Pre-edit  P(true): {m['pre_prob_true_mean']:.6f}")
    print(f"  Post-edit P(true): {m['post_prob_true_mean']:.6f}")
    print(f"  Delta_W norm mean: {m['delta_W_norm_mean']:.4f} (std: {m['delta_W_norm_std']:.4f})")

# Save deltas listing
saved_deltas = list(DELTAS_DIR.glob("delta_W_case_*.pt"))
print(f"\nDelta tensors saved: {len(saved_deltas)} files in {DELTAS_DIR}")

# Mark task done
mark_task_done(TASK_ID, RESULTS_DIR,
               status="success" if summary["pass"] else "fail",
               summary=f"Efficacy={efficacy:.1%}, {success_count}/{total_attempted} edits successful")

print(f"\n[{datetime.now().isoformat()}] Pilot ROME experiment complete.")
