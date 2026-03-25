#!/usr/bin/env python3
"""
Phase 1 Pilot: ROME Editing Validation on 100 CounterFact facts using EasyEdit.
Uses proper EasyEdit ROME with covariance matrix (C^{-1}) for correct editing.

Task: phase1_rome_validation (pilot mode)
GPU: single RTX 4090
Timeout: 900s
Pass criteria: editing efficacy > 75%
"""

import sys
import os
import json
import time
import random
import gc
import traceback
from pathlib import Path
from datetime import datetime

# === Configuration ===
TASK_ID = "phase1_rome_validation"
PROJECT_DIR = "/home/jinxulin/sibyl_system/projects/TECA"
EASYEDIT_DIR = f"{PROJECT_DIR}/EasyEdit"
MODEL_PATH = "/home/jinxulin/sibyl_system/shared/checkpoints/gpt2-xl"
COUNTERFACT_PATH = "/home/jinxulin/sibyl_system/shared/datasets/counterfact/counterfact.json"
RESULTS_DIR = f"{PROJECT_DIR}/results"
DELTAS_DIR = f"{RESULTS_DIR}/rome_deltas"
HPARAMS_PATH = f"{EASYEDIT_DIR}/hparams/ROME/gpt2-xl.yaml"

N_SAMPLES = 100
SEED = 42
GPU_ID = 2

# === Setup paths ===
sys.path.insert(0, EASYEDIT_DIR)
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(DELTAS_DIR).mkdir(parents=True, exist_ok=True)

# === Write PID file ===
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

start_time = time.time()

try:
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[{TASK_ID}] PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda:0")  # mapped via CUDA_VISIBLE_DEVICES
    print(f"[{TASK_ID}] GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # === Load EasyEdit ROME ===
    from easyeditor.models.rome import ROMEHyperParams, apply_rome_to_model, execute_rome
    from easyeditor.util import nethook

    print(f"[{TASK_ID}] EasyEdit ROME imported successfully")

    # === Load hparams and patch model_name ===
    hparams = ROMEHyperParams.from_hparams(HPARAMS_PATH)
    hparams.model_name = MODEL_PATH
    # stats_dir for covariance matrices - EasyEdit will compute and cache them
    hparams.stats_dir = f"{EASYEDIT_DIR}/data/stats"
    hparams.fp16 = True
    Path(hparams.stats_dir).mkdir(parents=True, exist_ok=True)
    print(f"[{TASK_ID}] Hparams loaded: layers={hparams.layers}, v_num_grad_steps={hparams.v_num_grad_steps}, fp16={hparams.fp16}")

    # === Load model and tokenizer ===
    # Use FP16 to save ~3GB VRAM (shared GPU with other users)
    print(f"[{TASK_ID}] Loading GPT-2-XL from {MODEL_PATH} (FP16)...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(device)
    model.eval()
    vram_used = torch.cuda.memory_allocated(0) / 1e9
    print(f"[{TASK_ID}] Model loaded (FP16). VRAM used: {vram_used:.2f} GB")

    # === Load CounterFact and sample ===
    with open(COUNTERFACT_PATH) as f:
        counterfact = json.load(f)
    print(f"[{TASK_ID}] CounterFact loaded: {len(counterfact)} facts total")

    random.seed(SEED)
    np.random.seed(SEED)
    sampled_indices = random.sample(range(len(counterfact)), N_SAMPLES)
    sampled_facts = [counterfact[i] for i in sampled_indices]
    print(f"[{TASK_ID}] Sampled {N_SAMPLES} facts (seed={SEED})")

    # === Helper: compute token probability ===
    @torch.no_grad()
    def get_target_prob(model, tok, prompt_text, target_text, device):
        """Compute P(target_text | prompt_text)."""
        full_text = prompt_text + target_text
        prompt_ids = tok.encode(prompt_text, return_tensors="pt").to(device)
        full_ids = tok.encode(full_text, return_tensors="pt").to(device)

        outputs = model(full_ids)
        logits = outputs.logits[0]

        prompt_len = prompt_ids.shape[1]
        target_ids = full_ids[0, prompt_len:]

        if len(target_ids) == 0:
            return 0.0

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        total_log_prob = 0.0
        for i, tid in enumerate(target_ids):
            pos = prompt_len - 1 + i
            total_log_prob += log_probs[pos, tid].item()

        return np.exp(total_log_prob)

    @torch.no_grad()
    def check_edit_success(model, tok, prompt, target_new, target_old, device):
        """Check if P(target_new | prompt) > P(target_old | prompt)."""
        p_new = get_target_prob(model, tok, prompt, target_new, device)
        p_old = get_target_prob(model, tok, prompt, target_old, device)
        return p_new > p_old, p_new, p_old

    # === Run ROME editing ===
    results = []
    n_success = 0
    n_errors = 0

    for idx, fact in enumerate(sampled_facts):
        fact_start = time.time()
        case_id = fact["case_id"]
        rw = fact["requested_rewrite"]
        subject = rw["subject"]
        prompt_template = rw["prompt"]
        target_new = rw["target_new"]["str"]
        target_old = rw["target_true"]["str"]

        # Format prompt with subject
        if "{}" in prompt_template:
            prompt_text = prompt_template.format(subject)
        else:
            prompt_text = prompt_template

        print(f"\n[{idx+1}/{N_SAMPLES}] Case {case_id}: '{subject}' | '{target_old}' -> '{target_new}'")

        # Clear GPU cache before each edit to prevent OOM accumulation
        torch.cuda.empty_cache()
        gc.collect()

        try:
            # Measure pre-edit probabilities
            pre_success, pre_p_new, pre_p_old = check_edit_success(
                model, tok, prompt_text, " " + target_new, " " + target_old, device
            )
            print(f"  Pre-edit: P(new)={pre_p_new:.6f}, P(old)={pre_p_old:.6f}")

            # Prepare EasyEdit request format
            request = {
                "prompt": prompt_template,
                "subject": subject,
                "target_new": target_new,
                "target_true": target_old,
            }

            # Execute ROME to get deltas (does NOT permanently modify model)
            deltas = execute_rome(model, tok, request, hparams)

            # Extract delta tensors and save
            delta_info = {}
            for w_name, (delta_u, delta_v) in deltas.items():
                upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                # Match shape (GPT-2 has transposed weights)
                w = nethook.get_parameter(model, w_name)
                if upd_matrix.shape != w.shape:
                    if upd_matrix.T.shape == w.shape:
                        upd_matrix = upd_matrix.T

                delta_info[w_name] = {
                    "shape": list(upd_matrix.shape),
                    "norm": upd_matrix.norm().item(),
                    "u_norm": delta_u.norm().item(),
                    "v_norm": delta_v.norm().item(),
                }

                # Save delta tensor
                delta_path = Path(DELTAS_DIR) / f"delta_case{case_id}.pt"
                torch.save({
                    "case_id": case_id,
                    "weight_name": w_name,
                    "delta_u": delta_u.cpu(),
                    "delta_v": delta_v.cpu(),
                    "upd_matrix": upd_matrix.cpu(),
                    "layer": hparams.layers[0],
                }, delta_path)

            # Apply edit temporarily to measure post-edit probs
            with torch.no_grad():
                orig_weights = {}
                for w_name, (delta_u, delta_v) in deltas.items():
                    w = nethook.get_parameter(model, w_name)
                    orig_weights[w_name] = w.detach().clone()
                    upd = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                    if upd.shape != w.shape:
                        if upd.T.shape == w.shape:
                            upd = upd.T
                    w[...] += upd

                post_success, post_p_new, post_p_old = check_edit_success(
                    model, tok, prompt_text, " " + target_new, " " + target_old, device
                )

                # Restore original weights
                for w_name, orig_w in orig_weights.items():
                    w = nethook.get_parameter(model, w_name)
                    w[...] = orig_w

            edit_success = post_p_new > post_p_old
            if edit_success:
                n_success += 1

            elapsed = time.time() - fact_start
            print(f"  Post-edit: P(new)={post_p_new:.6f}, P(old)={post_p_old:.6f} | Success={edit_success} | {elapsed:.1f}s")

            result = {
                "case_id": case_id,
                "subject": subject,
                "target_new": target_new,
                "target_old": target_old,
                "prompt": prompt_text,
                "edit_success": edit_success,
                "rewrite_acc": float(edit_success),
                "edit_layer": hparams.layers[0],
                "pre_prob_new": pre_p_new,
                "pre_prob_old": pre_p_old,
                "post_prob_new": post_p_new,
                "post_prob_old": post_p_old,
                "prob_increase_new": post_p_new - pre_p_new,
                "prob_decrease_old": pre_p_old - post_p_old,
                "delta_info": delta_info,
                "elapsed_sec": elapsed,
                "error": None,
            }

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            elapsed = time.time() - fact_start
            print(f"  OOM error! Skipping case {case_id}")
            n_errors += 1
            result = {
                "case_id": case_id,
                "subject": subject,
                "target_new": target_new,
                "target_old": target_old,
                "prompt": prompt_text,
                "edit_success": False,
                "rewrite_acc": 0.0,
                "edit_layer": hparams.layers[0],
                "error": "OOM",
                "elapsed_sec": elapsed,
            }

        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            elapsed = time.time() - fact_start
            print(f"  Error: {e}")
            traceback.print_exc()
            n_errors += 1
            result = {
                "case_id": case_id,
                "subject": subject,
                "target_new": target_new,
                "target_old": target_old,
                "prompt": prompt_text,
                "edit_success": False,
                "rewrite_acc": 0.0,
                "edit_layer": hparams.layers[0],
                "error": str(e),
                "elapsed_sec": elapsed,
            }

        results.append(result)

        # Report progress every 10 facts
        if (idx + 1) % 10 == 0 or idx == 0:
            current_efficacy = n_success / (idx + 1)
            report_progress(
                epoch=idx + 1, total=N_SAMPLES,
                metric={"efficacy": current_efficacy, "errors": n_errors}
            )
            print(f"  [Progress] {idx+1}/{N_SAMPLES} | Efficacy: {current_efficacy:.1%} | Errors: {n_errors}")

        # NOTE: Do NOT clear CONTEXT_TEMPLATES_CACHE - it's reusable across edits
        # and regenerating it costs ~15-20 seconds per call

    # === Compute summary statistics ===
    total_time = time.time() - start_time
    successful = [r for r in results if r["edit_success"]]
    failed_edits = [r for r in results if not r["edit_success"] and r.get("error") is None]
    errored = [r for r in results if r.get("error") is not None]

    n_valid = N_SAMPLES - len(errored)
    efficacy = len(successful) / n_valid if n_valid > 0 else 0
    efficacy_total = len(successful) / N_SAMPLES if N_SAMPLES > 0 else 0

    # Prob stats for successful edits
    post_p_new_vals = [r["post_prob_new"] for r in results if "post_prob_new" in r]
    post_p_old_vals = [r["post_prob_old"] for r in results if "post_prob_old" in r]

    summary = {
        "task_id": TASK_ID,
        "mode": "pilot",
        "n_samples": N_SAMPLES,
        "n_valid": n_valid,
        "seed": SEED,
        "model": "gpt2-xl",
        "edit_layer": hparams.layers[0],
        "gpu": f"RTX 4090 (GPU {GPU_ID})",
        "total_time_sec": total_time,
        "avg_time_per_fact_sec": total_time / N_SAMPLES if N_SAMPLES > 0 else 0,
        "efficacy": efficacy,
        "efficacy_total": efficacy_total,
        "n_successful": len(successful),
        "n_failed_edits": len(failed_edits),
        "n_errors": len(errored),
        "error_types": {},
        "pass_criteria": "efficacy > 75% (on valid edits)",
        "pass": efficacy > 0.75,
        "post_prob_new_mean": float(np.mean(post_p_new_vals)) if post_p_new_vals else None,
        "post_prob_new_std": float(np.std(post_p_new_vals)) if post_p_new_vals else None,
        "post_prob_old_mean": float(np.mean(post_p_old_vals)) if post_p_old_vals else None,
        "per_fact_results": results,
        "timestamp": datetime.now().isoformat(),
        "easyedit_hparams": {
            "layers": hparams.layers,
            "v_num_grad_steps": hparams.v_num_grad_steps,
            "v_lr": hparams.v_lr,
            "v_loss_layer": hparams.v_loss_layer,
            "mom2_adjustment": hparams.mom2_adjustment,
            "clamp_norm_factor": hparams.clamp_norm_factor,
        }
    }

    # Count error types
    for r in errored:
        et = r.get("error", "unknown")
        summary["error_types"][et] = summary["error_types"].get(et, 0) + 1

    # Save results
    results_path = Path(RESULTS_DIR) / "pilot_rome_results.json"
    results_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n{'='*60}")
    print(f"ROME Pilot Results")
    print(f"{'='*60}")
    print(f"Efficacy (valid): {efficacy:.1%} ({len(successful)}/{n_valid})")
    print(f"Efficacy (total): {efficacy_total:.1%} ({len(successful)}/{N_SAMPLES})")
    print(f"Errors: {len(errored)} (types: {summary['error_types']})")
    print(f"Total time: {total_time:.1f}s ({total_time/N_SAMPLES:.1f}s/fact)")
    print(f"Pass: {'YES' if efficacy > 0.75 else 'NO'} (threshold: 75%)")
    print(f"Results saved to: {results_path}")
    print(f"Delta tensors saved to: {DELTAS_DIR}/ ({len(list(Path(DELTAS_DIR).glob('*.pt')))} files)")

    status = "success" if efficacy > 0.75 else "completed_below_threshold"
    mark_done(status=status, summary=f"Efficacy={efficacy:.1%} ({len(successful)}/{n_valid} valid), errors={len(errored)}, time={total_time:.0f}s")

except Exception as e:
    total_time = time.time() - start_time
    print(f"\n[FATAL ERROR] {e}")
    traceback.print_exc()
    mark_done(status="failed", summary=f"Fatal error: {str(e)[:200]}")
    sys.exit(1)
