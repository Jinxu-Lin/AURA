#!/usr/bin/env python3
"""
Pre-compute MEMIT layer statistics (mom2) for GPT-2-XL layers 13-17.
Uses wikitext (wikipedia broken in datasets>=4).
Runs standalone to cache stats before the main MEMIT experiment.
"""
import sys
sys.path.insert(0, "/home/jinxulin/sibyl_system/projects/TECA/EasyEdit")

import os
import torch
import gc

PROJECT_DIR = "/home/jinxulin/sibyl_system/projects/TECA"
MODEL_PATH = "/home/jinxulin/sibyl_system/shared/checkpoints/gpt2-xl"
STATS_DIR = os.path.join(PROJECT_DIR, "data/stats")
LAYERS = [13, 14, 15, 16, 17]

os.makedirs(STATS_DIR, exist_ok=True)

# Monkey-patch wikipedia -> wikitext
from datasets import load_dataset as _orig_ld
def _patched_ld(ds_name, config_name=None, **kwargs):
    if ds_name == "wikipedia":
        print(f"  [PATCH] {ds_name}/{config_name} -> wikitext/wikitext-103-raw-v1")
        return _orig_ld("wikitext", "wikitext-103-raw-v1", **kwargs)
    return _orig_ld(ds_name, config_name, **kwargs)

from easyeditor.models.rome import layer_stats as _ls
_ls.load_dataset = _patched_ld

from easyeditor.models.rome.layer_stats import layer_stats
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).cuda()
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

print(f"Computing stats for layers {LAYERS}...")
for layer in LAYERS:
    layer_name = f"transformer.h.{layer}.mlp.c_proj"
    print(f"\n--- Layer {layer}: {layer_name} ---")
    try:
        stat = layer_stats(
            model, tokenizer, layer_name, STATS_DIR,
            ds_name="wikitext",
            to_collect=["mom2"],
            model_name=MODEL_PATH,
            sample_size=10000,
            precision="float32",
            batch_tokens=None,
            download=True,
            force_recompute=False,
        )
        print(f"  Layer {layer} stats computed and cached.")
    except Exception as e:
        print(f"  Layer {layer} FAILED: {e}")
    gc.collect()
    torch.cuda.empty_cache()

print("\nDone. Stats cached in:", STATS_DIR)
print("Files:", os.listdir(STATS_DIR))
