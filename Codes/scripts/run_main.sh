#!/bin/bash
# AURA: Run main experiment (Phase 2a BSS cross-seed stability)
# Expected time: ~7 GPU-hours
# Prerequisite: run_baseline.sh (5 trained models)
set -euo pipefail

CODES_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODES_DIR"

echo "=== AURA Main Experiment: Phase 2a BSS Cross-Seed Stability ==="
echo ""

# Verify checkpoints exist
for SEED in 42 123 456 789 1024; do
    CKPT="_Data/models/resnet18_cifar10_seed${SEED}.pt"
    if [ ! -f "$CKPT" ]; then
        echo "ERROR: Missing checkpoint $CKPT"
        echo "Run ./scripts/run_baseline.sh first."
        exit 1
    fi
done
echo "All 5 model checkpoints found."
echo ""

# Phase 2a: BSS cross-seed stability
echo "--- Phase 2a: BSS Cross-Seed Stability ---"
python evaluate.py --config configs/phase2a.yaml --phase phase2a
echo ""

# Phase 2a augmented: controls
echo "--- Phase 2a Augmented: Controls & Diagnostics ---"
python evaluate.py --config configs/phase2a.yaml --phase phase2a_augmented
echo ""

echo "=== Main Experiment Complete ==="
echo "Results in _Results/"
