#!/bin/bash
# AURA: Run Phase 2a BSS cross-seed stability (CRITICAL GATE)
# Expected time: ~7 GPU-hours
# Prerequisite: 5 trained ResNet-18 models (run_baseline.sh)
set -euo pipefail

CODES_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODES_DIR"

echo "=== AURA Phase 2a: BSS Cross-Seed Stability ==="
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

echo "=== Phase 2a Complete ==="
echo "Results in _Results/phase2a_*.md"
echo ""
echo "CHECK GATES before proceeding:"
echo "  1. BSS_partial cross-seed rho > 0.5 (or > 0.3 borderline)"
echo "  2. Within-class variance > 25%"
echo "  3. Partial correlation > 0.15"
