#!/bin/bash
# AURA: Run pilot verification (quick dry-run of full pipeline)
# Expected time: < 2 minutes on any GPU
set -euo pipefail

CODES_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODES_DIR"

echo "=== AURA Pilot: Quick Pipeline Verification ==="
echo "Codes dir: $CODES_DIR"
echo ""

# Step 1: Train (dry-run)
echo "--- Step 1: Train (dry-run) ---"
python train.py --config configs/pilot.yaml --dry-run --max-steps 2
echo ""

# Step 2: Evaluate (dry-run)
echo "--- Step 2: Evaluate Phase 2a (dry-run) ---"
python evaluate.py --config configs/phase2a.yaml --dry-run --phase phase2a
echo ""

echo "=== Pilot Complete ==="
echo "Check _Results/ for output files."
