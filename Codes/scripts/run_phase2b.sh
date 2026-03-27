#!/bin/bash
# AURA: Run Phase 2b Disagreement Analysis (0 GPU-hours, reuses Phase 1 data)
set -euo pipefail

CODES_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODES_DIR"

echo "=== AURA Phase 2b: Disagreement Analysis ==="
echo ""

python evaluate.py --config configs/phase2b.yaml --phase phase2b

echo ""
echo "=== Phase 2b Complete ==="
echo "Results in _Results/phase2b_disagreement.md"
