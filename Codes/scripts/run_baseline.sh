#!/bin/bash
# AURA: Run baseline model training (all 5 seeds)
# Expected time: ~2.5 GPU-hours (5 seeds x ~30min each on RTX 4090)
set -euo pipefail

CODES_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODES_DIR"

echo "=== AURA Baseline: Train ResNet-18 (5 seeds) ==="

# Train all 5 seeds (will skip existing checkpoints)
python train.py --config configs/base.yaml --seeds 42 123 456 789 1024

echo ""
echo "=== Baseline Training Complete ==="
echo "Checkpoints saved to _Data/models/"
