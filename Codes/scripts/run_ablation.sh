#!/bin/bash
# AURA: Run ablation studies (Phase 5)
# Priority order: damping > eigenvalue_count > bucket_granularity > grad_norm > train_subset
# Expected time: ~5 GPU-hours total
set -euo pipefail

CODES_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODES_DIR"

echo "=== AURA Ablation Studies ==="
echo ""

# Ablation 1: Damping (highest priority)
echo "--- Ablation 1: Damping Sensitivity ---"
python evaluate.py --config configs/ablations.yaml --phase ablation --ablation damping \
    --output ablation_damping.md
echo ""

# Ablation 2: Eigenvalue count
echo "--- Ablation 2: Eigenvalue Count ---"
python evaluate.py --config configs/ablations.yaml --phase ablation --ablation eigenvalue_count \
    --output ablation_eigenvalue_count.md
echo ""

# Ablation 3: Bucket granularity
echo "--- Ablation 3: Bucket Granularity ---"
python evaluate.py --config configs/ablations.yaml --phase ablation --ablation bucket_granularity \
    --output ablation_bucket_granularity.md
echo ""

# Ablation 4: Gradient-norm correction
echo "--- Ablation 4: Gradient-Norm Correction ---"
python evaluate.py --config configs/ablations.yaml --phase ablation --ablation grad_norm_correction \
    --output ablation_grad_norm_correction.md
echo ""

echo "=== Ablation Studies Complete ==="
echo "Results in _Results/ablation_*.md"
