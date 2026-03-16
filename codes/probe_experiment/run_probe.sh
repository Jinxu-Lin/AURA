#!/bin/bash
# AURA Probe Experiment Runner
# Runs the full Phase 1 pilot: train models, compute attributions, analyze results

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GPU=${1:-0}
SEEDS="42 123 456"

echo "============================================="
echo "AURA Phase 1 Probe Experiment"
echo "GPU: $GPU | Seeds: $SEEDS"
echo "============================================="

# Step 1: Train models
echo ""
echo ">>> Step 1: Training ResNet-18 models (3 seeds)..."
python train_model.py --seeds $SEEDS --gpu $GPU --epochs 100

# Step 2: Compute attributions
echo ""
echo ">>> Step 2: Computing attributions under 5 Hessian levels..."
python compute_attributions.py --seeds $SEEDS --gpu $GPU

# Step 3: Analyze results
echo ""
echo ">>> Step 3: Analyzing results..."
python analyze_results.py --seeds $SEEDS

echo ""
echo "============================================="
echo "Probe experiment complete!"
echo "Results: outputs/attributions/"
echo "Plots:   outputs/attributions/plots/"
echo "============================================="
