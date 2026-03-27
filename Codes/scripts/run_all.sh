#!/bin/bash
# AURA: Run complete experiment pipeline
# Total expected time: ~15 GPU-hours (Phase 2a + ablations)
# Phase 3 is conditional on pre-gate results
set -euo pipefail

CODES_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODES_DIR"

echo "================================================================"
echo "AURA: Complete Experiment Pipeline"
echo "================================================================"
echo ""

# Step 0: Pilot verification
echo ">>> Step 0: Pilot Verification"
bash scripts/run_pilot.sh
echo ""

# Step 1: Train all models
echo ">>> Step 1: Train ResNet-18 (5 seeds)"
bash scripts/run_baseline.sh
echo ""

# Step 2: Main experiment (Phase 2a)
echo ">>> Step 2: Main Experiment (Phase 2a)"
bash scripts/run_main.sh
echo ""

# Step 3: Ablation studies
echo ">>> Step 3: Ablation Studies"
bash scripts/run_ablation.sh
echo ""

echo "================================================================"
echo "Pipeline Complete"
echo "================================================================"
echo ""
echo "Results summary:"
echo "  _Results/phase2a_bss_crossseed.md  — Main results"
echo "  _Results/phase2a_augmented.md      — Controls"
echo "  _Results/ablation_*.md             — Ablation studies"
echo ""
echo "Next steps:"
echo "  1. Check Phase 2a gate results"
echo "  2. If PASS: proceed to Phase 3 (run_phase3.sh)"
echo "  3. If FAIL: iterate_direction"
