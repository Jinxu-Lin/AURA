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

# Step 2a: Main experiment (Phase 2a BSS cross-seed)
echo ">>> Step 2a: Phase 2a BSS Cross-Seed Stability"
bash scripts/run_phase2a.sh
echo ""

# Step 2b: Disagreement analysis (0 GPU-hours)
echo ">>> Step 2b: Phase 2b Disagreement Analysis"
bash scripts/run_phase2b.sh
echo ""

# Step 3: Phase 3 pre-gate (RepSim-wins check)
echo ">>> Step 3: Phase 3 Pre-Gate (RepSim-Wins)"
python evaluate.py --config configs/phase3.yaml --phase phase3_pregate
echo ""

# Step 4: Confound controls
echo ">>> Step 4: Confound Controls"
python evaluate.py --config configs/phase2a.yaml --phase confound
echo ""

# Step 5: Ablation studies
echo ">>> Step 5: Ablation Studies"
bash scripts/run_ablation.sh
echo ""

echo "================================================================"
echo "Pipeline Complete"
echo "================================================================"
echo ""
echo "Results summary:"
echo "  _Results/phase2a_bss_crossseed.md   -- Main BSS results"
echo "  _Results/phase2a_augmented.md       -- Controls"
echo "  _Results/phase2b_disagreement.md    -- Disagreement analysis"
echo "  _Results/phase3_pregate.md          -- RepSim-wins gate"
echo "  _Results/confound_controls.md       -- Confound controls"
echo "  _Results/ablation_*.md              -- Ablation studies"
echo ""
echo "Decision tree:"
echo "  1. Check Phase 2a gate (BSS_partial rho > 0.3)"
echo "     FAIL -> iterate_direction (STOP)"
echo "  2. Check Phase 3 pre-gate (RepSim-wins > 15%)"
echo "     FAIL -> diagnostic-only paper (Scenario B)"
echo "     PASS -> proceed to Phase 3 MRC"
