#!/bin/bash
# run_remaining_experiments.sh — Run financedocs D-AHM (with fixed paths) + ALL other experiments
#
# This script runs AFTER the background D-AHM job finishes tobacco experiments.
# The D-AHM script has skip-existing logic, so rerunning --run_all will only
# execute the missing financedocs experiments.
#
# Usage: nohup bash run_remaining_experiments.sh > logs/remaining_experiments.log 2>&1 &

set -e

BASE="${REPO_ROOT}"
LOGS="${REPO_ROOT}/logs"

cd "$BASE"
source .venv/bin/activate

mkdir -p "$LOGS"

echo "=========================================="
echo "REMAINING EXPERIMENTS — START"
echo "$(date)"
echo "=========================================="

# Step 1: D-AHM Rerun (only financedocs will be executed — tobacco skipped)
echo ""
echo ">>> STEP 1: D-AHM Evaluation (rerun — financedocs only)"
echo "$(date)"
python "$BASE/run_dahm_evaluation.py" --run_all 2>&1 | tee "$LOGS/step1_dahm_rerun_$(date +%Y%m%d_%H%M).log"
echo "Step 1 DONE: $(date)"

# Step 2: ViT Extended 17-Method Evaluation
echo ""
echo ">>> STEP 2: ViT Extended 17-Method Evaluation"
echo "$(date)"
python "$BASE/run_vit_extended_eval.py" --run_all 2>&1 | tee "$LOGS/step2_vit_extended_$(date +%Y%m%d_%H%M).log"
echo "Step 2 DONE: $(date)"

# Step 4: Distribution Shape Analysis (FSP)
echo ""
echo ">>> STEP 4: Distribution Shape Analysis (FSP)"
echo "$(date)"
python "$BASE/run_distribution_shape.py" --run_all 2>&1 | tee "$LOGS/step4_dist_shape_$(date +%Y%m%d_%H%M).log"
echo "Step 4 DONE: $(date)"

# Step 5: Near-OOD Experiments
echo ""
echo ">>> STEP 5: Near-OOD Experiments"
echo "$(date)"
python "$BASE/run_near_ood.py" --run_all 2>&1 | tee "$LOGS/step5_near_ood_$(date +%Y%m%d_%H%M).log"
echo "Step 5 DONE: $(date)"

# Step 6: NC-AHM Guided Mask Selection
echo ""
echo ">>> STEP 6: NC-AHM Guided Mask Selection"
echo "$(date)"
python "$BASE/run_nc_ahm.py" --run_all 2>&1 | tee "$LOGS/step6_nc_ahm_$(date +%Y%m%d_%H%M).log"
echo "Step 6 DONE: $(date)"

# Step 7: Per-Layer Sensitivity Decomposition
echo ""
echo ">>> STEP 7: Per-Layer Sensitivity Decomposition"
echo "$(date)"
python "$BASE/run_layer_decomposition.py" --run_all 2>&1 | tee "$LOGS/step7_layer_decomp_$(date +%Y%m%d_%H%M).log"
echo "Step 7 DONE: $(date)"

echo ""
echo "=========================================="
echo "ALL REMAINING EXPERIMENTS COMPLETED"
echo "$(date)"
echo "=========================================="
