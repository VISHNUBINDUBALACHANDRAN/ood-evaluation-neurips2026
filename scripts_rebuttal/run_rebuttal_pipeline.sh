#!/bin/bash
# =============================================================================
# run_rebuttal_pipeline.sh — additional seeds for the two headline comparisons.
#
# Adds 10 seeds to Tobacco-Cross (LayoutLMv3) and SST-2 -> 20-News (RoBERTa),
# taking both from n=5 to n=15.
#
# Every command below is copied verbatim from outputs/tobacco_cross_pipeline.log,
# i.e. the exact invocations behind the paper's five-seed numbers. Only the
# seed changes.
#
# Design notes:
#   * NOT `set -e`. A failure in one seed must not kill the remaining seeds.
#   * Resumable. Each step writes a marker under scripts_rebuttal/.state/;
#     a re-run skips completed steps.
#   * Disk-aware. Large intermediates (HF trainer
#     checkpoint-* dirs, cached layer_embeddings) are deleted as soon as the
#     step that needs them has finished. Only files this pipeline itself
#     created are ever removed; the 5 original paper seeds are never touched.
#   * A hard disk guard aborts before any step that could fill the volume.
#
# Usage:
#   nohup bash scripts_rebuttal/run_rebuttal_pipeline.sh > logs_rebuttal/PIPELINE.log 2>&1 &
# =============================================================================

BASE="${REPO_ROOT}"
cd "$BASE" || { echo "FATAL: cannot cd to $BASE"; exit 1; }

PY="$BASE/.venv/bin/python"
LOGS="$BASE/logs_rebuttal"
STATE="$BASE/scripts_rebuttal/.state"
mkdir -p "$LOGS" "$STATE"

# 10 new seeds -> n=15 in total.
NEW_SEEDS=(7 13 31 77 101 202 303 404 505 2026)
ORIG_SEEDS=(42 123 456 789 1024)

MIN_FREE_GB=10          # abort a step if the volume drops below this
PROTECT='seed42|seed123|seed456|seed789|seed1024'   # never delete original runs

# ---------------------------------------------------------------- helpers ---
ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

free_gb() { df -BG --output=avail "$BASE" | tail -1 | tr -dc '0-9'; }

disk_guard() {
    local f; f=$(free_gb)
    if [ "$f" -lt "$MIN_FREE_GB" ]; then
        log "FATAL: only ${f}GB free (<${MIN_FREE_GB}GB). Halting to avoid a corrupt run."
        exit 2
    fi
}

# Delete a path only if it is a pipeline-created artifact for a NEW seed.
safe_rm() {
    local p="$1"
    [ -e "$p" ] || return 0
    if echo "$p" | grep -qE "$PROTECT"; then
        log "  REFUSING to delete protected original-seed path: $p"
        return 0
    fi
    case "$p" in
        "$BASE"/outputs/*) rm -rf "$p" && log "  freed: $p" ;;
        *) log "  REFUSING to delete path outside outputs/: $p" ;;
    esac
}

# run <marker-name> <description> -- <command...>
run_step() {
    local marker="$1"; shift
    local desc="$1"; shift
    shift  # drop the literal '--'
    if [ -f "$STATE/$marker.done" ]; then
        log "SKIP (already done): $desc"
        return 0
    fi
    disk_guard
    log "START: $desc   [free $(free_gb)GB]"
    local t0=$SECONDS
    "$@" >> "$LOGS/$marker.log" 2>&1
    local rc=$?
    local mins=$(( (SECONDS - t0) / 60 ))
    if [ $rc -eq 0 ]; then
        touch "$STATE/$marker.done"
        log "OK   (${mins}m): $desc"
    else
        log "FAIL (rc=$rc, ${mins}m): $desc   -> see $LOGS/$marker.log"
        echo "$desc" >> "$STATE/FAILURES.txt"
    fi
    return $rc
}

# ------------------------------------------------------ wait for R1b to end --
log "============================================================"
log "PIPELINE START"
log "Free disk: $(free_gb)GB"
log "============================================================"

while pgrep -f "run_mcdropout_mechanism.py" > /dev/null 2>&1; do
    log "waiting for R1b (MC-dropout mechanism) to finish..."
    sleep 120
done
log "R1b finished. Proceeding."

# =============================================================================
# R3 — Tobacco-Cross (LayoutLMv3), 10 additional seeds
#   per seed: finetune -> extract -> ahm_opt -> ood_eval, then reclaim disk
# =============================================================================
log ""
log "### R3: Tobacco-Cross extra seeds (n=15) ###"

for S in "${NEW_SEEDS[@]}"; do
    log "--- tobacco_cross seed $S ---"
    CK="./outputs/checkpoints/tobacco_cross_seed${S}"
    EM="./outputs/embeddings/tobacco_cross_seed${S}"
    AH="./outputs/ahm_masks/tobacco_cross_seed${S}"
    RS="./outputs/results/tobacco_cross_seed${S}"

    run_step "r3_ft_${S}" "finetune tobacco_cross seed $S" -- \
        "$PY" "$BASE/run_finetune.py" \
            --data_path ./data/tobacco/colated_tensors \
            --output_dir "$CK" \
            --dataset tobacco --experiment_type cross --seed "$S"
    [ -f "$STATE/r3_ft_${S}.done" ] || { log "  skipping rest of seed $S (finetune failed)"; continue; }

    # HF trainer intermediate checkpoints are ~3GB and are not used downstream.
    for d in "$BASE"/outputs/checkpoints/tobacco_cross_seed${S}/checkpoint-*; do
        safe_rm "$d"
    done

    run_step "r3_emb_${S}" "extract embeddings seed $S" -- \
        "$PY" "$BASE/run_extract_embeddings.py" \
            --model_path "$CK/best_model" \
            --data_path ./data/tobacco/colated_tensors \
            --output_dir "$EM" \
            --dataset tobacco --experiment_type cross --seed "$S" \
            --ood_data_path ./data/colated_tensors

    run_step "r3_ahm_${S}" "AHM mask optimization seed $S" -- \
        "$PY" "$BASE/run_ahm_optimization.py" \
            --model_path "$CK/best_model" \
            --data_path ./data/tobacco/colated_tensors \
            --output_dir "$AH" \
            --dataset tobacco --experiment_type cross --seed "$S" \
            --ood_data_path ./data/colated_tensors

    run_step "r3_ood_${S}" "OOD evaluation seed $S" -- \
        "$PY" "$BASE/run_ood_evaluation.py" \
            --embedding_dir "$EM" \
            --output_dir "$RS" \
            --dataset tobacco --experiment_type cross --seed "$S" \
            --ahm_dir "$AH" \
            --model_path "$CK/best_model" \
            --data_path ./data/tobacco/colated_tensors \
            --ood_data_path ./data/colated_tensors

    # Keep embeddings + masks + results (~300MB) for X-Maha and re-analysis.
    if [ -f "$STATE/r3_ood_${S}.done" ]; then
        safe_rm "$BASE/outputs/checkpoints/tobacco_cross_seed${S}/best_model"
        log "  seed $S complete. free $(free_gb)GB"
    fi
done

# X-Mahalanobis on the new seeds (uses the cached embeddings, cheap).
run_step "r3_xmaha" "X-Mahalanobis on new tobacco_cross seeds" -- \
    "$PY" "$BASE/run_xmaha_baseline.py" \
        --architecture layoutlmv3 --experiment tobacco_cross \
        --seeds "${NEW_SEEDS[@]}"

# =============================================================================
# R4 — SST-2 -> 20-News (RoBERTa), 10 additional seeds
# =============================================================================
log ""
log "### R4: SST-2 -> 20-News extra seeds (n=15) ###"

for S in "${NEW_SEEDS[@]}"; do
    log "--- sst2_20news seed $S ---"

    run_step "r4_rob_${S}" "RoBERTa sst2->20news seed $S" -- \
        "$PY" "$BASE/run_roberta_experiments.py" --ood_source 20news --seed "$S"

    run_step "r4_mvag_${S}" "mahAvgAvg_AHM roberta seed $S" -- \
        "$PY" "$BASE/run_mahavgavg_ahm_crossarch.py" \
            --architecture roberta --ood_source 20news --seed "$S"

    # Cached per-layer embeddings are ~2.5GB per seed and are not needed once
    # both result JSONs are written.
    if [ -f "$STATE/r4_mvag_${S}.done" ]; then
        safe_rm "$BASE/outputs/roberta_results/sst2_20news_seed${S}/layer_embeddings"
        safe_rm "$BASE/outputs/roberta_checkpoints/sst2_seed${S}"
        log "  seed $S complete. free $(free_gb)GB"
    fi
done

# =============================================================================
# Done
# =============================================================================
log ""
log "============================================================"
log "PIPELINE FINISHED at $(ts). Free disk: $(free_gb)GB"
if [ -f "$STATE/FAILURES.txt" ]; then
    log "STEPS THAT FAILED:"
    sed 's/^/    /' "$STATE/FAILURES.txt"
else
    log "All steps completed without error."
fi
log "============================================================"
