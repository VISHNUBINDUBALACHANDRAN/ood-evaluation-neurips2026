#!/bin/bash
# =============================================================================
# run_rebuttal_phase2.sh — experiments R5 and R6, queued behind the seed pipeline.
#
# R5  CIFAR-100 class-count ablation  (10/20/50/100 classes vs SVHN)
# R6  RoBERTa-large                    (SST-2 -> MNLI / AG News / 20-News)
#
# Both reuse the existing drivers via additive flags (--num_classes,
# --model_name) whose defaults reproduce the original runs.
#
# Waits for run_rebuttal_pipeline.sh to finish so the two never contend for
# the GPU. Resumable via marker files; a failure in one cell does not stop
# the rest.
#
# Usage:
#   nohup bash scripts_rebuttal/run_rebuttal_phase2.sh > logs_rebuttal/PHASE2.log 2>&1 &
# =============================================================================

BASE="${REPO_ROOT}"
cd "$BASE" || { echo "FATAL: cannot cd to $BASE"; exit 1; }

PY="$BASE/.venv/bin/python"
LOGS="$BASE/logs_rebuttal"
STATE="$BASE/scripts_rebuttal/.state"
mkdir -p "$LOGS" "$STATE"

SEEDS=(42 123 456 789 1024)
CLASS_COUNTS=(10 20 50 100)      # 100 already has checkpoints; driver skips existing
MIN_FREE_GB=10

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }
free_gb() { df -BG --output=avail "$BASE" | tail -1 | tr -dc '0-9'; }

disk_guard() {
    local f; f=$(free_gb)
    if [ "$f" -lt "$MIN_FREE_GB" ]; then
        log "FATAL: only ${f}GB free (<${MIN_FREE_GB}GB). Halting."
        exit 2
    fi
}

run_step() {
    local marker="$1"; shift
    local desc="$1"; shift
    shift                       # drop the literal '--'
    if [ -f "$STATE/$marker.done" ]; then log "SKIP (done): $desc"; return 0; fi
    disk_guard
    log "START: $desc   [free $(free_gb)GB]"
    local t0=$SECONDS
    "$@" >> "$LOGS/$marker.log" 2>&1
    local rc=$? mins=$(( (SECONDS - t0) / 60 ))
    if [ $rc -eq 0 ]; then
        touch "$STATE/$marker.done"; log "OK   (${mins}m): $desc"
    else
        log "FAIL (rc=$rc, ${mins}m): $desc  -> $LOGS/$marker.log"
        echo "PHASE2: $desc" >> "$STATE/FAILURES.txt"
    fi
    return $rc
}

log "============================================================"
log "PHASE-2 EXPERIMENTS (R5 class-count, R6 RoBERTa-large)"
log "Free disk: $(free_gb)GB"
log "============================================================"

# ------------------------------------------------ wait for the seed pipeline --
while pgrep -f "run_rebuttal_pipeline.sh" > /dev/null 2>&1; do
    log "waiting for the n=15 seed pipeline to finish..."
    sleep 300
done
log "Seed pipeline finished. Starting R5."

# =============================================================================
# R5 — CIFAR-100 class-count ablation
#   ViT-B/16 fine-tuned on the first N CIFAR-100 classes, scored against SVHN.
#   Tests whether the mahAvgAvg_AHM vs X-Mahalanobis gap grows with class count.
# =============================================================================
log ""
log "### R5: CIFAR-100 class-count ablation ###"
for NC in "${CLASS_COUNTS[@]}"; do
    for S in "${SEEDS[@]}"; do
        run_step "r5_c${NC}_s${S}" "CIFAR-100 num_classes=$NC seed=$S vs SVHN" -- \
            "$PY" "$BASE/run_cifar100_benchmark.py" \
                --ood_source svhn --seed "$S" --num_classes "$NC"
    done
    log "  --- class count $NC done. free $(free_gb)GB ---"
done

# mahAvgAvg_AHM on the subset checkpoints (the other side of the comparison)
for NC in "${CLASS_COUNTS[@]}"; do
    for S in "${SEEDS[@]}"; do
        run_step "r5_mvag_c${NC}_s${S}" "mahAvgAvg_AHM cifar100 c=$NC seed=$S" -- \
            "$PY" "$BASE/run_mahavgavg_ahm_crossarch.py" \
                --architecture vit_cifar100 --ood_source svhn --seed "$S" \
                --num_classes "$NC"
    done
done

# =============================================================================
# R6 — RoBERTa-large
#   Second, larger text encoder on the same SST-2 OOD sets.
# =============================================================================
log ""
log "### R6: RoBERTa-large ###"
for OOD in 20news agnews mnli; do
    for S in "${SEEDS[@]}"; do
        run_step "r6_large_${OOD}_${S}" "RoBERTa-large sst2->$OOD seed=$S" -- \
            "$PY" "$BASE/run_roberta_experiments.py" \
                --ood_source "$OOD" --seed "$S" --model_name roberta-large
    done
    log "  --- roberta-large $OOD done. free $(free_gb)GB ---"
done

log ""
log "============================================================"
log "PHASE-2 FINISHED at $(ts). Free disk: $(free_gb)GB"
if [ -f "$STATE/FAILURES.txt" ]; then
    log "FAILURES RECORDED:"; sed 's/^/    /' "$STATE/FAILURES.txt"
else
    log "All steps completed without error."
fi
log "============================================================"
