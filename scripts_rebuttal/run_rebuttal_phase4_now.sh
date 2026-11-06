#!/bin/bash
# P4 RoBERTa-large (SST-2 vs MNLI / AG News / 20-News), run in parallel with phase 2.
# Shares .state markers with phase2, so if phase2 reaches these steps later it
# skips them rather than repeating work.
BASE="${REPO_ROOT}"
cd "$BASE" || exit 1
PY="$BASE/.venv/bin/python"; LOGS="$BASE/logs_rebuttal"; STATE="$BASE/scripts_rebuttal/.state"
mkdir -p "$LOGS" "$STATE"
ts(){ date '+%Y-%m-%d %H:%M:%S'; }; log(){ echo "[$(ts)] $*"; }
free_gb(){ df -BG --output=avail "$BASE" | tail -1 | tr -dc '0-9'; }
run_step(){ local m="$1"; shift; local d="$1"; shift; shift
  [ -f "$STATE/$m.done" ] && { log "SKIP (done): $d"; return 0; }
  [ "$(free_gb)" -lt 10 ] && { log "FATAL: disk <10GB"; exit 2; }
  log "START: $d"; local t0=$SECONDS
  "$@" >> "$LOGS/$m.log" 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then touch "$STATE/$m.done"; log "OK ($(( (SECONDS-t0)/60 ))m): $d"
  else log "FAIL rc=$rc: $d -> $LOGS/$m.log"; echo "PHASE4: $d" >> "$STATE/FAILURES.txt"; fi; }

log "============================================================"
log "PHASE-4: RoBERTa-large"
log "Free disk: $(free_gb)GB"
log "============================================================"
for OOD in 20news agnews mnli; do
  for S in 42 123 456 789 1024; do
    run_step "r6_large_${OOD}_${S}" "RoBERTa-large sst2->$OOD seed=$S" -- \
      "$PY" "$BASE/run_roberta_experiments.py" --ood_source "$OOD" --seed "$S" --model_name roberta-large
  done
  log "  --- $OOD done. free $(free_gb)GB ---"
done
log "PHASE-4 FINISHED at $(ts)"
