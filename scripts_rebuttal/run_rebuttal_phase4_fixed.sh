#!/bin/bash
# P4 RoBERTa-large, with head masks sized from model.config (24 layers x 16 heads).
#
# run_roberta_experiments catches per-experiment exceptions and still exits 0,
# so this script VERIFIES that the expected output file exists and parses
# before writing a .done marker; the exit code alone is not trusted.
BASE="${REPO_ROOT}"
cd "$BASE" || exit 1
PY="$BASE/.venv/bin/python"; LOGS="$BASE/logs_rebuttal"; STATE="$BASE/scripts_rebuttal/.state"
mkdir -p "$LOGS" "$STATE"
ts(){ date '+%Y-%m-%d %H:%M:%S'; }; log(){ echo "[$(ts)] $*"; }

# run_verified <marker> <expected-json> <desc> -- <cmd...>
run_verified(){
  local m="$1"; shift; local expect="$1"; shift; local d="$1"; shift; shift
  [ -f "$STATE/$m.done" ] && { log "SKIP (done): $d"; return 0; }
  log "START: $d"; local t0=$SECONDS
  "$@" >> "$LOGS/$m.log" 2>&1; local rc=$?
  local mins=$(( (SECONDS-t0)/60 ))
  if [ ! -s "$expect" ]; then
      log "FAIL (${mins}m): $d -- exit=$rc but NO OUTPUT at $expect"
      echo "PHASE4FIX: $d (no output)" >> "$STATE/FAILURES.txt"; return 1
  fi
  if ! "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if isinstance(d,dict) and len(d)>3 else 1)" "$expect"; then
      log "FAIL (${mins}m): $d -- output exists but is malformed"
      echo "PHASE4FIX: $d (malformed)" >> "$STATE/FAILURES.txt"; return 1
  fi
  touch "$STATE/$m.done"; log "OK (${mins}m): $d  [verified $(basename $(dirname $expect))]"
}

log "============================================================"
log "PHASE-4: RoBERTa-large (24x16 head masks)"
log "============================================================"
for OOD in 20news agnews mnli; do
  for S in 42 123 456 789 1024; do
    rm -f "$STATE/r6_large_${OOD}_${S}.done"      # clear the bogus markers
    run_verified "r6fix_large_${OOD}_${S}" \
      "$BASE/outputs/roberta_results/sst2_robertalarge_${OOD}_seed${S}/full_results.json" \
      "RoBERTa-large sst2->$OOD seed=$S" -- \
      "$PY" "$BASE/run_roberta_experiments.py" --ood_source "$OOD" --seed "$S" --model_name roberta-large
  done
done
log "PHASE-4 FINISHED at $(ts)"
