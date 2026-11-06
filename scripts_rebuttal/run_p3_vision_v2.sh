#!/bin/bash
# P3 vision with a 600-image OOD pool per held-out class. The CIFAR-100 test
# split has only 100 images per class; the held-out classes are absent from ID
# training, so their train-split images are equally unseen and are added.
# Reuses the existing fine-tuned checkpoints: eval only, no retraining.
BASE="${REPO_ROOT}"
cd "$BASE" || exit 1
PY="$BASE/.venv/bin/python"; LOGS="$BASE/logs_rebuttal"; STATE="$BASE/scripts_rebuttal/.state"
ts(){ date '+%Y-%m-%d %H:%M:%S'; }; log(){ echo "[$(ts)] $*"; }
log "=== P3 vision v2: OOD pool 100 -> 600 ==="
for S in 42 123 456 789 1024; do
  OUT="$BASE/outputs/nearood_polarity/vision_v2_seed${S}.json"
  [ -s "$OUT" ] && { log "SKIP (exists): seed $S"; continue; }
  log "START: vision v2 seed $S"
  "$PY" "$BASE/run_nearood_polarity.py" --arch vision --seeds "$S" \
      --epochs_vit 5 --n_eval 800 --n_masks 25 --bs 128 --out "$OUT" \
      >> "$LOGS/p3_vision_v2_${S}.log" 2>&1
  if [ -s "$OUT" ]; then log "OK: seed $S  [verified output]"
  else log "FAIL: seed $S -- no output"; echo "P3V2: vision seed $S" >> "$STATE/FAILURES.txt"; fi
done
log "=== P3 vision v2 FINISHED ==="
