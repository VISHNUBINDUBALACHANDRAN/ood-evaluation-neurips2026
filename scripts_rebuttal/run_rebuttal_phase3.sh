#!/bin/bash
# =============================================================================
# run_rebuttal_phase3.sh — experiments P3 and P6.
#
# P3  near-OOD outside documents
# P6  AHSS polarity vs ID input length
#
# Both ride on the same fine-tunes: the text arm's ID set is long-form 20-News
# prose, versus SST-2's single short sentences in the paper, so P6 falls out of
# the P3 text run at no extra training cost.
#
# The model is trained only on the near-neighbour classes, so the Mahalanobis
# reference is built from them (re-slicing the ID eval set alone would leave it
# unchanged). The script prints a SANITY line (far - near) so an inverted
# result cannot pass silently.
#
# Usage:
#   nohup bash scripts_rebuttal/run_rebuttal_phase3.sh > logs_rebuttal/PHASE3.log 2>&1 &
# =============================================================================

BASE="${REPO_ROOT}"
cd "$BASE" || { echo "FATAL: cannot cd to $BASE"; exit 1; }

PY="$BASE/.venv/bin/python"
LOGS="$BASE/logs_rebuttal"
STATE="$BASE/scripts_rebuttal/.state"
mkdir -p "$LOGS" "$STATE"

SEEDS=(42 123 456 789 1024)
MIN_FREE_GB=10

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }
free_gb() { df -BG --output=avail "$BASE" | tail -1 | tr -dc '0-9'; }

run_step() {
    local marker="$1"; shift; local desc="$1"; shift; shift
    if [ -f "$STATE/$marker.done" ]; then log "SKIP (done): $desc"; return 0; fi
    local f; f=$(free_gb)
    if [ "$f" -lt "$MIN_FREE_GB" ]; then log "FATAL: only ${f}GB free. Halting."; exit 2; fi
    log "START: $desc   [free ${f}GB]"
    local t0=$SECONDS
    "$@" >> "$LOGS/$marker.log" 2>&1
    local rc=$? mins=$(( (SECONDS - t0) / 60 ))
    if [ $rc -eq 0 ]; then touch "$STATE/$marker.done"; log "OK   (${mins}m): $desc"
    else log "FAIL (rc=$rc, ${mins}m): $desc -> $LOGS/$marker.log"
         echo "PHASE3: $desc" >> "$STATE/FAILURES.txt"; fi
    return $rc
}

log "============================================================"
log "PHASE-3: P3 near-OOD cross-modality + P6 polarity/length"
log "Free disk: $(free_gb)GB"
log "============================================================"

# Run alone: wait for both earlier pipelines so nothing contends for the GPU.
while pgrep -f "run_rebuttal_pipeline.sh|run_rebuttal_phase2.sh" > /dev/null 2>&1; do
    log "waiting for earlier pipelines..."
    sleep 300
done
log "GPU free. Starting P3/P6."

# --------------------------------------------------------------- P3 vision --
log ""
log "### P3 vision ###"
for S in "${SEEDS[@]}"; do
    run_step "p3_vision_${S}" "P3 vision seed $S" -- \
        "$PY" "$BASE/run_nearood_polarity.py" --arch vision --seeds "$S" \
            --epochs_vit 5 --n_eval 800 --n_masks 25 --bs 128 \
            --out "$BASE/outputs/nearood_polarity/vision_seed${S}.json"
done

# ------------------------------------------------------- P3 text  +  P6 -----
log ""
log "### P3 text + P6: 20news comp.* ID (LONG inputs), near=sci.electronics ###"
for S in "${SEEDS[@]}"; do
    run_step "p3_text_${S}" "P3/P6 text seed $S" -- \
        "$PY" "$BASE/run_nearood_polarity.py" --arch text --seeds "$S" \
            --epochs_txt 4 --n_eval 800 --n_masks 25 --bs_txt 32 --max_len 256 \
            --out "$BASE/outputs/nearood_polarity/text_seed${S}.json"
done

# ------------------------------------------------------------- aggregate ----
log ""
run_step "p3_aggregate" "aggregate P3/P6 across seeds" -- \
    "$PY" -c "
import json,glob,os,numpy as np
d='$BASE/outputs/nearood_polarity'
rows=[]
for f in sorted(glob.glob(os.path.join(d,'*_seed*.json'))):
    try: rows+=json.load(open(f))
    except Exception: pass
if not rows: raise SystemExit('no rows to aggregate')
out={}
for k in sorted({(r['modality'],r['config']) for r in rows}):
    rs=[r for r in rows if (r['modality'],r['config'])==k]
    e={'regime':rs[0]['regime'],'id_set':rs[0]['id_set'],'n_seeds':len(rs)}
    for m in ('mahalanobis','knn','ahss'):
        v=[r[m]['auroc'] for r in rs if m in r]
        if v: e[m]={'auroc_mean':float(np.mean(v)),'auroc_std':float(np.std(v))}
    out[f'{k[0]}|{k[1]}']=e
json.dump(out,open(os.path.join(d,'AGGREGATED.json'),'w'),indent=2)
for k,e in out.items():
    print(k, e['regime'], {m:round(e[m]['auroc_mean'],4) for m in ('mahalanobis','knn','ahss') if m in e})
for mo in {r['modality'] for r in rows}:
    n=[e for k,e in out.items() if k.startswith(mo) and e['regime']=='near']
    fa=[e for k,e in out.items() if k.startswith(mo) and e['regime']=='far']
    if n and fa and 'mahalanobis' in n[0] and 'mahalanobis' in fa[0]:
        gap=fa[0]['mahalanobis']['auroc_mean']-n[0]['mahalanobis']['auroc_mean']
        print(f'SANITY {mo}: far-near={gap:+.4f}', 'OK' if gap>0 else '** INVERTED **')
"

log ""
log "============================================================"
log "PHASE-3 FINISHED at $(ts). Free disk: $(free_gb)GB"
[ -f "$STATE/FAILURES.txt" ] && { log "FAILURES:"; sed 's/^/    /' "$STATE/FAILURES.txt"; } || log "No failures."
log "============================================================"
