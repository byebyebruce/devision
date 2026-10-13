#!/usr/bin/env bash
# v12 retrain, end to end on one Mac (plan section 11): v12 baseline on the new calibration set, then arm D, then
# arm D+S2, one after another. While an arm trains, every complete monitor step is judged against step 0 by the
# frozen rules; a stop rule ends that arm (no retry, no relaxed rerun). A crash (e.g. the MPS hang) is retried from
# the resume state up to 3 times.
#   nohup bash scripts/r13b/chain.sh > runs/r13b/chain.out 2>&1 &
set -u
set -m   # background jobs get their own process group, so a stop ends the whole pipeline tree
cd "$(dirname "$0")/../.."
mkdir -p runs/r13b
SETS="val exist science ext bind_gqa count_narrow count_wide"

log() { echo "== $(date '+%F %T') $*"; }

if [ ! -f runs/r13b-v12-base/stage2-recal/model.safetensors ]; then
  log "baseline: refit v12's temperatures on data/r13b/val.jsonl (copy)"
  uv run devision-calibrate --checkpoint runs/v12-ext/stage2 --data data/r13b/val.jsonl --data-root data \
    --out runs/r13b-v12-base/stage2-recal || exit 1
fi
log "baseline: evaluate"
uv run devision-pipeline configs/r13b-v12-base.yaml || exit 1

judge_loop() {   # $1 stage dir, $2 pipeline pid
  local seen=""
  while kill -0 "$2" 2>/dev/null; do
    sleep 120
    [ -d "$1/monitor" ] || continue
    for d in "$1"/monitor/step*; do
      [ -d "$d" ] || continue
      case " $seen " in *" $d "*) continue;; esac
      local all=1
      for s in $SETS; do [ -f "$d/$s.details.jsonl" ] || all=0; done
      [ $all = 1 ] || continue
      seen="$seen $d"
      [ "$(basename "$d")" = step0 ] && continue
      log "judge $d"
      uv run python scripts/r13b/judge.py monitor "$1"
      if [ $? = 3 ]; then
        log "STOP rule hit: ending this arm"
        touch "$1/STOPPED_BY_RULE"
        kill -TERM -- "-$2"
        return
      fi
    done
  done
}

for arm in data s2; do
  stage=runs/r13b-v12-$arm/stage2
  for attempt in 1 2 3; do
    [ -f "$stage/STOPPED_BY_RULE" ] && break
    log "arm $arm attempt $attempt"
    uv run devision-pipeline configs/r13b-v12-$arm.yaml &
    pid=$!
    judge_loop "$stage" "$pid" &
    jpid=$!
    wait "$pid"; rc=$?
    kill "$jpid" 2>/dev/null; wait "$jpid" 2>/dev/null
    [ -f "$stage/STOPPED_BY_RULE" ] && { log "arm $arm stopped by rule"; break; }
    [ $rc = 0 ] && { log "arm $arm done"; break; }
    log "arm $arm exited $rc; retrying from the resume state"
    sleep 60
  done
done
log "chain finished"
