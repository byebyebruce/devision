#!/usr/bin/env bash
# exp16 on one Mac: the r14 arm, then the r14b arm, both seed 1; each is stopped once its step-2000 monitor records
# are complete (no stop rules: the comparison is the point). A crash is retried from the resume state up to 3 times.
#   nohup bash scripts/exp16/chain.sh > runs/exp16/chain.out 2>&1 &
set -u
set -m   # background jobs get their own process group, so stopping ends the whole pipeline tree
cd "$(dirname "$0")/../.."
mkdir -p runs/exp16
SETS="val exist science ext bind_gqa count_narrow count_wide px_wide px_narrow px_exist flip vsr"

log() { echo "== $(date '+%F %T') $*"; }

complete() {   # $1 monitor step dir
  for s in $SETS; do [ -f "$1/$s.details.jsonl" ] || return 1; done
}

for arm in r14 r14b; do
  stage=runs/exp16-$arm-s1/stage2
  for attempt in 1 2 3; do
    [ -f "$stage/DONE_AT_2000" ] && break
    log "exp16-$arm-s1 attempt $attempt"
    uv run devision-pipeline configs/exp16-$arm-s1.yaml &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
      sleep 60
      if complete "$stage/monitor/step2000"; then
        sleep 30   # let the last file finish writing
        touch "$stage/DONE_AT_2000"
        log "exp16-$arm-s1 reached step 2000: stopping"
        kill -TERM -- "-$pid"
        break
      fi
    done
    wait "$pid"
    [ -f "$stage/DONE_AT_2000" ] && break
    log "exp16-$arm-s1 exited before step 2000; retrying from the resume state"
    sleep 60
  done
  [ -f "$stage/DONE_AT_2000" ] || { log "exp16-$arm-s1 failed 3 times"; exit 1; }
done
log "both arms at step 2000"
uv run python scripts/exp16/report.py > runs/exp16/report.txt 2>&1 && log "report: runs/exp16/report.txt"
log "chain finished"
