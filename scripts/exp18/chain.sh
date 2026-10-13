#!/usr/bin/env bash
# exp18 on one Mac (docs/experiments/2026-10-13-exp18-distill-plan.md): the teacher-term arm D, stopped once its
# step-2000 monitor records are complete (no stop rules), then the readout against B = runs/exp16-r14-s1. A crash
# (e.g. the MPS hang) is retried from the resume state up to 3 times.
#   nohup bash scripts/exp18/chain.sh > runs/exp18-distill.out 2>&1 &
set -u
set -m   # background jobs get their own process group, so stopping ends the whole pipeline tree
cd "$(dirname "$0")/../.."
SETS="val exist science ext bind_gqa count_narrow count_wide px_wide px_narrow px_exist flip vsr"
stage=runs/exp18-distill/stage2

log() { echo "== $(date '+%F %T') $*"; }

complete() {   # $1 monitor step dir
  for s in $SETS; do [ -f "$1/$s.details.jsonl" ] || return 1; done
}

for attempt in 1 2 3; do
  [ -f "$stage/DONE_AT_2000" ] && break
  log "exp18-distill attempt $attempt"
  uv run devision-pipeline configs/exp18-distill.yaml &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    if complete "$stage/monitor/step2000"; then
      sleep 30   # let the last file finish writing
      touch "$stage/DONE_AT_2000"
      log "exp18-distill reached step 2000: stopping"
      kill -TERM -- "-$pid"
      break
    fi
  done
  wait "$pid"
  [ -f "$stage/DONE_AT_2000" ] && break
  log "exp18-distill exited before step 2000; retrying from the resume state"
  sleep 60
done
[ -f "$stage/DONE_AT_2000" ] || { log "exp18-distill failed 3 times"; exit 1; }
uv run python scripts/exp18/report.py > runs/exp18-distill/report.txt 2>&1; log "report (exit $?): runs/exp18-distill/report.txt"
log "chain finished"
