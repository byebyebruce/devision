#!/usr/bin/env bash
# exp18 on one Mac (docs/experiments/2026-10-13-exp18-distill-plan.md): the teacher-term arm D, stopped once its
# step-2000 monitor records are complete (no stop rules), then the readout against B = runs/exp16-r14-s1.
# - The training checks its step 0 against B's before the first update (start_check). A mismatch writes
#   START_MISMATCH and ends the chain with status 3: it is not a device failure, so it is not retried.
# - Any other exit before step 2000 (e.g. the MPS hang) is retried from the resume state, at most 3 attempts.
# - DONE_AT_2000 only means the step was reached; the chain's exit status is the readout's (non-zero on failure).
#   nohup bash scripts/exp18/chain.sh > runs/exp18-distill.out 2>&1 &
# EXP18_TRAIN / EXP18_REPORT / EXP18_STAGE / EXP18_NAME / EXP18_POLL override the commands, stage directory, run
# name in the log and poll seconds (tests; scripts/exp19/chain.sh runs exp19 through this chain).
set -u
set -m   # background jobs get their own process group, so stopping ends the whole pipeline tree
cd "$(dirname "$0")/../.."
TRAIN=${EXP18_TRAIN:-"uv run devision-pipeline configs/exp18-distill.yaml"}
REPORT=${EXP18_REPORT:-"uv run python scripts/exp18/report.py"}
stage=${EXP18_STAGE:-runs/exp18-distill/stage2}
POLL=${EXP18_POLL:-60}
NAME=${EXP18_NAME:-exp18-distill}
SETS="val exist science ext bind_gqa count_narrow count_wide px_wide px_narrow px_exist flip vsr"

log() { echo "== $(date '+%F %T') $*"; }

complete() {   # $1 monitor step dir
  for s in $SETS; do [ -f "$1/$s.details.jsonl" ] || return 1; done
}

for attempt in 1 2 3; do
  [ -f "$stage/DONE_AT_2000" ] && break
  log "$NAME attempt $attempt"
  $TRAIN &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL"
    if complete "$stage/monitor/step2000"; then
      sleep "$POLL"   # let the last file finish writing
      touch "$stage/DONE_AT_2000"
      log "$NAME reached step 2000: stopping"
      kill -TERM -- "-$pid"
      break
    fi
  done
  wait "$pid"
  if [ -f "$stage/START_MISMATCH" ]; then
    log "step 0 does not match the baseline: $(cat "$stage/START_MISMATCH")"
    log "not retried; investigate before training again"
    exit 3
  fi
  [ -f "$stage/DONE_AT_2000" ] && break
  log "$NAME exited before step 2000; retrying from the resume state"
  sleep "$POLL"
done
[ -f "$stage/DONE_AT_2000" ] || { log "$NAME failed 3 times"; exit 1; }
mkdir -p "$(dirname "$stage")"
$REPORT > "$(dirname "$stage")/report.txt" 2>&1
rc=$?
log "report (exit $rc): $(dirname "$stage")/report.txt"
log "chain finished"
exit $rc
