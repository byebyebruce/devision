#!/usr/bin/env bash
# PROTECT one-arm mixed replay recovery (docs/experiments/2026-10-19-protect-replay-plan.md; critic EXP73-RESULT-01 /
# PROTECT-PREP-01 / PROTECT-R1-01). Only after the critic's model permission, with no other model task running.
# One model at a time, in priority order: start (model-free) -> fid (MPS: EXP72-K reproduces EXP72's 48 slice records, else
# stop) -> train -> freeze (+ re-bind EXP71's components) -> decide_b (B, required) -> decide_c (C, required) -> guard (A,
# required) -> decide_fit (report only, last) -> report (model-free; --finalize when a stage failed or was stopped). watch:
# own process group, no progress 5 / 3 min, memory 1.5 GiB for 30 s, ETA, its deadline 35 s before the outer bound, TERM
# then KILL 10 s later. Hard limit start + 120 minutes (PROTECT_LIMIT for CPU fixtures only), the last 180 s for the
# report. One start ever (runs/protect.started, never removed), no resume, no retry, nothing chained after it.
#   nohup bash scripts/protect/run.sh > runs/protect.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
MARK=runs/protect.started
[ ! -e "$MARK" ] || { log "$MARK exists: protect was already started once"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/protect/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
for d in runs/protect runs/protect-k; do [ ! -e "$d" ] || { log "$d exists"; exit 1; }; done
mkdir -p runs
DEADLINE=$(( $(date +%s) + ${PROTECT_LIMIT:-7200} ))
{ echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"; } > "$MARK" \
  || { log "cannot write $MARK"; exit 1; }
STATUS=runs/protect-status.txt
RESERVE=${PROTECT_RESERVE:-180}
. scripts/exp38/steps.sh
PY="uv run python scripts/protect/protect.py"
WATCH="uv run python scripts/exp53/watch.py --deadline $(( DEADLINE - RESERVE - 35 )) --grace 10"
log "protect: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
run_all() {
  local rc s
  step start $PY start --deadline "$DEADLINE"; rc=$?; codes=" start=$rc"; [ $rc = 0 ] || return $rc
  for s in fid train; do
    step $s $WATCH --progress runs/protect/progress-$s.txt --log runs/protect/watch-$s.txt -- $PY $s
    rc=$?; codes="$codes $s=$rc"; [ $rc = 0 ] || return $rc
  done
  step freeze $PY freeze; rc=$?; codes="$codes freeze=$rc"; [ $rc = 0 ] || return $rc
  for s in decide_b decide_c guard decide_fit; do
    step $s $WATCH --progress runs/protect/progress-$s.txt --log runs/protect/watch-$s.txt -- $PY $s
    rc=$?; codes="$codes $s=$rc"; [ $rc = 0 ] || return $rc
  done
}
run_all
rc=$?
log "stages:$codes"
if [ $rc = 0 ]; then final_report $PY report; else final_report $PY report --finalize "$codes"; fi
rrc=$?
log "report exit $rrc"
[ $rc = 0 ] && rc=$rrc
exit "$rc"
