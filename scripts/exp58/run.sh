#!/usr/bin/env bash
# exp58 the reading mode with EXP57's endpoint, live (docs/experiments/2026-10-18-exp58-live-transfer-plan.md; critic
# LIVE-TRANSFER-PREP-01). Prepared, NOT permitted: only after EXP57's pre-written branch allows it, NEW_SHA is filled and
# reviewed, and the critic gives a model permission for the named part. Two separate runs, never chained:
#   bash scripts/exp58/run.sh A [explore]    start -> live (stage A, 600 pictures) -> fidelity -> transfer (stage B1, dev_sqa) -> report;
#                                  hard limit 20 minutes, the last 120 s for the report
#   bash scripts/exp58/run.sh B2   guard (the seven panels) -> report; hard limit 45 minutes, the last 120 s for the report
# live / transfer / guard under scripts/exp53/watch.py (own process group, no progress 5 / 3 min, memory 1.5 GiB for 30 s,
# ETA, its deadline 35 s before the outer bound, TERM then KILL 10 s later). Fresh only, no resume, no retry; a failed
# stage starts nothing after it and the report is finalized.
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
PART=${1:?A or B2}
case ${2:-} in
  explore) MODE=explore-after-exp57-fit-fail ;;      # critic EXP58-EXPLORE-PREP-01: post-hoc, A + B1 only, never a candidate
  "") MODE=default ;;
  *) echo "mode must be empty or explore"; exit 1 ;;
esac
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp58/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
MARK=runs/exp58-$PART.started
# one start per part, ever: the marker is written before anything runs and never removed (a failed or stopped part is
# not started again; a product's absence is not taken as "not started")
[ ! -e "$MARK" ] || { log "$MARK exists: part $PART was already started once"; exit 1; }
case $PART in
  A) LIMIT=${EXP58_LIMIT:-1200}; [ ! -e runs/exp58 ] || { log "runs/exp58 exists"; exit 1; } ;;
  B2) LIMIT=${EXP58_LIMIT:-2700}; [ -e runs/exp58-A.started ] && [ -e runs/exp58/START.json ] || { log "B2 continues a started part A"; exit 1; } ;;
  *) log "part must be A or B2"; exit 1 ;;
esac
mkdir -p runs
DEADLINE=$(( $(date +%s) + LIMIT ))
{ echo "part $PART"; echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"; echo "mode $MODE"
  if [ "$PART" = B2 ]; then echo "continues_start $(shasum -a 256 runs/exp58/START.json | cut -d' ' -f1)"; fi; } > "$MARK" \
  || { log "cannot write $MARK"; exit 1; }
STATUS=runs/exp58-$PART-status.txt
RESERVE=${EXP58_RESERVE:-120}
. scripts/exp38/steps.sh
PY="uv run python scripts/exp58/live.py"
WATCH="uv run python scripts/exp53/watch.py --deadline $(( DEADLINE - RESERVE - 35 )) --grace 10"
log "exp58 $PART: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
run_a() {
  local rc
  step start $PY start --deadline "$DEADLINE" --mode "$MODE"; rc=$?; codes=" start=$rc"; [ $rc = 0 ] || return $rc
  step live $WATCH --progress runs/exp58/progress-live.txt --log runs/exp58/watch-live.txt -- $PY live; rc=$?; codes="$codes live=$rc"; [ $rc = 0 ] || return $rc
  step fidelity $PY fidelity; rc=$?; codes="$codes fidelity=$rc"; [ $rc = 0 ] || return $rc
  step transfer $WATCH --progress runs/exp58/progress-transfer.txt --log runs/exp58/watch-transfer.txt -- $PY transfer; rc=$?; codes="$codes transfer=$rc"
  return $rc
}
run_b2() {
  local rc
  step guard $WATCH --progress runs/exp58/progress-guard.txt --log runs/exp58/watch-guard.txt -- $PY guard; rc=$?; codes=" guard=$rc"
  return $rc
}
if [ "$PART" = A ]; then run_a; else run_b2; fi
rc=$?
log "stages:$codes"
if [ $rc = 0 ]; then final_report $PY report; else final_report $PY report --finalize "$codes"; fi
rrc=$?
[ $rc = 0 ] && rc=$rrc
exit "$rc"
