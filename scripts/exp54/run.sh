#!/usr/bin/env bash
# exp54 read values -> ordinary decide (docs/experiments/2026-10-18-exp54-read-to-decision-plan.md; critic EXP54-PREP-01..04).
# Only after the critic's model permission (EXP54-RUN-01). start (model-free: identities and the four conditions'
# inputs) -> decide (MPS, under scripts/exp53/watch.py: own process group, stall 5 / 3 min, memory 1.5 GiB for 30 s,
# ETA, its deadline 35 s before the outer bound, TERM then KILL 10 s later) -> report (model-free; --finalize when a
# stage failed or was stopped). Hard limit start + 30 minutes (EXP54_LIMIT for CPU fixtures only), the last 90 s for
# the report; fresh only, no resume, no retry. Exit: the first failing stage's status, else the report's.
#   nohup bash scripts/exp54/run.sh > runs/exp54.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp54/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
[ ! -e runs/exp54 ] || { log "runs/exp54 exists"; exit 1; }
mkdir -p runs/exp54
DEADLINE=$(( $(date +%s) + ${EXP54_LIMIT:-1800} ))
echo "$DEADLINE" > runs/exp54.deadline
STATUS=runs/exp54-status.txt
RESERVE=${EXP54_RESERVE:-90}
. scripts/exp38/steps.sh
log "exp54: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
step start uv run python scripts/exp54/exp54.py start --deadline "$DEADLINE"
rc=$?
codes=" start=$rc"
if [ $rc = 0 ]; then
  step decide uv run python scripts/exp53/watch.py --deadline $(( DEADLINE - RESERVE - 35 )) --grace 10 \
    --progress runs/exp54/progress-decide.txt --log runs/exp54/watch-decide.txt -- uv run python scripts/exp54/exp54.py decide
  rc=$?
  codes="$codes decide=$rc"
fi
log "stages:$codes"
if [ $rc = 0 ]; then
  final_report uv run python scripts/exp54/exp54.py report
else
  final_report uv run python scripts/exp54/exp54.py report --finalize "$codes"
fi
rrc=$?
log "report exit $rrc"
[ $rc = 0 ] && rc=$rrc
exit "$rc"
