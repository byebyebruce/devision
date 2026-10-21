#!/usr/bin/env bash
# exp56 the reading mode live on the original pictures (docs/experiments/2026-10-18-exp56-live-reading-plan.md; critic EXP56-PREP-01..03).
# Only after the critic's model permission (EXP56-RUN-01). start (model-free: identities and the reading bundle)
# -> live (MPS: P through devision.reading, then E / D; under scripts/exp53/watch.py: own process group, stall 5 / 3 min, memory 1.5 GiB for 30 s,
# ETA, its deadline 35 s before the outer bound, TERM then KILL 10 s later) -> report (model-free; --finalize when a
# stage failed or was stopped). Hard limit start + 30 minutes (EXP56_LIMIT for CPU fixtures only), the last 120 s for
# the report; fresh only, no resume, no retry. Exit: the first failing stage's status, else the report's.
#   nohup bash scripts/exp56/run.sh > runs/exp56.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp56/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
[ ! -e runs/exp56 ] || { log "runs/exp56 exists"; exit 1; }
mkdir -p runs/exp56
DEADLINE=$(( $(date +%s) + ${EXP56_LIMIT:-1800} ))
echo "$DEADLINE" > runs/exp56.deadline
STATUS=runs/exp56-status.txt
RESERVE=${EXP56_RESERVE:-120}
. scripts/exp38/steps.sh
log "exp56: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
step start uv run python scripts/exp56/exp56.py start --deadline "$DEADLINE"
rc=$?
codes=" start=$rc"
if [ $rc = 0 ]; then
  step live uv run python scripts/exp53/watch.py --deadline $(( DEADLINE - RESERVE - 35 )) --grace 10 \
    --progress runs/exp56/progress-live.txt --log runs/exp56/watch-live.txt -- uv run python scripts/exp56/exp56.py live
  rc=$?
  codes="$codes live=$rc"
fi
log "stages:$codes"
if [ $rc = 0 ]; then
  final_report uv run python scripts/exp56/exp56.py report
else
  final_report uv run python scripts/exp56/exp56.py report --finalize "$codes"
fi
rrc=$?
log "report exit $rrc"
[ $rc = 0 ] && rc=$rrc
exit "$rc"
