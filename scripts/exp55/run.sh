#!/usr/bin/env bash
# exp55 thousands separators in the fact tables only (docs/experiments/2026-10-18-exp55-format-plan.md; critic EXP55-PREP-01..03).
# Only after the critic's model permission (EXP55-RUN-01). start (model-free: identities and the four conditions'
# inputs) -> decide (MPS, under scripts/exp53/watch.py: own process group, stall 5 / 3 min, memory 1.5 GiB for 30 s,
# ETA, its deadline 35 s before the outer bound, TERM then KILL 10 s later) -> report (model-free; --finalize when a
# stage failed or was stopped). Hard limit start + 20 minutes (EXP55_LIMIT for CPU fixtures only), the last 90 s for
# the report; fresh only, no resume, no retry. Exit: the first failing stage's status, else the report's.
#   nohup bash scripts/exp55/run.sh > runs/exp55.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp55/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
[ ! -e runs/exp55 ] || { log "runs/exp55 exists"; exit 1; }
mkdir -p runs/exp55
DEADLINE=$(( $(date +%s) + ${EXP55_LIMIT:-1200} ))
echo "$DEADLINE" > runs/exp55.deadline
STATUS=runs/exp55-status.txt
RESERVE=${EXP55_RESERVE:-90}
. scripts/exp38/steps.sh
log "exp55: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
step start uv run python scripts/exp55/exp55.py start --deadline "$DEADLINE"
rc=$?
codes=" start=$rc"
if [ $rc = 0 ]; then
  step decide uv run python scripts/exp53/watch.py --deadline $(( DEADLINE - RESERVE - 35 )) --grace 10 \
    --progress runs/exp55/progress-decide.txt --log runs/exp55/watch-decide.txt -- uv run python scripts/exp55/exp55.py decide
  rc=$?
  codes="$codes decide=$rc"
fi
log "stages:$codes"
if [ $rc = 0 ]; then
  final_report uv run python scripts/exp55/exp55.py report
else
  final_report uv run python scripts/exp55/exp55.py report --finalize "$codes"
fi
rrc=$?
log "report exit $rrc"
[ $rc = 0 ] && rc=$rrc
exit "$rc"
