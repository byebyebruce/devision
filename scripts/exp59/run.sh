#!/usr/bin/env bash
# exp59 locate the real-picture transfer failure (docs/experiments/2026-10-18-exp59-locate-plan.md; critic EXP59-PREP-01).
# Only after the critic's model permission (EXP59-RUN-01). start (model-free: identities, the 24 items, G_num / G_facts
# inputs) -> run (MPS: P with diagnostics, the EXP58 reproduction check -- a failure stops here -- then G_num, G_facts;
# under scripts/exp53/watch.py: own process group, no progress 5 / 3 min, memory 1.5 GiB for 30 s, ETA, its deadline 35 s
# before the outer bound, TERM then KILL 10 s later) -> report (model-free; --finalize when a stage failed or was
# stopped). Hard limit start + 10 minutes (EXP59_LIMIT for CPU fixtures only), the last 120 s for the report. One start
# ever (runs/exp59.started, never removed), no resume, no retry, nothing chained after it.
#   nohup bash scripts/exp59/run.sh > runs/exp59.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
MARK=runs/exp59.started
[ ! -e "$MARK" ] || { log "$MARK exists: exp59 was already started once"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp59/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
[ ! -e runs/exp59 ] || { log "runs/exp59 exists"; exit 1; }
mkdir -p runs
DEADLINE=$(( $(date +%s) + ${EXP59_LIMIT:-600} ))
{ echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"; } > "$MARK" \
  || { log "cannot write $MARK"; exit 1; }
STATUS=runs/exp59-status.txt
RESERVE=${EXP59_RESERVE:-120}
. scripts/exp38/steps.sh
PY="uv run python scripts/exp59/exp59.py"
log "exp59: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
run_all() {
  local rc
  step start $PY start --deadline "$DEADLINE"; rc=$?; codes=" start=$rc"; [ $rc = 0 ] || return $rc
  step run uv run python scripts/exp53/watch.py --deadline $(( DEADLINE - RESERVE - 35 )) --grace 10 \
    --progress runs/exp59/progress-run.txt --log runs/exp59/watch-run.txt -- $PY run
  rc=$?; codes="$codes run=$rc"; return $rc
}
run_all
rc=$?
log "stages:$codes"
if [ $rc = 0 ]; then final_report $PY report; else final_report $PY report --finalize "$codes"; fi
rrc=$?
log "report exit $rrc"
[ $rc = 0 ] && rc=$rrc
exit "$rc"
