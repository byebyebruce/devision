#!/usr/bin/env bash
# exp70 raw vs explicit K = m * v^2 on the frozen EXP68-X (docs/experiments/2026-10-19-exp70-intermediate-plan.md; critic
# EXP70-PREP-01 / -PLAN-ACCEPT-01 / -IMPL-PREP-01). Only after the critic's model permission (EXP70-RUN-01), with no other
# model task running (EXP69 included). start (model-free identities) -> decide (MPS, under scripts/exp53/watch.py: the 48
# raw fidelity requests first -- a mismatch stops before any other request -- then the K diagnostic and the training
# slice) -> report (model-free; --finalize when a stage failed or was stopped). watch: own process group, no progress
# 5 / 3 min, memory 1.5 GiB for 30 s, ETA, its deadline 35 s before the outer bound, TERM then KILL 10 s later. Hard limit
# start + 15 minutes (EXP70_LIMIT for CPU fixtures only), the last 120 s for the report. One start ever (runs/exp70.started,
# never removed), no resume, no retry, nothing chained after it.
#   nohup bash scripts/exp70/run.sh > runs/exp70.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
MARK=runs/exp70.started
[ ! -e "$MARK" ] || { log "$MARK exists: exp70 was already started once"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp70/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
[ ! -e runs/exp70 ] || { log "runs/exp70 exists"; exit 1; }
mkdir -p runs
DEADLINE=$(( $(date +%s) + ${EXP70_LIMIT:-900} ))
{ echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"; } > "$MARK" \
  || { log "cannot write $MARK"; exit 1; }
STATUS=runs/exp70-status.txt
RESERVE=${EXP70_RESERVE:-120}
. scripts/exp38/steps.sh
PY="uv run python scripts/exp70/exp70.py"
WATCH="uv run python scripts/exp53/watch.py --deadline $(( DEADLINE - RESERVE - 35 )) --grace 10"
log "exp70: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
run_all() {
  local rc
  step start $PY start --deadline "$DEADLINE"; rc=$?; codes=" start=$rc"; [ $rc = 0 ] || return $rc
  step decide $WATCH --progress runs/exp70/progress-decide.txt --log runs/exp70/watch-decide.txt -- $PY decide
  rc=$?; codes="$codes decide=$rc"; return $rc
}
run_all
rc=$?
log "stages:$codes"
if [ $rc = 0 ]; then final_report $PY report; else final_report $PY report --finalize "$codes"; fi
rrc=$?
log "report exit $rrc"
[ $rc = 0 ] && rc=$rrc
exit "$rc"
