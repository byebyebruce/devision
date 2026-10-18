#!/usr/bin/env bash
# exp73 the seven protection panels through EXP71-on (docs/experiments/2026-10-19-exp73-guard-plan.md; critic
# EXP73-PREP-01). Only after the critic's model permission (EXP73-RUN-01), with no other model task running (EXP72 included). start
# (model-free identities) -> guard (MPS: exist / science / ext / count_wide / count_narrow / flip / vsr through
# EXP71-on's Reading.decide) -> report (model-free; --finalize when a stage failed or was stopped). guard runs under
# scripts/exp53/watch.py (own process group, no progress 5 / 3 min, memory 1.5 GiB for 30 s, ETA, its deadline 35 s
# before the outer bound, TERM then KILL 10 s later). Hard limit start + 45 minutes (EXP73_LIMIT for CPU fixtures only),
# the last 120 s for the report. One start ever (runs/exp73.started, never removed), no resume, no retry, nothing chained
# after it (no training, no release).
#   nohup bash scripts/exp73/run.sh > runs/exp73.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
MARK=runs/exp73.started
[ ! -e "$MARK" ] || { log "$MARK exists: exp73 was already started once"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp73/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
[ ! -e runs/exp73 ] || { log "runs/exp73 exists"; exit 1; }
mkdir -p runs
DEADLINE=$(( $(date +%s) + ${EXP73_LIMIT:-2700} ))
{ echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"; } > "$MARK" \
  || { log "cannot write $MARK"; exit 1; }
STATUS=runs/exp73-status.txt
RESERVE=${EXP73_RESERVE:-120}
. scripts/exp38/steps.sh
PY="uv run python scripts/exp73/exp73.py"
WATCH="uv run python scripts/exp53/watch.py --deadline $(( DEADLINE - RESERVE - 35 )) --grace 10"
log "exp73: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
run_all() {
  local rc
  step start $PY start --deadline "$DEADLINE"; rc=$?; codes=" start=$rc"; [ $rc = 0 ] || return $rc
  step guard $WATCH --progress runs/exp73/progress-guard.txt --log runs/exp73/watch-guard.txt -- $PY guard
  rc=$?; codes="$codes guard=$rc"; return $rc
}
run_all
rc=$?
log "stages:$codes"
if [ $rc = 0 ]; then final_report $PY report; else final_report $PY report --finalize "$codes"; fi
rrc=$?
log "report exit $rrc"
[ $rc = 0 ] && rc=$rrc
exit "$rc"
