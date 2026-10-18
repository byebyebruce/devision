#!/usr/bin/env bash
# exp67 token package x continuous geometry interface diagnostic (docs/experiments/2026-10-19-exp67-interface-plan.md;
# critic EXP67-PREP-01). Only after the critic's model permission (EXP67-RUN-01), with no other model task running (EXP65
# included). start (model-free identities) -> run (MPS: the 24 pictures, each encoded once, AA / AO / OA / OO with EXP64's
# head; AA / OO must reproduce EXP64, else stop) -> report (model-free; --finalize when a stage failed or was stopped).
# run is under scripts/exp53/watch.py (own process group, no progress 5 / 3 min, memory 1.5 GiB for 30 s, ETA, its
# deadline 35 s before the outer bound, TERM then KILL 10 s later). Hard limit start + 10 minutes (EXP67_LIMIT for CPU
# fixtures only), the last 120 s for the report. One start ever (runs/exp67.started, never removed), no resume, no
# retry, nothing chained after it.
#   nohup bash scripts/exp67/run.sh > runs/exp67.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
MARK=runs/exp67.started
[ ! -e "$MARK" ] || { log "$MARK exists: exp67 was already started once"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp67/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
[ ! -e runs/exp67 ] || { log "runs/exp67 exists"; exit 1; }
mkdir -p runs
DEADLINE=$(( $(date +%s) + ${EXP67_LIMIT:-600} ))
{ echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"; } > "$MARK" \
  || { log "cannot write $MARK"; exit 1; }
STATUS=runs/exp67-status.txt
RESERVE=${EXP67_RESERVE:-120}
. scripts/exp38/steps.sh
PY="uv run python scripts/exp67/exp67.py"
WATCH="uv run python scripts/exp53/watch.py --deadline $(( DEADLINE - RESERVE - 35 )) --grace 10"
log "exp67: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
run_all() {
  local rc
  step start $PY start --deadline "$DEADLINE"; rc=$?; codes=" start=$rc"; [ $rc = 0 ] || return $rc
  step run $WATCH --progress runs/exp67/progress-run.txt --log runs/exp67/watch-run.txt -- $PY run
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
