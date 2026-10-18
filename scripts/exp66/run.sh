#!/usr/bin/env bash
# exp66 selector with column / margin layouts, EXP64 head frozen (docs/experiments/2026-10-19-exp66-layout-plan.md; critic EXP66-PREP-01).
# Only after the critic's model permission (EXP66-RUN-01), with no other model task running (EXP65 included). start (model-free identities) -> global (MPS: the 800 exp66 training pictures' patches)
# -> train (CPU: the fixed plan, 2,000 selector updates from EXP53's end) -> reads (MPS: rl_tune / tune_V / tune_C new and c92 + EXP64 head, tune_C labels, EXP60's
# 24) -> report (model-free; --finalize when a stage failed or was stopped). global / train / reads run under
# scripts/exp53/watch.py (own process group, no progress 5 / 3 min, memory 1.5 GiB for 30 s, ETA, its deadline 35 s
# before the outer bound, TERM then KILL 10 s later). Hard limit start + 40 minutes (EXP66_LIMIT for CPU fixtures only),
# the last 120 s for the report. One start ever (runs/exp66.started, never removed), no resume, no retry, nothing
# chained after it.
#   nohup bash scripts/exp66/run.sh > runs/exp66.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
MARK=runs/exp66.started
[ ! -e "$MARK" ] || { log "$MARK exists: exp66 was already started once"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp66/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
[ ! -e runs/exp66 ] || { log "runs/exp66 exists"; exit 1; }
mkdir -p runs
DEADLINE=$(( $(date +%s) + ${EXP66_LIMIT:-2400} ))
{ echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"; } > "$MARK" \
  || { log "cannot write $MARK"; exit 1; }
STATUS=runs/exp66-status.txt
RESERVE=${EXP66_RESERVE:-120}
. scripts/exp38/steps.sh
PY="uv run python scripts/exp66/exp66.py"
WATCH="uv run python scripts/exp53/watch.py --deadline $(( DEADLINE - RESERVE - 35 )) --grace 10"
log "exp66: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
run_all() {
  local rc s
  step start $PY start --deadline "$DEADLINE"; rc=$?; codes=" start=$rc"; [ $rc = 0 ] || return $rc
  for s in global train reads; do
    step $s $WATCH --progress runs/exp66/progress-$s.txt --log runs/exp66/watch-$s.txt -- $PY $s
    rc=$?; codes="$codes $s=$rc"; [ $rc = 0 ] || return $rc
  done
  return 0
}
run_all
rc=$?
log "stages:$codes"
if [ $rc = 0 ]; then final_report $PY report; else final_report $PY report --finalize "$codes"; fi
rrc=$?
log "report exit $rrc"
[ $rc = 0 ] && rc=$rrc
exit "$rc"
