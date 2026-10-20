#!/usr/bin/env bash
# exp64 the character head's cross-layout adaptation, everything else frozen (docs/experiments/2026-10-19-exp64-head-plan.md; critic EXP64-PREP-01).
# Only after the critic's model permission (EXP64-RUN-01). start (model-free identities) -> features (MPS: the training pictures' region tokens to disk)
# -> train (the head, 1,000 updates from 0b548142) -> reads (MPS: one encoding per dev picture, old and new head; EXP60's
# 24) -> report (model-free; --finalize when a stage failed or was stopped). features / train / reads run under
# scripts/exp53/watch.py (own process group, no progress 5 / 3 min, memory 1.5 GiB for 30 s, ETA, its deadline 35 s
# before the outer bound, TERM then KILL 10 s later). Hard limit start + 45 minutes (EXP64_LIMIT for CPU fixtures only),
# the last 120 s for the report. One start ever (runs/exp64.started, never removed), no resume, no retry, nothing
# chained after it.
#   nohup bash scripts/exp64/run.sh > runs/exp64.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
MARK=runs/exp64.started
[ ! -e "$MARK" ] || { log "$MARK exists: exp64 was already started once"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp64/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
[ ! -e runs/exp64 ] || { log "runs/exp64 exists"; exit 1; }
mkdir -p runs
DEADLINE=$(( $(date +%s) + ${EXP64_LIMIT:-2700} ))
{ echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"; } > "$MARK" \
  || { log "cannot write $MARK"; exit 1; }
STATUS=runs/exp64-status.txt
RESERVE=${EXP64_RESERVE:-120}
. scripts/exp38/steps.sh
PY="uv run python scripts/exp64/exp64.py"
WATCH="uv run python scripts/exp53/watch.py --deadline $(( DEADLINE - RESERVE - 35 )) --grace 10"
log "exp64: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
run_all() {
  local rc s
  step start $PY start --deadline "$DEADLINE"; rc=$?; codes=" start=$rc"; [ $rc = 0 ] || return $rc
  for s in features train reads; do
    step $s $WATCH --progress runs/exp64/progress-$s.txt --log runs/exp64/watch-$s.txt -- $PY $s
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
