#!/usr/bin/env bash
# LIVE-K transfer check (docs/experiments/2026-10-19-live-k-plan.md, runner section; critic LIVE-K-RUNNER-PREP-02). Only after
# the critic's LIVE-K-RUN permission, with no other model task running. One model at a time: start (model-free; the K bundle
# and START before any forward) -> fid_text (MPS: PROTECT's end on P_K's given states reproduces PROTECT's C_P_new, else
# stop) -> live_p -> transfer -> guard_science (MPS, the K combination) -> report (model-free; --finalize when a stage failed
# or was stopped). watch: own process group, no progress 5 / 3 min, memory 1.5 GiB for 30 s, ETA, its deadline 35 s before
# the outer bound, TERM then KILL 10 s later. Hard limit start + 30 minutes (LIVEK_LIMIT for CPU fixtures only), the last
# 120 s for the report. One start ever (runs/livek.started, never removed), no resume, no retry, nothing chained after it.
#   nohup bash scripts/livek/run.sh > runs/livek.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
MARK=runs/livek.started
[ ! -e "$MARK" ] || { log "$MARK exists: livek was already started once"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/livek/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
[ ! -e runs/livek ] || { log "runs/livek exists"; exit 1; }
mkdir -p runs
DEADLINE=$(( $(date +%s) + ${LIVEK_LIMIT:-1800} ))
{ echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"; } > "$MARK" \
  || { log "cannot write $MARK"; exit 1; }
STATUS=runs/livek-status.txt
RESERVE=${LIVEK_RESERVE:-120}
. scripts/exp38/steps.sh
PY="uv run python scripts/livek/livek.py"
WATCH="uv run python scripts/exp53/watch.py --deadline $(( DEADLINE - RESERVE - 35 )) --grace 10"
log "livek: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
run_all() {
  local rc s
  step start $PY start --deadline "$DEADLINE"; rc=$?; codes=" start=$rc"; [ $rc = 0 ] || return $rc
  for s in fid_text live_p transfer guard_science; do
    step $s $WATCH --progress runs/livek/progress-$s.txt --log runs/livek/watch-$s.txt -- $PY $s
    rc=$?; codes="$codes $s=$rc"; [ $rc = 0 ] || return $rc
  done
}
run_all
rc=$?
log "stages:$codes"
if [ $rc = 0 ]; then final_report $PY report; else final_report $PY report --finalize "$codes"; fi
rrc=$?
log "report exit $rrc"
[ $rc = 0 ] && rc=$rrc
exit "$rc"
