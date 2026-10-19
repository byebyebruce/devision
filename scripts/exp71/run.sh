#!/usr/bin/env bash
# exp71 EXP69's speed corrector in the live reading, on / off (docs/experiments/2026-10-19-exp71-live-corrector-plan.md; critic
# EXP71-PREP-01). Only after the critic's model permission (EXP71-RUN-01), with no other model task running. start
# (model-free identities, two bundles) -> live (MPS, stage A: rl_tune 600 + EXP60's 24, both conditions) -> fidelity
# (model-free: the live reads = EXP69-RUN-02's registered reads; a failure stops here) -> transfer (MPS, stage B1: dev_sqa
# temp / force and their fixed wrong pictures, both conditions) -> report (model-free; --finalize when a stage failed or
# was stopped). live / transfer run under scripts/exp53/watch.py (own process group, no progress 5 / 3 min, memory 1.5 GiB
# for 30 s, ETA, its deadline 35 s before the outer bound, TERM then KILL 10 s later). Hard limit start + 30 minutes
# (EXP71_LIMIT for CPU fixtures only), the last 120 s for the report. One start ever (runs/exp71.started, never removed),
# no resume, no retry, nothing chained after it (no B2, no training, no release).
#   nohup bash scripts/exp71/run.sh > runs/exp71.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
MARK=runs/exp71.started
[ ! -e "$MARK" ] || { log "$MARK exists: exp71 was already started once"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp71/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
[ ! -e runs/exp71 ] || { log "runs/exp71 exists"; exit 1; }
mkdir -p runs
DEADLINE=$(( $(date +%s) + ${EXP71_LIMIT:-1800} ))
{ echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"; } > "$MARK" \
  || { log "cannot write $MARK"; exit 1; }
STATUS=runs/exp71-status.txt
RESERVE=${EXP71_RESERVE:-120}
. scripts/exp38/steps.sh
PY="uv run python scripts/exp71/exp71.py"
WATCH="uv run python scripts/exp53/watch.py --deadline $(( DEADLINE - RESERVE - 35 )) --grace 10"
log "exp71: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
run_all() {
  local rc
  step start $PY start --deadline "$DEADLINE"; rc=$?; codes=" start=$rc"; [ $rc = 0 ] || return $rc
  step live $WATCH --progress runs/exp71/progress-live.txt --log runs/exp71/watch-live.txt -- $PY live
  rc=$?; codes="$codes live=$rc"; [ $rc = 0 ] || return $rc
  step fidelity $PY fidelity; rc=$?; codes="$codes fidelity=$rc"; [ $rc = 0 ] || return $rc
  step transfer $WATCH --progress runs/exp71/progress-transfer.txt --log runs/exp71/watch-transfer.txt -- $PY transfer
  rc=$?; codes="$codes transfer=$rc"; return $rc
}
run_all
rc=$?
log "stages:$codes"
if [ $rc = 0 ]; then final_report $PY report; else final_report $PY report --finalize "$codes"; fi
rrc=$?
log "report exit $rrc"
[ $rc = 0 ] && rc=$rrc
exit "$rc"
