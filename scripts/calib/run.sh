#!/usr/bin/env bash
# V04-K calibration (docs/experiments/2026-10-19-v04-k-release-plan.md, PREP-02; critic V04-K-PREP-02-ACCEPT-01). Only after
# the critic's permission for this run, with no other model task running. One model at a time: start (model-free: the LIVE-K
# source verified, START, then source / clone -- files only) -> calibrate (MPS, 3,892 ordinary questions) -> rl_calib (CPU,
# 1,200 pictures) -> temp_bundle (CPU) -> collect (MPS, 1,200 reading requests, identity registered before loading) ->
# analyse (CPU) -> report (model-free; --finalize when a stage failed or was stopped). Model stages under watch (own process
# group, no progress 5 / 3 min, memory 1.5 GiB for 30 s, ETA, its deadline 35 s before the outer bound, TERM then KILL 10 s
# later). Hard limit start + 45 minutes (CALIB_LIMIT for CPU fixtures only), the last 180 s for the report. One start ever
# (runs/v04-k-calib.started, never removed), no resume, no retry, nothing chained after it (no bind, freeze or release).
#   nohup bash scripts/calib/run.sh > runs/v04-k-calib.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
MARK=runs/v04-k-calib.started
[ ! -e "$MARK" ] || { log "$MARK exists: the calibration was already started once"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/calib/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
for d in runs/v04-k-calib runs/v04-k data/rl_calib; do [ ! -e "$d" ] || { log "$d exists"; exit 1; }; done
mkdir -p runs
DEADLINE=$(( $(date +%s) + ${CALIB_LIMIT:-2700} ))
{ echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"; } > "$MARK" \
  || { log "cannot write $MARK"; exit 1; }
STATUS=runs/v04-k-calib-status.txt
RESERVE=${CALIB_RESERVE:-180}
. scripts/exp38/steps.sh
PY="uv run python scripts/calib/calib.py"
WATCH="uv run python scripts/exp53/watch.py --deadline $(( DEADLINE - RESERVE - 35 )) --grace 10"
log "calibration: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
run_all() {
  local rc s
  step start $PY start --deadline "$DEADLINE"; rc=$?; codes=" start=$rc"; [ $rc = 0 ] || return $rc
  step calibrate $WATCH --progress runs/v04-k-calib/progress-calibrate.txt --log runs/v04-k-calib/watch-calibrate.txt -- $PY calibrate
  rc=$?; codes="$codes calibrate=$rc"; [ $rc = 0 ] || return $rc
  for s in rl_calib temp_bundle; do
    step $s $PY $s; rc=$?; codes="$codes $s=$rc"; [ $rc = 0 ] || return $rc
  done
  step collect $WATCH --progress runs/v04-k-calib/progress-collect.txt --log runs/v04-k-calib/watch-collect.txt -- $PY collect
  rc=$?; codes="$codes collect=$rc"; [ $rc = 0 ] || return $rc
  step analyse $PY analyse; rc=$?; codes="$codes analyse=$rc"; return $rc
}
run_all
rc=$?
log "stages:$codes"
if [ $rc = 0 ]; then final_report $PY report; else final_report $PY report --finalize "$codes"; fi
rrc=$?
log "report exit $rrc"
[ $rc = 0 ] && rc=$rrc
exit "$rc"
