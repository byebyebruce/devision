#!/usr/bin/env bash
# exp53 automatic region selection for EXP51-T's frozen digit head (docs/experiments/2026-10-18-exp53-selector-plan.md;
# critic EXP53-PREP-01, R1-01..04). Only after the critic's model permission (EXP53-RUN-01). Stages of
# scripts/exp53/stages.py: start (model-free, START.json before any forward) -> global (MPS) -> train (CPU) -> local
# (MPS) -> readout (CPU), each under scripts/exp53/watch.py (its own process group; stall: no progress within 5 min /
# for 3 min; available memory < 1.5 GiB for 30 s; two checks projecting past the deadline; TERM then KILL 10 s later;
# its deadline 35 s before the outer bound's, and TERM / INT to the watch forwarded to the group)
# inside run_capped; then the model-free report -- with --finalize when a stage failed or was stopped. Hard limit start
# + 45 minutes (EXP53_LIMIT for CPU fixtures only); the last RESERVE seconds for the report; fresh only, no resume, no
# retry. Exit: the first failing stage's status, else the report's (a failed report is never exit 0).
#   nohup bash scripts/exp53/run.sh > runs/exp53.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp53/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
[ ! -e runs/exp53 ] || { log "runs/exp53 exists"; exit 1; }
mkdir -p runs/exp53
DEADLINE=$(( $(date +%s) + ${EXP53_LIMIT:-2700} ))
echo "$DEADLINE" > runs/exp53.deadline
STATUS=runs/exp53-status.txt
RESERVE=${EXP53_RESERVE:-120}
WATCH_ARGS=${EXP53_WATCH_ARGS:-}
. scripts/exp38/steps.sh
log "exp53: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
rc=0
for stage in start global train local readout; do
  if [ $stage = start ]; then
    step start uv run python scripts/exp53/stages.py start --deadline "$DEADLINE"
  else
    # the watch's own deadline ends its group (TERM, 10 s, KILL) before steps.sh's outer bound (D - RESERVE - 15) TERMs it
    step $stage uv run python scripts/exp53/watch.py --deadline $(( DEADLINE - RESERVE - 35 )) --grace 10 $WATCH_ARGS \
      --progress runs/exp53/progress-$stage.txt --log runs/exp53/watch-$stage.txt -- uv run python scripts/exp53/stages.py $stage
  fi
  rc=$?
  codes="$codes $stage=$rc"
  [ $rc = 0 ] || break
done
log "stages:$codes"
if [ $rc = 0 ]; then
  final_report uv run python scripts/exp53/stages.py report
else
  final_report uv run python scripts/exp53/stages.py report --finalize "$codes"
fi
rrc=$?
log "report exit $rrc"
[ $rc = 0 ] && rc=$rrc
exit "$rc"
