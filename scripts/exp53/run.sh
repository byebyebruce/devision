#!/usr/bin/env bash
# exp53 automatic region selection for EXP51-T's frozen digit head (docs/experiments/2026-10-18-exp53-selector-plan.md;
# critic EXP53-PREP-01). Only after the critic's model permission (EXP53-RUN-01). Stages of scripts/exp53/stages.py:
# start (model-free, START.json before any forward) -> global (MPS) -> train (CPU) -> local (MPS) -> readout (CPU),
# then the model-free report -- with --finalize when a stage failed or was stopped, so whatever is complete and bound
# is reported and the rest stated. Hard limit start + 45 minutes (EXP53_LIMIT seconds for CPU fixtures only); every
# stage bounded (TERM, KILL 15 s later); the last RESERVE seconds for the report; fresh only, no resume, no retry.
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
. scripts/exp38/steps.sh
log "exp53: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
rc=0
for stage in start global train local readout; do
  if [ $stage = start ]; then
    step start uv run python scripts/exp53/stages.py start --deadline "$DEADLINE"
  else
    step $stage uv run python scripts/exp53/stages.py $stage
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
log "report exit $?"
exit "$rc"
