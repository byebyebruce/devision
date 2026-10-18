#!/usr/bin/env bash
# exp51 local region readout (docs/experiments/2026-10-18-exp51-local-readout-plan.md; critic EXP51-PREP-01, R1-01 / 02).
# Only after the critic's model permission.
#   1. start   (model-free): vision identity X / T / C, questions / regions / model checked, runs/exp51/START.json with
#              the git HEAD and every executed file's sha256, before any model forward;
#   2. extract (MPS): T then C features from one frozen tower, features.json updated after each condition;
#   3. probe   (CPU): the two heads, readout, T - C, report, recompute check;
#   4. if any of them failed or was stopped: finalize (model-free) writes the incomplete report from whatever is complete
#      and bound -- every condition completed / missing / partial / invalid -- inside the same deadline.
# Hard deadline start + 30 minutes (EXP51_LIMIT seconds; CPU fixtures only); every step bounded (TERM, KILL 15 s later);
# the last RESERVE seconds for finalize; fresh only, no resume, no retry, nothing started after it. Exit: the first
# failing stage's status (124 at the limit), 0 when the probe completed.
#   nohup bash scripts/exp51/run.sh > runs/exp51.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp51/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
[ ! -e runs/exp51 ] || { log "runs/exp51 exists"; exit 1; }
mkdir -p runs/exp51
DEADLINE=$(( $(date +%s) + ${EXP51_LIMIT:-1800} ))
echo "$DEADLINE" > runs/exp51.deadline
STATUS=runs/exp51-status.txt
RESERVE=${EXP51_RESERVE:-90}
. scripts/exp38/steps.sh
log "exp51: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
rc=0
for stage in start extract probe; do
  case $stage in
    start) step start uv run python scripts/exp51/start.py --deadline "$DEADLINE" ;;
    extract) step extract uv run python scripts/exp51/extract.py ;;
    probe) step probe uv run python scripts/exp51/probe.py --device cpu ;;
  esac
  rc=$?
  codes="$codes $stage=$rc"
  [ $rc = 0 ] || break
done
log "stages:$codes"
if [ $rc != 0 ]; then
  final_report uv run python scripts/exp51/probe.py --finalize "$codes"
  log "finalize exit $?"
fi
exit "$rc"
