#!/usr/bin/env bash
# The synthetic confirmation (scripts/confirm/confirm_run.py; critic V04-K-CONFIRM-PREP-02). Only after the final combination
# is frozen, this runner accepted and a model permission given, with no other model task running. CONFIRM_IDENTITY (the
# frozen IDENTITY.json) and CONFIRM_CHECKPOINT (the frozen candidate) are required. start (model-free: the freeze verified)
# -> generate (CPU: the set once) -> cand_true -> cand_mism -> cand_rev -> v03_true -> v03_mism (MPS, one model at a time,
# under watch: own process group, no progress 5 / 3 min, memory 1.5 GiB for 30 s, ETA, its deadline 35 s before the outer
# bound, TERM then KILL 10 s later) -> report (model-free; --finalize when a stage failed or was stopped). Hard limit start +
# 20 minutes (CONFIRM_LIMIT for CPU fixtures only), the last 120 s for the report. One start ever (runs/v04-k-confirm.started,
# never removed), no resume, no retry, nothing chained after it.
#   CONFIRM_IDENTITY=... CONFIRM_CHECKPOINT=... nohup bash scripts/confirm/run.sh > runs/v04-k-confirm.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
[ -n "${CONFIRM_IDENTITY:-}" ] && [ -n "${CONFIRM_CHECKPOINT:-}" ] || { log "CONFIRM_IDENTITY / CONFIRM_CHECKPOINT not set"; exit 2; }
MARK=runs/v04-k-confirm.started
[ ! -e "$MARK" ] || { log "$MARK exists: the confirmation was already started once"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/confirm/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
for d in runs/v04-k-confirm data/v04_k_confirm; do [ ! -e "$d" ] || { log "$d exists"; exit 1; }; done
mkdir -p runs
DEADLINE=$(( $(date +%s) + ${CONFIRM_LIMIT:-1200} ))
{ echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"; } > "$MARK" \
  || { log "cannot write $MARK"; exit 1; }
STATUS=runs/v04-k-confirm-status.txt
RESERVE=${CONFIRM_RESERVE:-120}
. scripts/exp38/steps.sh
PY="uv run python scripts/confirm/confirm_run.py"
WATCH="uv run python scripts/exp53/watch.py --deadline $(( DEADLINE - RESERVE - 35 )) --grace 10"
log "confirmation: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
run_all() {
  local rc s
  step start $PY start --identity "$CONFIRM_IDENTITY" --checkpoint "$CONFIRM_CHECKPOINT" --deadline "$DEADLINE"
  rc=$?; codes=" start=$rc"; [ $rc = 0 ] || return $rc
  step generate $PY generate; rc=$?; codes="$codes generate=$rc"; [ $rc = 0 ] || return $rc
  for s in cand_true cand_mism cand_rev v03_true v03_mism; do
    step $s $WATCH --progress runs/v04-k-confirm/progress-$s.txt --log runs/v04-k-confirm/watch-$s.txt -- $PY $s
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
