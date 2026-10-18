#!/usr/bin/env bash
# exp57 decision-side comparison CE on text (docs/experiments/2026-10-18-decision-cmp-plan.md; critic EXP57-PREP-01..03).
# Only after the critic's model permission (EXP57-RUN-01). start (model-free identities) -> train (MPS: configs/exp57.yaml
# through devision-pipeline, 1,000 text-only updates) -> freeze (tensors only: frozen parts identical to EXP50-T, then
# FROZEN.json) -> decide (MPS: fit / G / P on the frozen checkpoint, then EXP50-T on fit; raw temperature, image=None)
# -> exist / science (devision-eval, raw temperature, recorded under the frozen identity) -> report (model-free;
# --finalize when a stage failed or was stopped). train and decide run under scripts/exp53/watch.py (own process group,
# no progress 5 min at first / 3 min after, memory 1.5 GiB for 30 s, ETA, its deadline 35 s before the outer bound,
# TERM then KILL 10 s later). Hard limit start + 45 minutes (EXP57_LIMIT for CPU fixtures only), the last 120 s for the
# report; fresh only, no resume, no retry, nothing chained after it. Exit: the first failing stage's status, else the report's.
#   nohup bash scripts/exp57/run.sh > runs/exp57.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp57/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
[ ! -e runs/exp57 ] || { log "runs/exp57 exists"; exit 1; }
[ ! -e runs/exp57.deadline ] || { log "runs/exp57.deadline exists (no second start)"; exit 1; }
DEADLINE=$(( $(date +%s) + ${EXP57_LIMIT:-2700} ))
echo "$DEADLINE" > runs/exp57.deadline
STATUS=runs/exp57-status.txt
RESERVE=${EXP57_RESERVE:-120}
. scripts/exp38/steps.sh
PY="uv run python scripts/exp57/exp57.py"
WATCH="uv run python scripts/exp53/watch.py --deadline $(( DEADLINE - RESERVE - 35 )) --grace 10"
log "exp57: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
run_all() {
  local rc
  step start $PY start --deadline "$DEADLINE"; rc=$?; codes=" start=$rc"; [ $rc = 0 ] || return $rc
  step train $WATCH --progress runs/exp57/progress-train.txt --log runs/exp57/watch-train.txt -- $PY train
  rc=$?; codes="$codes train=$rc"; [ $rc = 0 ] || return $rc
  step freeze $PY freeze; rc=$?; codes="$codes freeze=$rc"; [ $rc = 0 ] || return $rc
  step decide $WATCH --progress runs/exp57/progress-decide.txt --log runs/exp57/watch-decide.txt -- $PY decide
  rc=$?; codes="$codes decide=$rc"; [ $rc = 0 ] || return $rc
  mkdir -p runs/exp57/eval
  for s in exist=data/r13b/panel_exist.jsonl science=data/r13b/panel_science.jsonl; do
    step "eval:${s%%=*}" uv run devision-eval --checkpoint runs/exp57/stage2 --data "${s#*=}" --data-root data \
      --out runs/exp57/eval/${s%%=*}.json --role monitoring --no-temperature --max-questions 8
    rc=$?; codes="$codes ${s%%=*}=$rc"; [ $rc = 0 ] || return $rc
    uv run python scripts/v04b/freeze.py record runs/exp57 "${s%%=*}" || { status "eval:${s%%=*}" failed; codes="$codes record=1"; return 1; }
  done
  return 0
}
run_all
rc=$?
log "stages:$codes"
if [ $rc = 0 ]; then
  final_report $PY report
else
  final_report $PY report --finalize "$codes"
fi
rrc=$?
log "report exit $rrc"
[ $rc = 0 ] && rc=$rrc
exit "$rc"
