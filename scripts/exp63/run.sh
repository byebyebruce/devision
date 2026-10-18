#!/usr/bin/env bash
# exp63 complete-fact decision CE (docs/experiments/2026-10-19-decision-facts-prep.md; critic DECISION-FACTS-IMPL-01).
# Only after the critic's model permission (EXP63-RUN-01), with no other model task running. Priority order: start
# (model-free identities) -> train (MPS: configs/exp63.yaml, 1,200 updates) -> freeze (tensors only: the export against
# EXP57's end, FROZEN.json) -> decide (MPS: new fit / G / P, then EXP57 fit / G / P) -> exist / science (devision-eval,
# recorded) -> decide_h (MPS: the new end on H, report only) -> report (model-free; --finalize when a stage failed or was
# stopped; main and whole-chain completeness reported apart). train / decide / decide_h under scripts/exp53/watch.py (own
# process group, no progress 5 / 3 min, memory 1.5 GiB for 30 s, ETA, its deadline 35 s before the outer bound, TERM then
# KILL 10 s later). Hard limit start + 50 minutes (EXP63_LIMIT for CPU fixtures only), the last 120 s for the report.
# One start ever (runs/exp63.started, never removed), no resume, no retry, nothing chained after it.
#   nohup bash scripts/exp63/run.sh > runs/exp63.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
MARK=runs/exp63.started
[ ! -e "$MARK" ] || { log "$MARK exists: exp63 was already started once"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp63/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
[ ! -e runs/exp63 ] || { log "runs/exp63 exists"; exit 1; }
mkdir -p runs
DEADLINE=$(( $(date +%s) + ${EXP63_LIMIT:-3000} ))
{ echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"; } > "$MARK" \
  || { log "cannot write $MARK"; exit 1; }
STATUS=runs/exp63-status.txt
RESERVE=${EXP63_RESERVE:-120}
. scripts/exp38/steps.sh
PY="uv run python scripts/exp63/exp63.py"
WATCH="uv run python scripts/exp53/watch.py --deadline $(( DEADLINE - RESERVE - 35 )) --grace 10"
log "exp63: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
run_all() {
  local rc s
  step start $PY start --deadline "$DEADLINE"; rc=$?; codes=" start=$rc"; [ $rc = 0 ] || return $rc
  step train $WATCH --progress runs/exp63/progress-train.txt --log runs/exp63/watch-train.txt -- $PY train
  rc=$?; codes="$codes train=$rc"; [ $rc = 0 ] || return $rc
  step freeze $PY freeze; rc=$?; codes="$codes freeze=$rc"; [ $rc = 0 ] || return $rc
  step decide $WATCH --progress runs/exp63/progress-decide.txt --log runs/exp63/watch-decide.txt -- $PY decide
  rc=$?; codes="$codes decide=$rc"; [ $rc = 0 ] || return $rc
  mkdir -p runs/exp63/eval
  for s in exist=data/r13b/panel_exist.jsonl science=data/r13b/panel_science.jsonl; do
    step "eval:${s%%=*}" uv run devision-eval --checkpoint runs/exp63/stage2 --data "${s#*=}" --data-root data \
      --out runs/exp63/eval/${s%%=*}.json --role monitoring --no-temperature --max-questions 8
    rc=$?; codes="$codes ${s%%=*}=$rc"; [ $rc = 0 ] || return $rc
    uv run python scripts/v04b/freeze.py record runs/exp63 "${s%%=*}" || { status "eval:${s%%=*}" failed; codes="$codes record=1"; return 1; }
  done
  step decide_h $WATCH --progress runs/exp63/progress-decide.txt --log runs/exp63/watch-decide-h.txt -- $PY decide_h
  rc=$?; codes="$codes decide_h=$rc"; return $rc
}
run_all
rc=$?
log "stages:$codes"
if [ $rc = 0 ]; then final_report $PY report; else final_report $PY report --finalize "$codes"; fi
rrc=$?
log "report exit $rrc"
[ $rc = 0 ] && rc=$rrc
exit "$rc"
