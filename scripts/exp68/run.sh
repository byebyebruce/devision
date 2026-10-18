#!/usr/bin/env bash
# exp68 one plain-CE arm on the new temperature combinations (docs/experiments/2026-10-19-exp68-combinations-plan.md;
# critic EXP68-PREP-01 / -R1-ACCEPT-01). Only after the critic's model permission (EXP68-RUN-01), with no other model task
# running. One model at a time: start (model-free) -> train (MPS, under scripts/exp53/watch.py) -> freeze (tensors only)
# -> decide (MPS: X on fit / G / P / EXP65 diag / new diag, then EXP65-C on the new diag) -> exist / science for X
# (devision-eval, recorded) -> report (model-free; --finalize when a stage failed or was stopped). watch: own process
# group, no progress 5 / 3 min, memory 1.5 GiB for 30 s, ETA, its deadline 35 s before the outer bound, TERM then KILL
# 10 s later. Hard limit start + 60 minutes (EXP68_LIMIT for CPU fixtures only), the last 120 s for the report. One start
# ever (runs/exp68.started, never removed), no resume, no retry, nothing chained after it.
#   nohup bash scripts/exp68/run.sh > runs/exp68.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
MARK=runs/exp68.started
[ ! -e "$MARK" ] || { log "$MARK exists: exp68 was already started once"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp68/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
for d in runs/exp68 runs/exp68-x; do [ ! -e "$d" ] || { log "$d exists"; exit 1; }; done
mkdir -p runs
DEADLINE=$(( $(date +%s) + ${EXP68_LIMIT:-3600} ))
{ echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"; } > "$MARK" \
  || { log "cannot write $MARK"; exit 1; }
STATUS=runs/exp68-status.txt
RESERVE=${EXP68_RESERVE:-120}
. scripts/exp38/steps.sh
PY="uv run python scripts/exp68/exp68.py"
WATCH="uv run python scripts/exp53/watch.py --deadline $(( DEADLINE - RESERVE - 35 )) --grace 10"
log "exp68: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
run_all() {
  local rc s
  step start $PY start --deadline "$DEADLINE"; rc=$?; codes=" start=$rc"; [ $rc = 0 ] || return $rc
  step train $WATCH --progress runs/exp68/progress-train.txt --log runs/exp68/watch-train.txt -- $PY train
  rc=$?; codes="$codes train=$rc"; [ $rc = 0 ] || return $rc
  step freeze $PY freeze; rc=$?; codes="$codes freeze=$rc"; [ $rc = 0 ] || return $rc
  step decide $WATCH --progress runs/exp68/progress-decide.txt --log runs/exp68/watch-decide.txt -- $PY decide
  rc=$?; codes="$codes decide=$rc"; [ $rc = 0 ] || return $rc
  mkdir -p runs/exp68-x/eval
  for s in exist=data/r13b/panel_exist.jsonl science=data/r13b/panel_science.jsonl; do
    step "eval:x:${s%%=*}" uv run devision-eval --checkpoint runs/exp68-x/stage2 --data "${s#*=}" --data-root data \
      --out runs/exp68-x/eval/${s%%=*}.json --role monitoring --no-temperature --max-questions 8
    rc=$?; codes="$codes x:${s%%=*}=$rc"; [ $rc = 0 ] || return $rc
    uv run python scripts/v04b/freeze.py record runs/exp68-x "${s%%=*}" || { status "eval:x:${s%%=*}" failed; codes="$codes record=1"; return 1; }
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
