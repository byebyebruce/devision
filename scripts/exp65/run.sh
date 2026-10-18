#!/usr/bin/env bash
# exp65 equal-budget C vs W from EXP63's end (docs/experiments/2026-10-19-exp65-weight-plan.md; critic EXP65-PREP-01).
# Only after the critic's model permission (EXP65-RUN-01), with no other model task running. One model at a time:
# start (model-free) -> train_c -> freeze_c -> train_w -> freeze_w (training under scripts/exp53/watch.py, freeze tensors
# only) -> decide (MPS: EXP63 diag, then each arm fit / G / P / diag) -> exist / science per arm (devision-eval, recorded)
# -> report (model-free; --finalize when a stage failed or was stopped). watch: own process group, no progress 5 / 3 min,
# memory 1.5 GiB for 30 s, ETA, its deadline 35 s before the outer bound, TERM then KILL 10 s later. Hard limit start + 90
# minutes (EXP65_LIMIT for CPU fixtures only), the last 120 s for the report. One start ever (runs/exp65.started, never
# removed), no resume, no retry, nothing chained after it.
#   nohup bash scripts/exp65/run.sh > runs/exp65.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
MARK=runs/exp65.started
[ ! -e "$MARK" ] || { log "$MARK exists: exp65 was already started once"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp65/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
for d in runs/exp65 runs/exp65-c runs/exp65-w; do [ ! -e "$d" ] || { log "$d exists"; exit 1; }; done
mkdir -p runs
DEADLINE=$(( $(date +%s) + ${EXP65_LIMIT:-5400} ))
{ echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"; } > "$MARK" \
  || { log "cannot write $MARK"; exit 1; }
STATUS=runs/exp65-status.txt
RESERVE=${EXP65_RESERVE:-120}
. scripts/exp38/steps.sh
PY="uv run python scripts/exp65/exp65.py"
WATCH="uv run python scripts/exp53/watch.py --deadline $(( DEADLINE - RESERVE - 35 )) --grace 10"
log "exp65: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
run_all() {
  local rc a s
  step start $PY start --deadline "$DEADLINE"; rc=$?; codes=" start=$rc"; [ $rc = 0 ] || return $rc
  for a in c w; do
    step train_$a $WATCH --progress runs/exp65/progress-train_$a.txt --log runs/exp65/watch-train_$a.txt -- $PY train_$a
    rc=$?; codes="$codes train_$a=$rc"; [ $rc = 0 ] || return $rc
    step freeze_$a $PY freeze_$a; rc=$?; codes="$codes freeze_$a=$rc"; [ $rc = 0 ] || return $rc
  done
  step decide $WATCH --progress runs/exp65/progress-decide.txt --log runs/exp65/watch-decide.txt -- $PY decide
  rc=$?; codes="$codes decide=$rc"; [ $rc = 0 ] || return $rc
  for a in c w; do
    mkdir -p runs/exp65-$a/eval
    for s in exist=data/r13b/panel_exist.jsonl science=data/r13b/panel_science.jsonl; do
      step "eval:$a:${s%%=*}" uv run devision-eval --checkpoint runs/exp65-$a/stage2 --data "${s#*=}" --data-root data \
        --out runs/exp65-$a/eval/${s%%=*}.json --role monitoring --no-temperature --max-questions 8
      rc=$?; codes="$codes $a:${s%%=*}=$rc"; [ $rc = 0 ] || return $rc
      uv run python scripts/v04b/freeze.py record runs/exp65-$a "${s%%=*}" || { status "eval:$a:${s%%=*}" failed; codes="$codes record=1"; return 1; }
    done
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
