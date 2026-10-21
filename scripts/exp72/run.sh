#!/usr/bin/env bash
# exp72 raw vs K arms from EXP68-X, the same comparison-led plan (docs/experiments/2026-10-19-exp72-k-supervision-plan.md;
# critic EXP72-PREP-01 / -R1-01). Only after the critic's model permission (EXP72-RUN-01), with no other model task running.
# One model at a time: start (model-free) -> fid (MPS: EXP68-X reproduces EXP68's 48 raw records, else stop) -> train_raw ->
# freeze_raw -> train_k -> freeze_k -> decide (required: slice / diagnostic / G, both arms) -> exist / science per arm
# (required; devision-eval, recorded) -> decide_fit (the old fit, report only, last) -> report (model-free; --finalize when a
# stage failed or was stopped). watch: own process group, no progress 5 / 3 min, memory 1.5 GiB for 30 s, ETA, its
# deadline 35 s before the outer bound, TERM then KILL 10 s later. Hard limit start + 45 minutes (EXP72_LIMIT for CPU
# fixtures only), the last 120 s for the report. One start ever (runs/exp72.started, never removed), no resume, no retry,
# nothing chained after it.
#   nohup bash scripts/exp72/run.sh > runs/exp72.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
MARK=runs/exp72.started
[ ! -e "$MARK" ] || { log "$MARK exists: exp72 was already started once"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp72/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
for d in runs/exp72 runs/exp72-raw runs/exp72-k; do [ ! -e "$d" ] || { log "$d exists"; exit 1; }; done
mkdir -p runs
DEADLINE=$(( $(date +%s) + ${EXP72_LIMIT:-2700} ))
{ echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"; } > "$MARK" \
  || { log "cannot write $MARK"; exit 1; }
STATUS=runs/exp72-status.txt
RESERVE=${EXP72_RESERVE:-120}
. scripts/exp38/steps.sh
PY="uv run python scripts/exp72/exp72.py"
WATCH="uv run python scripts/exp53/watch.py --deadline $(( DEADLINE - RESERVE - 35 )) --grace 10"
log "exp72: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
run_all() {
  local rc a s
  step start $PY start --deadline "$DEADLINE"; rc=$?; codes=" start=$rc"; [ $rc = 0 ] || return $rc
  step fid $WATCH --progress runs/exp72/progress-fid.txt --log runs/exp72/watch-fid.txt -- $PY fid
  rc=$?; codes="$codes fid=$rc"; [ $rc = 0 ] || return $rc
  for a in raw k; do
    step train_$a $WATCH --progress runs/exp72/progress-train_$a.txt --log runs/exp72/watch-train_$a.txt -- $PY train_$a
    rc=$?; codes="$codes train_$a=$rc"; [ $rc = 0 ] || return $rc
    step freeze_$a $PY freeze_$a; rc=$?; codes="$codes freeze_$a=$rc"; [ $rc = 0 ] || return $rc
  done
  step decide $WATCH --progress runs/exp72/progress-decide.txt --log runs/exp72/watch-decide.txt -- $PY decide
  rc=$?; codes="$codes decide=$rc"; [ $rc = 0 ] || return $rc
  for a in raw k; do
    mkdir -p runs/exp72-$a/eval
    for s in exist=data/r13b/panel_exist.jsonl science=data/r13b/panel_science.jsonl; do
      step "eval:$a:${s%%=*}" uv run devision-eval --checkpoint runs/exp72-$a/stage2 --data "${s#*=}" --data-root data \
        --out runs/exp72-$a/eval/${s%%=*}.json --role monitoring --no-temperature --max-questions 8
      rc=$?; codes="$codes $a:${s%%=*}=$rc"; [ $rc = 0 ] || return $rc
      uv run python scripts/v04b/freeze.py record runs/exp72-$a "${s%%=*}" || { status "eval:$a:${s%%=*}" failed; codes="$codes record=1"; return 1; }
    done
  done
  step decide_fit $WATCH --progress runs/exp72/progress-decide_fit.txt --log runs/exp72/watch-decide_fit.txt -- $PY decide_fit
  rc=$?; codes="$codes decide_fit=$rc"; return $rc
}
run_all
rc=$?
log "stages:$codes"
if [ $rc = 0 ]; then final_report $PY report; else final_report $PY report --finalize "$codes"; fi
rrc=$?
log "report exit $rrc"
[ $rc = 0 ] && rc=$rrc
exit "$rc"
