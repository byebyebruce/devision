#!/usr/bin/env bash
# Model-free close-out of the v0.4 full evaluation (critic V04-K-FULL-MID-01-A). The active chain
# (devision-v04-freeze/scripts/v04full/run.sh) stops at its first failure or at its step deadline and never seals what
# was completed after its last seal. This watcher waits for the fixed chain PID to exit; if it exited non-zero (or was
# stopped at the limit) and no evaluation process of its tree is left, it runs only the existing
# `check_eval.py seal` and `check` (candidate; v0.3's items only when v0.3 was frozen by the chain) inside the original
# total deadline, and writes the missing / invalid list. It never starts run-extras, the pipeline or any model, never
# re-freezes, never removes or rewrites an output (seal only adds records for complete, valid outputs written after the
# freeze). An exit 0 chain has sealed and checked everything itself: nothing is run.
#   FINAL_PID=76293 FINAL_DEADLINE=1792425801 FINAL_REPO=/Users/tu/Documents/proj/devision-v04-freeze \
#     nohup bash scripts/v04full/finalize.sh > <repo>/runs/v04-k-finalize.out 2>&1 &
set -u
PID=${FINAL_PID:?}
DEADLINE=${FINAL_DEADLINE:?}
REPO=${FINAL_REPO:?}
POLL=${FINAL_POLL:-10}     # FINAL_MATCH: what an evaluation process of this chain has in its command line (default: the repo path, as in .venv/bin/python)
cd "$REPO" || exit 2
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh                                   # run_capped (TERM then KILL)
OUT=runs/v04-k-finalize.txt
C=configs/v04-full-eval.yaml; X=configs/v04-extra-evidence.yaml
C3=configs/v04-v03-confirm.yaml; X3=configs/v04-confirm-extras.yaml
CE="uv run python scripts/release/check_eval.py"
EVAL='eval_main|devision-eval|devision-pipeline|scripts/release/check_eval.py run-extras'
log "watching chain PID $PID; total deadline $(date -r "$DEADLINE" '+%F %T')"
while kill -0 "$PID" 2>/dev/null; do
  [ "$(date +%s)" -lt "$DEADLINE" ] || { log "total deadline reached while the chain still runs: nothing done"; exit 3; }
  sleep "$POLL"
done
rc=$(sed -n 's/^== .* exit \([0-9][0-9]*\)$/\1/p' runs/v04-k-full.out | tail -1)
log "chain exited: ${rc:-unknown (no exit line: killed?)}"
if [ "${rc:-}" = 0 ]; then
  log "the chain completed and sealed / checked itself: no close-out needed"
  { echo "chain exit 0: no close-out"; } > "$OUT"
  exit 0
fi
while [ -n "$(ps -axo pid=,command= | grep -E "$EVAL" | grep -v -e 'grep -E' | grep -F "${FINAL_MATCH:-$REPO}")" ]; do
  [ "$(date +%s)" -lt $(( DEADLINE - 30 )) ] || { log "evaluation processes still alive at the deadline: nothing sealed"; exit 4; }
  log "waiting for the chain's evaluation processes to end"
  sleep "$POLL"
done
left() { echo $(( DEADLINE - $(date +%s) - 5 )); }
run() {   # bounded by what is left of the total deadline; output kept
  local l; l=$(left)
  [ "$l" -gt 0 ] || { log "no time left: $*"; echo "not run (deadline): $*" >> "$OUT"; return 125; }
  log "$* (limit ${l}s)"
  { echo "### $*"; run_capped "$l" 5 "$@" 2>&1; echo "rc=$?"; } | tee -a "$OUT"
}
{ echo "close-out after chain exit ${rc:-unknown} at $(date '+%F %T')"; } > "$OUT"
run $CE seal "$C" --extras "$X"
run $CE check "$C" --extras "$X"
if [ -e runs/r15-vision/eval/IDENTITY.json ] && [ runs/r15-vision/eval/IDENTITY.json -nt runs/v04-k-full.started ]; then
  run $CE seal "$C3" --extras "$X3"
  run $CE check "$C3" --extras "$X3"
else
  echo "v0.3 confirmation items: not frozen by the chain -- all 6 missing (not evaluated here)" | tee -a "$OUT"
fi
log "close-out done: $OUT"
