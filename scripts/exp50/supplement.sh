#!/usr/bin/env bash
# exp50 supplement (critic EXP50-SUP-PREP-01, SUP-R1-01 / 02; scripts/exp50/supplement.py). Only with the critic's
# separate permission, after the 90-minute chain has ended and its report is on disk. Reads arm C's fixed end
# (runs/exp50-c/stage2, code 52ff548, temperature 1, max 8 questions per request, the chain's own devision-eval
# command) on the sets the chain did not complete, in the chain's order, into runs/exp50-supplement/c/eval/; each
# recorded only if its summary and details check; then the two-part report.
# - The repository HEAD must be exactly 52ff5484ea7b... (devision-eval writes the HEAD into each summary's code and the
#   record step requires 52ff548), and the decide / readout code in the working tree must equal it; the supplement
#   scripts themselves are untracked here and their sha256 go into START.json.
# - The whole entry -- plan, data lookup, evaluations, records, report -- runs under one absolute limit: TERM to the
#   entry's process tree at SUP_LIMIT (885 s), KILL SUP_GRACE (15 s) later, 900 s in all (scripts/exp35/guard.sh
#   run_capped). Inside, steps end by start + SUP_LIMIT - 60 and the report by start + SUP_LIMIT - 5.
# - argv, start, deadline, end and exit status go to runs/exp50-supplement/run.log. One process at a time, no
#   retry, nothing started after it; the original run's files are never written.
#   nohup bash scripts/exp50/supplement.sh > runs/exp50-supplement.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
HEAD_SHA=52ff5484ea7b36622a544f4134d176f0e3b5aeb8
OUT=runs/exp50-supplement
LIMIT=${SUP_LIMIT:-885}
GRACE=${SUP_GRACE:-15}

if [ -z "${SUP_INNER:-}" ]; then          # the outer entry: one bounded subtree, then the end record
  mkdir -p "$OUT"
  start=$(date +%s)
  echo "start $(date -r "$start" '+%F %T') argv: $0 $* limit ${LIMIT}s + ${GRACE}s" >> "$OUT/run.log"
  SUP_INNER=1 SUP_START=$start run_capped "$LIMIT" "$GRACE" bash "$0" "$@"
  rc=$?
  echo "end $(date '+%F %T') exit $rc$([ $rc = 124 ] && echo ' (stopped at the limit)')" >> "$OUT/run.log"
  log "supplement exit $rc"
  exit "$rc"
fi

others=$(model_processes | grep -v "scripts/exp50/supplement.sh")
[ -z "$others" ] || { log "model processes are running (the chain must be over):"; echo "$others"; exit 1; }
[ "$(git rev-parse HEAD)" = "$HEAD_SHA" ] || { log "HEAD is $(git rev-parse HEAD), not $HEAD_SHA: refused before any evaluation"; exit 1; }
# the decide / evaluation / readout code must be the chain's (52ff548), in the working tree as well
git diff --quiet "$HEAD_SHA" -- src/devision/image.py src/devision/model.py src/devision/infer.py src/devision/train \
    scripts/exp50/report.py scripts/exp48/report.py scripts/exp42/report.py scripts/exp41/report.py scripts/v04b/freeze.py \
  || { log "the decide / readout code differs from 52ff548"; exit 1; }
DEADLINE=$(( SUP_START + LIMIT - 5 ))    # final_report's bound; steps keep RESERVE more
STATUS=$OUT/status.txt
RESERVE=${SUP_RESERVE:-55}
. scripts/exp38/steps.sh
echo "deadline $(date -r "$DEADLINE" '+%F %T')" >> "$OUT/run.log"
uv run python scripts/exp50/supplement.py plan > "$OUT/plan.out"
rc=$?
cat "$OUT/plan.out"
[ $rc = 3 ] && { log "nothing missing: no supplement"; exit 0; }
[ $rc = 0 ] || { log "preconditions not met (exit $rc)"; exit 1; }
for s in $(python3 -c "import json; print(' '.join(json.load(open('$OUT/START.json'))['missing']))"); do
  data=$(uv run python scripts/exp50/supplement.py data "$s")
  step "eval:$s" uv run devision-eval --checkpoint runs/exp50-c/stage2 --data "$data" --data-root data \
    --out "$OUT/c/eval/$s.json" --role monitoring --no-temperature --max-questions 8
  rc=$?
  [ $rc = 124 ] || [ $rc = 125 ] && break
  [ $rc = 0 ] && { uv run python scripts/exp50/supplement.py record "$s" > /dev/null || { status "record:$s" failed; log "record refused: $s"; }; }
done
final_report uv run python scripts/exp50/supplement.py report
rc=$?
log "supplement report (exit $rc)"
exit "$rc"
