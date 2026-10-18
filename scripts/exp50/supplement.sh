#!/usr/bin/env bash
# exp50 supplement (critic EXP50-SUP-PREP-01; scripts/exp50/supplement.py). Only with the critic's separate permission,
# after the 90-minute chain has ended and its report is on disk. Reads arm C's fixed end (runs/exp50-c/stage2, code
# 52ff548, temperature 1, max 8 questions per request, the chain's own devision-eval command) on the sets the chain
# did not complete, in the chain's order, into runs/exp50-supplement/c/eval/; each recorded only if its summary and
# details check; then the two-part report. One process at a time, absolute limit start + 15 minutes (the last 60 s
# for the report), no retry, nothing started after it; the original run's files are never written.
#   nohup bash scripts/exp50/supplement.sh > runs/exp50-supplement.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp50/supplement.sh")
[ -z "$others" ] || { log "model processes are running (the chain must be over):"; echo "$others"; exit 1; }
# the decide / evaluation / readout code must be the chain's (52ff548), in the working tree as well
git diff --quiet 52ff548 -- src/devision/image.py src/devision/model.py src/devision/infer.py src/devision/train \
    scripts/exp50/report.py scripts/exp48/report.py scripts/exp42/report.py scripts/exp41/report.py scripts/v04b/freeze.py \
  || { log "the decide / readout code differs from 52ff548"; exit 1; }
mkdir -p runs/exp50-supplement
DEADLINE=$(( $(date +%s) + 15 * 60 ))
STATUS=runs/exp50-supplement/status.txt
RESERVE=60
. scripts/exp38/steps.sh
uv run python scripts/exp50/supplement.py plan > runs/exp50-supplement/plan.out
rc=$?
cat runs/exp50-supplement/plan.out
[ $rc = 3 ] && { log "nothing missing: no supplement"; exit 0; }
[ $rc = 0 ] || { log "preconditions not met (exit $rc)"; exit 1; }
log "supplement: deadline $(date -r "$DEADLINE" '+%F %T')"
for s in $(python3 -c "import json; print(' '.join(json.load(open('runs/exp50-supplement/START.json'))['missing']))"); do
  data=$(uv run python scripts/exp50/supplement.py data "$s")
  step "eval:$s" uv run devision-eval --checkpoint runs/exp50-c/stage2 --data "$data" --data-root data \
    --out runs/exp50-supplement/c/eval/$s.json --role monitoring --no-temperature --max-questions 8
  rc=$?
  [ $rc = 124 ] || [ $rc = 125 ] && break
  [ $rc = 0 ] && { uv run python scripts/exp50/supplement.py record "$s" > /dev/null || { status "record:$s" failed; log "record refused: $s"; }; }
done
final_report uv run python scripts/exp50/supplement.py report
rc=$?
log "supplement report (exit $rc)"
exit "$rc"
