#!/usr/bin/env bash
# exp49 frozen region readout (docs/experiments/2026-10-18-exp49-region-readout-plan.md; critic EXP49-PREP-01..04). Only
# after the critic's permission. Feature extraction from EXP48-X's frozen end (scripts/exp49/extract.py), then the two
# readout heads, the readout and the report (scripts/exp49/probe.py). Hard deadline start + 30 minutes; each step bounded
# (TERM, KILL 15 s later); no resume, no retry, nothing started after it.
#   nohup bash scripts/exp49/run.sh > runs/exp49.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp49/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
uv run python scripts/v04b/freeze.py check runs/exp48-x || { log "EXP48-X is not its frozen identity"; exit 1; }
want=$(python3 -c "import json; print(json.load(open('data/exp49/MANIFEST.json'))['sha256'])")
[ "$(shasum -a 256 data/exp49/questions.jsonl | cut -d' ' -f1)" = "$want" ] || { log "data/exp49/questions.jsonl changed"; exit 1; }
[ ! -e runs/exp49 ] || { log "runs/exp49 exists"; exit 1; }
mkdir -p runs/exp49
DEADLINE=$(( $(date +%s) + 30 * 60 ))
echo "$DEADLINE" > runs/exp49.deadline
STATUS=runs/exp49-status.txt
RESERVE=30
. scripts/exp38/steps.sh
log "exp49: deadline $(date -r "$DEADLINE" '+%F %T')"
step extract uv run python scripts/exp49/extract.py || { log "extraction did not complete"; exit 1; }
step probe uv run python scripts/exp49/probe.py --device cpu
rc=$?
log "probe exit $rc"
exit "$rc"
