#!/usr/bin/env bash
# exp46 causal diagnosis of the frozen EXP45-G (docs/experiments/2026-10-18-exp46-causal-plan.md; critic EXP46-PREP-01..03).
# Only after the critic's permission. One bounded run of scripts/exp46/diagnose.py (48 units, 96 questions, six modes; no
# training, no weight written), hard deadline start + 10 minutes with the last 60 s for the model-free report; no retry.
#   nohup bash scripts/exp46/run.sh > runs/exp46.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp46/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
uv run python scripts/v04b/freeze.py check runs/exp45-g || { log "EXP45-G is not its frozen identity"; exit 1; }
want=$(python3 -c "import json; print(json.load(open('data/exp46/MANIFEST.json'))['sha256'])")
[ "$(shasum -a 256 data/exp46/units.jsonl | cut -d' ' -f1)" = "$want" ] || { log "data/exp46/units.jsonl changed"; exit 1; }
want=$(python3 -c "import json; print(json.load(open('data/exp45/MANIFEST.json'))['regions_readout']['sha256'])")
[ "$(shasum -a 256 data/exp45/regions_readout.jsonl | cut -d' ' -f1)" = "$want" ] || { log "regions_readout changed"; exit 1; }
[ ! -e runs/exp46 ] || { log "runs/exp46 exists"; exit 1; }
DEADLINE=$(( $(date +%s) + 10 * 60 ))
echo "$DEADLINE" > runs/exp46.deadline
log "exp46: deadline $(date -r "$DEADLINE" '+%F %T')"
run_capped $(( DEADLINE - $(date +%s) - 60 - 15 )) 15 uv run python scripts/exp46/diagnose.py
rc=$?
log "diagnosis exit $rc"
# always, without a model: the report from what was written -- "incomplete" with each mode's missing questions unless
# all six modes covered all 96 (critic EXP46-R1-03)
uv run python scripts/exp46/diagnose.py --finalize "$rc"
frc=$?
log "final report (exit $frc)"
exit "$frc"
