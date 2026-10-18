#!/usr/bin/env bash
# exp43 oracle reference diagnosis (docs/experiments/2026-10-18-exp43-reference-plan.md; critic EXP43-PREP-01..04).
# Only after the critic's permission: the one fixed endpoint exp42-C on the six new sets at temperature 1 (7,128
# questions; main tune sets first), each a bounded devision-eval, then scripts/exp43/report.py (which reuses
# exp42-C's frozen name records). Hard deadline start + 15 minutes with the last 90 s for the report; no training,
# no temperature fit, no retry, nothing started after it.
#   nohup bash scripts/exp43/run.sh > runs/exp43.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp43/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
uv run python scripts/v04b/freeze.py check runs/exp42-c || { log "exp42-C is not its frozen identity"; exit 1; }
[ "$(shasum -a 256 runs/exp42-c/stage2/model.safetensors | cut -d' ' -f1)" = 6596907825f0b9c1ba9c0e9eb8dc136a91d6c2a48b331bc50ba34ea60ad36ca4 ] \
  || { log "exp42-C weights are not 65969078"; exit 1; }
SETS="tune_transcript tune_spatial_O tune_spatial_L fit_spatial_O fit_spatial_L fit_transcript"
for n in $SETS; do
  want=$(python3 -c "import json; print(json.load(open('data/exp43/MANIFEST.json'))['files']['$n']['sha256'])")
  [ "$(shasum -a 256 data/exp43/$n.jsonl | cut -d' ' -f1)" = "$want" ] || { log "data/exp43/$n.jsonl changed"; exit 1; }
done
[ ! -e runs/exp43 ] || { log "runs/exp43 exists"; exit 1; }
mkdir -p runs/exp43/eval
DEADLINE=$(( $(date +%s) + 15 * 60 ))
echo "$DEADLINE" > runs/exp43.deadline
STATUS=runs/exp43-status.txt
RESERVE=90
. scripts/exp38/steps.sh
log "exp43: exp42-C on six sets; deadline $(date -r "$DEADLINE" '+%F %T')"
for n in $SETS; do
  step "eval:$n" uv run devision-eval --checkpoint runs/exp42-c/stage2 --data data/exp43/$n.jsonl --data-root data \
    --out runs/exp43/eval/$n.json --role monitoring --no-temperature
  rc=$?
  { [ "$rc" = 124 ] || [ "$rc" = 125 ]; } && break
done
final_report uv run python scripts/exp43/report.py > runs/exp43-report.txt 2>&1
rc=$?
cat runs/exp43-report.txt
log "final report (exit $rc)"
exit "$rc"
