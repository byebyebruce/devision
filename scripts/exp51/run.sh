#!/usr/bin/env bash
# exp51 local region readout (docs/experiments/2026-10-18-exp51-local-readout-plan.md; critic EXP51-PREP-01). Only after
# the critic's model permission. Vision identity (X / T / C) and the region file checked, feature extraction for T then C
# from one frozen tower (scripts/exp51/extract.py, MPS), then the two heads, readout, T - C and report
# (scripts/exp51/probe.py, CPU). Hard deadline start + 30 minutes; each step bounded (TERM, KILL 15 s later); fresh only,
# no resume, no retry, nothing started after it.
#   nohup bash scripts/exp51/run.sh > runs/exp51.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp51/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
python3 scripts/exp51/vision_identity.py > /dev/null || { log "vision towers differ"; exit 1; }
want=$(python3 -c "import json; print(json.load(open('data/exp51/MANIFEST.json'))['regions_sha256'])")
[ "$(shasum -a 256 data/exp51/regions.json.gz | cut -d' ' -f1)" = "$want" ] || { log "data/exp51/regions.json.gz changed"; exit 1; }
[ ! -e runs/exp51 ] || { log "runs/exp51 exists"; exit 1; }
mkdir -p runs/exp51
DEADLINE=$(( $(date +%s) + 30 * 60 ))
echo "$DEADLINE" > runs/exp51.deadline
STATUS=runs/exp51-status.txt
RESERVE=30
. scripts/exp38/steps.sh
log "exp51: deadline $(date -r "$DEADLINE" '+%F %T')"
step extract uv run python scripts/exp51/extract.py || { log "extraction did not complete"; exit 1; }
step probe uv run python scripts/exp51/probe.py --device cpu
rc=$?
log "probe exit $rc"
exit "$rc"
