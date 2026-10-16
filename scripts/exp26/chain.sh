#!/usr/bin/env bash
# exp26 on one Mac (docs/experiments/2026-10-16-exp26-bind-align-plan.md):
# A. panel-binding alignment from exp24's reading-aligned checkpoint (configs/exp26-align.yaml), then
#    scripts/exp24/gate.py (mixed caption completion, real minus shuffled picture >= 0.30; exit 10 = stop);
# B. exp25 unchanged from the phase-A checkpoint through exp18's chain, stopped once step 5000 is recorded
#    (crash retries, the readout's exit status), readout scripts/exp26/report.py against exp25.
#   nohup bash scripts/exp26/chain.sh > runs/exp26.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
log() { echo "== $(date '+%F %T') $*"; }
a=runs/exp26-align/stage1
if [ ! -f "$a/align_report.json" ]; then
  log "phase A: panel-binding alignment"
  uv run devision-pipeline configs/exp26-align.yaml || { log "phase A failed"; exit 1; }
fi
uv run python scripts/exp24/gate.py "$a/align_report.json" > runs/exp26-align/gate.txt 2>&1
rc=$?
cat runs/exp26-align/gate.txt
[ $rc -eq 10 ] && { log "gate: phase B not run"; exit 10; }
[ $rc -eq 0 ] || { log "gate failed ($rc)"; exit $rc; }
EXP18_TRAIN="uv run devision-pipeline configs/exp26-fit.yaml" EXP18_REPORT="uv run python scripts/exp26/report.py" \
  EXP18_STAGE=runs/exp26-read-compare/stage2 EXP18_NAME=exp26-read-compare EXP18_SETS="val dev16 dev erased text" \
  EXP18_STEP=5000 exec bash scripts/exp18/chain.sh
