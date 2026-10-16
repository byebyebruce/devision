#!/usr/bin/env bash
# exp24 on one Mac (docs/experiments/2026-10-16-exp24-read-align-plan.md):
# A. reading alignment from r15 (configs/exp24-align.yaml), then scripts/exp24/gate.py: exit 10 = the alignment did
#    not learn to read (held-out gap < 0.30), stop without phase B;
# B. exp22's fit test from the aligned checkpoint through exp18's chain (stop at step 2000, crash retries, the
#    readout's exit status), readout scripts/exp24/report.py against exp22.
#   nohup bash scripts/exp24/chain.sh > runs/exp24.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
log() { echo "== $(date '+%F %T') $*"; }
a=runs/exp24-align/stage1
if [ ! -f "$a/align_report.json" ]; then
  log "phase A: reading alignment"
  uv run devision-pipeline configs/exp24-align.yaml || { log "phase A failed"; exit 1; }
fi
uv run python scripts/exp24/gate.py "$a/align_report.json" > runs/exp24-align/gate.txt 2>&1
rc=$?
cat runs/exp24-align/gate.txt
[ $rc -eq 10 ] && { log "gate: phase B not run"; exit 10; }
[ $rc -eq 0 ] || { log "gate failed ($rc)"; exit $rc; }
EXP18_TRAIN="uv run devision-pipeline configs/exp24-fit.yaml" EXP18_REPORT="uv run python scripts/exp24/report.py" \
  EXP18_STAGE=runs/exp24-fit/stage2 EXP18_NAME=exp24-fit EXP18_SETS="val dev16 dev erased text" \
  exec bash scripts/exp18/chain.sh
