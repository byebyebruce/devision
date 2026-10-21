#!/usr/bin/env bash
# exp25 on one Mac (docs/experiments/2026-10-16-exp25-read-compare-plan.md): train through exp18's chain, stopped
# once step 5000 is recorded (crash retries, the readout's exit status), readout scripts/exp25/report.py.
#   nohup bash scripts/exp25/chain.sh > runs/exp25.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
EXP18_TRAIN="uv run devision-pipeline configs/exp25-read-compare.yaml" EXP18_REPORT="uv run python scripts/exp25/report.py" \
  EXP18_STAGE=runs/exp25-read-compare/stage2 EXP18_NAME=exp25-read-compare EXP18_SETS="val dev16 dev erased text" \
  EXP18_STEP=5000 exec bash scripts/exp18/chain.sh
