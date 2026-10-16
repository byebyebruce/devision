#!/usr/bin/env bash
# exp22 on one Mac (docs/experiments/2026-10-16-exp22-fit-plan.md): train through exp18's chain (stop once step
# 2000 is recorded, crash retries, the readout's exit status), readout scripts/exp22/report.py.
#   nohup bash scripts/exp22/chain.sh > runs/exp22-fit.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
EXP18_TRAIN="uv run devision-pipeline configs/exp22-fit.yaml" EXP18_REPORT="uv run python scripts/exp22/report.py" \
  EXP18_STAGE=runs/exp22-fit/stage2 EXP18_NAME=exp22-fit EXP18_SETS="val dev16 dev erased text" \
  exec bash scripts/exp18/chain.sh
