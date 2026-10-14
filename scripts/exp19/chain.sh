#!/usr/bin/env bash
# exp19 on one Mac (docs/experiments/2026-10-14-exp19-vision-plan.md): exp18's chain (start check before the first
# update and no retry on a mismatch, stop at step 2000, crash retries, the readout's exit status) for the
# vision-unfreeze arm, then scripts/exp19/report.py against B = runs/exp16-r14b-s1.
#   nohup bash scripts/exp19/chain.sh > runs/exp19-vision.out 2>&1 &
cd "$(dirname "$0")/../.."
EXP18_TRAIN="uv run devision-pipeline configs/exp19-vision.yaml" EXP18_REPORT="uv run python scripts/exp19/report.py" \
  EXP18_STAGE=runs/exp19-vision/stage2 EXP18_NAME=exp19-vision exec bash scripts/exp18/chain.sh
