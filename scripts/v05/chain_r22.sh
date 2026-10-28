#!/usr/bin/env bash
# Wait for the exp126 pipeline to exit, then run round 22 (r22-guiego: exp124 data, seed 22) and the laya-vision report. Keeps the GPU busy.
set -u
cd "$(dirname "$0")/../.."
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
echo "$(date '+%F %T') starting r22-guiego"
uv run --no-sync devision-pipeline configs/r22-guiego.yaml > runs/r22-guiego.out 2>&1
echo "$(date '+%F %T') r22-guiego exit $?"
uv run --no-sync python scripts/lv_report.py runs/r22-guiego/eval --prev runs/r21-soft/eval >> runs/r22-guiego.out 2>&1
echo "$(date '+%F %T') lv_report exit $?"
