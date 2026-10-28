#!/usr/bin/env bash
# Wait for the exp128 chain to exit, then run exp129 (exp127 data + coco-size replay, seed 23) and its laya-vision report. Keeps the GPU busy.
set -u
cd "$(dirname "$0")/../.."
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
echo "$(date '+%F %T') starting exp129-v4-scale-size-seed23"
uv run --no-sync devision-pipeline configs/exp129-v4-scale-size-seed23.yaml > runs/exp129-v4-scale-size-seed23.out 2>&1
echo "$(date '+%F %T') exp129 exit $?"
uv run --no-sync python scripts/lv_report.py runs/exp129-v4-scale-size-seed23/eval --prev runs/r21-soft/eval >> runs/exp129-v4-scale-size-seed23.out 2>&1
echo "$(date '+%F %T') lv_report exit $?"
