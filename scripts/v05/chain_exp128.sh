#!/usr/bin/env bash
# Wait for the exp127 chain to exit, then run exp128 (r22 data + coco-size replay, seed 22) and its laya-vision report. Keeps the GPU busy.
set -u
cd "$(dirname "$0")/../.."
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
echo "$(date '+%F %T') starting exp128-v4-size-replay"
uv run --no-sync devision-pipeline configs/exp128-v4-size-replay.yaml > runs/exp128-v4-size-replay.out 2>&1
echo "$(date '+%F %T') exp128 exit $?"
uv run --no-sync python scripts/lv_report.py runs/exp128-v4-size-replay/eval --prev runs/r22-guiego/eval >> runs/exp128-v4-size-replay.out 2>&1
echo "$(date '+%F %T') lv_report exit $?"
