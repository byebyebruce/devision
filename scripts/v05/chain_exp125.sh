#!/usr/bin/env bash
# Wait for the exp124 pipeline to exit, then run exp125 (data frozen in data/v05/exp125 by
# scripts/data/v05/exp124_mix.py --seed 125 --exist 3000 --vqav2 1500 --out v05/exp125). Keeps the GPU busy.
set -u
cd "$(dirname "$0")/../.."
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
echo "$(date '+%F %T') starting exp125"
uv run --no-sync devision-pipeline configs/exp125-v4-exist-replay-half.yaml > runs/exp125-v4-exist-replay-half.out 2>&1
echo "$(date '+%F %T') exp125 exit $?"
