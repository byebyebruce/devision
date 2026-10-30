#!/usr/bin/env bash
# Wait for the exp119 pipeline to exit, then run exp120 (data frozen in data/v05/exp120 by
# scripts/data/v05/exp120_mix.py). Keeps the GPU busy.
set -u
cd "$(dirname "$0")/../.."
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
echo "$(date '+%F %T') starting exp120"
uv run --no-sync devision-pipeline configs/exp120-v4-depth-ego.yaml > runs/exp120-v4-depth-ego.out 2>&1
echo "$(date '+%F %T') exp120 exit $?"
