#!/usr/bin/env bash
# Wait for the exp121 pipeline to exit, then run exp122 (data frozen in data/v05/exp122 by
# scripts/data/v05/exp122_mix.py). Keeps the GPU busy.
set -u
cd "$(dirname "$0")/../.."
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
echo "$(date '+%F %T') starting exp122"
uv run --no-sync devision-pipeline configs/exp122-v4-spatial-balance.yaml > runs/exp122-v4-spatial-balance.out 2>&1
echo "$(date '+%F %T') exp122 exit $?"
