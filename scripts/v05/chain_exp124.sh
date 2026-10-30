#!/usr/bin/env bash
# Wait for the exp123 pipeline to exit, then run exp124 (data frozen in data/v05/exp124 by
# scripts/data/v05/exp124_mix.py). Keeps the GPU busy.
set -u
cd "$(dirname "$0")/../.."
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
echo "$(date '+%F %T') starting exp124"
uv run --no-sync devision-pipeline configs/exp124-v4-exist-replay.yaml > runs/exp124-v4-exist-replay.out 2>&1
echo "$(date '+%F %T') exp124 exit $?"
