#!/usr/bin/env bash
# Wait for the exp122 pipeline to exit, then run exp123 (data frozen in data/v05/exp123 by
# scripts/data/v05/exp123_mix.py). Keeps the GPU busy.
set -u
cd "$(dirname "$0")/../.."
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
echo "$(date '+%F %T') starting exp123"
uv run --no-sync devision-pipeline configs/exp123-v4-spatial-gui23.yaml > runs/exp123-v4-spatial-gui23.out 2>&1
echo "$(date '+%F %T') exp123 exit $?"
