#!/usr/bin/env bash
# Wait for the exp120 pipeline to exit, then run exp121 (data frozen in data/v05/exp121 by
# scripts/data/v05/exp121_mix.py). Keeps the GPU busy.
set -u
cd "$(dirname "$0")/../.."
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
echo "$(date '+%F %T') starting exp121"
uv run --no-sync devision-pipeline configs/exp121-v4-gui-scale.yaml > runs/exp121-v4-gui-scale.out 2>&1
echo "$(date '+%F %T') exp121 exit $?"
