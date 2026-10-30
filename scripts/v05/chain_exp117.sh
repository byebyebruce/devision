#!/usr/bin/env bash
# Wait for the exp116 pipeline to exit, then launch exp117 (exp116 mix + exp114 science rows; data already
# frozen in data/v05/exp117 by scripts/data/v05/exp117_mix.py). Keeps the GPU busy between rounds.
set -u
cd "$(dirname "$0")/../.."
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
echo "$(date '+%F %T') starting exp117"
uv run --no-sync devision-pipeline configs/exp117-v4-gui-ego-science.yaml > runs/exp117-v4-gui-ego-science.out 2>&1
echo "$(date '+%F %T') exp117 exit $?"
