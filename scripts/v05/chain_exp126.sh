#!/usr/bin/env bash
# Wait for the exp125 pipeline to exit, then run exp126 (exp122 train file, seed 124: seed-noise control). Keeps the GPU busy.
set -u
cd "$(dirname "$0")/../.."
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
echo "$(date '+%F %T') starting exp126"
uv run --no-sync devision-pipeline configs/exp126-v4-seed-control.yaml > runs/exp126-v4-seed-control.out 2>&1
echo "$(date '+%F %T') exp126 exit $?"
