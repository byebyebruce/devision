#!/usr/bin/env bash
# Wait for the r22 chain (pipeline + lv_report) to exit, then run exp127 (r22 data + ego / AITW-general scale-up). Keeps the GPU busy.
set -u
cd "$(dirname "$0")/../.."
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
echo "$(date '+%F %T') starting exp127-v4-gui-ego-scale"
uv run --no-sync devision-pipeline configs/exp127-v4-gui-ego-scale.yaml > runs/exp127-v4-gui-ego-scale.out 2>&1
echo "$(date '+%F %T') exp127 exit $?"
