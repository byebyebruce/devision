#!/usr/bin/env bash
# Wait for the exp118 pipeline to exit, then: v0.4 / exp117 / exp118 baselines on the AITW-shopping screen-state
# GUI test, then exp119 (data frozen in data/v05/exp119 by scripts/data/v05/exp119_mix.py). Keeps the GPU busy.
set -u
cd "$(dirname "$0")/../.."
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
for c in r21-soft-guishop exp117-v4-gui-ego-science-guishop exp118-gui-screen-state-guishop; do
  uv run --no-sync devision-pipeline configs/$c.yaml > runs/v05-chain/$c.out 2>&1
  echo "$(date '+%F %T') $c exit $?"
done
echo "$(date '+%F %T') starting exp119"
uv run --no-sync devision-pipeline configs/exp119-v4-gui-phone-science.yaml > runs/exp119-v4-gui-phone-science.out 2>&1
echo "$(date '+%F %T') exp119 exit $?"
