#!/usr/bin/env bash
# Wait for the exp117 pipeline to exit, then: v0.4 / exp116 baselines on the screen-state GUI test, then exp118
# (data frozen in data/v05/exp118 by scripts/data/v05/exp118_gui_state.py). Keeps the GPU busy between rounds.
set -u
cd "$(dirname "$0")/../.."
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
for c in r21-soft-guistate exp116-v4-gui-ego-guistate; do
  uv run --no-sync devision-pipeline configs/$c.yaml > runs/v05-chain/$c.out 2>&1
  echo "$(date '+%F %T') $c exit $?"
done
echo "$(date '+%F %T') starting exp118"
uv run --no-sync devision-pipeline configs/exp118-gui-screen-state.yaml > runs/exp118-gui-screen-state.out 2>&1
echo "$(date '+%F %T') exp118 exit $?"
