#!/usr/bin/env bash
# Wait for the exp115 pipeline and the running GUI data build to exit, then freeze the exp116 mix
# (copies GUI / ego dev+test into data/v05/exp116), evaluate the baselines (v0.4, exp115) on the new
# sets, and launch exp116. Keeps the GPU busy between rounds.
set -u
cd "$(dirname "$0")/../.."
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
uv run --no-sync python scripts/data/v05/exp116_mix.py --root data || { echo "mix failed"; exit 1; }
for cfg in r21-soft-guiego exp115-v4-chart-ui-count-guiego; do
  uv run --no-sync devision-pipeline configs/$cfg.yaml > runs/v05-chain/$cfg.out 2>&1
  echo "$(date '+%F %T') $cfg exit $?"
done
echo "$(date '+%F %T') starting exp116"
uv run --no-sync devision-pipeline configs/exp116-v4-gui-ego.yaml > runs/exp116-v4-gui-ego.out 2>&1
echo "$(date '+%F %T') exp116 exit $?"
