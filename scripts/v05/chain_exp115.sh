#!/usr/bin/env bash
# Wait for the exp114 pipeline to exit, then (1) attribute exp114's CountBench drop by
# evaluating exp81-chart on CountBench, (2) launch exp115. Keeps the GPU busy between rounds.
set -u
cd "$(dirname "$0")/../.."
WAIT_PID=${1:-530492}
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
echo "$(date '+%F %T') exp114 pipeline exited"
mkdir -p runs/exp81-chart/eval
[ -f runs/exp81-chart/eval/bench_countbench.json ] || uv run --no-sync devision-eval --checkpoint runs/exp81-chart/train \
  --data data/r13/bench/countbench.jsonl --data-root data --role heldout --device cuda \
  --out runs/exp81-chart/eval/bench_countbench.json
echo "$(date '+%F %T') exp81 countbench done, starting exp115"
uv run --no-sync devision-pipeline configs/exp115-v4-chart-ui-count.yaml > runs/exp115-v4-chart-ui-count.out 2>&1
echo "$(date '+%F %T') exp115 exit $?"
