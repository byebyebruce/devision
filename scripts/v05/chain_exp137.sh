#!/usr/bin/env bash
# After the r23 chain (given pids) ends: r23's Decision Index public vision index on the GPU, board-gap dev splits on
# v0.4 and r23, then exp137 (docs/experiments/2026-10-29-exp137-board-mix.md) train + evaluate, lv_report, paired
# comparisons against v0.4 and r23, and exp137's Decision Index index.
set -u
cd "$(dirname "$0")/../.."
export UV_NO_SYNC=1 HF_HUB_OFFLINE=1
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
bash scripts/v05/board_run.sh runs/r23-v05/train r23-v05
for c in exp137-v4-boarddev exp137-r23-boarddev; do uv run devision-pipeline configs/$c.yaml > runs/$c.out 2>&1; echo "$(date '+%F %T') $c exit $?"; done
R=runs/exp137-board-mix
uv run devision-pipeline configs/exp137-board-mix.yaml > $R.out 2>&1
echo "$(date '+%F %T') exp137 exit $?"
uv run python scripts/lv_report.py $R/eval --prev runs/r23-v05/eval >> $R.out 2>&1
echo "$(date '+%F %T') lv_report exit $?"
mkdir -p $R/eval/compare
for f in $R/eval/*.details.jsonl; do n=$(basename $f .details.jsonl); case $n in *.mismatched|*.reversed) continue;; esac
  for ref in r21-soft r23-v05; do [ -f runs/$ref/eval/$n.details.jsonl ] || continue
    uv run devision-compare runs/$ref/eval/$n.details.jsonl $f --by source > $R/eval/compare/$n.vs-$ref.txt 2>/dev/null
  done
done
bash scripts/v05/board_run.sh $R/train exp137-board-mix
echo "$(date '+%F %T') exp137 chain done"
