#!/usr/bin/env bash
# After the exp137 chain (given pids) ends: second-batch board dev splits on v0.4 and r23, then exp138
# (docs/experiments/2026-10-29-exp138-board-mix2.md) train + evaluate, lv_report, paired comparisons against v0.4,
# r23 and exp137, and exp138's Decision Index public vision index.
set -u
cd "$(dirname "$0")/../.."
export UV_NO_SYNC=1 HF_HUB_OFFLINE=1
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
[ -s data/v05/exp138/MANIFEST.json ] || { echo "$(date '+%F %T') data/v05/exp138 not built, stop"; exit 1; }
for c in exp138-v4-boarddev exp138-r23-boarddev; do uv run devision-pipeline configs/$c.yaml > runs/$c.out 2>&1; echo "$(date '+%F %T') $c exit $?"; done
R=runs/exp138-board-mix2
uv run devision-pipeline configs/exp138-board-mix2.yaml > $R.out 2>&1
echo "$(date '+%F %T') exp138 exit $?"
uv run python scripts/lv_report.py $R/eval --prev runs/r23-v05/eval >> $R.out 2>&1
echo "$(date '+%F %T') lv_report exit $?"
mkdir -p $R/eval/compare
for f in $R/eval/*.details.jsonl; do n=$(basename $f .details.jsonl); case $n in *.mismatched|*.reversed) continue;; esac
  for ref in r21-soft r23-v05 exp137-board-mix; do [ -f runs/$ref/eval/$n.details.jsonl ] || continue
    uv run devision-compare runs/$ref/eval/$n.details.jsonl $f --by source > $R/eval/compare/$n.vs-$ref.txt 2>/dev/null
  done
done
bash scripts/v05/board_run.sh $R/train exp138-board-mix2
echo "$(date '+%F %T') exp138 chain done"
