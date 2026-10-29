#!/usr/bin/env bash
# r23 (docs/experiments/2026-10-29-r23-v05-plan.md): after the given pids exit (the exp135 chain), pick lr_vision from
# exp135 vs exp132, train + evaluate r23-v05, then lv_report and paired comparisons against v0.4 and exp132.
set -u
cd "$(dirname "$0")/../.."
export UV_NO_SYNC=1 HF_HUB_OFFLINE=1
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
R=runs/r23-v05
uv run python scripts/v05/pick_r23.py
echo "$(date '+%F %T') pick exit $?"
uv run devision-pipeline configs/r23-v05.yaml > $R.out 2>&1
echo "$(date '+%F %T') r23 exit $?"
uv run python scripts/lv_report.py $R/eval --prev runs/r21-soft/eval >> $R.out 2>&1
echo "$(date '+%F %T') lv_report exit $?"
mkdir -p $R/eval/compare
for f in $R/eval/*.details.jsonl; do n=$(basename $f .details.jsonl); case $n in *.mismatched|*.reversed) continue;; esac
  for ref in r21-soft exp132-v4-offline-distill exp135-v4-lowlr-vision; do [ -f runs/$ref/eval/$n.details.jsonl ] || continue
    uv run devision-compare runs/$ref/eval/$n.details.jsonl $f --by source > $R/eval/compare/$n.vs-$ref.txt 2>/dev/null
  done
done
echo "$(date '+%F %T') r23 done"
