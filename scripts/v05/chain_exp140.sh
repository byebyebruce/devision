#!/usr/bin/env bash
# exp140 (docs/experiments/2026-10-30-exp140-board-scale.md): after the exp139 chain (given pids) ends, train +
# evaluate exp140 (r24 recipe with the passing board families scaled up), then lv_report, paired comparisons against
# r24 / v0.4, acceptance against v0.4, the Decision Index public vision index and criterion 1 against r24.
set -u
cd "$(dirname "$0")/../.."
export UV_NO_SYNC=1 HF_HUB_OFFLINE=1
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
[ -s data/v05/exp140/train.jsonl ] && [ -s data/v05/r24/calib.jsonl ] || { echo "$(date '+%F %T') exp140 data missing, stop"; exit 1; }
R=runs/exp140-board-scale
uv run devision-pipeline configs/exp140-board-scale.yaml > $R.out 2>&1
echo "$(date '+%F %T') exp140 exit $?"
uv run python scripts/lv_report.py $R/eval --prev runs/r24-v05/eval >> $R.out 2>&1
mkdir -p $R/eval/compare
for f in $R/eval/*.details.jsonl; do n=$(basename $f .details.jsonl); case $n in *.mismatched|*.reversed) continue;; esac
  for ref in r21-soft r24-v05; do [ -f runs/$ref/eval/$n.details.jsonl ] || continue
    uv run devision-compare runs/$ref/eval/$n.details.jsonl $f --by source > $R/eval/compare/$n.vs-$ref.txt 2>/dev/null
  done
done
uv run python scripts/v05/accept_r23.py --cand $R/eval --out $R/accept_vs_v0.4
echo "$(date '+%F %T') accept exit $?"
bash scripts/v05/board_run.sh $R/train exp140-board-scale
uv run python scripts/v05/judge_board.py exp140-board-scale caption,figure,depth,drive --ref r24-v05 > runs/v05-chain/judge_exp140.json
echo "$(date '+%F %T') exp140 chain done"
