#!/usr/bin/env bash
# exp139 (docs/experiments/2026-10-29-exp139-infographic.md): after the r24 chain (given pids) ends, evaluate r24 and
# v0.4 on the full infographic family's dev, train + evaluate exp139 (r24 + full infographic family), then lv_report,
# paired comparisons against r24 / v0.4, acceptance against v0.4 and the Decision Index public vision index.
set -u
cd "$(dirname "$0")/../.."
export UV_NO_SYNC=1 HF_HUB_OFFLINE=1
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
[ -s data/v05/exp139/train.jsonl ] && [ -s data/v05/exp139/calib.jsonl ] || { echo "$(date '+%F %T') exp139 data missing, stop"; exit 1; }
D=data/v05/exp139/dev_board_infographic_full.jsonl
for ref in r24-v05 r21-soft; do o=runs/$ref/eval/dev_board_infographic_full.json; [ -s $o ] && continue
  uv run devision-eval --checkpoint runs/$ref/train --data $D --data-root data --role heldout --device cuda --out $o
done
echo "$(date '+%F %T') refs on infographic_full done"
R=runs/exp139-infographic
uv run devision-pipeline configs/exp139-infographic.yaml > $R.out 2>&1
echo "$(date '+%F %T') exp139 exit $?"
uv run python scripts/lv_report.py $R/eval --prev runs/r24-v05/eval >> $R.out 2>&1
mkdir -p $R/eval/compare
for f in $R/eval/*.details.jsonl; do n=$(basename $f .details.jsonl); case $n in *.mismatched|*.reversed) continue;; esac
  for ref in r21-soft r24-v05; do [ -f runs/$ref/eval/$n.details.jsonl ] || continue
    uv run devision-compare runs/$ref/eval/$n.details.jsonl $f --by source > $R/eval/compare/$n.vs-$ref.txt 2>/dev/null
  done
done
uv run python scripts/v05/accept_r23.py --cand $R/eval --out $R/accept_vs_v0.4
echo "$(date '+%F %T') accept exit $?"
bash scripts/v05/board_run.sh $R/train exp139-infographic
uv run python scripts/v05/judge_board.py exp139-infographic infographic_full,infographic --ref r24-v05 > runs/v05-chain/judge_exp139.json
echo "$(date '+%F %T') exp139 chain done"
