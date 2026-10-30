#!/usr/bin/env bash
# exp131 (docs/experiments/2026-10-28-exp131-chartqa-test.md): after the exp130 chain exits, evaluate v0.4, exp129 and
# (if chosen) the exp130 interpolation on the held-out ChartQA test split, then pair each against v0.4.
set -u
cd "$(dirname "$0")/../.."
export UV_NO_SYNC=1 HF_HUB_OFFLINE=1
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
D=data/v05/chartqa_test/test_chartqa.jsonl
[ -s $D ] || { echo "no $D"; exit 1; }
ev(){ [ -f $2/test_chartqa.json ] || { mkdir -p $2; uv run devision-eval --checkpoint $1 --data $D --data-root data --device cuda --role heldout --out $2/test_chartqa.json > /dev/null || echo "eval $1 failed"; }; }
ev runs/r21-soft/train runs/r21-soft/eval
ev runs/exp129-v4-scale-size-seed23/train runs/exp129-v4-scale-size-seed23/eval
a=$(sed -n 's/^CHOSEN //p' runs/exp130-interp/select.out 2>/dev/null)
[ -n "$a" ] && [ "$a" != none ] && ev runs/exp130-interp/a$a runs/exp130-interp/eval
mkdir -p runs/exp131-chartqa
for m in runs/exp129-v4-scale-size-seed23/eval runs/exp130-interp/eval; do
  [ -f $m/test_chartqa.details.jsonl ] || continue
  uv run devision-compare runs/r21-soft/eval/test_chartqa.details.jsonl $m/test_chartqa.details.jsonl --by kind > runs/exp131-chartqa/$(basename $(dirname $m)).vs_v04.json
done
echo "$(date '+%F %T') exp131 done"
