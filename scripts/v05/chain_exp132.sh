#!/usr/bin/env bash
# exp132 (docs/experiments/2026-10-29-exp132-v4-offline-distill.md): after the exp131 chain exits, evaluate v0.4 on the
# old-source rows of the exp129 training file (teacher pass), mix its answers 50/50 into their gold, train and evaluate
# with the exp129 settings (same seed 23), then lv_report against v0.4.
set -u
cd "$(dirname "$0")/../.."
export UV_NO_SYNC=1 HF_HUB_OFFLINE=1
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
D=data/v05/exp132; T=runs/exp132-v4-offline-distill/teacher
[ -s $D/teacher_input.jsonl ] || uv run python scripts/data/v05/exp132_distill.py subset || exit 1
mkdir -p $T
[ -f $T/v04.json ] || uv run devision-eval --checkpoint runs/r21-soft/train --data $D/teacher_input.jsonl --data-root data \
  --device cuda --role monitoring --out $T/v04.json > $T/v04.log 2>&1 || { echo "teacher pass failed"; exit 1; }
echo "$(date '+%F %T') teacher pass done"
[ -s $D/train.jsonl ] || uv run python scripts/data/v05/exp132_distill.py mix --teacher $T/v04.details.jsonl > $T/mix.log || exit 1
uv run devision-pipeline configs/exp132-v4-offline-distill.yaml > runs/exp132-v4-offline-distill.out 2>&1
echo "$(date '+%F %T') exp132 exit $?"
uv run python scripts/lv_report.py runs/exp132-v4-offline-distill/eval --prev runs/r21-soft/eval >> runs/exp132-v4-offline-distill.out 2>&1
echo "$(date '+%F %T') lv_report exit $?"
