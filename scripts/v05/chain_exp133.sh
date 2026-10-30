#!/usr/bin/env bash
# exp133 (docs/experiments/2026-10-29-exp133-tiles.md): after the given pids exit (the exp132 chain), train + evaluate
# the tiles=2 arm (risky one first; on CUDA OOM both arms drop to micro batch 2), then the tiles=0 arm, then the
# paired comparisons, v0.4 / exp132 on the sets they lack, and the CPU latency of both arms on 40 Mind2Web screenshots.
set -u
cd "$(dirname "$0")/../.."
export UV_NO_SYNC=1 HF_HUB_OFFLINE=1
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
echo "$(date '+%F %T') waited for: $*"
run() { uv run devision-pipeline configs/$1.yaml > runs/$1.out 2>&1; echo "$(date '+%F %T') $1 exit $?"; }
run exp133-tiles2
if grep -q "out of memory" runs/exp133-tiles2/logs/train.log 2>/dev/null && [ ! -f runs/exp133-tiles2/train/config.json ]; then
  echo "$(date '+%F %T') tiles2 OOM at micro batch 4 -> 2 for both arms"
  sed -i 's/^    micro_batch: 4$/    micro_batch: 2/' configs/exp133-tiles2.yaml configs/exp133-tiles0.yaml
  uv run devision-pipeline configs/exp133-tiles2.yaml --force > runs/exp133-tiles2.out 2>&1; echo "$(date '+%F %T') tiles2 retry exit $?"
fi
run exp133-tiles0
# references the pair is read against: v0.4 on FUNSD dev, exp132 on ChartQA test
for spec in "runs/r21-soft:test_funsd:data/v05/funsd/derived/dev.jsonl" \
            "runs/exp132-v4-offline-distill:test_chartqa:data/v05/chartqa_test/test_chartqa.jsonl"; do
  IFS=: read -r R N P <<< "$spec"
  [ -f $R/eval/$N.json ] || uv run devision-eval --checkpoint $R/train --data $P --data-root data --device cuda \
    --role heldout --out $R/eval/$N.json > /dev/null 2>&1
done
mkdir -p runs/exp133-tiles2/eval/compare
for f in runs/exp133-tiles2/eval/*.details.jsonl; do n=$(basename $f .details.jsonl)
  for ref in exp133-tiles0 r21-soft; do [ -f runs/$ref/eval/$n.details.jsonl ] || continue
    uv run devision-compare runs/$ref/eval/$n.details.jsonl $f --by source > runs/exp133-tiles2/eval/compare/$n.vs-$ref.txt 2>/dev/null
  done
done
for arm in tiles0 tiles2; do
  uv run devision-eval --checkpoint runs/exp133-$arm/train --data data/v05/exp133/latency_m2w40.jsonl --data-root data \
    --device cpu --no-group --role heldout --out runs/exp133-$arm/latency_cpu.json > /dev/null 2>&1
done
echo "$(date '+%F %T') exp133 done"
