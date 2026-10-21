#!/usr/bin/env bash
# exp35 (docs/experiments/2026-10-17-exp35-number-fill-plan.md): one read-only run of scripts/exp35/mlm_numbers.py
# on exp32's A2 checkpoint, only after every model process (exp32's chain, its training and evaluations) has exited
# and the critic released it; at most 10 minutes in total (585 s, then TERM, then KILL 15 s later); no retry;
# whatever it wrote stays (START.json is written before the first forward).
#   bash scripts/exp35/run.sh > runs/exp35.out 2>&1
set -u
cd "$(dirname "$0")/../.."
. scripts/exp35/guard.sh
others=$(model_processes)
[ -z "$others" ] || { echo "other model processes are running:"; echo "$others"; exit 1; }
want=8476145efc019d0acb03eec2031d9d78027fc15ed8d93ccd1c2e98edde00e297
have=$(shasum -a 256 runs/exp32-align/stage1/model.safetensors | cut -d' ' -f1)
[ "$have" = "$want" ] || { echo "checkpoint sha256 $have is not the released $want"; exit 1; }
echo "== $(date '+%F %T') exp35 on runs/exp32-align/stage1 ($have)"
HF_HUB_OFFLINE=1 run_capped 585 15 uv run python scripts/exp35/mlm_numbers.py --checkpoint runs/exp32-align/stage1 \
  --out runs/exp35-number-fill/report.json --device mps
rc=$?
echo "== $(date '+%F %T') exit $rc"
exit $rc
