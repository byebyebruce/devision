#!/usr/bin/env bash
# exp35 (docs/experiments/2026-10-17-exp35-number-fill-plan.md): one read-only run of scripts/exp35/mlm_numbers.py
# on exp32's A2 checkpoint, only after exp32's chain has exited and the critic released it; at most 10 minutes
# (timeout), no retry; whatever it wrote stays.
#   bash scripts/exp35/run.sh > runs/exp35.out 2>&1
set -u
cd "$(dirname "$0")/../.."
pgrep -f "scripts/exp32/chain.sh" >/dev/null && { echo "exp32 is still running"; exit 1; }
want=8476145efc019d0acb03eec2031d9d78027fc15ed8d93ccd1c2e98edde00e297
have=$(shasum -a 256 runs/exp32-align/stage1/model.safetensors | cut -d' ' -f1)
[ "$have" = "$want" ] || { echo "checkpoint sha256 $have is not the released $want"; exit 1; }
echo "== $(date '+%F %T') exp35 on runs/exp32-align/stage1 ($have)"
kill_tree() { local c; for c in $(pgrep -P "$1"); do kill_tree "$c"; done; kill -TERM "$1" 2>/dev/null; }
HF_HUB_OFFLINE=1 uv run python scripts/exp35/mlm_numbers.py --checkpoint runs/exp32-align/stage1 \
  --out runs/exp35-number-fill/report.json --device mps &
pid=$!
end=$(( $(date +%s) + 600 ))                 # 10 minutes, then the whole process tree is stopped
while kill -0 "$pid" 2>/dev/null; do
  if [ "$(date +%s)" -ge "$end" ]; then echo "== 10 minute limit reached, stopping"; kill_tree "$pid"; break; fi
  sleep 5
done
wait "$pid"
rc=$?
echo "== $(date '+%F %T') exit $rc"
exit $rc
