#!/usr/bin/env bash
# exp20 on one Mac (docs/experiments/2026-10-15-exp20-tokens256-plan.md):
# 1. fold r15's projector into a 256-token init (once; the folded checkpoint is the training init);
# 2. CPU latency, r15 and the folded init back to back on the v0.3 fixture (4 threads, one question per request):
#    the token count, not the weights, sets the cost, so the init stands for the trained arm;
# 3. train through exp18's chain (no retry on a start mismatch, stop at step 2000, crash retries, the readout's
#    exit status), monitoring post-r15-order-a's panels, then scripts/exp20/report.py against A.
#   nohup bash scripts/exp20/chain.sh > runs/exp20-tokens256.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
log() { echo "== $(date '+%F %T') $*"; }
init=runs/exp20-tokens256/init
if [ ! -f "$init/fold.json" ]; then
  rm -rf "$init"
  uv run python scripts/exp20/fold.py runs/r15-vision/stage2 "$init" || { log "fold failed"; exit 1; }
fi
mkdir -p runs/exp20-tokens256/latency
for m in r15:runs/r15-vision/stage2 tokens256:$init; do
  out=runs/exp20-tokens256/latency/${m%%:*}.json
  [ -f "$out" ] && continue
  log "CPU latency ${m%%:*}"
  OMP_NUM_THREADS=4 uv run devision-eval --checkpoint "${m#*:}" --data data/r12/cpu_latency.jsonl --data-root data \
    --device cpu --no-group --out "$out" > "${out%.json}.log" 2>&1 || { log "latency ${m%%:*} failed"; exit 1; }
done
EXP18_TRAIN="uv run devision-pipeline configs/exp20-tokens256.yaml" EXP18_REPORT="uv run python scripts/exp20/report.py" \
  EXP18_STAGE=runs/exp20-tokens256/stage2 EXP18_NAME=exp20-tokens256 \
  EXP18_SETS="val ext science exist flip count_wide count_narrow count_wide_reverse count_narrow_reverse" \
  exec bash scripts/exp18/chain.sh
