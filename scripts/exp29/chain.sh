#!/usr/bin/env bash
# exp29 (docs/experiments/2026-10-17-exp29-perm-avg-plan.md): v0.3 on every cyclic option order of the development
# panels' choice questions (evaluation only), then scripts/exp29/report.py.
#   nohup bash scripts/exp29/chain.sh > runs/exp29.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
log() { echo "== $(date '+%F %T') $*"; }
run=runs/exp29-perm-avg
mkdir -p "$run/eval"
for p in science ext count_wide count_narrow flip sqa_diagram; do
  out=$run/eval/$p.json
  [ -f "${out%.json}.details.jsonl" ] && continue
  log "v03 $p"
  uv run devision-eval --checkpoint runs/r15-vision/stage2 --data "data/exp29/$p.jsonl" --data-root data --out "$out" \
    --role monitoring > "${out%.json}.log" 2>&1 || { log "$p failed"; exit 1; }
done
uv run python scripts/exp29/report.py > "$run/report.txt" 2>&1
rc=$?
cat "$run/report.txt"
log "report (exit $rc)"
exit $rc
