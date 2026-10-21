#!/usr/bin/env bash
# exp21 on one Mac (docs/experiments/2026-10-15-exp21-reading-plan.md):
# 1. precheck on r15 with devision-eval (synthetic reading + its erased / text controls + the 24 transcribed
#    ScienceQA questions); scripts/exp21/precheck.py applies the rules written before running and exits 10 when
#    training would not answer anything new (r15 already reads, the text comparison itself fails, or the
#    synthetic data leaks the answer) -- the chain then stops without training;
# 2. otherwise train through exp18's chain (stop at step 2000, crash retries, the readout's exit status),
#    readout scripts/exp21/report.py.
#   nohup bash scripts/exp21/chain.sh > runs/exp21-reading.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
log() { echo "== $(date '+%F %T') $*"; }
pre=runs/exp21-reading/precheck
mkdir -p "$pre"
for s in dev_syn dev_syn_erased dev_syn_text sqa_text sqa_blank; do
  [ -f "$pre/$s.details.jsonl" ] && continue
  log "precheck $s"
  uv run devision-eval --checkpoint runs/r15-vision/stage2 --data "data/exp21/$s.jsonl" --data-root data \
    --out "$pre/$s.json" --role monitoring > "$pre/$s.log" 2>&1 || { log "precheck $s failed"; exit 1; }
done
uv run python scripts/exp21/precheck.py "$pre" > "$pre/precheck.txt" 2>&1
rc=$?
cat "$pre/precheck.txt"
[ $rc -eq 10 ] && { log "precheck: no training (see $pre/precheck.txt)"; exit 10; }
[ $rc -eq 0 ] || { log "precheck failed ($rc)"; exit $rc; }
EXP18_TRAIN="uv run devision-pipeline configs/exp21-reading.yaml" EXP18_REPORT="uv run python scripts/exp21/report.py" \
  EXP18_STAGE=runs/exp21-reading/stage2 EXP18_NAME=exp21-reading \
  EXP18_SETS="val syn syn_erased syn_text sqa_temp sqa_force sqa_temp_mism sqa_force_mism sqa_text sqa_blank exist science ext count_wide count_narrow flip vsr" \
  exec bash scripts/exp18/chain.sh
