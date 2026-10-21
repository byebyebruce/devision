#!/usr/bin/env bash
# exp27 on one Mac (docs/experiments/2026-10-17-exp27-transfer-plan.md), evaluation only, no training:
# 1. export exp26's step-5000 weights from its resume state onto its init (scripts/exp27/export.py);
# 2. devision-eval v0.3 (= runs/r15-vision/stage2) and that export on the same sets: ScienceQA temperature / force
#    questions (official test split) with fixed mismatched pictures, concentration / attract-repel (report only),
#    the 24 transcribed questions, and r15's 7 protection panels;
# 3. scripts/exp27/report.py (rules written before running).
#   nohup bash scripts/exp27/chain.sh > runs/exp27.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
log() { echo "== $(date '+%F %T') $*"; }
run=runs/exp27-transfer
exp=$run/exp26-step5000
if [ ! -f "$exp/export.json" ]; then
  rm -rf "$exp"
  log "export"
  uv run python scripts/exp27/export.py runs/exp26-align/stage1 runs/exp26-read-compare/stage2/resume.pt "$exp" \
    || { log "export failed"; exit 1; }
fi
SETS="sqa_temp=data/exp21/dev_sqa_temp.jsonl sqa_force=data/exp21/dev_sqa_force.jsonl
sqa_temp_mism=data/exp21/dev_sqa_temp_mism.jsonl sqa_force_mism=data/exp21/dev_sqa_force_mism.jsonl
sqa_conc=data/v10/dev_sqa_conc.jsonl sqa_repel=data/v10/dev_sqa_repel.jsonl
sqa_text=data/exp21/sqa_text.jsonl sqa_blank=data/exp21/sqa_blank.jsonl
exist=data/r13b/panel_exist.jsonl science=data/r13b/panel_science.jsonl ext=data/ext/dev_ext.jsonl
count_wide=data/r13b/count_wide.jsonl count_narrow=data/v11/dev_count_transfer.jsonl flip=data/v6/dev_lr_pairs.jsonl
vsr=data/v5/dev_vsr.jsonl"
for m in v03:runs/r15-vision/stage2 exp26:$exp; do
  name=${m%%:*}; ckpt=${m#*:}
  mkdir -p "$run/eval/$name"
  for s in $SETS; do
    set=${s%%=*}; data=${s#*=}
    out=$run/eval/$name/$set.json
    [ -f "${out%.json}.details.jsonl" ] && continue
    log "$name $set"
    uv run devision-eval --checkpoint "$ckpt" --data "$data" --data-root data --out "$out" --role monitoring \
      > "${out%.json}.log" 2>&1 || { log "$name $set failed"; exit 1; }
  done
done
uv run python scripts/exp27/report.py > "$run/report.txt" 2>&1
rc=$?
cat "$run/report.txt"
log "report (exit $rc)"
exit $rc
