#!/usr/bin/env bash
# exp28 on one Mac (docs/experiments/2026-10-17-exp28-diagnose-plan.md), evaluation only, no training, no export:
# 1. the two intermediate checkpoints (exp24 phase A, exp26 phase A) on exp27's 15 sets (exp27 already holds v0.3
#    and exp26 phase B, reused as they are);
# 2. exp26 phase B (exp27's export) on control30 (positive control against its own step-5000 monitor records);
# 3. v0.3 and exp26 phase B on the field readings, the force questions with a general rule, and the "same"-option
#    wording x order set;
# 4. scripts/exp28/report.py (rules written before running).
#   nohup bash scripts/exp28/chain.sh > runs/exp28.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
log() { echo "== $(date '+%F %T') $*"; }
run=runs/exp28-diagnose
B=runs/exp27-transfer/exp26-step5000
[ -f "$B/export.json" ] || { log "missing $B (exp27's export)"; exit 1; }
STAGE_SETS="sqa_temp=data/exp21/dev_sqa_temp.jsonl sqa_force=data/exp21/dev_sqa_force.jsonl
sqa_temp_mism=data/exp21/dev_sqa_temp_mism.jsonl sqa_force_mism=data/exp21/dev_sqa_force_mism.jsonl
sqa_conc=data/v10/dev_sqa_conc.jsonl sqa_repel=data/v10/dev_sqa_repel.jsonl
sqa_text=data/exp21/sqa_text.jsonl sqa_blank=data/exp21/sqa_blank.jsonl
exist=data/r13b/panel_exist.jsonl science=data/r13b/panel_science.jsonl ext=data/ext/dev_ext.jsonl
count_wide=data/r13b/count_wide.jsonl count_narrow=data/v11/dev_count_transfer.jsonl flip=data/v6/dev_lr_pairs.jsonl
vsr=data/v5/dev_vsr.jsonl"
DIAG_SETS="fields=data/exp28/fields.jsonl fields_blank=data/exp28/fields_blank.jsonl force_rule=data/exp28/force_rule.jsonl equal_wording=data/exp28/equal_wording.jsonl"
evaluate() {   # $1 model name, $2 checkpoint, $3 sets
  mkdir -p "$run/eval/$1"
  for s in $3; do
    set=${s%%=*}; data=${s#*=}
    out=$run/eval/$1/$set.json
    [ -f "${out%.json}.details.jsonl" ] && continue
    log "$1 $set"
    uv run devision-eval --checkpoint "$2" --data "$data" --data-root data --out "$out" --role monitoring \
      > "${out%.json}.log" 2>&1 || { log "$1 $set failed"; exit 1; }
  done
}
evaluate exp24A runs/exp24-align/stage1 "$STAGE_SETS"
evaluate exp26A runs/exp26-align/stage1 "$STAGE_SETS"
evaluate exp26B "$B" "control30=data/exp28/control30.jsonl $DIAG_SETS"
evaluate v03 runs/r15-vision/stage2 "$DIAG_SETS"
uv run python scripts/exp28/report.py > "$run/report.txt" 2>&1
rc=$?
cat "$run/report.txt"
log "report (exit $rc)"
exit $rc
