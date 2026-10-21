#!/usr/bin/env bash
# exp30 on one Mac (docs/experiments/2026-10-17-exp30-align-replay-plan.md): alignment with real caption replay
# (configs/exp30-align-replay.yaml), exp24's reading gate (recorded, not a stop: the protection result is reported
# either way), the 15 exp27/28 sets on the aligned checkpoint, then scripts/exp30/report.py.
#   nohup bash scripts/exp30/chain.sh > runs/exp30.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
log() { echo "== $(date '+%F %T') $*"; }
run=runs/exp30-align-replay
ckpt=$run/stage1
if [ ! -f "$ckpt/align_report.json" ]; then
  log "alignment with replay"
  uv run devision-pipeline configs/exp30-align-replay.yaml || { log "alignment failed"; exit 1; }
fi
uv run python scripts/exp24/gate.py "$ckpt/align_report.json" > "$run/gate.txt" 2>&1
cat "$run/gate.txt"
SETS="sqa_temp=data/exp21/dev_sqa_temp.jsonl sqa_force=data/exp21/dev_sqa_force.jsonl
sqa_temp_mism=data/exp21/dev_sqa_temp_mism.jsonl sqa_force_mism=data/exp21/dev_sqa_force_mism.jsonl
sqa_conc=data/v10/dev_sqa_conc.jsonl sqa_repel=data/v10/dev_sqa_repel.jsonl
sqa_text=data/exp21/sqa_text.jsonl sqa_blank=data/exp21/sqa_blank.jsonl
exist=data/r13b/panel_exist.jsonl science=data/r13b/panel_science.jsonl ext=data/ext/dev_ext.jsonl
count_wide=data/r13b/count_wide.jsonl count_narrow=data/v11/dev_count_transfer.jsonl flip=data/v6/dev_lr_pairs.jsonl
vsr=data/v5/dev_vsr.jsonl"
mkdir -p "$run/eval"
for s in $SETS; do
  set=${s%%=*}; data=${s#*=}
  out=$run/eval/$set.json
  [ -f "${out%.json}.details.jsonl" ] && continue
  log "exp30 $set"
  uv run devision-eval --checkpoint "$ckpt" --data "$data" --data-root data --out "$out" --role monitoring \
    > "${out%.json}.log" 2>&1 || { log "$set failed"; exit 1; }
done
uv run python scripts/exp30/report.py > "$run/report.txt" 2>&1
rc=$?
cat "$run/report.txt"
log "report (exit $rc)"
exit $rc
