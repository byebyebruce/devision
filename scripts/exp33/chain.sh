#!/usr/bin/env bash
# exp33 on one Mac (docs/experiments/2026-10-17-exp33-decision-protect-plan.md): alignment with caption replay and
# old decision supervision (configs/exp33-align-decide.yaml), exp24's reading gate (recorded, not a stop), the
# same 15 sets as exp30 on the aligned checkpoint, then scripts/exp33/report.py. One absolute 2 h deadline
# (runs/exp33.deadline, kept across restarts) covers the alignment and every evaluation: each runs under `capped`,
# which stops the whole process tree at the deadline and starts nothing after it. Not started before the critic
# chooses exp33 after exp32.
#   nohup bash scripts/exp33/chain.sh > runs/exp33.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
log() { echo "== $(date '+%F %T') $*"; }
run=runs/exp33-align-decide
ckpt=$run/stage1
DEADLINE_FILE=runs/exp33.deadline
[ -f "$DEADLINE_FILE" ] || echo $(( $(date +%s) + 120 * 60 )) > "$DEADLINE_FILE"
DEADLINE=$(cat "$DEADLINE_FILE")
log "deadline $(date -r "$DEADLINE" '+%F %T')"
kill_tree() { local c; for c in $(pgrep -P "$1"); do kill_tree "$c"; done; kill -TERM "$1" 2>/dev/null; }
capped() {
  [ "$(date +%s)" -lt "$DEADLINE" ] || { log "past the deadline, not starting: $*"; exit 2; }
  "$@" &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
      log "deadline reached, stopping: $*"; kill_tree "$pid"; wait "$pid"; exit 2
    fi
    sleep "${CAPPED_POLL:-20}"
  done
  wait "$pid"
}
if [ ! -f "$ckpt/align_report.json" ]; then
  log "alignment with caption replay and decision supervision"
  capped uv run devision-pipeline configs/exp33-align-decide.yaml || { log "alignment failed"; exit 1; }
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
  log "exp33 $set"
  capped uv run devision-eval --checkpoint "$ckpt" --data "$data" --data-root data --out "$out" --role monitoring \
    > "${out%.json}.log" 2>&1 || { log "$set failed"; exit 1; }
done
uv run python scripts/exp33/report.py > "$run/report.txt" 2>&1
rc=$?
cat "$run/report.txt"
log "report (exit $rc)"
exit $rc
