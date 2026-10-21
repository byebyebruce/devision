#!/usr/bin/env bash
# exp37 on one Mac (docs/experiments/2026-10-17-exp37-digit-size-plan.md; critic EXP37-PREP-01 / READOUT-01): arm S
# (small digits) then arm L (16 px digits), each 1,000 steps from exp32's A2 end with exp36-B's full-number mask;
# then each model's full-number readout on the 600 tune pictures drawn S, drawn L and as originally drawn, and on its
# own 200-picture fit slice; the 7 protection panels of each; scripts/exp37/report.py. One absolute 2 h deadline (runs/exp37.deadline, kept across restarts) covers everything:
# each step runs under `capped`, stopped with its whole process tree (TERM, then KILL 15 s later) at the deadline;
# nothing is retried and no decision stage follows.
#   nohup bash scripts/exp37/chain.sh > runs/exp37.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1   # the MLM head of the readouts comes from the local cache only
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp37/chain.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
want=8476145efc019d0acb03eec2031d9d78027fc15ed8d93ccd1c2e98edde00e297
have=$(shasum -a 256 runs/exp32-align/stage1/model.safetensors | cut -d' ' -f1)
[ "$have" = "$want" ] || { log "A2 sha256 $have is not $want"; exit 1; }
log "start checkpoint runs/exp32-align/stage1 sha256 $have; data S $(shasum -a 256 data/exp37/align_mix_S.jsonl | cut -c1-16) L $(shasum -a 256 data/exp37/align_mix_L.jsonl | cut -c1-16)"
DEADLINE_FILE=runs/exp37.deadline
[ -f "$DEADLINE_FILE" ] || echo $(( $(date +%s) + 120 * 60 )) > "$DEADLINE_FILE"
DEADLINE=$(cat "$DEADLINE_FILE")
log "deadline $(date -r "$DEADLINE" '+%F %T')"
capped() {
  local left=$(( DEADLINE - $(date +%s) - 15 ))
  [ "$left" -gt 0 ] || { log "past the deadline, not starting: $*"; exit 2; }
  run_capped "$left" 15 "$@"
  local rc=$?
  [ "$rc" = 124 ] && { log "deadline reached, stopped: $*"; exit 2; }
  return $rc
}
for arm in s l; do
  if [ ! -f runs/exp37-$arm/stage1/align_report.json ]; then
    log "arm $arm"
    capped uv run devision-pipeline configs/exp37-$arm.yaml || { log "arm $arm failed"; exit 1; }
  fi
done
for arm in s l; do
  A=$(echo $arm | tr a-z A-Z)
  for split in tune_S:data/exp37/tune_S.jsonl tune_L:data/exp37/tune_L.jsonl tune_orig:data/real_layout/captions_tune.jsonl \
               fit:data/exp37/fit_$A.jsonl; do
    name=${split%%:*}; data=${split#*:}; out=runs/exp37-$arm/numbers_$name
    [ -f "$out/numbers.jsonl" ] && continue
    log "readout $arm $name"
    capped uv run python scripts/exp36/readout.py --checkpoint runs/exp37-$arm/stage1 --data "$data" --out "$out" \
      --full-numbers-only > "$out.log" 2>&1 || { log "readout $arm $name failed"; exit 1; }
    tail -1 "$out.log"
  done
done
SETS="exist=data/r13b/panel_exist.jsonl science=data/r13b/panel_science.jsonl ext=data/ext/dev_ext.jsonl
count_wide=data/r13b/count_wide.jsonl count_narrow=data/v11/dev_count_transfer.jsonl flip=data/v6/dev_lr_pairs.jsonl
vsr=data/v5/dev_vsr.jsonl"
for arm in s l; do
  mkdir -p runs/exp37-$arm/eval
  for s in $SETS; do
    set=${s%%=*}; data=${s#*=}; out=runs/exp37-$arm/eval/$set.json
    [ -f "${out%.json}.details.jsonl" ] && continue
    log "protection $arm $set"
    capped uv run devision-eval --checkpoint runs/exp37-$arm/stage1 --data "$data" --data-root data --out "$out" \
      --role monitoring > "${out%.json}.log" 2>&1 || { log "$arm $set failed"; exit 1; }
  done
done
capped uv run python scripts/exp37/report.py > runs/exp37-report.txt 2>&1
rc=$?
cat runs/exp37-report.txt
log "report (exit $rc)"
exit $rc
