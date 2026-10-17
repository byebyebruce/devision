#!/usr/bin/env bash
# exp38 on one Mac (docs/experiments/2026-10-17-exp38-512-plan.md; critic EXP38-PREP-01 / READOUT-01): arm H (512
# input, exp37-L's weights read at 512) then arm C (256), each 1,000 steps on exp36's original pictures with the
# full-number mask; then, in this order of priority so that a deadline cuts only the last items:
#   1. each model's full-number readout on the 600 original tune pictures and the 200 original fit pictures;
#   2. scripts/exp38/report.py (the four investment conditions on H - C);
#   3. CPU latency of each (200-question fixture, 4 threads, one question per request; MPS idle by then);
#   4. report only: ScienceQA temperature / force (true and fixed wrong pictures) and dev16, then the 7 protection
#      panels; the report is rewritten after each group.
# One absolute deadline: start + 180 minutes and never later than 2026-10-18 00:30 (runs/exp38.deadline, kept
# across restarts); every step runs under `capped` (TERM, then KILL 15 s later); nothing is retried.
#   nohup bash scripts/exp38/chain.sh > runs/exp38.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp38/chain.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
want=f9fc93c25b4bd96c55d41fd7c974c43408ea31034eac2c3a1ade2bcf5950da31
have=$(shasum -a 256 runs/exp37-l/stage1/model.safetensors | cut -d' ' -f1)
[ "$have" = "$want" ] || { log "exp37-L sha256 $have is not $want"; exit 1; }
DEADLINE_FILE=runs/exp38.deadline
if [ ! -f "$DEADLINE_FILE" ]; then
  cap=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-10-18 00:30:00" +%s)
  end=$(( $(date +%s) + 180 * 60 )); [ "$end" -lt "$cap" ] || end=$cap
  echo "$end" > "$DEADLINE_FILE"
fi
DEADLINE=$(cat "$DEADLINE_FILE")
log "start exp37-L sha256 $have; data $(shasum -a 256 data/exp36/align_mix.jsonl | cut -c1-16); deadline $(date -r "$DEADLINE" '+%F %T')"
capped() {
  local left=$(( DEADLINE - $(date +%s) - 15 ))
  [ "$left" -gt 0 ] || { log "past the deadline, not starting: $*"; exit 2; }
  run_capped "$left" 15 "$@"
  local rc=$?
  [ "$rc" = 124 ] && { log "deadline reached, stopped: $*"; exit 2; }
  return $rc
}
[ -d runs/exp38-init-512 ] || uv run python scripts/exp38/convert.py runs/exp37-l/stage1 runs/exp38-init-512 --image-size 512 \
  || { log "conversion failed"; exit 1; }
for arm in h c; do
  if [ ! -f runs/exp38-$arm/stage1/align_report.json ]; then
    log "arm $arm"
    capped uv run devision-pipeline configs/exp38-$arm.yaml || { log "arm $arm failed"; exit 1; }
  fi
done
for arm in h c; do
  for split in tune:data/real_layout/captions_tune.jsonl fit:data/exp36/fit_slice.jsonl; do
    name=${split%%:*}; data=${split#*:}; out=runs/exp38-$arm/numbers_$name
    [ -f "$out/numbers.jsonl" ] && continue
    log "readout $arm $name"
    capped uv run python scripts/exp36/readout.py --checkpoint runs/exp38-$arm/stage1 --data "$data" --out "$out" \
      --full-numbers-only > "$out.log" 2>&1 || { log "readout $arm $name failed"; exit 1; }
    tail -1 "$out.log"
  done
done
report() { capped uv run python scripts/exp38/report.py > runs/exp38-report.txt 2>&1; cat runs/exp38-report.txt; }
report
for arm in h c; do
  out=runs/exp38-$arm/cpu/single.json; mkdir -p runs/exp38-$arm/cpu
  [ -f "$out" ] && continue
  log "cpu latency $arm"
  OMP_NUM_THREADS=4 capped uv run devision-eval --checkpoint runs/exp38-$arm/stage1 --data data/r12/cpu_latency.jsonl \
    --data-root data --out "$out" --device cpu --no-group > "${out%.json}.log" 2>&1 || log "cpu latency $arm failed"
done
evalset() {   # arm set data
  local out=runs/exp38-$1/eval/$2.json; mkdir -p runs/exp38-$1/eval
  [ -f "${out%.json}.details.jsonl" ] && return 0
  log "eval $1 $2"
  capped uv run devision-eval --checkpoint runs/exp38-$1/stage1 --data "$3" --data-root data --out "$out" --role monitoring \
    > "${out%.json}.log" 2>&1 || log "$1 $2 failed"
}
for arm in h c; do
  for s in sqa_temp=data/exp21/dev_sqa_temp.jsonl sqa_force=data/exp21/dev_sqa_force.jsonl \
           sqa_temp_mism=data/exp21/dev_sqa_temp_mism.jsonl sqa_force_mism=data/exp21/dev_sqa_force_mism.jsonl \
           dev16=data/exp22/dev16.jsonl; do evalset $arm "${s%%=*}" "${s#*=}"; done
done
report
for arm in h c; do
  for s in exist=data/r13b/panel_exist.jsonl science=data/r13b/panel_science.jsonl ext=data/ext/dev_ext.jsonl \
           count_wide=data/r13b/count_wide.jsonl count_narrow=data/v11/dev_count_transfer.jsonl \
           flip=data/v6/dev_lr_pairs.jsonl vsr=data/v5/dev_vsr.jsonl; do evalset $arm "${s%%=*}" "${s#*=}"; done
done
report
log "done"
