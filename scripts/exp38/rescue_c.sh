#!/usr/bin/env bash
# exp38 C rescue (critic EXP38-MEM-03): after the resource-interrupted exp38 chain, arm C alone with the reviewed
# configs/exp38-c.yaml (exp37-L's end, exp36's original pictures, 256 / 64, full-number mask, 1,000 steps) -- then
# its full-number readouts on the original 600 tune and 200 fit pictures, the C-only report, and by priority the
# ScienceQA / dev16, the 7 protection panels, CPU latency. Own log / status / report; one deadline min(start + 60
# minutes, 2026-10-18 00:07:44) including the final 60 s for the report (scripts/exp38/steps.sh); no retry.
#   nohup bash scripts/exp38/rescue_c.sh > runs/exp38c-rescue.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp38/rescue_c.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
SRC=f9fc93c25b4bd96c55d41fd7c974c43408ea31034eac2c3a1ade2bcf5950da31
have=$(shasum -a 256 runs/exp37-l/stage1/model.safetensors | cut -d' ' -f1)
[ "$have" = "$SRC" ] || { log "exp37-L sha256 $have is not $SRC"; exit 1; }
[ ! -e runs/exp38-c ] || { log "runs/exp38-c already exists"; exit 1; }
DEADLINE_FILE=runs/exp38c-rescue.deadline
if [ ! -f "$DEADLINE_FILE" ]; then
  cap=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-10-18 00:07:44" +%s)
  end=$(( $(date +%s) + 60 * 60 )); [ "$end" -lt "$cap" ] || end=$cap
  echo "$end" > "$DEADLINE_FILE"
fi
DEADLINE=$(cat "$DEADLINE_FILE")
STATUS=runs/exp38c-rescue-status.txt
. scripts/exp38/steps.sh
log "C rescue: exp37-L sha256 $have; data $(shasum -a 256 data/exp36/align_mix.jsonl | cut -c1-16); deadline $(date -r "$DEADLINE" '+%F %T')"
finish() {
  final_report uv run python scripts/exp38/report_c.py > runs/exp38c-rescue-report.txt 2>&1
  local rc=$?
  cat runs/exp38c-rescue-report.txt
  log "final report (exit $rc)"
  [ "$rc" = 0 ] && rc=${1:-0}
  exit "$rc"
}
step train:c uv run devision-pipeline configs/exp38-c.yaml || finish 1
for split in tune:data/real_layout/captions_tune.jsonl fit:data/exp36/fit_slice.jsonl; do
  name=${split%%:*}; data=${split#*:}
  step "readout:c:$name" uv run python scripts/exp36/readout.py --checkpoint runs/exp38-c/stage1 --data "$data" \
    --out runs/exp38-c/numbers_$name --full-numbers-only || finish 1
done
step report:c uv run python scripts/exp38/report_c.py || finish 1
stop=0
extra() {
  local name=$1; shift
  if [ "$stop" = 1 ]; then status "$name" missing; return; fi
  step "$name" "$@"
  local rc=$?
  { [ "$rc" = 124 ] || [ "$rc" = 125 ]; } && stop=1
  return 0
}
mkdir -p runs/exp38-c/eval runs/exp38-c/cpu
for s in sqa_temp=data/exp21/dev_sqa_temp.jsonl sqa_force=data/exp21/dev_sqa_force.jsonl sqa_temp_mism=data/exp21/dev_sqa_temp_mism.jsonl \
         sqa_force_mism=data/exp21/dev_sqa_force_mism.jsonl dev16=data/exp22/dev16.jsonl \
         exist=data/r13b/panel_exist.jsonl science=data/r13b/panel_science.jsonl ext=data/ext/dev_ext.jsonl \
         count_wide=data/r13b/count_wide.jsonl count_narrow=data/v11/dev_count_transfer.jsonl flip=data/v6/dev_lr_pairs.jsonl \
         vsr=data/v5/dev_vsr.jsonl; do
  set=${s%%=*}; data=${s#*=}
  extra "eval:c:$set" uv run devision-eval --checkpoint runs/exp38-c/stage1 --data "$data" --data-root data \
    --out runs/exp38-c/eval/$set.json --role monitoring
done
extra cpu:c env OMP_NUM_THREADS=4 uv run devision-eval --checkpoint runs/exp38-c/stage1 --data data/r12/cpu_latency.jsonl \
  --data-root data --out runs/exp38-c/cpu/single.json --device cpu --no-group
finish 0
