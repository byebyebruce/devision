#!/usr/bin/env bash
# exp38 on one Mac (docs/experiments/2026-10-17-exp38-512-plan.md; critic EXP38-PREP-01 / READOUT-01 / R1-01..03):
# arm H (512 input, exp37-L's weights read at 512) then arm C (256), each 1,000 steps on exp36's original pictures
# with the full-number mask; then, by priority: the full-number readouts (600 original tune, 200 original fit), the
# main report, CPU latency, ScienceQA temperature / force and dev16, the 7 protection panels. Every step's outcome is
# written to runs/exp38-status.txt (completed / failed / interrupted / missing). One deadline: start + 210 minutes
# and never later than 2026-10-18 00:30 (runs/exp38.deadline, kept across restarts); steps stop 60 s before it,
# the final report runs in those 60 s, and its exit status (or a failed main step) is the chain's. No retry.
#   nohup bash scripts/exp38/chain.sh > runs/exp38.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp38/chain.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
SRC=f9fc93c25b4bd96c55d41fd7c974c43408ea31034eac2c3a1ade2bcf5950da31
have=$(shasum -a 256 runs/exp37-l/stage1/model.safetensors | cut -d' ' -f1)
[ "$have" = "$SRC" ] || { log "exp37-L sha256 $have is not $SRC"; exit 1; }
DEADLINE_FILE=runs/exp38.deadline
if [ ! -f "$DEADLINE_FILE" ]; then
  cap=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-10-18 00:30:00" +%s)
  end=$(( $(date +%s) + 210 * 60 )); [ "$end" -lt "$cap" ] || end=$cap
  echo "$end" > "$DEADLINE_FILE"
fi
DEADLINE=$(cat "$DEADLINE_FILE")
STATUS=runs/exp38-status.txt
. scripts/exp38/steps.sh
log "start exp37-L sha256 $have; data $(shasum -a 256 data/exp36/align_mix.jsonl | cut -c1-16); deadline $(date -r "$DEADLINE" '+%F %T')"
finish() {   # final report in the reserved window; exit non-zero if the main part or the report failed
  final_report uv run python scripts/exp38/report.py > runs/exp38-report.txt 2>&1
  local rc=$?
  cat runs/exp38-report.txt
  log "final report (exit $rc)"
  [ "$rc" = 0 ] && rc=${1:-0}
  exit "$rc"
}
if [ -d runs/exp38-init-512 ]; then   # reuse only the conversion of this exact source at 512 with interpolation
  uv run python - <<'PY' || { log "runs/exp38-init-512 is not the expected conversion"; status convert failed; finish 1; }
import json, sys
c = json.load(open("runs/exp38-init-512/CONVERTED.json")); g = json.load(open("runs/exp38-init-512/config.json"))
ok = (c.get("source_model_sha256") == "f9fc93c25b4bd96c55d41fd7c974c43408ea31034eac2c3a1ade2bcf5950da31"
      and g.get("image_size") == 512 and g.get("interpolate_pos_encoding") is True and all(c["files_identical"].values()))
sys.exit(0 if ok else 1)
PY
else
  step convert uv run python scripts/exp38/convert.py runs/exp37-l/stage1 runs/exp38-init-512 --image-size 512 || finish 1
fi
for arm in h c; do
  [ -f runs/exp38-$arm/stage1/align_report.json ] && continue
  log "arm $arm"
  step "train:$arm" uv run devision-pipeline configs/exp38-$arm.yaml || finish 1
done
for arm in h c; do
  for split in tune:data/real_layout/captions_tune.jsonl fit:data/exp36/fit_slice.jsonl; do
    name=${split%%:*}; data=${split#*:}; out=runs/exp38-$arm/numbers_$name
    [ -f "$out/numbers.jsonl" ] && continue
    log "readout $arm $name"
    step "readout:$arm:$name" uv run python scripts/exp36/readout.py --checkpoint runs/exp38-$arm/stage1 --data "$data" \
      --out "$out" --full-numbers-only || finish 1
  done
done
step report:main uv run python scripts/exp38/report.py || finish 1
stop=0
extra() {   # name cmd...: report-only; once one is interrupted or not started, the rest are recorded missing
  local name=$1; shift
  if [ "$stop" = 1 ]; then status "$name" missing; return; fi
  step "$name" "$@"
  local rc=$?
  [ "$rc" = 124 ] || [ "$rc" = 125 ] && stop=1
  return 0
}
for arm in h c; do
  mkdir -p runs/exp38-$arm/cpu
  extra "cpu:$arm" env OMP_NUM_THREADS=4 uv run devision-eval --checkpoint runs/exp38-$arm/stage1 --data data/r12/cpu_latency.jsonl \
    --data-root data --out runs/exp38-$arm/cpu/single.json --device cpu --no-group
done
for group in "sqa_temp=data/exp21/dev_sqa_temp.jsonl sqa_force=data/exp21/dev_sqa_force.jsonl sqa_temp_mism=data/exp21/dev_sqa_temp_mism.jsonl sqa_force_mism=data/exp21/dev_sqa_force_mism.jsonl dev16=data/exp22/dev16.jsonl" \
             "exist=data/r13b/panel_exist.jsonl science=data/r13b/panel_science.jsonl ext=data/ext/dev_ext.jsonl count_wide=data/r13b/count_wide.jsonl count_narrow=data/v11/dev_count_transfer.jsonl flip=data/v6/dev_lr_pairs.jsonl vsr=data/v5/dev_vsr.jsonl"; do
  for arm in h c; do
    mkdir -p runs/exp38-$arm/eval
    for s in $group; do
      set=${s%%=*}; data=${s#*=}
      extra "eval:$arm:$set" uv run devision-eval --checkpoint runs/exp38-$arm/stage1 --data "$data" --data-root data \
        --out runs/exp38-$arm/eval/$set.json --role monitoring
    done
  done
done
finish 0
