#!/usr/bin/env bash
# exp40 on one Mac (docs/experiments/2026-10-18-exp40-pair-plan.md; critic EXP40-PREP-01, V04-TIME-04). Only after
# the critic's model permission. Serially: arm M (configs/exp40-m.yaml) under scripts/exp40/watch.py, its weights /
# config frozen (scripts/v04b/freeze.py), its REQUIRED screening sets; then arm O the same way; then, report only,
# rl_tune_L / exp31 / semantics tune of M and O; scripts/exp40/report.py. Deadline min(start + 330 minutes,
# 2026-10-18 10:00:00) with the last 120 s for the report; one run per arm, no retry, no full evaluation started.
# An arm that fails or is stopped does not stop the other.
#   nohup bash scripts/exp40/chain.sh > runs/exp40.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp40/chain.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
INIT=runs/exp37-l/stage1
INIT_SHA=f9fc93c25b4bd96c55d41fd7c974c43408ea31034eac2c3a1ade2bcf5950da31
INIT_CFG=3a10bacd56e7f48aa07ad981821a592fff10c5eb317c39a9591ad425c175c324
have=$(shasum -a 256 $INIT/model.safetensors | cut -d' ' -f1)
cfg=$(shasum -a 256 $INIT/config.json | cut -d' ' -f1)
[ "$have" = "$INIT_SHA" ] && [ "$cfg" = "$INIT_CFG" ] || { log "exp37-L is $have / $cfg"; exit 1; }
for arm in M O; do
  want=$(python3 -c "import json; print(json.load(open('data/exp40/MANIFEST.json'))['arms']['$arm']['sha256'])")
  got=$(shasum -a 256 data/exp40/train_$arm.jsonl | cut -d' ' -f1)
  [ "$want" = "$got" ] || { log "data/exp40/train_$arm.jsonl is $got, MANIFEST $want"; exit 1; }
done
python3 -c "import json, sys; p = json.load(open('data/exp40/PLAN.json')); sys.exit(0 if p['arms_same_plan'] and p['arms_same_items_apart_from_image'] else 1)" \
  || { log "PLAN.json does not show the same items / plan for both arms"; exit 1; }
[ ! -e runs/exp40-m ] && [ ! -e runs/exp40-o ] || { log "runs/exp40-m or runs/exp40-o already exists"; exit 1; }
DEADLINE_FILE=runs/exp40.deadline
if [ ! -f "$DEADLINE_FILE" ]; then
  cap=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-10-18 10:00:00" +%s)
  end=$(( $(date +%s) + 330 * 60 )); [ "$end" -lt "$cap" ] || end=$cap
  echo "$end" > "$DEADLINE_FILE"
fi
DEADLINE=$(cat "$DEADLINE_FILE")
RESERVE=120
. scripts/exp38/steps.sh
log "exp40: start exp37-L $have; deadline $(date -r "$DEADLINE" '+%F %T')"
evalset() {   # run name data: evaluate the frozen arm, then tie the files to its frozen identity
  step "eval:$2" uv run devision-eval --checkpoint "$1/stage2" --data "$3" --data-root data \
    --out "$1/eval/$2.json" --role monitoring
  local rc=$?
  [ "$rc" = 0 ] || return $rc
  uv run python scripts/v04b/freeze.py record "$1" "$2" || { status "eval:$2" failed; log "identity changed: $1 $2"; return 1; }
}
arm() {   # M|O: train, freeze, required screening
  local a=$1 run=runs/exp40-$(echo "$1" | tr MO mo)
  STATUS=$run-status.txt
  mkdir -p "$run"
  python3 -c "import json, sys; json.dump({'arm': '$a', 'init': '$INIT', 'init_sha256': {'model.safetensors': '$INIT_SHA', 'config.json': '$INIT_CFG'}, 'data': 'data/exp40/train_$a.jsonl', 'data_sha256': '$(shasum -a 256 data/exp40/train_$a.jsonl | cut -d' ' -f1)', 'git_commit': '$(git rev-parse HEAD)', 'started': '$(date '+%F %T')'}, open('$run/START.json', 'w'), indent=1)"
  log "arm $a"
  step train uv run python scripts/exp40/watch.py "$run/stage2" "$run-watch.txt" -- uv run devision-pipeline configs/exp40-$(echo "$a" | tr MO mo).yaml || return 1
  uv run python scripts/v04b/freeze.py freeze "$run" || { status freeze failed; return 1; }
  mkdir -p "$run/eval"
  for s in sqa_temp=data/exp21/dev_sqa_temp.jsonl sqa_force=data/exp21/dev_sqa_force.jsonl \
           sqa_temp_mism=data/exp21/dev_sqa_temp_mism.jsonl sqa_force_mism=data/exp21/dev_sqa_force_mism.jsonl \
           dev16=data/exp22/dev16.jsonl syn_erased=data/exp21/dev_syn_erased.jsonl \
           exist=data/r13b/panel_exist.jsonl science=data/r13b/panel_science.jsonl ext=data/ext/dev_ext.jsonl \
           count_wide=data/r13b/count_wide.jsonl count_narrow=data/v11/dev_count_transfer.jsonl \
           flip=data/v6/dev_lr_pairs.jsonl vsr=data/v5/dev_vsr.jsonl rl_tune=data/real_layout/questions_tune.jsonl; do
    evalset "$run" "${s%%=*}" "${s#*=}"
    local rc=$?
    { [ "$rc" = 124 ] || [ "$rc" = 125 ]; } && return "$rc"      # out of time: the rest stay missing
  done
  return 0
}
arm M
arm O
for a in m o; do      # report only, after both arms' required sets
  STATUS=runs/exp40-$a-status.txt
  [ -f runs/exp40-$a/FROZEN.json ] || continue
  for s in rl_tune_L=data/exp40/rl_tune_L.jsonl exp31=data/exp31/text_force.jsonl semantics_tune=data/exp34/semantics_tune.jsonl; do
    evalset runs/exp40-$a "${s%%=*}" "${s#*=}"
  done
done
final_report uv run python scripts/exp40/report.py > runs/exp40-report.txt 2>&1
rc=$?
cat runs/exp40-report.txt
log "final report (exit $rc)"
exit "$rc"
