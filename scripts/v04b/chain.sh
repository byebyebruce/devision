#!/usr/bin/env bash
# Candidate B on one Mac (docs/experiments/2026-10-18-v04b-candidate-plan.md; critic V04-B-R1-01 / 02). Only after
# exp39-H met its four conditions, the exp39 chain ended and the critic released B: decision training from exp39-H's
# end (configs/v04b-fit.yaml) under scripts/v04b/watch.py, then the final checkpoint on the REQUIRED screening sets
# (ScienceQA true / wrong, dev16, erased digits, the 7 protection panels, real-layout tune) and, report only, exp31
# and the exp34 semantics tune; scripts/v04b/report.py. Deadline min(start + 330 minutes, 2026-10-18 08:30:00) with
# the last 120 s for the report; one run, no retry, no full evaluation started.
#   V04B_INIT_SHA256=<exp39-H model sha256> nohup bash scripts/v04b/chain.sh > runs/v04b.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
: "${V04B_INIT_SHA256:?set V04B_INIT_SHA256 to the exp39-H model sha256}"
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/v04b/chain.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
have=$(shasum -a 256 runs/exp39-h/stage1/model.safetensors 2>/dev/null | cut -d' ' -f1)
[ "$have" = "$V04B_INIT_SHA256" ] || { log "exp39-H sha256 ${have:-missing} is not $V04B_INIT_SHA256"; exit 1; }
uv run python - <<'PY' || { log "exp39's report does not allow B (four conditions)"; exit 1; }
import json, sys
r = json.load(open("runs/exp39-report.json"))
sys.exit(0 if r.get("invest_in_512") is True else 1)
PY
data=$(shasum -a 256 data/v04b/train_mix.jsonl | cut -d' ' -f1)
[ "$data" = fd53d39427ab82dc142756a97bc2c5c5c6d6156acd7a0e1678b899a4bab98b69 ] || { log "train mix sha256 $data"; exit 1; }
[ ! -e runs/v04b-fit ] || { log "runs/v04b-fit already exists"; exit 1; }
DEADLINE_FILE=runs/v04b.deadline
if [ ! -f "$DEADLINE_FILE" ]; then
  cap=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-10-18 08:30:00" +%s)
  end=$(( $(date +%s) + 330 * 60 )); [ "$end" -lt "$cap" ] || end=$cap
  echo "$end" > "$DEADLINE_FILE"
fi
DEADLINE=$(cat "$DEADLINE_FILE")
STATUS=runs/v04b-status.txt
RESERVE=120
. scripts/exp38/steps.sh
log "B: start exp39-H $have; data $data; deadline $(date -r "$DEADLINE" '+%F %T')"
finish() {
  final_report uv run python scripts/v04b/report.py runs/v04b-fit "$V04B_INIT_SHA256" > runs/v04b-report.txt 2>&1
  local rc=$?
  cat runs/v04b-report.txt
  log "final report (exit $rc)"
  [ "$rc" = 0 ] && rc=${1:-0}
  exit "$rc"
}
step train uv run python scripts/v04b/watch.py runs/v04b-fit/stage2 runs/v04b-watch.txt -- uv run devision-pipeline configs/v04b-fit.yaml || finish 1
mkdir -p runs/v04b-fit/eval
evalset() {   # name data
  step "eval:$1" uv run devision-eval --checkpoint runs/v04b-fit/stage2 --data "$2" --data-root data \
    --out runs/v04b-fit/eval/$1.json --role monitoring
}
for s in sqa_temp=data/exp21/dev_sqa_temp.jsonl sqa_force=data/exp21/dev_sqa_force.jsonl \
         sqa_temp_mism=data/exp21/dev_sqa_temp_mism.jsonl sqa_force_mism=data/exp21/dev_sqa_force_mism.jsonl \
         dev16=data/exp22/dev16.jsonl syn_erased=data/exp21/dev_syn_erased.jsonl \
         exist=data/r13b/panel_exist.jsonl science=data/r13b/panel_science.jsonl ext=data/ext/dev_ext.jsonl \
         count_wide=data/r13b/count_wide.jsonl count_narrow=data/v11/dev_count_transfer.jsonl \
         flip=data/v6/dev_lr_pairs.jsonl vsr=data/v5/dev_vsr.jsonl rl_tune=data/real_layout/questions_tune.jsonl \
         exp31=data/exp31/text_force.jsonl semantics_tune=data/exp34/semantics_tune.jsonl; do
  evalset "${s%%=*}" "${s#*=}"
  rc=$?
  { [ "$rc" = 124 ] || [ "$rc" = 125 ]; } && break       # out of time: the rest stay missing, the report says so
done
finish 0
