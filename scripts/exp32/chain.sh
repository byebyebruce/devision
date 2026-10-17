#!/usr/bin/env bash
# exp32 on one Mac (docs/experiments/2026-10-17-exp32-bridge-plan.md):
# A2. real-layout binding alignment from exp30's checkpoint (configs/exp32-align.yaml); exp24's gate on the
#     real-layout tune captions is recorded, not a stop;
# B.  decision training to the end (configs/exp32-fit.yaml; re-running the YAML resumes; at most 3 attempts, all
#     within the 3.5 h budget);
# E.  devision-eval of the final checkpoint on exp27's 15 sets + the real-layout tune and confirm questions;
#     scripts/exp32/report.py.
#   EXP32_INIT_SHA256=<released sha> nohup bash scripts/exp32/chain.sh > runs/exp32.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
log() { echo "== $(date '+%F %T') $*"; }
# Preconditions (critic EXP32-N01): exp30 passed its protection screen, and the start checkpoint is the one the
# critic released (EXP32_INIT_SHA256, sha256 of exp30's model.safetensors); a model file alone is not a pass.
init=runs/exp30-align-replay/stage1/model.safetensors
[ -f "$init" ] || { log "missing exp30's checkpoint"; exit 1; }
outcome=$(uv run python -c "import json; print(json.load(open('runs/exp30-align-replay/report.json'))['outcome'])") \
  || { log "no exp30 report"; exit 1; }
case "$outcome" in "passes this protection screen"*) ;; *) log "exp30 did not pass: $outcome"; exit 1;; esac
sha=$(shasum -a 256 "$init" | cut -d' ' -f1)
log "start checkpoint $init sha256 $sha"
[ "${EXP32_INIT_SHA256:-}" = "$sha" ] || { log "EXP32_INIT_SHA256 does not match the released checkpoint"; exit 1; }
BUDGET=$((210 * 60))                    # 3.5 hours for everything, retries included
if [ ! -f runs/exp32-align/stage1/align_report.json ]; then
  log "A2: real-layout binding alignment"
  uv run devision-pipeline configs/exp32-align.yaml || { log "A2 failed"; exit 1; }
fi
uv run python scripts/exp24/gate.py runs/exp32-align/stage1/align_report.json > runs/exp32-align/gate.txt 2>&1
cat runs/exp32-align/gate.txt
for attempt in 1 2 3; do
  [ -f runs/exp32-fit/stage2/model.safetensors ] && break
  [ "$SECONDS" -lt "$BUDGET" ] || { log "over the 3.5 h budget, no more attempts"; break; }
  log "B attempt $attempt"
  uv run devision-pipeline configs/exp32-fit.yaml || log "B exited non-zero"
done
[ -f runs/exp32-fit/stage2/model.safetensors ] || { log "B failed 3 times"; exit 1; }
run=runs/exp32-fit
SETS="sqa_temp=data/exp21/dev_sqa_temp.jsonl sqa_force=data/exp21/dev_sqa_force.jsonl
sqa_temp_mism=data/exp21/dev_sqa_temp_mism.jsonl sqa_force_mism=data/exp21/dev_sqa_force_mism.jsonl
sqa_conc=data/v10/dev_sqa_conc.jsonl sqa_repel=data/v10/dev_sqa_repel.jsonl
sqa_text=data/exp21/sqa_text.jsonl sqa_blank=data/exp21/sqa_blank.jsonl
exist=data/r13b/panel_exist.jsonl science=data/r13b/panel_science.jsonl ext=data/ext/dev_ext.jsonl
count_wide=data/r13b/count_wide.jsonl count_narrow=data/v11/dev_count_transfer.jsonl flip=data/v6/dev_lr_pairs.jsonl
vsr=data/v5/dev_vsr.jsonl rl_tune=data/real_layout/questions_tune.jsonl rl_confirm=data/real_layout/questions_confirm.jsonl
dev16=data/exp22/dev16.jsonl"
mkdir -p "$run/eval"
for s in $SETS; do
  set=${s%%=*}; data=${s#*=}
  out=$run/eval/$set.json
  [ -f "${out%.json}.details.jsonl" ] && continue
  log "eval $set"
  uv run devision-eval --checkpoint $run/stage2 --data "$data" --data-root data --out "$out" --role monitoring \
    > "${out%.json}.log" 2>&1 || { log "$set failed"; exit 1; }
done
# v0.3 on the real-layout and dev16 sets (the 15 exp27 sets are reused)
mkdir -p runs/exp32-fit/eval_v03
for s in rl_tune=data/real_layout/questions_tune.jsonl rl_confirm=data/real_layout/questions_confirm.jsonl dev16=data/exp22/dev16.jsonl; do
  set=${s%%=*}; data=${s#*=}; out=runs/exp32-fit/eval_v03/$set.json
  [ -f "${out%.json}.details.jsonl" ] && continue
  log "v03 $set"
  uv run devision-eval --checkpoint runs/r15-vision/stage2 --data "$data" --data-root data --out "$out" --role monitoring \
    > "${out%.json}.log" 2>&1 || { log "v03 $set failed"; exit 1; }
done
uv run python scripts/exp32/report.py > "$run/report.txt" 2>&1
rc=$?
cat "$run/report.txt"
log "report (exit $rc)"
exit $rc
