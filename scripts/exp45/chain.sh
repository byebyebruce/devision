#!/usr/bin/env bash
# exp45 on one Mac (docs/experiments/2026-10-18-exp45-region-plan.md; critic EXP45-PREP-01 / 02). Only after the critic's
# model permission. Arm S (configs/exp45-s.yaml, region weight 0) then arm G (configs/exp45-g.yaml, 0.5), each: training
# under scripts/exp45/watch.py, weights / config frozen (scripts/v04b/freeze.py), the four main decision sets at
# temperature 1, the attention observation on rl_tune and fit_O, exist / science -- every file recorded under the arm's
# frozen identity; then scripts/exp45/report.py (M and C reuse exp42's frozen records). Deadline start + 90 minutes with
# the last 120 s for the report; one run per arm, no retry, nothing started after it.
#   nohup bash scripts/exp45/chain.sh > runs/exp45.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp45/chain.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
INIT=runs/exp40-m/stage2
INIT_SHA=3fb4232e27bff5fae767b8d62230884bae8dbaae0e92cca76de89be9e3b151dd
INIT_CFG=74e32e7cd4a7d08e9236cb95a2b6cd1a8c407056f07c7008a3f2dc5679b98a32
uv run python scripts/v04b/freeze.py check runs/exp40-m && uv run python scripts/v04b/freeze.py check runs/exp42-c \
  || { log "exp40-M or exp42-C is not its frozen identity"; exit 1; }
[ "$(shasum -a 256 $INIT/model.safetensors | cut -d' ' -f1)" = "$INIT_SHA" ] && [ "$(shasum -a 256 $INIT/config.json | cut -d' ' -f1)" = "$INIT_CFG" ] \
  || { log "exp40-M is not 3fb4232e / 74e32e7c"; exit 1; }
for f in train fit_O fit_L val_sub; do
  want=$(python3 -c "import json; print(json.load(open('data/exp42/MANIFEST.json'))['files']['$f'])")
  [ "$(shasum -a 256 data/exp42/$f.jsonl | cut -d' ' -f1)" = "$want" ] || { log "data/exp42/$f.jsonl changed"; exit 1; }
done
for f in regions regions_readout; do
  want=$(python3 -c "import json; print(json.load(open('data/exp45/MANIFEST.json'))['$f']['sha256'])")
  [ "$(shasum -a 256 data/exp45/$f.jsonl | cut -d' ' -f1)" = "$want" ] || { log "data/exp45/$f.jsonl changed"; exit 1; }
done
[ ! -e runs/exp45-s ] && [ ! -e runs/exp45-g ] || { log "an exp45 run directory already exists"; exit 1; }
DEADLINE_FILE=runs/exp45.deadline
[ -f "$DEADLINE_FILE" ] || echo $(( $(date +%s) + 90 * 60 )) > "$DEADLINE_FILE"
DEADLINE=$(cat "$DEADLINE_FILE")
RESERVE=120
. scripts/exp38/steps.sh
log "exp45: start exp40-M $INIT_SHA; deadline $(date -r "$DEADLINE" '+%F %T')"
REG_SHA=$(shasum -a 256 data/exp45/regions.jsonl | cut -d' ' -f1)
over() { [ "$1" = 124 ] || [ "$1" = 125 ]; }
arm() {   # s|g weight
  local run=runs/exp45-$1 rc
  STATUS=$run-status.txt
  mkdir -p "$run"
  python3 -c "import json; json.dump({'arm': '$1', 'objective': 'ce', 'question_conditioning': True, 'region_weight': $2, 'regions_sha256': '$REG_SHA', 'init': '$INIT', 'init_sha256': {'model.safetensors': '$INIT_SHA', 'config.json': '$INIT_CFG'}, 'data': 'data/exp42/train.jsonl', 'data_sha256': '$(shasum -a 256 data/exp42/train.jsonl | cut -d' ' -f1)', 'git_commit': '$(git rev-parse HEAD)', 'started': '$(date '+%F %T')'}, open('$run/START.json', 'w'), indent=1)"
  log "arm $1 (region weight $2)"
  step train uv run python scripts/exp45/watch.py "$run/stage2" "$run-watch.txt" -- uv run devision-pipeline configs/exp45-$1.yaml || return 1
  uv run python scripts/v04b/freeze.py freeze "$run" || { status freeze failed; return 1; }
  mkdir -p "$run/eval"
  record() { uv run python scripts/v04b/freeze.py record "$run" "$1" || { status "eval:$1" failed; log "identity changed: $1"; return 1; }; }
  for s in fit_O=data/exp42/fit_O.jsonl fit_L=data/exp42/fit_L.jsonl rl_tune=data/real_layout/questions_tune.jsonl rl_tune_L=data/exp40/rl_tune_L.jsonl; do
    step "eval:${s%%=*}" uv run devision-eval --checkpoint $run/stage2 --data "${s#*=}" --data-root data --out $run/eval/${s%%=*}.json \
      --role monitoring --no-temperature --max-questions 8
    rc=$?; over $rc && return $rc; [ $rc = 0 ] && record "${s%%=*}"
  done
  for s in rl_tune=data/real_layout/questions_tune.jsonl fit_O=data/exp42/fit_O.jsonl; do
    step "eval:attn_${s%%=*}" uv run python scripts/exp45/attention.py $run/stage2 "${s#*=}" "${s%%=*}" $run/eval "attn_${s%%=*}"
    rc=$?; over $rc && return $rc; [ $rc = 0 ] && record "attn_${s%%=*}"
  done
  for s in exist=data/r13b/panel_exist.jsonl science=data/r13b/panel_science.jsonl; do
    step "eval:${s%%=*}" uv run devision-eval --checkpoint $run/stage2 --data "${s#*=}" --data-root data --out $run/eval/${s%%=*}.json \
      --role monitoring --no-temperature --max-questions 8
    rc=$?; over $rc && return $rc; [ $rc = 0 ] && record "${s%%=*}"
  done
  return 0
}
arm s 0.0
arm g 0.5
final_report uv run python scripts/exp45/report.py > runs/exp45-report.txt 2>&1
rc=$?
cat runs/exp45-report.txt
log "final report (exit $rc)"
exit "$rc"
