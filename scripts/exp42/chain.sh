#!/usr/bin/env bash
# exp42 on one Mac (docs/experiments/2026-10-18-exp42-objective-plan.md; critic EXP42-PREP-01). Only after the critic's
# model permission. In order of priority: the start (exp40-M's frozen end) on fit_O / fit_L / rl_tune / rl_tune_L at
# temperature 1; arm R (objective rlcd) then arm C (objective ce), each trained under scripts/exp42/watch.py, its
# weights / config frozen (scripts/v04b/freeze.py), the same four sets plus exist / science at temperature 1, each
# recorded under the frozen identity; then the start's exist / science; scripts/exp42/report.py. Deadline start + 90
# minutes with the last 120 s for the report; one run per arm, no retry, nothing started after it.
#   nohup bash scripts/exp42/chain.sh > runs/exp42.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp42/chain.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
INIT=runs/exp40-m/stage2
INIT_SHA=3fb4232e27bff5fae767b8d62230884bae8dbaae0e92cca76de89be9e3b151dd
INIT_CFG=74e32e7cd4a7d08e9236cb95a2b6cd1a8c407056f07c7008a3f2dc5679b98a32
uv run python scripts/v04b/freeze.py check runs/exp40-m || { log "exp40-M is not its frozen identity"; exit 1; }
[ "$(shasum -a 256 $INIT/model.safetensors | cut -d' ' -f1)" = "$INIT_SHA" ] && [ "$(shasum -a 256 $INIT/config.json | cut -d' ' -f1)" = "$INIT_CFG" ] \
  || { log "exp40-M is not 3fb4232e / 74e32e7c"; exit 1; }
for f in train fit_O fit_L val_sub; do
  want=$(python3 -c "import json; print(json.load(open('data/exp42/MANIFEST.json'))['files']['$f'])")
  [ "$(shasum -a 256 data/exp42/$f.jsonl | cut -d' ' -f1)" = "$want" ] || { log "data/exp42/$f.jsonl changed"; exit 1; }
done
[ -f data/exp42/PLAN.json ] || { log "no data/exp42/PLAN.json"; exit 1; }
[ ! -e runs/exp42-r ] && [ ! -e runs/exp42-c ] && [ ! -e runs/exp42-start ] || { log "an exp42 run directory already exists"; exit 1; }
DEADLINE_FILE=runs/exp42.deadline
[ -f "$DEADLINE_FILE" ] || echo $(( $(date +%s) + 90 * 60 )) > "$DEADLINE_FILE"
DEADLINE=$(cat "$DEADLINE_FILE")
RESERVE=120
. scripts/exp38/steps.sh
log "exp42: start exp40-M $INIT_SHA; deadline $(date -r "$DEADLINE" '+%F %T')"
MAIN="fit_O=data/exp42/fit_O.jsonl fit_L=data/exp42/fit_L.jsonl rl_tune=data/real_layout/questions_tune.jsonl rl_tune_L=data/exp40/rl_tune_L.jsonl"
RISK="exist=data/r13b/panel_exist.jsonl science=data/r13b/panel_science.jsonl"
evalset() {   # run checkpoint name data [record]: evaluate at temperature 1; arms tie the files to their frozen identity
  step "eval:$3" uv run devision-eval --checkpoint "$2" --data "$4" --data-root data --out "$1/eval/$3.json" \
    --role monitoring --no-temperature
  local rc=$?
  [ "$rc" = 0 ] || return $rc
  [ "${5:-}" = record ] || return 0
  uv run python scripts/v04b/freeze.py record "$1" "$3" || { status "eval:$3" failed; log "identity changed: $1 $3"; return 1; }
}
sets() {   # run checkpoint record "name=data ...": stop at the deadline
  local s rc
  for s in $4; do
    evalset "$1" "$2" "${s%%=*}" "${s#*=}" "$3"
    rc=$?
    { [ "$rc" = 124 ] || [ "$rc" = 125 ]; } && return "$rc"
  done
  return 0
}
mkdir -p runs/exp42-start/eval
STATUS=runs/exp42-start-status.txt
sets runs/exp42-start "$INIT" "" "$MAIN"
arm() {   # r|c objective
  local run=runs/exp42-$1
  STATUS=$run-status.txt
  mkdir -p "$run"
  python3 -c "import json; json.dump({'arm': '$1', 'objective': '$2', 'init': '$INIT', 'init_sha256': {'model.safetensors': '$INIT_SHA', 'config.json': '$INIT_CFG'}, 'data': 'data/exp42/train.jsonl', 'data_sha256': '$(shasum -a 256 data/exp42/train.jsonl | cut -d' ' -f1)', 'git_commit': '$(git rev-parse HEAD)', 'started': '$(date '+%F %T')'}, open('$run/START.json', 'w'), indent=1)"
  log "arm $1 ($2)"
  step train uv run python scripts/exp42/watch.py "$run/stage2" "$run-watch.txt" -- uv run devision-pipeline configs/exp42-$1.yaml || return 1
  uv run python scripts/v04b/freeze.py freeze "$run" || { status freeze failed; return 1; }
  mkdir -p "$run/eval"
  sets "$run" "$run/stage2" record "$MAIN $RISK"
}
arm r rlcd
arm c ce
STATUS=runs/exp42-start-status.txt
sets runs/exp42-start "$INIT" "" "$RISK"
final_report uv run python scripts/exp42/report.py > runs/exp42-report.txt 2>&1
rc=$?
cat runs/exp42-report.txt
log "final report (exit $rc)"
exit "$rc"
