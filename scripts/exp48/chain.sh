#!/usr/bin/env bash
# exp48 on one Mac (docs/experiments/2026-10-18-exp48-interaction-plan.md; critic EXP48-PREP-01..04). Only after the critic's
# model permission. Arm C (configs/exp48-c.yaml, interaction weight 0) then arm X (configs/exp48-x.yaml, 1.0), both from
# EXP45-G's frozen end, each: training under scripts/exp48/watch.py, weights / config frozen (scripts/v04b/freeze.py), the
# four main decision sets at temperature 1, exist / science -- every file recorded under the arm's frozen identity; then
# scripts/exp48/report.py (G's records reused after their frozen check). Deadline start + 90 minutes with the last 120 s
# for the report; one run per arm, no retry, nothing started after it.
#   nohup bash scripts/exp48/chain.sh > runs/exp48.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp48/chain.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
INIT=runs/exp45-g/stage2
INIT_SHA=c34b43ddde846af40e154aefb8ca1a381c27325e95b03aca55d42b75d4e7c15e
INIT_CFG=2a5dfa64d33846052bf8c13862bd0def53888fce7160a18c25d38279ce89ab08
uv run python scripts/v04b/freeze.py check runs/exp45-g || { log "EXP45-G is not its frozen identity"; exit 1; }
[ "$(shasum -a 256 $INIT/model.safetensors | cut -d' ' -f1)" = "$INIT_SHA" ] && [ "$(shasum -a 256 $INIT/config.json | cut -d' ' -f1)" = "$INIT_CFG" ] \
  || { log "EXP45-G is not c34b43dd / 2a5dfa64"; exit 1; }
for f in train fit_O fit_L val_sub; do
  want=$(python3 -c "import json; print(json.load(open('data/exp42/MANIFEST.json'))['files']['$f'])")
  [ "$(shasum -a 256 data/exp42/$f.jsonl | cut -d' ' -f1)" = "$want" ] || { log "data/exp42/$f.jsonl changed"; exit 1; }
done
want=$(python3 -c "import json; print(json.load(open('data/exp48/MANIFEST.json'))['sha256'])")
[ "$(shasum -a 256 data/exp48/quads.jsonl | cut -d' ' -f1)" = "$want" ] || { log "data/exp48/quads.jsonl changed"; exit 1; }
python3 -c "import json, sys; p = json.load(open('data/exp48/PLAN.json')); sys.exit(0 if p['updates'] == 576 and p['quads_split'] == 0 and p['batches_two_whole_quads_O_and_L'] else 1)" \
  || { log "data/exp48/PLAN.json does not show 576 updates of two whole quads"; exit 1; }
[ ! -e runs/exp48-c ] && [ ! -e runs/exp48-x ] || { log "an exp48 run directory already exists"; exit 1; }
DEADLINE_FILE=runs/exp48.deadline
[ -f "$DEADLINE_FILE" ] || echo $(( $(date +%s) + 90 * 60 )) > "$DEADLINE_FILE"
DEADLINE=$(cat "$DEADLINE_FILE")
RESERVE=120
. scripts/exp38/steps.sh
log "exp48: start EXP45-G $INIT_SHA; deadline $(date -r "$DEADLINE" '+%F %T')"
Q_SHA=$(shasum -a 256 data/exp48/quads.jsonl | cut -d' ' -f1)
over() { [ "$1" = 124 ] || [ "$1" = 125 ]; }
arm() {   # c|x weight
  local run=runs/exp48-$1 rc
  STATUS=$run-status.txt
  mkdir -p "$run"
  python3 -c "import json; json.dump({'arm': '$1', 'objective': 'ce', 'interaction_weight': $2, 'quads_sha256': '$Q_SHA', 'init': '$INIT', 'init_sha256': {'model.safetensors': '$INIT_SHA', 'config.json': '$INIT_CFG'}, 'data': 'data/exp42/train.jsonl', 'data_sha256': '$(shasum -a 256 data/exp42/train.jsonl | cut -d' ' -f1)', 'git_commit': '$(git rev-parse HEAD)', 'started': '$(date '+%F %T')'}, open('$run/START.json', 'w'), indent=1)"
  log "arm $1 (interaction weight $2)"
  step train uv run python scripts/exp48/watch.py "$run/stage2" "$run-watch.txt" -- uv run devision-pipeline configs/exp48-$1.yaml || return 1
  uv run python scripts/v04b/freeze.py freeze "$run" || { status freeze failed; return 1; }
  mkdir -p "$run/eval"
  record() { uv run python scripts/v04b/freeze.py record "$run" "$1" || { status "eval:$1" failed; log "identity changed: $1"; return 1; }; }
  for s in fit_O=data/exp42/fit_O.jsonl fit_L=data/exp42/fit_L.jsonl rl_tune=data/real_layout/questions_tune.jsonl rl_tune_L=data/exp40/rl_tune_L.jsonl; do
    step "eval:${s%%=*}" uv run devision-eval --checkpoint $run/stage2 --data "${s#*=}" --data-root data --out $run/eval/${s%%=*}.json \
      --role monitoring --no-temperature --max-questions 8
    rc=$?; over $rc && return $rc; [ $rc = 0 ] && record "${s%%=*}"
  done
  for s in exist=data/r13b/panel_exist.jsonl science=data/r13b/panel_science.jsonl; do
    step "eval:${s%%=*}" uv run devision-eval --checkpoint $run/stage2 --data "${s#*=}" --data-root data --out $run/eval/${s%%=*}.json \
      --role monitoring --no-temperature --max-questions 8
    rc=$?; over $rc && return $rc; [ $rc = 0 ] && record "${s%%=*}"
  done
  return 0
}
arm c 0.0
arm x 1.0
final_report uv run python scripts/exp48/report.py > runs/exp48-report.txt 2>&1
rc=$?
cat runs/exp48-report.txt
log "final report (exit $rc)"
exit "$rc"
