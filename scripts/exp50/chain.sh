#!/usr/bin/env bash
# exp50 on one Mac (docs/experiments/2026-10-18-exp50-local-views-plan.md; critic EXP50-PREP-01..05). Only after the
# critic's model permission. Grid n12 / overlap 0.25 (critic EXP50-R1-01). Arm T (configs/exp50-t.yaml, local views from the original picture) then arm C
# (configs/exp50-c.yaml, from the reduced picture), both from EXP48-X's frozen end, each: training under
# scripts/exp50/watch.py, weights / config frozen (scripts/v04b/freeze.py), the four main decision sets at temperature 1,
# exist / science -- every file recorded under the arm's frozen identity, all through ordinary decide (devision-eval);
# then scripts/exp50/report.py (X's records reused after their frozen check). Deadline start + 90 minutes with the last
# 120 s for the report; fresh only (an existing run directory is refused), one run per arm, no resume, no retry, nothing
# started after it.
#   nohup bash scripts/exp50/chain.sh > runs/exp50.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp50/chain.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
INIT=runs/exp48-x/stage2
INIT_SHA=a8b92cb93a56aa4653f516cbf2a4a38e88fbd64d69ac0f820acc6307f519b232
INIT_CFG=c017a469b35fcccf7e2fb2953dcd334be345bfd4fac493d64f4fd302155140df
uv run python scripts/v04b/freeze.py check runs/exp48-x || { log "EXP48-X is not its frozen identity"; exit 1; }
[ "$(shasum -a 256 $INIT/model.safetensors | cut -d' ' -f1)" = "$INIT_SHA" ] && [ "$(shasum -a 256 $INIT/config.json | cut -d' ' -f1)" = "$INIT_CFG" ] \
  || { log "EXP48-X is not a8b92cb9 / c017a469"; exit 1; }
for f in train fit_O fit_L val_sub; do
  want=$(python3 -c "import json; print(json.load(open('data/exp42/MANIFEST.json'))['files']['$f'])")
  [ "$(shasum -a 256 data/exp42/$f.jsonl | cut -d' ' -f1)" = "$want" ] || { log "data/exp42/$f.jsonl changed"; exit 1; }
done
check() { want=$(python3 -c "import json; print(json.load(open('data/exp50/MANIFEST.json'))['files']['$1'])")
  [ "$(shasum -a 256 "$2" | cut -d' ' -f1)" = "$want" ] || { log "$2 changed"; exit 1; }; }
check val_quarter data/exp50/val_quarter.jsonl
check quads data/exp48/quads.jsonl
check plan data/exp50/PLAN.json
check geometry data/exp50/geometry.json
python3 -c "import json, sys; p = json.load(open('data/exp50/PLAN.json')); sys.exit(0 if p['updates'] == 576 and p['quads_split'] == 0 and p['batches_two_whole_quads_O_and_L'] and all(p['same_as_exp48'].values()) else 1)" \
  || { log "data/exp50/PLAN.json does not show EXP48's 576 updates of two whole quads"; exit 1; }
[ ! -e runs/exp50-t ] && [ ! -e runs/exp50-c ] || { log "an exp50 run directory already exists"; exit 1; }
DEADLINE_FILE=runs/exp50.deadline
[ ! -e "$DEADLINE_FILE" ] || { log "$DEADLINE_FILE exists (no second start)"; exit 1; }
echo $(( $(date +%s) + 90 * 60 )) > "$DEADLINE_FILE"
DEADLINE=$(cat "$DEADLINE_FILE")
RESERVE=120
. scripts/exp38/steps.sh
log "exp50: start EXP48-X $INIT_SHA; deadline $(date -r "$DEADLINE" '+%F %T')"
Q_SHA=$(shasum -a 256 data/exp48/quads.jsonl | cut -d' ' -f1)
over() { [ "$1" = 124 ] || [ "$1" = 125 ]; }
arm() {   # t|c T|C
  local run=runs/exp50-$1 rc
  STATUS=$run-status.txt
  mkdir -p "$run"
  python3 -c "import json; json.dump({'arm': '$2', 'objective': 'ce', 'interaction_weight': 1.0, 'quads_sha256': '$Q_SHA', 'local_views': 12, 'local_overlap': 0.25, 'local_source': '$2', 'init': '$INIT', 'init_sha256': {'model.safetensors': '$INIT_SHA', 'config.json': '$INIT_CFG'}, 'data': 'data/exp42/train.jsonl', 'data_sha256': '$(shasum -a 256 data/exp42/train.jsonl | cut -d' ' -f1)', 'git_commit': '$(git rev-parse HEAD)', 'started': '$(date '+%F %T')'}, open('$run/START.json', 'w'), indent=1)"
  log "arm $2 (local views from $([ "$2" = T ] && echo the original picture || echo the reduced picture))"
  step train uv run python scripts/exp50/watch.py "$run/stage2" "$run-watch.txt" -- uv run devision-pipeline configs/exp50-$1.yaml || return 1
  uv run python scripts/v04b/freeze.py freeze "$run" || { status freeze failed; return 1; }
  mkdir -p "$run/eval"
  record() { uv run python scripts/v04b/freeze.py record "$run" "$1" || { status "eval:$1" failed; log "identity changed: $1"; return 1; }; }
  for s in fit_O=data/exp42/fit_O.jsonl fit_L=data/exp42/fit_L.jsonl rl_tune=data/real_layout/questions_tune.jsonl rl_tune_L=data/exp40/rl_tune_L.jsonl \
           exist=data/r13b/panel_exist.jsonl science=data/r13b/panel_science.jsonl; do
    step "eval:${s%%=*}" uv run devision-eval --checkpoint $run/stage2 --data "${s#*=}" --data-root data --out $run/eval/${s%%=*}.json \
      --role monitoring --no-temperature --max-questions 8
    rc=$?; over $rc && return $rc; [ $rc = 0 ] && record "${s%%=*}"
  done
  return 0
}
arm t T
arm c C
final_report uv run python scripts/exp50/report.py > runs/exp50-report.txt 2>&1
rc=$?
cat runs/exp50-report.txt
log "final report (exit $rc)"
exit "$rc"
