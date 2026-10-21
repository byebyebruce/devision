#!/usr/bin/env bash
# exp44 on one Mac (docs/experiments/2026-10-18-exp44-number-plan.md; critic EXP44-PREP-01..04). Only after the critic's
# model permission. In order of priority: arm J (configs/exp44-j.yaml) trained under scripts/exp44/watch.py; its weights /
# config frozen (scripts/v04b/freeze.py); J's decision readout at temperature 1 (fit_O, fit_L, rl_tune, rl_tune_L); J's
# number readout through the frozen MLM head (scripts/exp44/aux_readout.py, same four sets); exist / science (risk);
# then, lowest priority, exp40-M's number readout on fit_O / rl_tune; every J file recorded under J's frozen identity;
# scripts/exp44/report.py (M and C reuse exp42's frozen records). Deadline start + 75 minutes with the last 120 s for the
# report; one run, no retry, nothing started after it.
#   nohup bash scripts/exp44/chain.sh > runs/exp44.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp44/chain.sh")
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
uv run python -c "
import json, sys
sys.argv = ['x']
from importlib import util
s = util.spec_from_file_location('c', 'scripts/exp44/check.py'); c = util.module_from_spec(s); s.loader.exec_module(c)
want = json.load(open('data/exp44/CHECK.json'))['head']
sys.exit(0 if c.head_identity() == want else 1)" || { log "the cached MLM head is not the checked one"; exit 1; }
[ ! -e runs/exp44-j ] && [ ! -e runs/exp44-m-aux ] || { log "an exp44 run directory already exists"; exit 1; }
DEADLINE_FILE=runs/exp44.deadline
[ -f "$DEADLINE_FILE" ] || echo $(( $(date +%s) + 75 * 60 )) > "$DEADLINE_FILE"
DEADLINE=$(cat "$DEADLINE_FILE")
RESERVE=120
. scripts/exp38/steps.sh
log "exp44: start exp40-M $INIT_SHA; deadline $(date -r "$DEADLINE" '+%F %T')"
RUN=runs/exp44-j
STATUS=$RUN-status.txt
mkdir -p "$RUN"
python3 -c "import json; json.dump({'arm': 'J', 'objective': 'ce', 'aux_weight': 1.0, 'aux_slots': 8, 'init': '$INIT', 'init_sha256': {'model.safetensors': '$INIT_SHA', 'config.json': '$INIT_CFG'}, 'data': 'data/exp42/train.jsonl', 'data_sha256': '$(shasum -a 256 data/exp42/train.jsonl | cut -d' ' -f1)', 'git_commit': '$(git rev-parse HEAD)', 'started': '$(date '+%F %T')'}, open('$RUN/START.json', 'w'), indent=1)"
stop() { final_report uv run python scripts/exp44/report.py > runs/exp44-report.txt 2>&1; local rc=$?; cat runs/exp44-report.txt; log "final report (exit $rc)"; exit "$rc"; }
step train uv run python scripts/exp44/watch.py "$RUN/stage2" "$RUN-watch.txt" -- uv run devision-pipeline configs/exp44-j.yaml || stop
uv run python scripts/v04b/freeze.py freeze "$RUN" || { status freeze failed; stop; }
mkdir -p "$RUN/eval"
record() { uv run python scripts/v04b/freeze.py record "$RUN" "$1" || { status "eval:$1" failed; log "identity changed: $1"; return 1; }; }
over() { [ "$1" = 124 ] || [ "$1" = 125 ]; }
for s in fit_O=data/exp42/fit_O.jsonl fit_L=data/exp42/fit_L.jsonl rl_tune=data/real_layout/questions_tune.jsonl rl_tune_L=data/exp40/rl_tune_L.jsonl; do
  n=${s%%=*}; d=${s#*=}
  step "eval:$n" uv run devision-eval --checkpoint $RUN/stage2 --data "$d" --data-root data --out $RUN/eval/$n.json \
    --role monitoring --no-temperature --max-questions 8
  rc=$?; over $rc && stop; [ $rc = 0 ] && record "$n"
done
for s in fit_O=data/exp42/fit_O.jsonl fit_L=data/exp42/fit_L.jsonl rl_tune=data/real_layout/questions_tune.jsonl rl_tune_L=data/exp40/rl_tune_L.jsonl; do
  n=aux_${s%%=*}; d=${s#*=}
  step "eval:$n" uv run python scripts/exp44/aux_readout.py $RUN/stage2 "$d" $RUN/eval "$n"
  rc=$?; over $rc && stop; [ $rc = 0 ] && record "$n"
done
for s in exist=data/r13b/panel_exist.jsonl science=data/r13b/panel_science.jsonl; do
  n=${s%%=*}; d=${s#*=}
  step "eval:$n" uv run devision-eval --checkpoint $RUN/stage2 --data "$d" --data-root data --out $RUN/eval/$n.json \
    --role monitoring --no-temperature --max-questions 8
  rc=$?; over $rc && stop; [ $rc = 0 ] && record "$n"
done
STATUS=runs/exp44-m-aux-status.txt
mkdir -p runs/exp44-m-aux/eval
for s in fit_O=data/exp42/fit_O.jsonl rl_tune=data/real_layout/questions_tune.jsonl; do
  n=aux_${s%%=*}; d=${s#*=}
  step "eval:$n" uv run python scripts/exp44/aux_readout.py $INIT "$d" runs/exp44-m-aux/eval "$n"
  rc=$?; over $rc && break
done
stop
