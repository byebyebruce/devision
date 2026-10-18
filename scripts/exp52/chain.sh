#!/usr/bin/env bash
# exp52 on one Mac (docs/experiments/2026-10-18-exp52-number-reader-plan.md; critic EXP52-PREP-01). Only after the
# critic's model permission (EXP52-RUN-01). Arms N (digit loss weight 1) and Z (weight 0), both from EXP50-T's frozen end
# with the same number reader init; in the order the judgement needs them:
#   1. N: train (scripts/exp52/watch.py), freeze, ordinary decide on fit_O / fit_L / rl_tune / rl_tune_L;
#   2. Z: the same;
#   3. number readout + content-use check (scripts/exp52/readout.py) for N, then Z;
#   4. ScienceQA temp / force transfer (true and fixed wrong picture) for N, Z and EXP50-T (runs/exp52-base);
#   5. exist / science for N, then Z;
#   6. scripts/exp52/report.py.
# Every evaluation is devision-eval (raw temperature 1, max 8 questions per request) recorded under the arm's frozen
# identity. Deadline start + 120 minutes with the last 120 s for the report; each step bounded; fresh only (existing run
# directories are refused), no resume, no retry; at the limit the rest stays missing and the report says so.
#   nohup bash scripts/exp52/chain.sh > runs/exp52.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp52/chain.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
INIT=runs/exp50-t/stage2
INIT_SHA=1a8e5439d5c4f64fd1dbce551f9c2df55152b2aea7cc3a93fc01eb5d364ca007
INIT_CFG=99c6246e201c5c38c7f62c6831050792858b5aecf2b4be352c5cb7cbc3231fc6
uv run python scripts/v04b/freeze.py check runs/exp50-t || { log "EXP50-T is not its frozen identity"; exit 1; }
[ "$(shasum -a 256 $INIT/model.safetensors | cut -d' ' -f1)" = "$INIT_SHA" ] && [ "$(shasum -a 256 $INIT/config.json | cut -d' ' -f1)" = "$INIT_CFG" ] \
  || { log "EXP50-T is not 1a8e5439 / 99c6246e"; exit 1; }
want() { python3 -c "import json; d = json.load(open('$1')); print(d$2)"; }
[ "$(shasum -a 256 data/exp42/train.jsonl | cut -d' ' -f1)" = "$(want data/exp42/MANIFEST.json "['files']['train']")" ] || { log "train changed"; exit 1; }
[ "$(shasum -a 256 data/exp50/val_quarter.jsonl | cut -d' ' -f1)" = "$(want data/exp50/MANIFEST.json "['files']['val_quarter']")" ] || { log "val_quarter changed"; exit 1; }
[ "$(shasum -a 256 data/exp48/quads.jsonl | cut -d' ' -f1)" = "$(want data/exp48/MANIFEST.json "['sha256']")" ] || { log "quads changed"; exit 1; }
[ "$(shasum -a 256 data/exp52/numbers.jsonl | cut -d' ' -f1)" = "$(want data/exp52/MANIFEST.json "['sha256']")" ] || { log "numbers changed"; exit 1; }
for d in runs/exp52-n runs/exp52-z runs/exp52-base runs/exp52.deadline; do [ ! -e $d ] || { log "$d exists"; exit 1; }; done
echo $(( $(date +%s) + 120 * 60 )) > runs/exp52.deadline
DEADLINE=$(cat runs/exp52.deadline)
RESERVE=120
. scripts/exp38/steps.sh
log "exp52: start EXP50-T $INIT_SHA; deadline $(date -r "$DEADLINE" '+%F %T')"
over() { [ "$1" = 124 ] || [ "$1" = 125 ]; }
STOP=0
record() { uv run python scripts/v04b/freeze.py record "$1" "$2" || { status "eval:$2" failed; log "identity changed: $1 $2"; return 1; }; }
evalset() {   # run name data
  [ $STOP = 1 ] && { status "eval:$2" missing; return 125; }
  step "eval:$2" uv run devision-eval --checkpoint $1/stage2 --data "$3" --data-root data --out $1/eval/$2.json \
    --role monitoring --no-temperature --max-questions 8
  local rc=$?
  over $rc && STOP=1
  [ $rc = 0 ] && record "$1" "$2"
  return $rc
}
train_arm() {   # n|z weight
  local run=runs/exp52-$1
  STATUS=$run-status.txt
  mkdir -p "$run"
  python3 -c "import json; json.dump({'arm': '$1'.upper(), 'number_weight': $2, 'numbers_sha256': '$(shasum -a 256 data/exp52/numbers.jsonl | cut -d' ' -f1)', 'init': '$INIT', 'init_sha256': {'model.safetensors': '$INIT_SHA', 'config.json': '$INIT_CFG'}, 'data': 'data/exp42/train.jsonl', 'data_sha256': '$(shasum -a 256 data/exp42/train.jsonl | cut -d' ' -f1)', 'git_commit': '$(git rev-parse HEAD)', 'started': '$(date '+%F %T')'}, open('$run/START.json', 'w'), indent=1)"
  [ $STOP = 1 ] && { status train missing; return 125; }
  log "arm $1 (number weight $2)"
  step train uv run python scripts/exp52/watch.py "$run/stage2" "$run-watch.txt" -- uv run devision-pipeline configs/exp52-$1.yaml
  local rc=$?
  over $rc && STOP=1
  [ $rc = 0 ] || return $rc
  uv run python scripts/v04b/freeze.py freeze "$run" || { status freeze failed; return 1; }
  mkdir -p "$run/eval"
  for s in fit_O=data/exp42/fit_O.jsonl fit_L=data/exp42/fit_L.jsonl rl_tune=data/real_layout/questions_tune.jsonl rl_tune_L=data/exp40/rl_tune_L.jsonl; do
    evalset "$run" "${s%%=*}" "${s#*=}"
  done
  return 0
}
trained() { grep -q "^train completed" runs/exp52-$1-status.txt 2>/dev/null && [ -f runs/exp52-$1/FROZEN.json ]; }
train_arm n 1.0
train_arm z 0.0
for a in n z; do
  trained $a || continue
  STATUS=runs/exp52-$a-status.txt
  [ $STOP = 1 ] && { status readout missing; continue; }
  step readout uv run python scripts/exp52/readout.py --run runs/exp52-$a
  over $? && STOP=1
done
SQA="sqa_temp=data/exp21/dev_sqa_temp.jsonl sqa_force=data/exp21/dev_sqa_force.jsonl sqa_temp_mism=data/exp21/dev_sqa_temp_mism.jsonl sqa_force_mism=data/exp21/dev_sqa_force_mism.jsonl"
for a in n z; do
  trained $a || continue
  STATUS=runs/exp52-$a-status.txt
  for s in $SQA; do evalset runs/exp52-$a "${s%%=*}" "${s#*=}"; done
done
mkdir -p runs/exp52-base/eval      # EXP50-T on the transfer slices, recorded by hash (its own run directory is not touched)
STATUS=runs/exp52-base-status.txt
python3 -c "import json; json.dump({'checkpoint': '$INIT', 'sha256': {'model.safetensors': '$INIT_SHA', 'config.json': '$INIT_CFG'}, 'git_commit': '$(git rev-parse HEAD)', 'started': '$(date '+%F %T')'}, open('runs/exp52-base/START.json', 'w'), indent=1)"
for s in $SQA; do
  n=${s%%=*}
  [ $STOP = 1 ] && { status "eval:$n" missing; continue; }
  step "eval:$n" uv run devision-eval --checkpoint $INIT --data "${s#*=}" --data-root data --out runs/exp52-base/eval/$n.json \
    --role monitoring --no-temperature --max-questions 8
  rc=$?
  over $rc && STOP=1
  [ $rc = 0 ] && python3 -c "
import hashlib, json, os
p = 'runs/exp52-base/RECORDS.json'
r = json.load(open(p)) if os.path.exists(p) else {}
r['$n'] = {f: hashlib.sha256(open('runs/exp52-base/eval/' + f, 'rb').read()).hexdigest() for f in ('$n.json', '$n.details.jsonl')}
json.dump(r, open(p, 'w'), indent=1)"
done
for a in n z; do
  trained $a || continue
  STATUS=runs/exp52-$a-status.txt
  for s in exist=data/r13b/panel_exist.jsonl science=data/r13b/panel_science.jsonl; do evalset runs/exp52-$a "${s%%=*}" "${s#*=}"; done
done
final_report uv run python scripts/exp52/report.py > runs/exp52-report.txt 2>&1
rc=$?
cat runs/exp52-report.txt
log "final report (exit $rc)"
exit "$rc"
