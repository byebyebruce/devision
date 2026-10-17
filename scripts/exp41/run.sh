#!/usr/bin/env bash
# exp41 oracle region diagnosis (docs/experiments/2026-10-18-exp41-region-plan.md; critic EXP41-PREP-01). Only after
# the exp40 chain has ended AND the critic's separate permission: exp40-M's frozen endpoint on data/exp41/questions.jsonl
# (1,320 questions, 840 pictures), capped at 15 minutes (TERM, KILL 15 s later), then scripts/exp41/report.py.
# One run, no retry, nothing else started.
#   nohup bash scripts/exp41/run.sh > runs/exp41.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes)
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
want=$(python3 -c "import json; print(json.load(open('data/exp41/MANIFEST.json'))['sha256'])")
[ "$(shasum -a 256 data/exp41/questions.jsonl | cut -d' ' -f1)" = "$want" ] || { log "data/exp41/questions.jsonl changed"; exit 1; }
uv run python scripts/v04b/freeze.py check runs/exp40-m || { log "exp40-M is not its frozen identity"; exit 1; }
[ ! -e runs/exp41 ] || { log "runs/exp41 exists"; exit 1; }
mkdir -p runs/exp41
log "exp41: exp40-M on data/exp41/questions.jsonl, 15-minute cap"
run_capped 900 15 uv run devision-eval --checkpoint runs/exp40-m/stage2 --data data/exp41/questions.jsonl --data-root data \
  --out runs/exp41/m.json --role monitoring
rc=$?
log "evaluation exit $rc"
[ "$rc" = 0 ] || exit "$rc"
uv run python scripts/exp41/report.py
