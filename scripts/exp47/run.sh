#!/usr/bin/env bash
# exp47 residual-difference diagnosis of the frozen EXP45-G (docs/experiments/2026-10-18-exp47-residual-plan.md; critic
# EXP47-PREP-01..03). Only after the critic's permission. One bounded run of scripts/exp47/diagnose.py (EXP46's 48 units /
# 96 questions, six modes; no training, no weight written), then always the model-free --finalize. Hard deadline start +
# 10 minutes with the last 60 s for the report; no retry.
#   nohup bash scripts/exp47/run.sh > runs/exp47.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/exp47/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
uv run python scripts/v04b/freeze.py check runs/exp45-g || { log "EXP45-G is not its frozen identity"; exit 1; }
[ "$(shasum -a 256 data/exp46/units.jsonl | cut -d' ' -f1)" = ad68597eb5765422bfd2bf7733a854764ed8a51b754a70629ee5b1a7306ae32b ] \
  || { log "data/exp46/units.jsonl changed"; exit 1; }
[ "$(shasum -a 256 runs/exp46/details.jsonl | cut -d' ' -f1)" = 4d7fe4a276f603017ddfae77ca633c186085c0343deaf20ddc7fbae8607131b8 ] \
  && [ "$(shasum -a 256 runs/exp46/report.json | cut -d' ' -f1)" = 1ec8da995767eb0a3f84e894aed56650d5f167d26bd990a8d966e03c46540d75 ] \
  || { log "EXP46's accepted outputs changed"; exit 1; }
[ ! -e runs/exp47 ] || { log "runs/exp47 exists"; exit 1; }
DEADLINE=$(( $(date +%s) + 10 * 60 ))
echo "$DEADLINE" > runs/exp47.deadline
log "exp47: deadline $(date -r "$DEADLINE" '+%F %T')"
run_capped $(( DEADLINE - $(date +%s) - 60 - 15 )) 15 uv run python scripts/exp47/diagnose.py
rc=$?
log "diagnosis exit $rc"
uv run python scripts/exp47/diagnose.py --finalize "$rc"
frc=$?
log "final report (exit $frc)"
exit "$frc"
