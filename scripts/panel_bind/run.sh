#!/usr/bin/env bash
# PANEL-BIND-PREP-01 direct number-binding probe (docs/experiments/2026-10-20-panel-bind-probe-plan.md). One start ever
# (runs/panel-bind.started), serial: START (no model) -> probe start -> probe d1 -> probe exp26 -> report (no model).
# Hard limit start + 1,200 s (BIND_LIMIT for fixtures), each probe ends 120 s before it (TERM, KILL 15 s later); the last
# 120 s are for the report. A failed / stopped probe leaves its end point missing; the report still runs and says so.
#   nohup bash scripts/panel_bind/run.sh > runs/panel-bind.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
MARK=runs/panel-bind.started
[ ! -e "$MARK" ] || { log "$MARK exists: the probe was already started once"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/panel_bind/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
ENTRY='devision-calibrate|devision-eval|devision-pipeline|devision-train|devision-align|train_main|align_main|scripts/(calib|livek|protect|confirm|panel_learn)/|scripts/exp[0-9]+/'
own=$(ps -axo pid=,command= | grep -E "$ENTRY" | grep -v -e "grep -E" -e "scripts/panel_bind/run.sh")
[ -z "$own" ] || { log "other model processes are running:"; echo "$own"; exit 1; }
mkdir -p runs
DEADLINE=$(( $(date +%s) + ${BIND_LIMIT:-1200} ))
{ echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"
  for f in scripts/panel_bind/run.sh scripts/panel_bind/probe.py scripts/panel_bind/report.py src/devision/train/align.py; do
    echo "sha256 $(shasum -a 256 "$f")"; done; } > "$MARK" || { log "cannot write $MARK"; exit 1; }
STATUS=runs/panel-bind-status.txt
RESERVE=${BIND_RESERVE:-120}
. scripts/exp38/steps.sh
log "panel-bind probe: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
step start uv run python scripts/panel_bind/report.py start; rc=$?; codes="$codes start=$rc"
if [ $rc = 0 ]; then
  for e in start:runs/v04-k/cand d1:runs/exp74-panel-d1/d1 exp26:runs/exp26-align/stage1; do
    name=${e%%:*}; ck=${e#*:}
    step "$name" uv run python scripts/panel_bind/probe.py --checkpoint "$ck" --out "runs/panel-bind/$name"; r=$?; codes="$codes $name=$r"
  done
fi
log "steps:$codes"
final_report uv run python scripts/panel_bind/report.py report
rrc=$?
log "report exit $rrc"
exit $rrc
