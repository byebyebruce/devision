#!/usr/bin/env bash
# EXP74 runner (PANEL-LEARN-PREP-01; docs/experiments/2026-10-19-panel-learn-plan.md). Two phases, each its own model-run
# permission, one start each (runs/exp74-<phase>.started, never removed), no retry, nothing chained after it:
#   d1: consistency (the frozen candidate's decision weights reproduce the sealed predictions on the sealed read states)
#       -> D1 binding alignment (configs/exp74-panel-d1.yaml) -> gate (scripts/exp24/gate.py: gap >= 0.30, exit 10 = stop);
#       hard limit start + 75 min (EXP74_LIMIT for fixtures), the last 120 s for the gate report.
#   d2: refuses unless D1's gate passes; D2 (configs/exp74-panel-d2.yaml: training, monitors, exist / science probes)
#       -> report; hard limit start + 95 min, the last 120 s for the report.
#   EXP74_PHASE=d1 nohup bash scripts/panel_learn/run.sh > runs/exp74-d1.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
PHASE=${EXP74_PHASE:-}
case "$PHASE" in d1) LIMIT=${EXP74_LIMIT:-4500};; d2) LIMIT=${EXP74_LIMIT:-5700};; *) log "EXP74_PHASE must be d1 or d2"; exit 2;; esac
MARK=runs/exp74-$PHASE.started
[ ! -e "$MARK" ] || { log "$MARK exists: phase $PHASE was already started once"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/panel_learn/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
ENTRY='devision-calibrate|devision-eval|devision-pipeline|devision-train|devision-align|train_main|align_main|scripts/(calib|livek|protect|confirm)/|scripts/exp[0-9]+/'
own=$(ps -axo pid=,command= | grep -E "$ENTRY" | grep -v -e "grep -E" -e "scripts/panel_learn/run.sh")
[ -z "$own" ] || { log "other model processes are running:"; echo "$own"; exit 1; }
if [ "$PHASE" = d2 ]; then
  uv run python scripts/panel_learn/report.py d1 > /dev/null || { log "D1's gate did not pass (or D1 is missing): no D2"; exit 1; }
fi
mkdir -p runs
DEADLINE=$(( $(date +%s) + LIMIT ))
{ echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"
  for f in configs/exp74-panel-$PHASE.yaml scripts/panel_learn/run.sh scripts/panel_learn/report.py data/panel_learn/MANIFEST.json; do
    echo "sha256 $(shasum -a 256 "$f")"; done; } > "$MARK" || { log "cannot write $MARK"; exit 1; }
STATUS=runs/exp74-$PHASE-status.txt
RESERVE=${EXP74_RESERVE:-120}
. scripts/exp38/steps.sh
log "exp74 $PHASE: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
run_d1() {
  step consistency uv run python scripts/panel_learn/consistency.py runs/exp74-panel-d1/consistency.json; local rc=$?
  codes="$codes consistency=$rc"; [ $rc = 0 ] || return $rc
  step d1 uv run devision-pipeline configs/exp74-panel-d1.yaml; rc=$?; codes="$codes d1=$rc"; return $rc
}
run_d2() {
  step d2 uv run devision-pipeline configs/exp74-panel-d2.yaml; local rc=$?; codes="$codes d2=$rc"; return $rc
}
if [ "$PHASE" = d1 ]; then run_d1; else run_d2; fi
rc=$?
log "steps:$codes"
if [ "$PHASE" = d1 ]; then final_report uv run python scripts/panel_learn/report.py d1; else final_report uv run python scripts/panel_learn/report.py d2; fi
rrc=$?
log "report exit $rrc"
[ $rc = 0 ] && rc=$rrc
exit "$rc"
