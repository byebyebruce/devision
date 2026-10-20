#!/usr/bin/env bash
# EXP76 (PANEL-DECISION-PREP-01; docs/experiments/2026-10-20-panel-decision-plan.md). One start ever
# (runs/exp76-panel-decision.started), serial: preflight (no model; refused -> exit 12, nothing trained) -> the decision
# pass + its exist / science probes (configs/exp76-panel-decision.yaml) -> the accepted D2 report (no model) on this run.
# Hard limit start + 5,700 s (EXP76_LIMIT for fixtures), the training / probe step ends 120 s before it (TERM, KILL 15 s
# later); the last 120 s are for the report. EXP74's D2 and its guard are not used. Nothing is chained after it.
#   nohup bash scripts/panel_decision/run.sh > runs/exp76-panel-decision.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
MARK=runs/exp76-panel-decision.started
[ ! -e "$MARK" ] || { log "$MARK exists: EXP76 was already started once"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/panel_decision/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
ENTRY='devision-calibrate|devision-eval|devision-pipeline|devision-train|devision-align|train_main|align_main|scripts/(calib|livek|protect|confirm|panel_learn|panel_bind|panel_number|mlm_repro)/|scripts/exp[0-9]+/'
own=$(ps -axo pid=,command= | grep -E "$ENTRY" | grep -v -e "grep -E" -e "scripts/panel_decision/run.sh")
[ -z "$own" ] || { log "other model processes are running:"; echo "$own"; exit 1; }
uv run python scripts/panel_decision/preflight.py || { log "preflight refused: see runs/exp76-preflight.json; nothing trained"; exit 12; }
mkdir -p runs
DEADLINE=$(( $(date +%s) + ${EXP76_LIMIT:-5700} ))
{ echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"
  for f in configs/exp76-panel-decision.yaml scripts/panel_decision/run.sh scripts/panel_decision/preflight.py scripts/panel_learn/report.py \
           runs/exp76-preflight.json data/panel_learn/MANIFEST.json; do echo "sha256 $(shasum -a 256 "$f")"; done; } > "$MARK" \
  || { log "cannot write $MARK"; exit 1; }
STATUS=runs/exp76-panel-decision-status.txt
RESERVE=${EXP76_RESERVE:-120}
. scripts/exp38/steps.sh
log "EXP76: deadline $(date -r "$DEADLINE" '+%F %T')"
step decision uv run devision-pipeline configs/exp76-panel-decision.yaml; rc=$?
log "steps: decision=$rc"
final_report uv run python scripts/panel_learn/report.py d2 runs/exp76-panel-decision runs/v04-k/eval configs/exp76-panel-decision.yaml
rrc=$?
log "report exit $rrc"
[ $rc = 0 ] && rc=$rrc
exit $rc
