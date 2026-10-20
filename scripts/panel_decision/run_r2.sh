#!/usr/bin/env bash
# EXP76 r2 (EXP76-RETRY-PREP-01; docs/experiments/2026-10-20-exp76-r2-plan.md): an independent retry of EXP76 in its own
# paths. One start ever (runs/exp76-panel-decision-r2.started), serial: process checks -> preflight_r2 (no model; the
# accepted checks, fresh r2 paths, the first attempt's evidence unchanged; writes START with retry_of; refused -> exit 12,
# nothing trained) -> the pipeline under scripts/panel_decision/guarded.py (its own process group; a GPU command-buffer
# error / recovery line in its output or logs ends that group: TERM, KILL after 15 s; the limit the same way) ->
# report_r2 (no model; execution validity first, then the accepted D2 report). Hard limit start + 5,700 s (EXP76_LIMIT
# for fixtures): the guarded step ends 120 s before it, TERM / KILL grace included; the last 120 s are for the report.
# This runner is not in the guarded group and is never signalled by the guard. Nothing is retried or chained.
#   nohup bash scripts/panel_decision/run_r2.sh > runs/exp76-panel-decision-r2.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
NAME=exp76-panel-decision-r2
log() { echo "== $(date '+%F %T') $*"; }
MARK=runs/$NAME.started
[ ! -e "$MARK" ] || { log "$MARK exists: EXP76 r2 was already started once"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v -e "scripts/panel_decision/run_r2.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
ENTRY='devision-calibrate|devision-eval|devision-pipeline|devision-train|devision-align|train_main|align_main|scripts/(calib|livek|protect|confirm|panel_learn|panel_bind|panel_number|mlm_repro)/|scripts/exp[0-9]+/'
own=$(ps -axo pid=,command= | grep -E "$ENTRY" | grep -v -e "grep -E" -e "scripts/panel_decision/run_r2.sh")
[ -z "$own" ] || { log "other model processes are running:"; echo "$own"; exit 1; }
uv run python scripts/panel_decision/preflight_r2.py || { log "preflight refused: see runs/$NAME-preflight.json; nothing trained"; exit 12; }
mkdir -p runs
DEADLINE=$(( $(date +%s) + ${EXP76_LIMIT:-5700} ))
RESERVE=${EXP76_RESERVE:-120}
{ echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"
  for f in configs/$NAME.yaml scripts/panel_decision/run_r2.sh scripts/panel_decision/preflight_r2.py scripts/panel_decision/guarded.py \
           scripts/panel_decision/report_r2.py scripts/panel_learn/report.py runs/$NAME-preflight.json runs/$NAME.START.json; do
    echo "sha256 $(shasum -a 256 "$f")"; done; } > "$MARK" || { log "cannot write $MARK"; exit 1; }
log "EXP76 r2: deadline $(date -r "$DEADLINE" '+%F %T')"
# one absolute deadline for the whole model phase (EXP76-RETRY-R1-01): the guard starts any stop early enough to end
# by GUARD_END (polling, TERM grace and KILL check included) and returns by then even if the group cannot be reaped;
# run_capped ends the guard itself at the reserve boundary as an outer bound; the report keeps its full RESERVE.
GUARD_END=$(( DEADLINE - RESERVE - 5 ))
run_capped $(( DEADLINE - RESERVE - $(date +%s) )) 2 uv run python scripts/panel_decision/guarded.py --record runs/$NAME-guard.json \
  --watch runs/$NAME/logs --out runs/$NAME-pipeline.out --deadline "$GUARD_END" -- uv run devision-pipeline configs/$NAME.yaml
rc=$?
log "guarded pipeline exit $rc"
[ $rc = 71 ] && log "cleanup of the model process group not confirmed: no further model is started"
. scripts/exp38/steps.sh
final_report uv run python scripts/panel_decision/report_r2.py
rrc=$?
log "report exit $rrc"
[ $rc = 0 ] && rc=$rrc
exit $rc
