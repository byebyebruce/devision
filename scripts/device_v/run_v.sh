#!/usr/bin/env bash
# EXP76 device V stage (DEVICE-V-IMPL-PREP-01; docs/experiments/2026-10-20-exp76-device-plan.md section 2). Needs its own
# one-time run permission; holding this script is not one, and nothing starts it automatically. One start ever
# (runs/exp76-device-v.started), serial: process checks -> preflight_v (no model: R, data, plan, the eight V1 reference
# files, a boot cycle after the EXP76 faults; refused -> exit 12, nothing run) -> vrun.py (training V1 + V2 with the
# planned stop after the 250th update, then V3 only when allowed; each in its own process group under the accepted
# guard's pieces) -> report_v (no model). Hard limit start + 2,700 s (V_LIMIT for fixtures): the report keeps the last
# 120 s, V3 300 s before it. Nothing is retried or chained; a full EXP76 run needs its own permission.
#   nohup bash scripts/device_v/run_v.sh > runs/exp76-device-v.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
NAME=exp76-device-v
log() { echo "== $(date '+%F %T') $*"; }
MARK=runs/$NAME.started
[ ! -e "$MARK" ] || { log "$MARK exists: the V stage was already started once"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v -e "scripts/device_v/run_v.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
ENTRY='devision-calibrate|devision-eval|devision-pipeline|devision-train|devision-align|train_main|align_main|scripts/(calib|livek|protect|confirm|panel_learn|panel_bind|panel_number|panel_decision|mlm_repro|device_v)/|scripts/exp[0-9]+/'
own=$(ps -axo pid=,command= | grep -E "$ENTRY" | grep -v -e "grep -E" -e "scripts/device_v/run_v.sh")
[ -z "$own" ] || { log "other model processes are running:"; echo "$own"; exit 1; }
uv run python scripts/device_v/preflight_v.py || { log "preflight refused: see runs/$NAME-preflight.json; nothing run"; exit 12; }
mkdir -p runs
T0=$(date +%s)
LIMIT=${V_LIMIT:-2700}
V3_RESERVE=${V_V3_RESERVE:-300}
RESERVE=${V_RESERVE:-120}
DEADLINE=$(( T0 + LIMIT ))
{ echo "start $T0"; echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"
  for f in configs/$NAME.yaml scripts/device_v/run_v.sh scripts/device_v/preflight_v.py scripts/device_v/vcheck.py scripts/device_v/vrun.py \
           scripts/device_v/v3.py scripts/device_v/report_v.py scripts/panel_decision/guarded.py runs/$NAME-preflight.json runs/$NAME.START.json; do
    echo "sha256 $(shasum -a 256 "$f")"; done; } > "$MARK" || { log "cannot write $MARK"; exit 1; }
log "V stage: limit $LIMIT s, deadline $(date -r "$DEADLINE" '+%F %T'); report reserve $RESERVE s, V3 reserve $V3_RESERVE s"
run_capped $(( DEADLINE - RESERVE - $(date +%s) )) 2 uv run python scripts/device_v/vrun.py --record runs/$NAME-record.json \
  --run-dir runs/$NAME --start "$T0" --total "$LIMIT" --v3-reserve "$V3_RESERVE" --report-reserve "$RESERVE" \
  -- uv run devision-pipeline configs/$NAME.yaml \
  -- uv run python scripts/device_v/v3.py --save-dir runs/$NAME/v3_saved
rc=$?
log "V model steps exit $rc"
[ $rc = 71 ] && log "cleanup of a model process group not confirmed: no further model is started"
. scripts/exp38/steps.sh
final_report uv run python scripts/device_v/report_v.py
rrc=$?
log "report exit $rrc"
[ $rc = 0 ] && rc=$rrc
exit $rc
