#!/usr/bin/env bash
# EXP75 (PANEL-NUMBER-PREP-01; docs/experiments/2026-10-20-panel-number-plan.md). One start ever (runs/panel-number.started),
# serial: arm N (configs/exp75-panel-n.yaml) -> arm R (configs/exp75-panel-r.yaml) -> START for the arm end points that
# exist -> the accepted panel probe on each -> compare (no model). Hard limit start + 7,200 s (EXP75_LIMIT for fixtures);
# every model step ends 120 s before it (TERM, KILL 15 s later); the last 120 s are for the comparison. A failed or
# stopped arm is missing in the report; the other arm still runs and is reported. Nothing is chained after it.
#   nohup bash scripts/panel_number/run.sh > runs/panel-number.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
MARK=runs/panel-number.started
[ ! -e "$MARK" ] || { log "$MARK exists: EXP75 was already started once"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/panel_number/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
ENTRY='devision-calibrate|devision-eval|devision-pipeline|devision-train|devision-align|train_main|align_main|scripts/(calib|livek|protect|confirm|panel_learn|panel_bind)/|scripts/exp[0-9]+/'
own=$(ps -axo pid=,command= | grep -E "$ENTRY" | grep -v -e "grep -E" -e "scripts/panel_number/run.sh")
[ -z "$own" ] || { log "other model processes are running:"; echo "$own"; exit 1; }
for d in runs/exp75-panel-n runs/exp75-panel-r runs/panel-number; do [ ! -e "$d" ] || { log "$d exists (fresh only)"; exit 1; }; done
mkdir -p runs
DEADLINE=$(( $(date +%s) + ${EXP75_LIMIT:-7200} ))
{ echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"
  for f in configs/exp75-panel-n.yaml configs/exp75-panel-r.yaml src/devision/train/align.py scripts/panel_number/run.sh \
           scripts/panel_number/compare.py scripts/panel_bind/probe.py scripts/panel_bind/report.py data/exp26/align_train.jsonl \
           data/exp26/align_val.jsonl runs/exp74-panel-d1/d1/model.safetensors; do echo "sha256 $(shasum -a 256 "$f")"; done; } > "$MARK" \
  || { log "cannot write $MARK"; exit 1; }
STATUS=runs/panel-number-status.txt
RESERVE=${EXP75_RESERVE:-120}
. scripts/exp38/steps.sh
log "EXP75: deadline $(date -r "$DEADLINE" '+%F %T')"
codes=""
step arm_n uv run devision-pipeline configs/exp75-panel-n.yaml; codes="$codes arm_n=$?"
step arm_r uv run devision-pipeline configs/exp75-panel-r.yaml; codes="$codes arm_r=$?"
step probe_start uv run python scripts/panel_number/compare.py start; rc=$?; codes="$codes probe_start=$rc"
if [ $rc = 0 ]; then
  for e in n:runs/exp75-panel-n/n r:runs/exp75-panel-r/r; do
    name=${e%%:*}; ck=${e#*:}
    if [ -e "$ck/model.safetensors" ]; then
      step "probe_$name" uv run python scripts/panel_bind/probe.py --checkpoint "$ck" --out "runs/panel-number/$name"; codes="$codes probe_$name=$?"
    fi
  done
fi
log "steps:$codes"
final_report uv run python scripts/panel_number/compare.py report
rrc=$?
log "compare exit $rrc"
exit $rrc
