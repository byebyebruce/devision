#!/usr/bin/env bash
# MLM-REPRO-PREP-01 short reproduction (docs/experiments/2026-10-20-mlm-repro-plan.md). One start ever
# (runs/mlm-repro.started); the saved N end point and its training report must be the accepted ones (sha256) before
# anything loads; one process, three passes on one loaded instance (scripts/mlm_repro/repro.py run), each written when it
# finishes; then always the no-model finalize. Hard limit start + 600 s (REPRO_LIMIT for fixtures): the run ends 60 s
# before it (TERM, KILL 10 s later), the finalize in what is left. Nothing retrained, nothing chained.
#   nohup bash scripts/mlm_repro/run.sh > runs/mlm-repro.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
MARK=runs/mlm-repro.started
[ ! -e "$MARK" ] || { log "$MARK exists: the reproduction was already started once"; exit 1; }
[ ! -e runs/mlm-repro ] || { log "runs/mlm-repro exists (fresh only)"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/mlm_repro/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
ENTRY='devision-calibrate|devision-eval|devision-pipeline|devision-train|devision-align|train_main|align_main|scripts/(calib|livek|protect|confirm|panel_learn|panel_bind|panel_number|panel_decision)/|scripts/exp[0-9]+/'
own=$(ps -axo pid=,command= | grep -E "$ENTRY" | grep -v -e "grep -E" -e "scripts/mlm_repro/run.sh")
[ -z "$own" ] || { log "other model processes are running:"; echo "$own"; exit 1; }
N_MODEL=${REPRO_N_MODEL:-3ab0ab0304a159d73c989876dee15db1eedf71b67c035e25389d1dc1809a8d2d}
N_REPORT=${REPRO_N_REPORT:-a24ff10b49eaa9884d60c32e42c3f0d4886ad21190cd3446d7ab75771d5f524f}
[ "$(shasum -a 256 runs/exp75-panel-n/n/model.safetensors | cut -c1-64)" = "$N_MODEL" ] && \
  [ "$(shasum -a 256 runs/exp75-panel-n/n/align_report.json | cut -c1-64)" = "$N_REPORT" ] || { log "the N end point is not the accepted one"; exit 12; }
mkdir -p runs
DEADLINE=$(( $(date +%s) + ${REPRO_LIMIT:-600} ))
{ echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"
  for f in scripts/mlm_repro/run.sh scripts/mlm_repro/repro.py scripts/panel_bind/probe.py src/devision/train/align.py; do echo "sha256 $(shasum -a 256 "$f")"; done
  echo "n model $N_MODEL"; echo "n report $N_REPORT"; } > "$MARK" || { log "cannot write $MARK"; exit 1; }
. scripts/exp35/guard.sh
log "repro: deadline $(date -r "$DEADLINE" '+%F %T')"
run_capped $(( DEADLINE - $(date +%s) - 60 )) 10 uv run python scripts/mlm_repro/repro.py run --checkpoint runs/exp75-panel-n/n --out runs/mlm-repro
rc=$?
log "repro exit $rc"
# always: the no-model finalize inside the remaining time keeps every finished segment and marks the rest (R1-02)
left=$(( DEADLINE - $(date +%s) - 5 ))
if [ "$left" -gt 0 ]; then run_capped "$left" 5 uv run python scripts/mlm_repro/repro.py finalize --out runs/mlm-repro; frc=$?; else frc=1; log "no time left to finalize"; fi
log "finalize exit $frc"
[ $rc = 0 ] && rc=$frc
exit $rc
