#!/usr/bin/env bash
# v0.4 full release evaluation (critic V04-K-FULL-RUN-01): a thin chain of the existing commands, nothing new in sampling,
# caching or scoring. The candidate (configs/v04-full-eval.yaml + configs/v04-extra-evidence.yaml, frozen in
# runs/v04-k/eval/IDENTITY.json):
#   dryrun (devision-pipeline --dry-run) -> preflight0 (check_eval preflight: everything missing, nothing invalid)
#   -> first (the plan's own eval:exist command, the first formal output) -> seal1 + preflight1 + first_ok (exist.json
#   sealed under the frozen identity; otherwise stop here) -> pipeline (the rest of the 29 sets and controls; the
#   verified exist output is skipped) -> extras (check_eval run-extras, 15) -> seal -> check;
# then v0.3's six confirmation items (configs/v04-v03-confirm.yaml + configs/v04-confirm-extras.yaml): freeze ->
#   preflight -> run-extras -> seal -> check (v0.3's historical 29 sets are not touched).
# Serial, one model at a time, the first failure stops the chain. Hard limit start + 37,800 s (FULL_LIMIT for CPU
# fixtures only); every step must end FULL_RESERVE (900 s) before it; run_capped stops a step with TERM then KILL.
# One start ever (runs/v04-k-full.started, never removed); no resume, no retry, no --force, nothing chained after it.
#   nohup bash scripts/v04full/run.sh > runs/v04-k-full.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
log() { echo "== $(date '+%F %T') $*"; }
MARK=runs/v04-k-full.started
[ ! -e "$MARK" ] || { log "$MARK exists: the full evaluation was already started once"; exit 1; }
. scripts/exp35/guard.sh
others=$(model_processes | grep -v "scripts/v04full/run.sh")
[ -z "$others" ] || { log "other model processes are running:"; echo "$others"; exit 1; }
ENTRY='devision-calibrate|devision-eval|devision-pipeline|devision-train|train_main|scripts/(calib|livek|protect)/|scripts/confirm/|scripts/exp[0-9]+/'
own=$(ps -axo pid=,command= | grep -E "$ENTRY" | grep -v -e "grep -E" -e "scripts/v04full/run.sh")
[ -z "$own" ] || { log "other model processes are running:"; echo "$own"; exit 1; }
C=configs/v04-full-eval.yaml
X=configs/v04-extra-evidence.yaml
C3=configs/v04-v03-confirm.yaml
X3=configs/v04-confirm-extras.yaml
mkdir -p runs
DEADLINE=$(( $(date +%s) + ${FULL_LIMIT:-37800} ))
{ echo "deadline $DEADLINE"; echo "started $(date '+%F %T')"; echo "head $(git rev-parse HEAD 2>/dev/null)"
  for f in "$C" "$X" "$C3" "$X3" scripts/v04full/run.sh runs/v04-k/eval/IDENTITY.json; do echo "sha256 $(shasum -a 256 "$f")"; done; } > "$MARK" \
  || { log "cannot write $MARK"; exit 1; }
command -v caffeinate >/dev/null && { caffeinate -i -w $$ & }          # keep the Mac awake while this chain lives
STATUS=runs/v04-k-full-status.txt
RESERVE=${FULL_RESERVE:-900}
. scripts/exp38/steps.sh
CE="uv run python scripts/release/check_eval.py"
log "full evaluation: deadline $(date -r "$DEADLINE" '+%F %T'), steps end by $(date -r $(( DEADLINE - RESERVE )) '+%F %T')"
first_cmd() {        # the plan's own command for eval:exist (the same argv the pipeline would run)
  uv run python - "$C" <<'EOF'
import shlex, sys, yaml
from devision.train.pipeline import build_plan
step = next(s for s in build_plan(yaml.safe_load(open(sys.argv[1]))).steps if s.name == "eval:exist")
print(" ".join(shlex.quote(a) for a in step.argv))
EOF
}
first_ok() {         # exist.json and its details sealed under the frozen identity
  uv run python -c "import json, sys; r = json.load(open('runs/v04-k/eval/RESULTS.json')); sys.exit(0 if 'exist.json' in r else 1)"
}
codes=""
s() { local name=$1; shift; step "$name" "$@"; local rc=$?; codes="$codes $name=$rc"; return $rc; }
run_all() {
  s dryrun uv run devision-pipeline "$C" --dry-run || return
  s preflight0 $CE preflight "$C" --extras "$X" || return
  local cmd; cmd=$(first_cmd) || { codes="$codes first_cmd=failed"; return 1; }
  log "first formal output: $cmd"
  s first bash -c "$cmd" || return
  s seal1 $CE seal "$C" --extras "$X" || return
  s preflight1 $CE preflight "$C" --extras "$X" || return
  s first_ok first_ok || return
  s pipeline uv run devision-pipeline "$C" || return
  s extras $CE run-extras "$C" --extras "$X" || return
  s seal $CE seal "$C" --extras "$X" || return
  s check $CE check "$C" --extras "$X" || return
  local c
  for c in freeze preflight run-extras seal check; do
    s "v03_$c" $CE "$c" "$C3" --extras "$X3" || return
  done
}
run_all
rc=$?
log "steps:$codes"
log "exit $rc"
exit "$rc"
