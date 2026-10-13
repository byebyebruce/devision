#!/usr/bin/env bash
# exp17 on one Mac (docs/experiments/2026-10-13-exp17-interp.md): for each alpha, interpolate v12 -> r14, refit the
# temperatures on data/r13b/val.jsonl, evaluate the development panels and judge them against v12 recalibrated on the
# same val (r14b's frozen end-of-run panel rules). Then the largest of 0.3 / 0.5 / 0.7 that passes gets the full
# evaluation and the r14 acceptance against v12; none passes: stop. Rerunning skips finished steps.
#   nohup bash scripts/exp17/chain.sh > runs/exp17-interp/chain.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
mkdir -p runs/exp17-interp
BASE=runs/r13b-v12-base/stage2-recal

log() { echo "== $(date '+%F %T') $*"; }

for a in 030 050 070 100; do
  ck=runs/exp17-interp/a$a
  if [ ! -f "$ck/interp.json" ]; then
    rm -rf "$ck"
    log "interpolate alpha 0.$a"
    uv run python scripts/exp17/interp.py runs/v12-ext/stage2 runs/r14-count/stage2 "$(echo "$a" | sed 's/^\(.\)/\1./')" "$ck" || exit 1
  fi
  if [ ! -f "$ck/CALIBRATED" ]; then
    log "calibrate alpha 0.$a"
    uv run devision-calibrate --checkpoint "$ck" --data data/r13b/val.jsonl --data-root data --in-place \
      > "$ck/calibrate.txt" 2>&1 || exit 1
    touch "$ck/CALIBRATED"
  fi
  log "panels alpha 0.$a"
  uv run devision-pipeline configs/exp17-a$a.yaml || exit 1
  uv run python scripts/r13b/judge.py --rules scripts/r14b/rules.json final "$ck" "$BASE" > "$ck/judge_final.txt"
  log "alpha 0.$a panels: $(tail -1 "$ck/judge_final.txt")"
done

chosen=""
for a in 070 050 030; do
  [ -f runs/exp17-interp/a$a/judge_final.json ] || continue
  python3 -c "import json,sys; sys.exit(0 if json.load(open('runs/exp17-interp/a$a/judge_final.json'))['pass'] else 1)" \
    && { chosen=$a; break; }
done
if [ -z "$chosen" ]; then
  log "no alpha passes the panels: stop (no full evaluation)"
  exit 0
fi
log "chosen alpha 0.$chosen: full evaluation"
echo "$chosen" > runs/exp17-interp/CHOSEN
# judge_final.json of the panel step is kept; the full config's own judge command writes it again with r14's rules
cp runs/exp17-interp/a$chosen/judge_final.json runs/exp17-interp/a$chosen/judge_final_panels_r14b.json
sed "s#^  checkpoint: .*#  checkpoint: runs/exp17-interp/a$chosen#" configs/exp17-full.yaml > runs/exp17-interp/full.yaml
uv run devision-pipeline runs/exp17-interp/full.yaml || exit 1
uv run python scripts/r14/accept.py --cand runs/exp17-interp/a$chosen | tee runs/exp17-interp/a$chosen/accept.txt
log "chain finished"
