#!/usr/bin/env bash
# Round 15 (r15-vision, docs/experiments/2026-10-14-r15-vision.md) on one Mac, as r14b's chain: every complete
# monitor step is judged against step 0 by r14b's frozen stop rules (a stop ends the round: no retry, no relaxed rerun);
# a crash (e.g. the MPS hang) is retried from the resume state up to 3 times. First, once: the stage directory is
# seeded with exp19-vision's step-2000 training state (resume.pt) and its step-0 / step-2000 monitor records, so the
# round continues that run (same config, seed 1) instead of retraining its first 2,000 steps. After the full
# evaluation: the r14 acceptance against v12 (and, report only, against r14 and exp17's alpha 0.5).
#   nohup bash scripts/r15/chain.sh > runs/r15-vision.out 2>&1 &
set -u
set -m   # background jobs get their own process group, so a stop ends the whole pipeline tree
cd "$(dirname "$0")/../.."
SETS="val exist science ext bind_gqa count_narrow count_wide px_wide px_narrow px_exist flip vsr"

log() { echo "== $(date '+%F %T') $*"; }

judge_loop() {   # $1 stage dir, $2 pipeline pid
  local seen=""
  while kill -0 "$2" 2>/dev/null; do
    sleep 120
    [ -d "$1/monitor" ] || continue
    for d in "$1"/monitor/step*; do
      [ -d "$d" ] || continue
      case " $seen " in *" $d "*) continue;; esac
      local all=1
      for s in $SETS; do [ -f "$d/$s.details.jsonl" ] || all=0; done
      [ $all = 1 ] || continue
      seen="$seen $d"
      [ "$(basename "$d")" = step0 ] && continue
      log "judge $d"
      uv run python scripts/r13b/judge.py --rules scripts/r14b/rules.json monitor "$1"
      if [ $? = 3 ]; then
        log "STOP rule hit: ending this arm"
        touch "$1/STOPPED_BY_RULE"
        kill -TERM -- "-$2"
        return
      fi
    done
  done
}

stage=runs/r15-vision/stage2
if [ ! -e "$stage" ]; then
  src=runs/exp19-vision/stage2
  mkdir -p "$stage/monitor"
  cp "$src/resume.pt" "$stage/resume.pt" && cp -R "$src/monitor/step0" "$src/monitor/step2000" "$stage/monitor/" \
    && cp "$src/start_check.json" "$stage/" || { log "seeding from $src failed"; exit 1; }
  printf '{"seeded_from": "%s", "step": 2000, "resume_sha256": "%s"}\n' "$src" \
    "$(shasum -a 256 "$stage/resume.pt" | cut -d' ' -f1)" > "$stage/seeded_from.json"
  log "seeded $stage from $src (step 2000)"
fi
for attempt in 1 2 3; do
  [ -f "$stage/STOPPED_BY_RULE" ] && break
  log "r15-vision attempt $attempt"
  uv run devision-pipeline configs/r15-vision.yaml &
  pid=$!
  judge_loop "$stage" "$pid" &
  jpid=$!
  wait "$pid"; rc=$?
  kill "$jpid" 2>/dev/null; wait "$jpid" 2>/dev/null
  [ -f "$stage/STOPPED_BY_RULE" ] && { log "r15-vision stopped by rule"; break; }
  [ $rc = 0 ] && { log "r15-vision done"; break; }
  log "r15-vision exited $rc; retrying from the resume state"
  sleep 60
done
if [ -f "$stage/STOPPED_BY_RULE" ] || [ $rc != 0 ]; then log "chain finished without a full evaluation"; exit 1; fi
uv run python scripts/r14/accept.py --cand runs/r15-vision/eval | tee runs/r15-vision/eval/accept.txt
rc=${PIPESTATUS[0]}
for b in r14-count/eval exp17-interp/a050; do   # report only; the verdicts above are against v12
  n=${b%%/*}
  uv run python scripts/r14/accept.py --cand runs/r15-vision/eval --base "runs/$b" \
    --out "runs/r15-vision/eval/compare/vs_$n.json" > "runs/r15-vision/eval/compare/vs_$n.txt" 2>&1
done
log "chain finished (acceptance exit $rc)"
exit $rc
