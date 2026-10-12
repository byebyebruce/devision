#!/usr/bin/env bash
# Round 14 (r14-count) on one Mac. While an arm trains, every complete monitor step is judged against step 0 by the
# frozen rules; a stop rule ends that arm (no retry, no relaxed rerun). A crash (e.g. the MPS hang) is retried from
# the resume state up to 3 times.
#   nohup bash scripts/r14/chain.sh > runs/r14/chain.out 2>&1 &
set -u
set -m   # background jobs get their own process group, so a stop ends the whole pipeline tree
cd "$(dirname "$0")/../.."
mkdir -p runs/r14
SETS="val exist science ext bind_gqa count_narrow count_wide px_wide px_narrow px_exist"

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
      uv run python scripts/r13b/judge.py --rules scripts/r14/rules.json monitor "$1"
      if [ $? = 3 ]; then
        log "STOP rule hit: ending this arm"
        touch "$1/STOPPED_BY_RULE"
        kill -TERM -- "-$2"
        return
      fi
    done
  done
}

stage=runs/r14-count/stage2
for attempt in 1 2 3; do
  [ -f "$stage/STOPPED_BY_RULE" ] && break
  log "r14-count attempt $attempt"
  uv run devision-pipeline configs/r14-count.yaml &
  pid=$!
  judge_loop "$stage" "$pid" &
  jpid=$!
  wait "$pid"; rc=$?
  kill "$jpid" 2>/dev/null; wait "$jpid" 2>/dev/null
  [ -f "$stage/STOPPED_BY_RULE" ] && { log "r14-count stopped by rule"; break; }
  [ $rc = 0 ] && { log "r14-count done"; break; }
  log "r14-count exited $rc; retrying from the resume state"
  sleep 60
done
log "chain finished"
