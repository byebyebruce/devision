# exp38 step control (critic EXP38-R1-02), sourced by chain.sh and its CPU test. Needs DEADLINE (epoch seconds),
# STATUS (a file), log() and scripts/exp35/guard.sh. Model and evaluation steps must end RESERVE seconds before the
# deadline (TERM, then KILL 15 s later); the last RESERVE seconds are kept for the final report.
RESERVE=${RESERVE:-60}
status() { echo "$1 $2" >> "$STATUS"; }
# step NAME cmd...: 0 completed, 124 interrupted at the deadline, 125 not started (no time left), else failed;
# the outcome is written to STATUS either way (a details file existing is never taken as success).
step() {
  local name=$1; shift
  local left=$(( DEADLINE - RESERVE - $(date +%s) - 15 ))
  if [ "$left" -le 0 ]; then status "$name" missing; log "no time left, not started: $name"; return 125; fi
  run_capped "$left" 15 "$@"
  local rc=$?
  case $rc in
    0) status "$name" completed ;;
    124) status "$name" interrupted; log "deadline reached, stopped: $name" ;;
    *) status "$name" failed; log "failed ($rc): $name" ;;
  esac
  return $rc
}
# final_report cmd...: the report, inside what is left until the deadline itself; its exit status is returned.
final_report() {
  local left=$(( DEADLINE - $(date +%s) - 5 ))
  [ "$left" -gt 0 ] || { log "no time left for the final report"; return 1; }
  run_capped "$left" 5 "$@"
}
