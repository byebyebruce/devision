# Bounded run for exp35 (critic EXP35-R1-03), sourced by run.sh and its CPU test.
# run_capped LIMIT GRACE cmd...: runs cmd; at LIMIT seconds sends TERM to its whole process tree, and KILL to
# whatever is left GRACE seconds later -- the total never exceeds LIMIT + GRACE. Returns cmd's status, or 124 when
# it had to be stopped.
tree() { local c; echo "$1"; for c in $(pgrep -P "$1"); do tree "$c"; done; }
run_capped() {
  local limit=$1 grace=$2; shift 2
  "$@" &
  local pid=$! end=$(( $(date +%s) + limit )) stopped=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$end" ]; then
      local procs; procs=$(tree "$pid")
      echo "== limit of ${limit}s reached: TERM to $(echo $procs)"
      kill -TERM $procs 2>/dev/null
      local until=$(( $(date +%s) + grace ))
      while [ "$(date +%s)" -lt "$until" ] && for p in $procs; do kill -0 "$p" 2>/dev/null && break; done; do sleep 1; done
      local left=""; for p in $procs; do kill -0 "$p" 2>/dev/null && left="$left $p"; done
      [ -n "$left" ] && { echo "== still running after ${grace}s: KILL$left"; kill -KILL $left 2>/dev/null; }
      stopped=1
      break
    fi
    sleep 1
  done
  wait "$pid" 2>/dev/null
  local rc=$?
  [ "$stopped" = 1 ] && return 124
  return $rc
}
# No other model process of this repository may run (training, evaluation, pipelines, experiment chains).
model_processes() { pgrep -fl "train_main|align_main|eval_main|devision-pipeline|devision-eval|scripts/exp[0-9]+/chain.sh|scripts/exp[0-9]+/run.sh" | grep -v "^$$ " ; }
