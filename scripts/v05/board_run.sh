#!/usr/bin/env bash
# Decision Index public vision index of one checkpoint on the GPU (docs/research/decision-index-vision.md): serve it
# with devision-serve --device cuda, drive the kit's runner (/home/bruce/ext/scripts/run_devision.py, kit venv), copy
# the question-free scores to runs/ext-decision-index/<name>/. Only evaluates; the board rows never enter training.
#   usage: scripts/v05/board_run.sh CHECKPOINT NAME [PORT]
set -u
cd "$(dirname "$0")/../.."
CK=$1 NAME=$2 PORT=${3:-8031}
EXT=/home/bruce/ext DATA=$EXT/decision-index-data/vision OUT=$EXT/decision-index-runs/devision-$NAME
mkdir -p "$OUT" runs/ext-decision-index/$NAME
UV_NO_SYNC=1 HF_HUB_OFFLINE=1 uv run --no-sync devision-serve --checkpoint "$CK" --device cuda --port $PORT --no-demo \
  > runs/ext-decision-index/$NAME/serve.log 2>&1 &
SP=$!
for _ in $(seq 120); do curl -sf -o /dev/null http://127.0.0.1:$PORT/health && break; sleep 5; done
echo "$(date '+%F %T') serve up ($CK, pid $SP)"
$EXT/decision-index/.venv/bin/python $EXT/scripts/run_devision.py $DATA "$OUT" --rows $DATA/vision-rows.jsonl \
  --workers 4 --urls http://127.0.0.1:$PORT > runs/ext-decision-index/$NAME/run.log 2>&1
RC=$?
kill $SP; wait $SP 2>/dev/null
cp "$OUT"/vision-scores.json "$OUT"/scores.json runs/ext-decision-index/$NAME/ 2>/dev/null
tail -14 runs/ext-decision-index/$NAME/run.log
echo "$(date '+%F %T') board $NAME exit $RC"
