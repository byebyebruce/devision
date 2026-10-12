#!/usr/bin/env bash
# Public-data pack (docs/research/image-jev-top10-training-data-2026-10-12.md): download every source, convert it
# to the training format, then build the weak-point training mix.
#
#   bash scripts/data/pub/build_pub.sh [data] [source ...]       # default: all sources, then the mix
#
# Resumable at every level: an interrupted download continues from its .part file (HTTP Range) or the Hugging Face
# cache; pictures already on disk are not fetched again (dead links are remembered); a source whose train.jsonl and
# MANIFEST.json exist is skipped (rm them, or run its script with --force, to rebuild). Just run it again.
#
# Needs the project's data on this machine first: the held-out evaluation sets (to keep their pictures out of
# training) and the v12 training pool (for the replay in the mix). On the machine that has them:
#   uv run python scripts/data/pub/export_list.py --root data > /tmp/devision-files.txt
#   rsync -a --partial --info=progress2 --files-from=/tmp/devision-files.txt data/ user@host:/path/devision/data/
# Output: data/ext/<source>/{train,dev}.jsonl + MANIFEST.json (+ images/), and data/pub/train_mix.jsonl.
# Disk: ~13 GB CoSyn (4 branches) + ~3.3 GB ABO + ~12 GB VizWiz + ~17.7 GB OneJev + PixMo pictures (~10-25 GB).
set -u
ROOT=${1:-data}
shift || true
SOURCES=${*:-"pixmo_points cosyn abo vizwiz onejev"}
cd "$(dirname "$0")/../../.."
LOG="$ROOT/build_pub.log"
mkdir -p "$ROOT"
echo "== $(date '+%F %T') build_pub: $SOURCES" | tee -a "$LOG"
if ! uv run python -c "import sys; sys.path.insert(0, 'scripts/data/pub'); import held; held.require_history('$ROOT')"; then
  echo "== stopped: copy the project's data to this machine first (scripts/data/pub/export_list.py)" | tee -a "$LOG"
  exit 3
fi
for s in $SOURCES; do
  case $s in
    pixmo_points) script=scripts/data/ext/pixmo_points.py; args="--images 80000" ;;
    *) script=scripts/data/pub/$s.py; args="" ;;
  esac
  for attempt in 1 2 3; do
    echo "== $(date '+%F %T') $s (attempt $attempt)" | tee -a "$LOG"
    if uv run python "$script" --root "$ROOT" $args 2>&1 | tee -a "$LOG"; [ "${PIPESTATUS[0]}" = 0 ]; then
      break
    fi
    echo "== $s failed; retrying in 60 s (downloads resume where they stopped)" | tee -a "$LOG"
    sleep 60
  done
done
echo "== $(date '+%F %T') mix" | tee -a "$LOG"
uv run python scripts/data/pub/pub_mix.py --root "$ROOT" 2>&1 | tee -a "$LOG"
echo "== $(date '+%F %T') done" | tee -a "$LOG"
