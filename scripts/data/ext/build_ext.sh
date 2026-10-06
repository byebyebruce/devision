#!/usr/bin/env bash
# Build the extension data pack (scripts/data/ext/; sources and licences in docs/research/data-pack-sources-2026-10-06.md)
# and mix it with replay into data/ext/train_mix.jsonl. Needs the data `build_all.sh` makes (held-out sets, replay
# pools, data/v9/r_pairs.jsonl) and data/v11 (the counting sets), so their pictures stay out.
#
#   bash scripts/data/ext/build_ext.sh [data] [source ...]      # default: every source, then replay and mix
#
# Every step is resumable: downloads finished once are never fetched again (interrupted ones resume), pictures are
# extracted once, and a source whose train.jsonl exists is skipped (run its script with --force to rebuild it).
# Output is copied to <root>/ext/build_ext.log. Stops at the first failing step; just run it again.
set -euo pipefail
ROOT="${1:-data}"
shift || true
cd "$(dirname "$0")/../../.."
mkdir -p "$ROOT/ext"
export PYTHONUNBUFFERED=1
exec > >(tee -a "$ROOT/ext/build_ext.log") 2>&1

SOURCES=("$@")
if [ ${#SOURCES[@]} -eq 0 ]; then
  SOURCES=(iconqa figureqa clevr clevr_math tallyqa dvqa mapqa superclevr visonlyqa spatialsense snli_ve
           pixmo_count pixmo_points objects365 vision_flan)
fi
run() { echo "== $(date '+%F %T') $*"; uv run python "scripts/data/ext/$1" "${@:2}" --root "$ROOT"; }

for s in "${SOURCES[@]}"; do
  case $s in
    clevr|clevr_math) run clevr.py --subset "$s" ;;
    dvqa|mapqa)       run chartmap.py --subset "$s" ;;
    *)                run "$s.py" ;;
  esac
done

# replay: v9b's left/right questions + every earlier ability (REPLAY_QUOTAS' proportions; 36,000 is about the most
# the ScienceQA / VSR / above-below pools allow), with the current held-out pictures excluded
if [ ! -s "$ROOT/ext/replay/train.jsonl" ]; then
  echo "== $(date '+%F %T') replay"
  uv run python scripts/data/v9_mix.py --root "$ROOT" --out "$ROOT/ext/replay" --pairs-file "$ROOT/v9/r_pairs.jsonl" \
      --replay 36000 --seed 0
fi
echo "== $(date '+%F %T') mix"
uv run python scripts/data/ext/mix.py --root "$ROOT"
echo "== $(date '+%F %T') done: $ROOT/ext/train_mix.jsonl, $ROOT/ext/dev_ext.jsonl, $ROOT/ext/MIX.json"
