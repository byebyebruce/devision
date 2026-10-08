#!/usr/bin/env bash
# Build round 13's external sources (plan docs/experiments/2026-10-08-r13-binding-plan.md 1.4 / 1.5 and step 3):
# the evaluation-only benches first (training sources drop pictures that are in them or near them), then the
# training sources. Needs data/r13/identity.json (scripts/data/r13/identity.py) and should run after
# scripts/data/r13/gqa_bind.py, whose dev / test pictures the training sources must not use.
#
#   bash scripts/data/r13/sources/build_sources.sh [data] [step ...]     # default: every step below
#
# Every step is resumable: downloads finished once are never fetched again (interrupted ones resume), pictures are
# fetched / extracted once, and a set whose output exists is skipped (run its script with --force to rebuild).
# Large downloads: CLEVR-CoGenT 24.7 GB, VisMin 39 GB, Flickr30k 4.4 GB, CounterCurate 6 GB; Visual Genome and web
# pictures are fetched one by one (VG's server gives about 4 pictures / s).
# Output is copied to <root>/r13/build_sources.log. Stops at the first failing step; just run it again.
set -euo pipefail
ROOT="${1:-data}"
shift || true
cd "$(dirname "$0")/../../../.."
mkdir -p "$ROOT/r13"
export PYTHONUNBUFFERED=1
exec > >(tee -a "$ROOT/r13/build_sources.log") 2>&1

STEPS=("$@")
if [ ${#STEPS[@]} -eq 0 ]; then
  STEPS=(sugarcrepe whatsup countbench cogent tallyqa_complex vaw pixmo_two_kind vismin countercurate countbench_recheck)
fi
[ -s "$ROOT/r13/identity.json" ] || { echo "missing $ROOT/r13/identity.json: run scripts/data/r13/identity.py first"; exit 1; }
for f in dev_bind_gqa test_bind_gqa; do
  [ -s "$ROOT/r13/gqa/$f.jsonl" ] || echo "warning: $ROOT/r13/gqa/$f.jsonl not built yet; training sources cannot avoid its pictures"
done
run() { echo "== $(date '+%F %T') $*"; uv run python "scripts/data/r13/sources/$1" --root "$ROOT" "${@:2}"; }

for s in "${STEPS[@]}"; do
  case $s in
    cogent)             run cogent.py ;;               # cogent_a (training) + bench/cogent_b
    countbench_recheck) run countbench.py --force ;;   # near-duplicate flags against the training sources just built
    *)                  run "$s.py" ;;
  esac
done
echo "== $(date '+%F %T') done"
