#!/usr/bin/env bash
# Build every data file the training configs use, from the original sources, on a fresh machine:
#
#   bash scripts/data/build_all.sh [data]        # default data root: data
#
# Downloads (first run only, cached under <root>/raw and <root>/coco|gqa): COCO 2014 annotations and the
# images the questions use, VQAv2, GQA (lmms-lab/GQA) and its scene graphs, Visual Genome image data, POPE
# (lmms-lab/POPE), A-OKVQA, ScienceQA, The Cauldron's AI2D / TQA, and laya-vision's published per-question
# predictions. Tens of GB and a few hours. Every step is seeded; the result follows the same rules and
# proportions as the files the project trained on, not necessarily byte for byte (e.g. scene-graph questions
# only use the GQA pictures present). Stops at the first failing step.
set -euo pipefail
ROOT="${1:-data}"
cd "$(dirname "$0")/../.."
run() { echo "== $(date '+%F %T') $*"; uv run python "scripts/data/$@" --root "$ROOT"; }

# stage 1: image-conditioned captions, all COCO train2014 images (val 200)
run captions.py --images 100000 --val-images 200
mv "$ROOT/align_train.jsonl" "$ROOT/align_full_train.jsonl"

# data-v2: balanced pools, dev splits, test_* / bench_pope, MANIFEST; the round-3 mix; half of dev
run v2_build.py
run v2_mix.py
run v2_dev_half.py

# above/below relations from scene graphs, and round 4's relation mix (data-v3 takes its above/below part)
run v2_vg_relation.py
run v4_relation_mix.py

# data-v3: A-OKVQA / ScienceQA / AI2D / TQA, the full v3 mix, dev; round 5's continuation mix
run v3_build.py
run v5_mix.py

# laya-vision's benchmark (bench_lv_*), with the subsets our training never saw
run lv_bench.py

# round 6: mirrored left/right questions (train, dev, test) and the pair mix
run v6_flip.py
run v6_flip.py --dev
run v6_flip.py --files test_position --out test_position_flip.jsonl
run v6_flip.py --files test_relation --out test_relation_flip.jsonl
run v6_mix.py

echo "== done: data under $ROOT; check the counts in $ROOT/v2/MANIFEST.json and $ROOT/v3/MANIFEST.json"
