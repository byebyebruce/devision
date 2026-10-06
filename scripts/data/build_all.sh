#!/usr/bin/env bash
# Build the data files of the data-v2-and-later training configs from the original sources, on a fresh
# machine: configs/scratch.yaml and the round 3-9 configs (v3-data2, v4-relation-tb, v5-data3, v6-flip-lr,
# v7-sqa, v8-vsr-v7w, v9b-lr-replay and their *-lvbench evaluations). Round 10's data is not built here. Rounds 1-2 used older files (train_100k, val_mix, ...) this does not build.
#
#   bash scripts/data/build_all.sh [data]        # default data root: data
#
# Downloads (first run only, cached under <root>/raw and <root>/coco|gqa): COCO 2014 annotations and the
# images the questions use, VQAv2, GQA (lmms-lab/GQA) and its scene graphs, Visual Genome image data, POPE
# (lmms-lab/POPE), A-OKVQA, ScienceQA, The Cauldron's AI2D / TQA, VSR (cambridgeltl/vsr_random), Visual7W telling and the
# Visual Genome pictures it uses, and laya-vision's published per-question
# predictions. Tens of GB and a few hours. Every step is seeded; the result follows the same rules and
# proportions as the files the project trained on, not necessarily byte for byte (e.g. scene-graph questions
# only use the GQA pictures present). Stops at the first failing step.
set -euo pipefail
ROOT="${1:-data}"
cd "$(dirname "$0")/../.."
mkdir -p "$ROOT"
export PYTHONUNBUFFERED=1                              # print each step's output as it happens
exec > >(tee -a "$ROOT/build_all.log") 2>&1            # and keep a copy in <root>/build_all.log
run() { echo "== $(date '+%F %T') $*"; uv run python "scripts/data/$@" --root "$ROOT"; }

# stage 1: image-conditioned captions, all COCO train2014 images (val 200)
run captions.py --images 100000 --val-images 200
mv "$ROOT/align_train.jsonl" "$ROOT/align_full_train.jsonl"

# data-v2: balanced pools, dev splits, test_* / bench_pope, MANIFEST; the round-3 mix; half of dev
run v2_build.py
run v2_mix.py
run v2_dev_half.py

# laya-vision's benchmark (bench_lv_*) BEFORE the training data that must exclude its pictures
# (ScienceQA reuses pictures across splits; v2_vg_relation and v3_build refuse to run without it)
run lv_bench.py

# above/below relations from scene graphs, and round 4's relation mix (data-v3 takes its above/below part)
run v2_vg_relation.py
run v4_relation_mix.py

# data-v3: A-OKVQA / ScienceQA / AI2D / TQA, the full v3 mix, dev; round 5's continuation mix
run v3_build.py
run v5_mix.py

# round 6: mirrored left/right questions (train, dev, test) and the pair mix
run v6_flip.py
run v6_flip.py --dev
run v6_flip.py --files test_position --out test_position_flip.jsonl
run v6_flip.py --files test_relation --out test_relation_flip.jsonl
run v6_mix.py

# data-v4 (diagrams matched by identical pixels) and round 7's ScienceQA mix
run v3_build.py --version v4 --match exact
run v7_mix.py

# data-v5: VSR and Visual7W telling (train pools, strict test and dev sets), round 8's mix and dev_mix
run v5_build.py
run v8_mix.py

# round 9 (v9b, the current best): R's 2,500 real left/right questions, then the same questions (pairs on
# near-duplicate photos of held-out pictures dropped) plus 2,500 replay. Byte for byte v9b's files when <root>/v10
# and <root>/v11 do not exist yet; their evaluation sets are also held out, so building after them removes more pictures
run v9_pairs.py
run v9_mix.py --pairs-file "$ROOT/v9/r_pairs.jsonl" --replay 2500 --out "$ROOT/v9/x1_r2500"

# every data file the supported configs use exists and is not empty
uv run python scripts/data/check_configs.py --root "$ROOT" configs/scratch.yaml \
    configs/v3-data2.yaml configs/v3-data2-lvbench.yaml configs/v4-relation-tb.yaml configs/v5-data3.yaml \
    configs/v5-data3-lvbench.yaml configs/v6-flip-lr.yaml configs/v7-sqa.yaml configs/v8-vsr-v7w.yaml \
    configs/v9b-lr-replay.yaml
echo "== done: data under $ROOT; counts in $ROOT/v2/MANIFEST.json, $ROOT/v3/MANIFEST.json, $ROOT/v5/MANIFEST.json and $ROOT/v9/x1_r2500/MIX.json"
