#!/bin/zsh
# How the released checkpoint (stage2-cocoqa) was trained, end to end, on a Mac (MPS).
# Run from the repo root. About 16 hours of training in total; each stage logs to SwanLab
# (project devision) and to runs/logs/.
#
# Data: the converters live outside the repo in scripts/data/ (see CLAUDE.md). They produce
#   data/align_full_train.jsonl, data/align_val.jsonl   COCO train2014 captions (82,583 / 200 images)
#   data/train_100k.jsonl                               60k GQA + 40k VQAv2 yes/no decision questions
#   data/cocoqa_train.jsonl, data/val_cocoqa.jsonl      questions generated from COCO instance boxes
#   data/val_mix.jsonl, data/val_gqa.jsonl, data/val_vqav2.jsonl, data/pope.jsonl   evaluation
# and train_cocoqa_mix.jsonl = cocoqa_train + 22,803 questions sampled from train_100k (seed 0).
# Every evaluation image is excluded from every training file, under both its COCO and VG id.
set -e
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p runs/logs

# Stage 1: align the projector with image-conditioned masked captions (~4 h).
uv run devision-align --data data/align_full_train.jsonl --val data/align_val.jsonl --data-root data \
  --out runs/align-full-mac --run-name align-full-mac --device mps --epochs 2 --eval-every 500 \
  > runs/logs/align-full-mac.log 2>&1

# Stage 2a: RLCD on 100k GQA / VQAv2 questions, one epoch (~5 h). At the default learning rates a run
# this long collapses to 50/50 outputs; these lower ones with warm-up are stable.
uv run devision-train --init runs/align-full-mac --lr-new 5e-5 --lr-head 5e-5 --lr-lora 1e-4 --warmup 500 \
  --data data/train_100k.jsonl --val data/val_mix.jsonl --data-root data --eval pope=data/pope.jsonl \
  --out runs/stage2-100k-lowlr --run-name stage2-100k-lowlr --device mps --epochs 1 --eval-every 500 \
  > runs/logs/stage2-100k-lowlr.log 2>&1

# Stage 2b: continue on COCO-box grounding questions plus replayed stage-2a questions (~4 h).
uv run devision-train --init runs/stage2-100k-lowlr --lr-new 5e-5 --lr-head 5e-5 --lr-lora 1e-4 --warmup 500 \
  --data data/train_cocoqa_mix.jsonl --val data/val_mix.jsonl --data-root data \
  --eval pope=data/pope.jsonl --eval cocoqa=data/val_cocoqa.jsonl \
  --out runs/stage2-cocoqa --run-name stage2-cocoqa --device mps --epochs 1 --eval-every 1000 \
  > runs/logs/stage2-cocoqa.log 2>&1

# Evaluation through decide() on CPU, the production code path.
for set in val_gqa val_vqav2 pope val_cocoqa; do
  uv run devision-eval --checkpoint runs/stage2-cocoqa --data data/$set.jsonl --data-root data \
    --out runs/stage2-cocoqa/$set.json
done
