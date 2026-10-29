#!/usr/bin/env bash
# CPU latency of a checkpoint for the release card (same 200 questions and protocol as release/v0.4/cpu_latency.json): one question per
# request, no batching, 4 threads, GPU hidden. Run it on an otherwise idle machine; the number is a deployment
# reference only.   usage: scripts/v05/cpu_latency.sh CHECKPOINT OUT_JSON
set -eu
cd "$(dirname "$0")/../.."
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 UV_NO_SYNC=1 HF_HUB_OFFLINE=1 uv run --no-sync devision-eval --checkpoint "$1" \
  --data runs/v04-cpu/latency200.jsonl --data-root data --device cpu --no-group --role monitoring --out "$2"
