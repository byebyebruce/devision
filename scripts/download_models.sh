#!/usr/bin/env bash
# Pre-download the models used by configs/scratch.yaml and configs/scratch-a100.yaml.
# Uses the standard Hugging Face cache; HF_HOME / HF_HUB_CACHE overrides are respected.
# Re-running reuses completed, unchanged files. No model is loaded and no GPU is used.
set -euo pipefail

if [[ $# -gt 0 ]]; then
    if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
        cat <<'HELP'
Usage: bash scripts/download_models.sh

Download Laya, SigLIP2 and the ModernBERT weights needed for caption alignment.
Requires uv; run from any directory. Stops on the first failed download.
Models use the default Hugging Face cache (normally ~/.cache/huggingface/hub).
To override it, set HF_HOME or HF_HUB_CACHE before running this script and use
the same settings when training. Re-run after an interruption to reuse the cache.
HELP
        exit 0
    fi
    echo "Usage: bash scripts/download_models.sh [--help]" >&2
    exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "Downloading Laya: text backbone, decision head and tokenizer"
uv run hf download convaiinnovations/laya \
    --include "rl_agent_config.json" \
    --include "model.safetensors" \
    --include "tokenizer/*" \
    --include "encoder/*"

echo "Downloading SigLIP2: vision encoder"
uv run hf download google/siglip2-base-patch16-256 \
    config.json model.safetensors

# Alignment extracts only the MLM head, but it is stored in the full weight file.
echo "Downloading ModernBERT: weights containing the alignment MLM head"
uv run hf download answerdotai/ModernBERT-large model.safetensors

echo "Done. Train with the same Hugging Face cache settings to reuse these files."
