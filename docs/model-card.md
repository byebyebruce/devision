---
license: LICENSE_ID
language:
- en
library_name: devision
pipeline_tag: visual-question-answering
base_model:
- convaiinnovations/laya
- google/siglip2-base-patch16-256
tags:
- decision-model
- calibration
- jev
- system-one
- vision
datasets:
- lmms-lab/GQA
- lmms-lab/POPE
---

# deVision 0.1

deVision (Decision + Vision) answers typed questions about an image with **calibrated probabilities**, in one
forward pass and without generating text. It is compatible with the Jev `/v1/systemone` API; the only
extension is that `state` may contain `{"type": "image", "base64" | "url": ...}`.

- **Code, training recipe and experiment log:** https://github.com/byebyebruce/devision
- **Question types:** `noul` (probability that a statement holds) and `choice` (one of 2–255 options). `score` is not supported.
- **Input:** one English question set and one image, letterboxed to 256×256.

## Usage

```bash
git clone https://github.com/byebyebruce/devision && cd devision && uv sync
uv run devision-serve --checkpoint HF_REPO_ID --port 8000   # POST /v1/systemone, web demo at /
```

```python
from devision.model import Decider

decider = Decider.load("HF_REPO_ID")
decider.decide(
    state=[{"type": "image", "url": "https://example.com/kitchen.jpg"}],
    questions={
        "fork": {"type": "noul", "instructions": "Is there a fork in the image?"},
        "room": {"type": "choice", "instructions": "Which room is this?",
                 "criteria": {"kitchen": None, "bathroom": None, "bedroom": None}},
    },
)
```

## Architecture

SigLIP2-B/16-256 vision tower (frozen) → 2×2 pixel shuffle and a two-layer MLP projector → 64 visual tokens
placed after `[CLS]` in Laya's ModernBERT-large encoder → Laya's decision head (2 Transformer layers and a
scorer over one `[MASK]` per option) → softmax with a per-type temperature. 518M parameters, fp32.

## Training

On a Mac (MPS), about 16 hours in total (`configs/release-0.1.yaml` in the code repo, run with `devision-pipeline`):

1. **Alignment.** Image-conditioned masked captions on all COCO train2014 captions (82,583 images), training only the projector, with ModernBERT-large's MLM head.
2. **Decisions.** RLCD (noisy-logit policy gradient with proper scoring rewards, plus soft cross-entropy), training the projector, decision head and a LoRA on ModernBERT (merged into this checkpoint):
   - 60k GQA balanced train questions (yes/no → `noul`, choose → `choice`) + 40k VQAv2 train yes/no questions, one epoch;
   - then 57k questions generated from COCO train2014 instance boxes (object presence with co-occurrence negatives, position, relative position, size) + 23k replayed questions, one epoch.
3. **Calibration.** Per-type temperatures fitted on held-out GQA testdev + VQAv2 val questions: choice 2.72, noul 1.46.

Every evaluation image is excluded from every training source, under both its COCO and Visual Genome id.

## Evaluation

Through `decide()` on CPU, with the fitted temperatures:

| Set | Questions | Accuracy | ECE |
|---|---|---|---|
| POPE adversarial | 300 | 0.773 | 0.098 |
| VQAv2 val yes/no (re-split by image, balanced) | 1,000 | 0.648 | 0.025 |
| GQA testdev (yes/no + 2-way choice) | 1,000 | 0.635 | 0.042 |
| COCO val2014 box questions | 1,000 | 0.761 | 0.054 |

The VQAv2 and GQA sets are subsets drawn by this project and are not comparable to published leaderboard numbers.
Latency on a Mac CPU: P50 about 160–190 ms per question.

## Limitations

- **It cannot tell where a named object is.** Questions like "Is the cup to the left of the laptop?" are at chance. Results on above/below and size come mostly from category priors. The visual tokens carry the positions; the model does not learn to pick the object the words refer to (see the experiment log).
- English only, one image per request, `noul` and `choice` only.
- 256×256 input: small text and fine detail are lost.
- POPE-style questions are less well calibrated (ECE 0.098) than the GQA/VQAv2 questions the temperatures were fitted on.

## License

LICENSE_NOTE
