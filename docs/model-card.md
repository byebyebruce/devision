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

# deVision (round 2, `v2-align-lora`)

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
import devision

decider = devision.load("HF_REPO_ID")
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

On a Mac (MPS), about 13 hours in total (`configs/v2-align-lora.yaml` in the code repo, run with `devision-pipeline`):

1. **Alignment.** Image-conditioned masked captions on all COCO train2014 captions (82,583 images, 2 epochs) with ModernBERT-large's MLM head, training the projector and a LoRA (r=16) on ModernBERT, merged afterwards.
2. **Decisions.** RLCD (noisy-logit policy gradient with proper scoring rewards, plus soft cross-entropy), training the projector, decision head and a LoRA on ModernBERT (merged into this checkpoint):
   - 60k GQA balanced train questions (yes/no → `noul`, choose → `choice`) + 40k VQAv2 train yes/no questions, one epoch;
   - then 57k questions generated from COCO train2014 instance boxes (object presence with co-occurrence negatives, position, relative position, size) + 23k replayed questions, one epoch.
3. **Calibration.** Per-type temperatures fitted on held-out GQA testdev + VQAv2 val questions: choice 2.71, noul 1.57.

Every evaluation image is excluded from every training source, under both its COCO and Visual Genome id.

## Evaluation

Through `decide()` on CPU, with the fitted temperatures. The test sets hold no picture any training file used.

| Set | Role | Questions | Accuracy | ECE |
|---|---|---|---|---|
| COCO object presence | held out | 1,120 | 0.931 | 0.023 |
| COCO size (which is bigger) | held out | 1,100 | 0.837 | 0.029 |
| VQAv2 multiple choice (13 categories) | held out | 1,420 | 0.826 | 0.142 |
| GQA val (yes/no + choice) | held out | 992 | 0.750 | 0.051 |
| COCO position | held out | 1,274 | 0.708 (above/below 0.902, left/right 0.516) | 0.035 |
| VQAv2 val yes/no | held out | 1,000 | 0.691 | 0.030 |
| COCO relative position | held out | 1,950 | 0.598 (above/below 0.860, left/right 0.518) | 0.017 |
| POPE (all three splits) | watched in training | 8,676 | 0.828 | 0.084 |
| GQA testdev + VQAv2 val | **temperature fit** | 2,000 | 0.683 | 0.029 |

With every picture swapped for another one, accuracy falls to chance or to the question-only baseline on every set, so the scores come from looking at the image.

On laya-vision's published questions, paired question by question with their per-question predictions (difference = ours − laya-vision 201M, 95% interval by picture bootstrap): VQAv2 yes/no 0.721 vs 0.717, +0.4 [−1.3, +2.0]; A-OKVQA 0.527 vs 0.598, −7.1 [−10.5, −3.5]; ScienceQA 0.471 vs 0.824 (deVision was never trained on A-OKVQA or ScienceQA-style questions; laya-vision was). POPE random / popular / adversarial: 0.842 / 0.835 / 0.805 (laya-vision reports 0.836 / 0.819 / 0.777).

The VQAv2 and GQA sets are subsets drawn by this project and are not comparable to published leaderboard numbers.
Latency on a Mac CPU: P50 about 160–190 ms per question (an older measurement without warm-up separation or thread count; not a server benchmark).

## Limitations

- **Left and right are at chance.** "Is the cup to the left of the laptop?" is a coin flip; above/below and size work. The visual tokens carry the positions; the model does not learn to pick the object the words refer to (see the experiment log).
- Colour and counting choices are weak (about 0.49 each).
- POPE-style questions lean towards "no" (recall 0.71) and VQAv2 multiple choice is over-confident (ECE 0.142); the temperatures were fitted on GQA/VQAv2 yes/no and two-way questions. Validate any decision threshold on your own data.
- English only, one image per request, `noul` and `choice` only.
- 256×256 input: small text and fine detail are lost.

## License

LICENSE_NOTE
