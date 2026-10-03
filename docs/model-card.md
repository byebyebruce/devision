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
- devision
- system-one
- calibrated-decisions
- rlcd
- vision
- visual-question-answering
- jev
datasets:
- lmms-lab/GQA
---

# deVision

**Non-autoregressive System 1 decision model that looks at pictures.** Give it an **image** and **typed questions**; it returns typed answers with calibrated probabilities in a single forward pass, without generating text. It speaks the Jev `/v1/systemone` request and response format; the only extension is that `state` may hold an image (`{"type": "image", "base64" | "url": ...}`). Built on [Laya](https://huggingface.co/convaiinnovations/laya)'s decision model and a SigLIP2 vision encoder, trained with RLCD (reinforcement learning against strictly proper scoring rules).

This checkpoint is training round 2 (`v2-align-lora`).

<p align="center">
  <img src="eval/benchmark_laya_vision.png" alt="deVision and laya-vision 201M on laya-vision's published questions" width="80%" />
</p>

## Installation

```bash
git clone https://github.com/byebyebruce/devision && cd devision && uv sync
```

Python 3.12, managed with [uv](https://docs.astral.sh/uv/). Inference runs on CPU.

## Quickstart

```python
import devision

decider = devision.load("HF_REPO_ID")   # downloads this repository on first use
result = decider.decide(
    state=[{"type": "image", "url": "https://example.com/kitchen.jpg"}],
    questions={
        "fork": {"type": "noul", "instructions": "Is there a fork in the image?"},
        "room": {"type": "choice", "instructions": "Which room is this?",
                 "criteria": {"kitchen": None, "bathroom": None, "bedroom": None}},
    },
)
print(result["answers"]["fork"]["noul"])            # probability that the statement holds
print(result["answers"]["room"]["choice"])          # kitchen
print(result["answers"]["room"]["probabilities"])   # one probability per option
```

All questions about one image share one image encoding.

## Self-hosting: Jev-compatible HTTP server

```bash
uv run devision-serve --checkpoint HF_REPO_ID --port 8000   # POST /v1/systemone; a web demo at /
```

```bash
curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": [{"type": "image", "url": "https://example.com/street.jpg"}],
  "questions": {"car": {"type": "noul", "instructions": "Is there a car in the image?"}}
}'
```

`score` questions return 422; so does a malformed question, naming the problem.

---

## Architecture

- **Vision:** SigLIP2-B/16 at 256×256 (frozen, 93M). The image is letterboxed to a square; its 256 patch features are pixel-shuffled 2×2 and mapped by a two-layer MLP projector (4M) into **64 visual tokens**, placed right after `[CLS]`.
- **Decision model:** Laya's ModernBERT-large encoder (395M, bidirectional) and decision head (27M: 2 Transformer layers and a scorer). Every option is scored at its own `[MASK]` token, then softmaxed over that question's options, with one temperature per question type.
- **Size:** 518M parameters, stored in fp32 (2.07 GB).

## Training

About 13 hours on a Mac (MPS); the recipe is `configs/v2-align-lora.yaml` in the code repository, run with `devision-pipeline`.

1. **Alignment.** Image-conditioned masked captions: all COCO train2014 captions (82,583 images, 2 epochs) with ModernBERT-large's MLM head, training the projector and a LoRA (r=16) on ModernBERT, merged afterwards.
2. **Decisions (RLCD).** Exploration adds zero-mean Gaussian noise to the logits; the reward is a strictly proper scoring rule, plus soft cross-entropy. Trains the projector, the decision head and a LoRA on ModernBERT (merged into this checkpoint):
   - 60k GQA balanced train questions (yes/no → `noul`, choose → `choice`) + 40k VQAv2 train yes/no questions, one epoch;
   - then 57k questions generated from COCO train2014 instance boxes (presence with co-occurrence negatives, position, relative position, size) + 23k replayed questions, one epoch.
3. **Calibration.** One temperature per question type, fitted on GQA testdev + VQAv2 val questions: choice 2.71, noul 1.57.

Every evaluation picture is excluded from every training source, under both its COCO and Visual Genome id.

---

## Evaluation

Full numbers: [`eval/results.md`](eval/results.md) and [`eval/results.json`](eval/results.json). Through `decide()` on CPU, with the fitted temperatures; the test sets hold no picture any training file used.

| Set | Questions | Accuracy | ECE |
|---|---|---|---|
| COCO object presence | 1,120 | 0.931 | 0.023 |
| COCO size (which is bigger) | 1,100 | 0.837 | 0.029 |
| VQAv2 multiple choice (13 categories) | 1,420 | 0.826 | 0.142 |
| GQA val (choice) | 992 | 0.750 | 0.051 |
| COCO position | 1,274 | 0.708 (above/below 0.902, left/right 0.516) | 0.035 |
| VQAv2 val yes/no | 1,000 | 0.691 | 0.030 |
| COCO relative position | 1,950 | 0.598 (above/below 0.860, left/right 0.518) | 0.017 |
| POPE, all splits (watched during training) | 8,676 | 0.828 | 0.084 |

With every picture swapped for another one, accuracy falls to chance or to the question-only baseline on every set: the scores come from the image.

<p align="center">
  <img src="eval/reliability_heldout.png" alt="reliability on held-out test questions" width="45%" />
</p>

### Against laya-vision 201M

On laya-vision's published questions, paired question by question with its published predictions (difference = deVision − laya-vision, 95% interval by picture bootstrap):

| Set | Questions | laya-vision 201M | deVision | Difference |
|---|---|---|---|---|
| VQAv2 yes/no | 4,887 | 0.717 | 0.721 | +0.004 [−0.013, +0.020] |
| A-OKVQA (4-way) | 1,138 | 0.598 | 0.527 | −0.071 [−0.105, −0.035] |
| ScienceQA (with image) | 2,097 | 0.824 | 0.471 | −0.353 [−0.379, −0.328] |
| POPE random / popular / adversarial | 3 × 3,000 | 0.836 / 0.819 / 0.777 (published) | 0.842 / 0.835 / 0.805 | not paired |

deVision was never trained on A-OKVQA or ScienceQA-style questions (common-sense reasoning, textbook diagrams); laya-vision was.

**Speed:** P50 about 160 ms for a one-question request on a Mac CPU (warm-up excluded; not a server benchmark).

The VQAv2 and GQA sets are subsets drawn by this project and are not comparable to published leaderboard numbers.

---

## Honest Limits

- **Left and right are at chance.** "Is the cup to the left of the laptop?" is a coin flip, while above/below and size work. The visual tokens carry the positions; the model does not learn to pick out the object the words refer to.
- **Colour and counting choices are weak** (about 0.49 each on VQAv2 multiple choice).
- **POPE-style questions lean towards "no"** (recall 0.71), and VQAv2 multiple choice is over-confident (ECE 0.142): the temperatures were fitted on yes/no and two-way questions. Validate decision thresholds on your own data.
- **No reasoning or reading:** common-sense (A-OKVQA) and diagram (ScienceQA) questions are far below laya-vision; small text and fine detail are lost at 256×256.
- English only, one image per request, `noul` and `choice` only.

---

## Links

- **Code, training recipe and experiment log:** https://github.com/byebyebruce/devision
- **Laya (the decision model this builds on):** https://huggingface.co/convaiinnovations/laya

LICENSE_NOTE
