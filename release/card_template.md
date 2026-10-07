{{front_matter}}

# deVision {{version}}

**Image + typed questions → structured answers and calibrated probabilities.** deVision is a non-autoregressive visual decision model built from Laya's ModernBERT-large decision model and a SigLIP2 vision encoder. It scores the supplied options without generating text. Multiple questions about one image share a single image encoding.

This is release **{{version}}** (training round `{{round}}`). It supports English, one image per request, yes/no decisions (`noul`) and multiple-choice decisions (`choice`). Weights and code are released under the Apache-2.0 licence; see [Licence and data](#licence-and-data).

## Install

```bash
pip install "devision @ git+https://github.com/byebyebruce/devision"
```

Python 3.11 or newer is required. For the HTTP server and browser demo, install `devision[serve]` instead. If the model repository is private, authenticate first with `hf auth login` using an account that has access.

## Python quickstart

```python
import devision

model = devision.load("{{hub_repo}}", revision="{{version}}", device="cpu")
result = model.predict(
    image="photo.jpg",  # an image path, an http(s) URL, base64, bytes or a PIL image
    state="",           # text context; empty when there is none
    questions={
        "has_fork": {"type": "noul", "instructions": "Is there a fork in the image?"},
        "room": {
            "type": "choice",
            "instructions": "Which room is this?",
            "criteria": {"kitchen": None, "bathroom": None, "bedroom": None},
        },
    },
)
print(result["answers"]["has_fork"]["noul"])       # probability of yes
print(result["answers"]["room"]["choice"])         # selected option
print(result["answers"]["room"]["probabilities"])  # probability of each option
```

`model.decide(...)` takes the same arguments. Pin `revision="{{version}}"` so that later releases do not change your results. The default device is `auto` (CUDA, then MPS, then CPU).

### Image input

The Python `image=` argument accepts:

| Input | Example |
|---|---|
| HTTP(S) URL | `image="https://example.com/photo.jpg"` |
| Local path | `image="photo.jpg"` or `image=Path("photo.jpg")` |
| Base64 data URI | `image="data:image/png;base64,..."` |
| Plain base64 | `image=base64_string` |
| Encoded image bytes | `image=image_bytes` (PNG / JPEG file contents) |
| PIL image | `image=pil_image` |

### Context

`state` carries text context as a string, object or array, exactly as in Jev:

```python
result = model.decide(
    image="photo.jpg",
    state={"note": "The customer says this item arrived damaged."},
    questions={"damaged": {"type": "noul", "instructions": "Is the item visibly damaged?"}},
)
```

An `image` key inside an object state stays text: `state={"image": "photo.jpg"}` does not load the file. The Jev-compatible image part `state=[{"type": "image", "url": "https://..."}]` also works; do not combine it with `image=`.

### Output

- `noul`: the probability that the statement holds, between 0 and 1.
- `choice`: the highest-probability option; `probabilities` covers every option and sums to 1.
- `confidence` (choice) follows Jev: `(n * p_max - 1) / (n - 1)`.
- Invalid requests raise `devision.InvalidRequest`.

## HTTP API and browser demo

```bash
pip install "devision[serve] @ git+https://github.com/byebyebruce/devision"
devision-serve --checkpoint {{hub_repo}} --device cpu --port 8000
```

Open `http://127.0.0.1:8000` for the demo. The server speaks the Jev-compatible `POST /v1/systemone` format; an image is an element of the `state` array:

```json
{"state": [{"type": "image", "url": "https://example.com/photo.jpg"}, "optional text"],
 "questions": {"has_fork": {"type": "noul", "instructions": "Is there a fork in the image?"}}}
```

Use `{"type": "image", "base64": "..."}` for an uploaded picture. HTTP never reads server-local paths. Invalid requests and `score` questions return HTTP 422.

## Architecture

| Component | |
|---|---|
| Vision encoder | Frozen SigLIP2-B/16, {{image_size}} × {{image_size}} input, about 93M parameters |
| Image preprocessing | Aspect-preserving resize and letterbox padding |
| Connector | {{shuffle}} × {{shuffle}} patch grouping, then LayerNorm → Linear → GELU → Linear |
| Visual sequence | {{visual_tokens}} visual tokens, inserted after `[CLS]` |
| Text encoder | Laya-initialised ModernBERT-large, about 395M parameters |
| Decision head | Laya's two Transformer layers and option scorer, about 27M parameters |
| Checkpoint | About 518M parameters, FP32, {{weights_gb}} GB of weights (LoRA merged) |

Each option is scored at its `[MASK]` position; a temperature-scaled softmax turns the scores into probabilities.

## Training and calibration

{{training}}

Temperatures fitted on the project's calibration set (a question's type / option-count bucket first, else its type):

{{temperature_table}}

Calibration changes probabilities, not visual ability. Validate confidence thresholds on your own data.

## Evaluation

Results of this checkpoint through `decide`, with the fitted temperatures; ECE uses 15 bins. *Mismatched* is the accuracy when every picture is swapped for an unrelated one (how much the answer depends on the picture). The project test sets use held-out pictures; they are project-specific subsets or generated questions, not official leaderboard scores, and several have informed decisions across training rounds.

{{eval_table}}

### Comparison with Laya Vision 201M

Laya Vision's published per-question predictions on the same questions; differences in percentage points with 95% intervals from paired resampling by picture. We did not rerun Laya Vision.

{{laya_table}}

{{pope_line}}

CPU latency: {{latency}}. Several questions about one picture in one request share the image encoding, so each extra question costs less.

## Limitations

{{limitations}}

## Licence and data

The weights and the code are released under the **Apache-2.0** licence, the licence of the three base models ([Laya](https://huggingface.co/convaiinnovations/laya), [SigLIP2](https://huggingface.co/google/siglip2-base-patch16-256), [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-large)). The training data come from public datasets with their own terms, some of them non-commercial (for example ScienceQA, CC BY-NC-SA 4.0) or covering the pictures separately (COCO / Flickr images). Whether such terms carry over to trained weights is not settled; check the datasets listed in this card's metadata against your use.

## Links

- [Code, training configurations and experiment records](https://github.com/byebyebruce/devision) — this release is tag `{{git_tag}}`
- [Evaluation details](evaluation/results.md) · [metrics](evaluation/results.json) · [provenance](provenance.json)
- [Laya decision model](https://huggingface.co/convaiinnovations/laya) · [SigLIP2](https://huggingface.co/google/siglip2-base-patch16-256) · [Laya Vision](https://github.com/r33drichards/laya-vision)
