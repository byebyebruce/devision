{{front_matter}}

# deVision {{version}}

**Image + English questions → structured answers with calibrated probabilities.** deVision pairs a SigLIP2 vision encoder with Laya's ModernBERT decision model. It scores the options of yes/no (`noul`) and multiple-choice (`choice`) questions instead of generating text, follows the Jev answer format, and runs without a GPU. Several questions about one image share a single image encoding.

## Usage

```bash
pip install "devision @ git+https://github.com/byebyebruce/devision"
```

```python
import devision

model = devision.load("{{hub_repo}}")   # latest release; revision="{{version}}" pins this one; device defaults to auto
result = model.predict(
    image="photo.jpg",   # a path, an http(s) URL, a data URI, base64, bytes or a PIL image
    state="",            # text context; "" when there is none
    questions={
        "has_fork": {"type": "noul", "instructions": "Is there a fork in the image?"},
        "room": {"type": "choice", "instructions": "Which room is this?",
                 "criteria": {"kitchen": None, "bathroom": None, "bedroom": None}},
    },
)
print(result["answers"]["has_fork"]["noul"])        # probability of yes
print(result["answers"]["room"]["probabilities"])   # one probability per option, summing to 1
```

HTTP server with a browser demo at `http://127.0.0.1:8000`:

```bash
devision-serve --checkpoint {{hub_repo}} --port 8000
curl -s http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' \
  -d '{"image": "https://example.com/photo.jpg", "state": "", "questions": {"has_fork": {"type": "noul", "instructions": "Is there a fork in the image?"}}}'
```

Over HTTP, `image` is an http(s) URL, a data URI or base64; the server never reads its own files. Responses follow Jev; `confidence` for `choice` is `(n * p_max - 1) / (n - 1)`. Invalid requests return 422 (`devision.InvalidRequest` in Python).

**Good for:** whether something is present and how many (ask counts as a choice), what kind of thing or scene, colours and materials, up/down and left/right in photos (as a choice). **Not for:** comparing two places on a diagram, chart or map; relative-position yes/no questions; small text; other languages or several images. Use the probabilities as thresholds and send uncertain cases to a person or a stronger model, after checking the thresholds on your own data.

## Evaluation

Accuracy through `decide` with the fitted temperatures. *Mismatched*: the same questions with every picture swapped for an unrelated one. Test sets use held-out pictures; they are project subsets, not official leaderboard scores.

{{eval_table}}

Against Laya Vision 201M on the same questions (its published per-question predictions; difference in points, 95% interval from paired resampling by picture):

{{laya_table}}

{{pope_line}}

CPU latency: {{latency}}.

## Model

| | |
|---|---|
| Vision | Frozen SigLIP2-B/16, {{image_size}} × {{image_size}} letterboxed input, {{visual_tokens}} visual tokens after a {{shuffle}} × {{shuffle}} merge and an MLP projector |
| Decision | Laya-initialised ModernBERT-large and Laya's decision head; each option is scored at its `[MASK]`, then a temperature-scaled softmax |
| Size | About 518M parameters, FP32, {{weights_gb}} GB (LoRA merged) |

{{training}}

## Limitations

{{limitations}}

## Licence

Weights and code: Apache-2.0, like the three base models ([Laya](https://huggingface.co/convaiinnovations/laya), [SigLIP2](https://huggingface.co/google/siglip2-base-patch16-256), [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-large)). The training datasets (listed in this card's metadata) have their own terms, some non-commercial (e.g. ScienceQA, CC BY-NC-SA 4.0); check them against your use.

## Links

[Code and training records](https://github.com/byebyebruce/devision) (this release: tag `{{git_tag}}`) · [evaluation details](evaluation/results.md) · [provenance](provenance.json) · [Laya Vision](https://github.com/r33drichards/laya-vision)
