"""The training / evaluation sample format. The code that produces samples lives outside the repo.

One JSON object per JSONL line:

    {"id": "gqa:02241348", "source": "gqa",
     "image_id": "vg:2384884",            # namespaced ("coco:42", "vg:..."), used to keep eval images out of train
     "image": "gqa/images/2384884.jpg",   # relative to the data root
     "questions": {"q": {"type": "choice", "instructions": "...", "criteria": {"a": null, "b": null}}},
     "gold": {"q": {"probabilities": {"a": 0.0, "b": 1.0}}}}

`questions` follows the Jev request format (noul / choice); `gold` is a distribution over the
options (noul: "false" / "true") and is used as a soft target.
An optional "state_text" is text that comes with the image (e.g. a ScienceQA hint): decide() gets it as
[image, state_text], training feeds it to the encoder the same way.

The alignment stage (`align.py`) reads caption samples instead, one image per line:

    {"id": "coco-cap:42", "source": "coco-captions", "image_id": "coco:42",
     "image": "coco/train2014/COCO_train2014_000000000042.jpg",
     "captions": ["A dog on a couch.", "..."]}
"""
import json
from typing import Any, Dict, List

Sample = Dict[str, Any]


def read_jsonl(path: str) -> List[Sample]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
