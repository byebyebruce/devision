"""`Decider.decide`: the one entry point for inference. Jev `/v1/systemone` state + questions in,
Jev answers out. The HTTP server and the evaluation both go through it."""
import base64
import binascii
import io
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, NoReturn, Optional, cast

import torch
from laya.common import QTYPES, build_sequence, collate_items, serialize_state
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer, SiglipVisionConfig

from .image import to_pixel_values
from .network import ModelConfig, VisionDecisionModel, build_model, text_encoder

CONFIG_FILE = "devision_config.json"


class InvalidRequest(ValueError):
    """The request breaks the contract; the HTTP layer maps this to 422."""


MAX_CHOICE_OPTIONS = 255
MAX_IMAGE_BYTES = 20 * 1024 * 1024
URL_TIMEOUT_S = 10.0


def _fetch(url: Any) -> bytes:
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise InvalidRequest("image url must be http(s)")
    try:
        with urllib.request.urlopen(url, timeout=URL_TIMEOUT_S) as r:
            data = r.read(MAX_IMAGE_BYTES + 1)
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise InvalidRequest("image url could not be fetched: %s" % e) from None
    if len(data) > MAX_IMAGE_BYTES:
        raise InvalidRequest("image larger than %d bytes" % MAX_IMAGE_BYTES)
    return data


def _load_image(part: Dict[str, Any]) -> Image.Image:
    has_b64, has_url = "base64" in part, "url" in part
    if has_b64 == has_url:
        raise InvalidRequest("image part needs exactly one of 'base64' or 'url'")
    try:
        if has_url:
            raw = _fetch(part["url"])
        elif isinstance(part["base64"], str):
            raw = base64.b64decode(part["base64"], validate=True)
        else:
            raise InvalidRequest("image base64 must be a string")
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (binascii.Error, OSError, ValueError, Image.DecompressionBombError) as e:
        raise InvalidRequest("image could not be decoded: %s" % e) from None
    return img


def _open(raw: bytes) -> Image.Image:
    if len(raw) > MAX_IMAGE_BYTES:
        raise InvalidRequest("image larger than %d bytes" % MAX_IMAGE_BYTES)
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (OSError, ValueError, Image.DecompressionBombError) as e:
        raise InvalidRequest("image could not be decoded: %s" % e) from None
    return img


def _image_arg(image: Any) -> Image.Image:
    """The `image=` argument of decide (Python only): a PIL image, bytes, a path, or a string that is an http(s)
    URL, a data URI, an existing local file or base64 -- tried in that order."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return _open(bytes(image))
    if isinstance(image, os.PathLike):
        image = os.fspath(image)
    if not isinstance(image, str) or not image.strip():
        raise InvalidRequest("image must be a URL, data URI, file path, base64 string, bytes or PIL image")
    s = image.strip()
    if s.startswith(("http://", "https://")):
        return _open(_fetch(s))
    if s.startswith("data:"):
        head, _, body = s.partition(",")
        if ";base64" not in head or not body:
            raise InvalidRequest("image data URI must be base64 (data:image/...;base64,...)")
        s = body
    else:
        try:
            is_file = len(s) < 4096 and os.path.isfile(os.path.expanduser(s))
        except (OSError, ValueError):
            is_file = False
        if is_file:
            with open(os.path.expanduser(s), "rb") as f:
                return _open(f.read(MAX_IMAGE_BYTES + 1))
    try:
        raw = base64.b64decode("".join(s.split()), validate=True)
    except (binascii.Error, ValueError):
        raise InvalidRequest("image is not an http(s) URL, a data URI, an existing file or base64") from None
    return _open(raw)


def _split_state(state: Any):
    """(text state for the encoder, image or None)."""
    if not isinstance(state, (str, dict, list)):
        raise InvalidRequest("state must be a string, object or array")
    if not isinstance(state, list):
        return state, None
    images = [p for p in state if isinstance(p, dict) and p.get("type") == "image"]
    if len(images) > 1:
        raise InvalidRequest("at most one image per request")
    text = [p for p in state if not (isinstance(p, dict) and p.get("type") == "image")]
    if len(text) == 1 and isinstance(text[0], str):
        text = text[0]
    return (text if text else ""), (_load_image(images[0]) if images else None)


def _validate_question(qid: str, q: Any) -> None:
    def bad(msg: str) -> NoReturn:
        raise InvalidRequest("question %r: %s" % (qid, msg))

    if not isinstance(q, dict):
        bad("must be an object")
    t = q.get("type")
    if t == "score":
        bad("type 'score' is not supported yet")
    if t not in ("noul", "choice"):
        bad("type must be 'noul' or 'choice'")
    if not isinstance(q.get("instructions"), (str, dict, list)):
        bad("instructions (string, object or array) is required")
    crit = q.get("criteria")
    if t == "choice":
        if not isinstance(crit, dict):
            bad("choice criteria must map each option to a description or null")
        if not 2 <= len(crit) <= MAX_CHOICE_OPTIONS:
            bad("choice needs 2..%d options" % MAX_CHOICE_OPTIONS)
    elif crit is not None and not (isinstance(crit, dict) and set(crit) <= {"true", "false"}):
        bad("noul criteria may only describe 'true' and 'false'")


def _option_count(q: Dict[str, Any]) -> int:
    return 2 if q["type"] == "noul" else len(q["criteria"])


def question_item(tok, cfg: ModelConfig, text_state: Any, q: Dict[str, Any]) -> Dict[str, Any]:
    """Tokenized sequence + option markers for one question; shared by training and inference."""
    laya_q = {"t": q["type"], "ins": serialize_state(q["instructions"]), "crit": q.get("criteria")}
    out: tuple = build_sequence(tok, text_state, laya_q, cfg.max_len, cfg.head_max_len, return_stats=True)
    ids, markers, stats = out[0], out[1], cast(Dict[str, Any], out[2])
    # options cut to fit the head budget can end up as the same tokens: the model could not tell them apart
    return {"ids": ids, "markers": markers, "qtype": QTYPES[q["type"]],
            "options_distinct": stats.get("options_distinct", len(markers))}


def image_tensor(img: Image.Image, cfg: ModelConfig) -> torch.Tensor:
    return to_pixel_values(img, cfg.image_size)


def best_device() -> str:
    """The fastest device torch can use here: "cuda", then "mps" (Apple GPU), then "cpu"."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Decider:
    def __init__(self, model: VisionDecisionModel, tokenizer, cfg: ModelConfig):
        self.model = model.eval()
        self.tok = tokenizer
        self.cfg = cfg

    def save(self, path) -> None:
        """Self-contained checkpoint: configs, tokenizer and every weight (LoRA already merged)."""
        path = str(path)
        os.makedirs(path, exist_ok=True)
        text_encoder(self.model.decision).config.save_pretrained(os.path.join(path, "encoder"))
        self.model.vision.config.save_pretrained(os.path.join(path, "vision"))
        self.tok.save_pretrained(os.path.join(path, "tokenizer"))
        save_file({k: v.contiguous().cpu() for k, v in self.model.state_dict().items()},
                  os.path.join(path, "model.safetensors"))
        with open(os.path.join(path, CONFIG_FILE), "w") as f:
            json.dump(self.cfg.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path, device: str = "auto", revision: Optional[str] = None,
             token: Optional[str] = None) -> "Decider":
        """`path` is a checkpoint directory (as written by `save`) or a Hugging Face model repo id;
        `revision` and `token` apply to the latter. `device`: "auto" (default: the fastest available --
        CUDA, then MPS, then CPU, so it also runs on machines without a GPU), or "cuda" / "mps" / "cpu"."""
        device = best_device() if device == "auto" else device
        path = str(path)
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download

            path = snapshot_download(path, revision=revision, token=token)
        with open(os.path.join(path, CONFIG_FILE)) as f:
            cfg = ModelConfig.from_dict(json.load(f))
        model = build_model(AutoConfig.from_pretrained(os.path.join(path, "encoder")),
                            SiglipVisionConfig.from_pretrained(os.path.join(path, "vision")), cfg)
        model.load_state_dict(load_file(os.path.join(path, "model.safetensors")), strict=True)
        tok = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
        return cls(model.to(device).float(), tok, cfg)

    def predict(self, state: Any, questions: Any, image: Any = None) -> Dict[str, Any]:
        """Same as `decide`, under Laya's name."""
        return self.decide(state, questions, image=image)

    @torch.inference_mode()
    def decide(self, state: Any, questions: Any, image: Any = None) -> Dict[str, Any]:
        """Jev answers. Each question's probabilities are softmax(option logits / T), T being its
        (type, option count) bucket's temperature, or its type's when the bucket has none.

        `image` (Python only, not part of the HTTP request): the picture as an http(s) URL, data URI, local file
        path, base64 string, bytes, pathlib.Path or PIL image -- the same as an image part in `state`, which must
        then not have one. A key named "image" inside an object state stays plain text, as in Jev."""
        if not isinstance(questions, dict) or not questions:
            raise InvalidRequest("questions must be a non-empty object")
        for qid, q in questions.items():
            _validate_question(qid, q)
        text_state, state_image = _split_state(state)
        if image is not None and state_image is not None:
            raise InvalidRequest("give the image either as image= or as an image part in state, not both")
        image = _image_arg(image) if image is not None else state_image
        qids = list(questions)
        items = [question_item(self.tok, self.cfg, text_state, questions[qid]) for qid in qids]
        for qid, it in zip(qids, items):
            if len(it["markers"]) != _option_count(questions[qid]):
                raise InvalidRequest("question %r: options do not fit the model's %d-token input; "
                                     "use fewer or shorter options" % (qid, self.cfg.max_len))
            if it["options_distinct"] < len(it["markers"]):
                raise InvalidRequest("question %r: some options are identical once cut to fit the input; "
                                     "make them shorter or let them differ earlier" % qid)
        batch = collate_items([items], self.tok.pad_token_id)
        assert batch is not None
        dev = next(self.model.parameters()).device
        visual = None
        if image is not None:
            pixels = image_tensor(image, self.cfg)[None].to(dev)
            visual = self.model.encode_image(pixels).expand(len(items), -1, -1)
        logits = self.model(*(batch[k].to(dev) for k in
                              ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")),
                            visual=visual).cpu()
        n_visual = 0 if visual is None else visual.size(1)
        answers = {}
        for i, qid in enumerate(qids):
            q, it = questions[qid], items[i]
            k = len(it["markers"])
            t = self.cfg.temperature_for(it["qtype"], k)
            p = torch.softmax(logits[i, :k] / t, -1).tolist()
            answers[qid] = self._answer(q, p)
        input_tokens = sum(len(it["ids"]) + n_visual for it in items)
        return {"model": self.cfg.model_name, "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}}

    @staticmethod
    def _answer(q: Dict[str, Any], p: List[float]) -> Dict[str, Any]:
        if q["type"] == "noul":
            return {"type": "noul", "noul": p[1]}
        options = list(q["criteria"])
        n, best = len(options), max(range(len(p)), key=p.__getitem__)
        return {"type": "choice", "choice": options[best],
                "probabilities": dict(zip(options, p)),
                "confidence": (n * p[best] - 1) / (n - 1)}
