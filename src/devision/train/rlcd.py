"""RLCD fine-tuning of the vision decision model, following Laya's single-device script
(research/scripts/finetune_single_device.py in NandhaKishorM/laya, Apache-2.0).

Frozen: the SigLIP vision tower. Trained: projector, Laya's decision head, LoRA on ModernBERT.
LoRA is merged before saving, so checkpoints load without peft.

Calibration: at the end, temperatures are fitted on the val set, one per question type and one per
(type, option count) bucket as in Laya (`fit_temperatures`); `calibrate` refits them on an existing
checkpoint without training (the `devision-calibrate` command).

Every `save_every` steps the training state (trainable weights with the LoRA unmerged, optimizer,
scheduler, step, epoch order, RNG, metrics so far) is written to <out>/resume.pt, replacing the previous
one. Training the same out directory again picks up from there (the model must be built the same way,
i.e. the same --init); the file is removed when training finishes.
"""
import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
from laya.common import QTYPES, clamp_temperature, collate_items, ece_score, proper_reward
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image

from .compare import same_start
from .evaluate import score_records
from .samples import Sample
from .tracking import Tracker, TrainLog
from ..infer import Decider, image_tensor, question_item
from ..model import ModelConfig, option_bucket, pick_temperature, question_mask, text_encoder
from ..infer import best_device

LORA_TARGETS = ["Wqkv", "Wo", "Wi"]  # ModernBERT attention + MLP projections
MIN_TYPE_ITEMS = 10      # fewer val questions of a type: its temperature is not fitted
MIN_BUCKET_ITEMS = 50    # fewer val questions in a bucket: no bucket temperature, decide() uses the type's


@dataclass
class TrainConfig:
    epochs: int = 4
    micro_batch: int = 8
    eval_batch: int = 32         # questions per forward pass when evaluating (val / --eval sets); results do not depend on it
    group_size: int = 4          # RLCD noisy-logit samples per item
    sigma_start: float = 0.4
    sigma_end: float = 0.1
    lr_new: float = 1e-3         # projector (randomly initialised)
    lr_head: float = 1e-4        # Laya decision head
    lr_lora: float = 2e-4
    warmup: int = 200            # linear warm-up steps, then cosine decay to 0 over the run
    lora_r: int = 16
    lora_alpha: int = 32
    unfreeze_top: int = 0        # also train the original weights of ModernBERT's top N layers
    lr_encoder: float = 2e-5     # learning rate for those weights
    calib_fraction: float = 0.1  # held out to fit temperatures when no val set is given
    seed: int = 0
    device: str = "auto"
    log_every: int = 10
    eval_every: int = 500        # steps between evaluations (plus step 0 and each epoch end); 0: epoch ends only
    save_every: int = 1000       # steps between resume states (<out>/resume.pt); 0: none
    amp: str = "cuda"            # bf16 autocast of the forward pass: "cuda" (on CUDA only), "mps" (on CUDA or MPS), "off"
    swanlab_project: str = ""    # "" disables SwanLab; the CLI defaults to "devision"
    run_name: str = ""
    # round 13 screens (docs/experiments/2026-10-08-r13-binding-plan.md)
    grounding_weight: float = 0.0   # S1: weight of the grounding loss (questions whose sample has boxes); 0: off
    question_conditioning: bool = False   # S2: add question-conditioned visual tokens (model.QuestionConditioning)
    lr_cond: float = 1e-4        # learning rate of the S2 module and the S1 grounding head
    # v12 retrain (docs/experiments/2026-10-11-v12-retrain-plan.md, section 10): at every evaluation, also write
    # each set's per-question records (devision-eval's details format, temperature 1) to
    # <out>/monitor/step<N>/<set>.details.jsonl, and the fitted-temperature ones to <out>/monitor/final/
    monitor_details: bool = False
    # exp18 (docs/experiments/2026-10-13-exp18-distill-plan.md): questions of the samples whose "block" is in
    # distill_blocks (comma-separated) also get distill_weight * KL(teacher || student) over their options, the
    # teacher being the frozen checkpoint distill_teacher (eval, no RLCD noise, temperature 1); the sum over those
    # questions is divided by the micro batch size. The RLCD loss and gold of every question are unchanged.
    distill_weight: float = 0.0
    distill_teacher: str = ""
    distill_blocks: str = ""
    # a run that reuses another as its baseline (exp18): after the step-0 evaluation, before the first update, its
    # per-question records must match the baseline's step-0 records in this directory (same questions, valid
    # probabilities, none further apart than start_tolerance); otherwise START_MISMATCH is written and training stops
    start_check: str = ""
    start_tolerance: float = 1e-5
    # exp19 (docs/experiments/2026-10-14-exp19-vision-plan.md): also train the original weights of the SigLIP
    # tower's top N encoder layers and its final layer norm, at lr_vision; 0: the tower stays frozen
    vision_unfreeze_top: int = 0
    lr_vision: float = 5e-6
    # exp42 (docs/experiments/2026-10-18-exp42-objective-plan.md): "rlcd" = the RLCD policy term + gold CE (every run
    # before it); "ce" = gold CE only. The RLCD noise is drawn either way, so both use the same random stream.
    objective: str = "rlcd"
    # exp44 (docs/experiments/2026-10-18-exp44-number-plan.md): aux_weight > 0 adds aux_weight x a per-question
    # number reconstruction loss -- the same picture, state text and question, no options, aux_slots [MASK] slots
    # read by ModernBERT's frozen MLM head (aux_head, a training device only) into the gold number's tokens and an
    # end token. 0: off, nothing loaded, the run is exactly the one before it.
    aux_weight: float = 0.0
    aux_slots: int = 8
    aux_head: str = "answerdotai/ModernBERT-large"
    # exp45 (docs/experiments/2026-10-18-exp45-region-plan.md): region_file (a sidecar JSONL: per training row id,
    # the asked field's box [x, y, w, h] and the picture's width / height) makes the training keep the attention of
    # the question conditioning module (model.QuestionConditioning, needs question_conditioning) and adds
    # region_weight x KL(target || attention), the target being the box's share of each SigLIP patch mixed 0.9 / 0.1
    # with the uniform distribution. The same attention path runs whatever the weight (0 = the matched control).
    # "" = off: nothing loaded, the run is the one before it; decide never uses the boxes.
    region_file: str = ""
    region_weight: float = 0.0
    # exp48 (docs/experiments/2026-10-18-exp48-interaction-plan.md): interaction_file (a sidecar JSONL: per training
    # row id, its four-corner quad, view 0 / 1, object 0 / 1 and the a / b option indices) adds interaction_weight x
    # mean over the batch's quads of softplus(1 - J), J = (m[0,A] - m[0,B] - m[1,A] + m[1,B]) / 4 with
    # m[v,o] = z[v,o,a] - z[v,o,b] on the ordinary option logits. Computed (for monitoring) whatever the weight;
    # "" = off: nothing loaded, the run is the one before it.
    interaction_file: str = ""
    interaction_weight: float = 0.0
    # exp50 (docs/experiments/2026-10-18-exp50-local-views-plan.md): local_views > 0 gives the model that many fixed
    # local views (ModelConfig.local_views / local_overlap / local_source, saved with the checkpoint, so decide reads
    # them the same way) read through the question conditioning, whose zero-initialised geometry layer joins the cond
    # group; needs question_conditioning. Each micro batch's pictures are encoded once. 0: off, the run is the one before it.
    local_views: int = 0
    local_overlap: float = 0.0
    local_source: str = ""


GRID = 8   # visual tokens per side of the released model (16x16 patches, 2x2 merged), row-major as Projector orders them


def box_grid(boxes, width: int, height: int, size: int, grid: int = GRID, smooth: float = 0.1) -> List[float]:
    """Where the named object is, as a distribution over the grid x grid visual tokens: each [x, y, w, h] box
    (picture pixels) is letterboxed exactly as the picture is (image.letterbox), the share of each cell it
    covers is summed over boxes and normalised, then mixed with the uniform distribution (`smooth`)."""
    scale = size / max(width, height)
    ox, oy = (size - round(width * scale)) // 2, (size - round(height * scale)) // 2
    cell = size / grid
    cover = [0.0] * (grid * grid)
    for x, y, w, h in boxes:
        x0, y0, x1, y1 = ox + x * scale, oy + y * scale, ox + (x + w) * scale, oy + (y + h) * scale
        for r in range(grid):
            for c in range(grid):
                ix = max(0.0, min(x1, (c + 1) * cell) - max(x0, c * cell))
                iy = max(0.0, min(y1, (r + 1) * cell) - max(y0, r * cell))
                cover[r * grid + c] += ix * iy / (cell * cell)
    total = sum(cover)
    if total <= 0:
        return []
    return [(1 - smooth) * v / total + smooth / len(cover) for v in cover]


class GroundingHead(nn.Module):
    """S1, training only: from the encoder's last layer (a global-attention layer), the mean of the question's
    tokens scores each visual token; trained towards where the named object is (KL to box_grid)."""

    def __init__(self, dim: int, inner: int = 256):
        super().__init__()
        self.q, self.k, self.inner = nn.Linear(dim, inner), nn.Linear(dim, inner), inner

    def forward(self, hidden, n_visual: int, qmask):
        """hidden [B, L+V, D] (visual at 1..V), qmask [B, L] over text positions -> log-probs [B, V]."""
        text = torch.cat([hidden[:, :1], hidden[:, 1 + n_visual:]], 1)
        m = qmask.unsqueeze(-1).to(text.dtype)
        q = (text * m).sum(1) / m.sum(1).clamp(min=1)
        k = hidden[:, 1:1 + n_visual]
        scores = (self.k(k) @ self.q(q).unsqueeze(-1)).squeeze(-1) / self.inner ** 0.5
        return torch.log_softmax(scores.float(), -1)


def plan_batches(items: Sequence[Dict[str, Any]], micro_batch: int, rng: Any = random) -> List[List[int]]:
    """Micro batches (lists of item indices) in which every counterfactual unit is whole: the items that ask
    the same question (same "group" and question id) about a scene family's pictures -- base, recolours, count
    edit -- always share a batch, so the model sees the pair in one update (critic R13-I02). Items without a
    group are units of one and fill the remaining slots. Units are shuffled, packed first-fit, and the batches
    shuffled; a unit larger than a batch is split into batch-sized pieces."""
    units: Dict[Any, List[int]] = {}
    for i, it in enumerate(items):
        key = (it["group"], it.get("qid")) if "group" in it else ("", i)
        units.setdefault(key, []).append(i)
    pieces = [u[j:j + micro_batch] for u in units.values() for j in range(0, len(u), micro_batch)]
    rng.shuffle(pieces)
    pieces.sort(key=len, reverse=True)   # first-fit decreasing: the big units first, singletons fill the gaps
    batches: List[List[int]] = []
    open_: List[List[int]] = []
    for u in pieces:
        for bt in open_:
            if len(bt) + len(u) <= micro_batch:
                bt.extend(u)
                break
        else:
            bt = list(u)
            batches.append(bt)
            open_.append(bt)
        open_ = [bt for bt in open_ if len(bt) < micro_batch]
    rng.shuffle(batches)
    return batches


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device(best_device())
    return torch.device(name)


def _items(decider: Decider, samples: Sequence[Sample]) -> List[Dict[str, Any]]:
    items = []
    for s in samples:
        for qid, q in s["questions"].items():
            # text that comes with the image (e.g. a ScienceQA hint), as decide() gets it from the state
            it = question_item(decider.tok, decider.cfg, s.get("state_text", ""), q)
            probs = s["gold"][qid]["probabilities"]
            keys = ["false", "true"] if q["type"] == "noul" else list(q["criteria"])
            target = [float(probs.get(k, 0.0)) for k in keys]
            total = sum(target)
            if not total > 0:   # read_decision_jsonl rejects these; never turn a broken label into a uniform one
                raise ValueError("sample %s question %r: gold probabilities sum to 0" % (s.get("id"), qid))
            it["target"] = [t / total for t in target]
            it["image"] = s["image"]
            it["source"] = s.get("source", "")
            it["qid"] = qid
            # what a per-question monitor record needs (the same fields devision-eval writes)
            it["sample_id"], it["image_id"] = s.get("id", ""), s.get("image_id", s["image"])
            it["kind"], it["axis"], it["type"] = s.get("kind", ""), s.get("axis", ""), q["type"]
            it["keys"], it["gold"] = keys, probs
            if "group" in s:
                it["group"] = s["group"]   # a scene family: its counterfactual pairs share a micro batch
            if "block" in s:
                it["block"] = s["block"]   # the data block it was mixed from (round 14 mixes); picks distilled questions
            boxes = (s.get("boxes") or {}).get(qid)
            present = ((s.get("meta") or {}).get(qid) or {}).get("referent_present", True)
            if boxes and present and s.get("width") and s.get("height"):   # an absent referent is never a target
                side = decider.cfg.image_size // int(cast(int, decider.model.vision.config.patch_size)) // decider.cfg.visual_shuffle
                grid = box_grid(boxes, s["width"], s["height"], decider.cfg.image_size, grid=side)
                if grid:
                    it["grid"] = grid
            items.append(it)
    return items


class StartMismatch(RuntimeError):
    """The run does not start from its baseline's model (TrainConfig.start_check): not a device failure, so not
    something to retry."""


START_MISMATCH = "START_MISMATCH"


class _Images:
    """Pixel tensor for a relative path. Decoded on every use: a JPEG decode + letterbox costs a few
    ms next to a ~1 s training step, while caching float tensors costs 0.8 MB per image."""

    def __init__(self, root, decider: Decider):
        self.root, self.decider = str(root), decider

    def __call__(self, rel: str) -> torch.Tensor:
        with Image.open(os.path.join(self.root, rel)) as img:
            return image_tensor(img, self.decider.cfg)


def _vision_params(vision, top: int) -> List[nn.Parameter]:
    """The original weights of the SigLIP tower's top `top` encoder layers and its final layer norm."""
    n = int(vision.config.num_hidden_layers)
    if not 0 < top <= n:
        raise ValueError("vision_unfreeze_top must be in 1..%d, got %d" % (n, top))
    keys = tuple("encoder.layers.%d." % i for i in range(n - top, n)) + ("post_layernorm.",)
    return [p for name, p in vision.named_parameters() if any(k in name for k in keys)]


def _forward(model, batch, pixels, device, return_hidden: bool = False, index: Optional[torch.Tensor] = None):
    """Option logits of `batch`; `pixels` one per question, or (exp50) one per distinct picture with `index` giving
    each question's picture."""
    trains_vision = torch.is_grad_enabled() and any(p.requires_grad for p in model.vision.parameters())
    patches, visual, geom = model.visual_inputs(pixels.to(device), grad_vision=trains_vision)
    if index is not None:
        index = index.to(device)
        patches, visual = patches.index_select(0, index), visual.index_select(0, index)
        geom = None if geom is None else geom.index_select(0, index)
    return model(*(batch[k].to(device) for k in
                   ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")),
                 visual=visual, patches=patches, return_hidden=return_hidden, patch_geom=geom)


def _rlcd_loss(logits, mask, target, qtype, sigma, group_size, objective: str = "rlcd"):
    """(loss, gold CE). objective "ce": the gold CE alone, after drawing the same RLCD noise (same random stream)."""
    k = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((group_size,) + logits.shape, device=logits.device) * sigma * mask
    if objective == "ce":
        ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
        return ce, ce
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    return -(adv * logp).mean() + ce, ce

INTERACTION_FORMULA = "J4/softplus(1-J)/v1"


def interaction_table(path: str, samples: Sequence[Sample]) -> Dict[Tuple[str, str], Tuple[str, int, int, int, int]]:
    """(sample id, qid) -> (quad, view, object, a index, b index) for every training question; every question needs a
    row and every quad its four distinct corners."""
    rows = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            k = (r["id"], r["qid"])
            if k in rows:
                raise ValueError("interaction: %s repeated in %s" % (k, path))
            rows[k] = (r["quad"], int(r["view"]), int(r["object"]), int(r["a_index"]), int(r["b_index"]))
    out, corners = {}, defaultdict(set)
    for s in samples:
        for qid, q in s["questions"].items():
            k = (s.get("id", ""), qid)
            if k not in rows:
                raise ValueError("interaction: no quad for %s" % (k,))
            quad, v, o, a, b = rows[k]
            n = len(q.get("criteria") or {})
            if not (0 <= a < n and 0 <= b < n and a != b):
                raise ValueError("interaction: %s a / b indices %d / %d do not fit %d options" % (k, a, b, n))
            if (v, o) in corners[quad]:
                raise ValueError("interaction: quad %s corner %s twice" % (quad, (v, o)))
            corners[quad].add((v, o))
            out[k] = rows[k]
    bad = [q for q, c in corners.items() if c != {(0, 0), (0, 1), (1, 0), (1, 1)}]
    if bad:
        raise ValueError("interaction: %d quads without their four corners, e.g. %s" % (len(bad), bad[0]))
    return out


def interaction_loss(logits: torch.Tensor, corners: Sequence[Tuple[str, int, int, int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """(mean softplus(1 - J) over the quads of a batch, mean J): `corners[i]` is row i's (quad, view, object, a, b)."""
    by: Dict[str, Dict[Tuple[int, int], torch.Tensor]] = defaultdict(dict)
    for i, (quad, v, o, a, b) in enumerate(corners):
        by[quad][(v, o)] = logits[i, a] - logits[i, b]
    js = []
    for quad, m in by.items():
        if set(m) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            raise ValueError("interaction: quad %s split across batches" % quad)
        js.append((m[(0, 0)] - m[(0, 1)] - m[(1, 0)] + m[(1, 1)]) / 4)
    j = torch.stack(js)
    return nn.functional.softplus(1 - j).mean(), j.mean()


REGION_SMOOTH = 0.1     # exp45: the target is 0.9 x the box's patch shares + 0.1 x uniform, fixed before the run


def region_targets(path: str, samples: Sequence[Sample], side: int, size: int) -> Dict[str, List[float]]:
    """Sample id -> the target distribution over the side x side SigLIP patches (box_grid of the sidecar's box on the
    letterboxed picture); every training sample needs a box inside its picture."""
    boxes = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            boxes[r["id"]] = r
    out = {}
    for s in samples:
        r = boxes.get(s.get("id", ""))
        if r is None:
            raise ValueError("region: no box for %s in %s" % (s.get("id"), path))
        x, y, w, h = r["box"]
        if not (w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= r["width"] and y + h <= r["height"]):
            raise ValueError("region: box %s of %s is not inside its %dx%d picture" % (r["box"], s.get("id"), r["width"], r["height"]))
        grid = box_grid([r["box"]], r["width"], r["height"], size, grid=side, smooth=REGION_SMOOTH)
        if not grid:
            raise ValueError("region: box of %s covers no patch" % s.get("id"))
        out[s["id"]] = grid
    return out


def region_kl(weights: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-question KL(target || attention) over the patches, averaged over the questions."""
    return (target * (target.log() - weights.float().clamp_min(1e-12).log())).sum(-1).mean()


AUX_NUMBER = re.compile(r"^(\d[\d,]*(?:\.\d+)?) ")


def aux_number(option: str) -> str:
    """The number of a reading answer ("1,200 m/s" -> "1,200"); anything else is refused."""
    m = AUX_NUMBER.match(option + " ")
    if not m:
        raise ValueError("aux: %r is not a number answer" % option)
    return m.group(1)


def aux_example(tok, state_text: str, instructions: str, gold: str, slots: int) -> Tuple[List[int], List[int]]:
    """(input ids, slot targets): [CLS] state text [SEP] question [SEP] [MASK] x slots [SEP] -- no option, no number;
    the input never depends on the answer. Targets: the number's tokens (as after a space) and [SEP], then -100."""
    ids = ([tok.cls_token_id] + tok(state_text or "", add_special_tokens=False)["input_ids"] + [tok.sep_token_id]
           + tok(instructions, add_special_tokens=False)["input_ids"] + [tok.sep_token_id]
           + [tok.mask_token_id] * slots + [tok.sep_token_id])
    target = tok(" " + aux_number(gold), add_special_tokens=False)["input_ids"] + [tok.sep_token_id]
    if len(target) > slots:
        raise ValueError("aux: %r needs %d slots, only %d" % (gold, len(target), slots))
    if len(ids) > 512:
        raise ValueError("aux: input of %d tokens" % len(ids))
    return ids, target + [-100] * (slots - len(target))


def aux_table(tok, samples: Sequence[Sample], slots: int) -> Dict[Tuple[str, str], Tuple[List[int], List[int]]]:
    """(sample id, qid) -> aux_example, for every question; a gold that is not exactly one option is refused."""
    out = {}
    for s in samples:
        for qid, q in s["questions"].items():
            probs = s["gold"][qid]["probabilities"]
            ones = [o for o, p in probs.items() if p == 1.0]
            if len(ones) != 1 or sum(probs.values()) != 1.0:
                raise ValueError("aux: %s %s has no single gold option" % (s.get("id"), qid))
            out[(s.get("id", ""), qid)] = aux_example(tok, s.get("state_text", ""), q["instructions"], ones[0], slots)
    return out


def aux_head(encoder, repo: str):
    """ModernBERT's MLM head (align._MLMHead, tied to the encoder's frozen word embeddings), frozen; and its identity:
    the file and the sha256 of its three tensors."""
    import hashlib
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from .align import _MLMHead
    head = _MLMHead(encoder)
    ident: Dict[str, Any] = {"repo": repo}
    if repo:                               # "" -> a random frozen head (tiny test models only)
        path = hf_hub_download(repo, "model.safetensors")
        head.load_pretrained(repo)
        ident["file"] = os.path.realpath(path)
        with safe_open(path, "pt") as f:
            for k in ("head.dense.weight", "head.norm.weight", "decoder.bias"):
                ident[k] = hashlib.sha256(f.get_tensor(k).float().numpy().tobytes()).hexdigest()
    for p in head.head.parameters():
        p.requires_grad_(False)
    head.bias.requires_grad_(False)
    return head, ident


def aux_logits(model, head, ids: Sequence[List[int]], pixels, device, slots: int, pad_id: int) -> torch.Tensor:
    """Vocabulary logits [B, slots, vocab] at the [MASK] slots (only there: no full-sequence vocabulary output)."""
    n, L = len(ids), max(map(len, ids))
    input_ids = torch.full((n, L), pad_id, dtype=torch.long)
    attention = torch.zeros((n, L), dtype=torch.long)
    for i, x in enumerate(ids):
        input_ids[i, :len(x)], attention[i, :len(x)] = torch.tensor(x), 1
    trains_vision = torch.is_grad_enabled() and any(p.requires_grad for p in model.vision.parameters())
    visual = model.visual_inputs(pixels.to(device), grad_vision=trains_vision)[1]
    h, _ = model.encode(input_ids.to(device), attention.to(device), visual)
    start = torch.tensor([len(x) - 1 - slots for x in ids], device=device) + visual.size(1)
    pos = start[:, None] + torch.arange(slots, device=device)[None]
    hs = h.gather(1, pos[..., None].expand(-1, -1, h.size(-1)))
    return head(hs).float()


def aux_loss(logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(loss: per question the mean CE over its number tokens and end token, then the mean over questions;
    whole-number accuracy: every target slot right)."""
    valid = target != -100
    ce = nn.functional.cross_entropy(logits.transpose(1, 2), target.clamp(min=0), reduction="none") * valid
    loss = (ce.sum(1) / valid.sum(1)).mean()
    right = ((logits.argmax(-1) == target) | ~valid).all(1).float().mean()
    return loss, right


def _distill_kl(student_logits, teacher_logits, mask):
    """Per question KL(teacher || student) over its options (mask: the real options, padding excluded), both
    softmaxed at temperature 1 from raw logits."""
    s = torch.log_softmax(student_logits.masked_fill(~mask, -1e4), -1)
    t = torch.softmax(teacher_logits.masked_fill(~mask, -1e4), -1)
    return ((t * (t.clamp_min(1e-12).log() - s)) * mask).sum(-1)


def _fit_temperature(rows) -> float:
    """One temperature for (logits, target) rows, by LBFGS on held-out NLL."""
    if len(rows) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in rows)
    Z, T = torch.full((len(rows), kmax), -1e4), torch.zeros((len(rows), kmax))
    for i, (z, t) in enumerate(rows):
        Z[i, :len(z)], T[i, :len(t)] = torch.tensor(z), torch.tensor(t)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return clamp_temperature(log_t.exp().item())


def fit_temperatures(logits, items, previous: Optional[ModelConfig] = None,
                     min_bucket: int = MIN_BUCKET_ITEMS) -> Tuple[List[float], Dict[str, float]]:
    """(per-type temperatures [choice, score, noul], per-bucket temperatures {"choice:3-5": t, ...}) fitted
    on `logits` against the items' targets.

    A bucket with fewer than `min_bucket` questions gets no entry, so decide() falls back to its type's
    temperature rather than one fitted on a handful of rows. A type with fewer than MIN_TYPE_ITEMS questions
    is not fitted: it keeps `previous`'s temperature and buckets (1.0 and none without `previous`)."""
    by_type: Dict[int, list] = defaultdict(list)
    by_bucket: Dict[str, list] = defaultdict(list)
    for z, it in zip(logits, items):
        by_type[it["qtype"]].append((z, it["target"]))
        by_bucket[option_bucket(it["qtype"], len(z))].append((z, it["target"]))
    fitted = {t for t in QTYPES.values() if len(by_type[t]) >= MIN_TYPE_ITEMS}
    temperature = [_fit_temperature(by_type[t]) if t in fitted else (previous.temperature[t] if previous else 1.0)
                   for t in range(len(QTYPES))]
    by_options = {b: _fit_temperature(rows) for b, rows in by_bucket.items() if len(rows) >= min_bucket}
    if previous:
        by_options.update({b: t for b, t in previous.temperature_by_options.items()
                           if QTYPES.get(b.split(":")[0]) not in fitted})
    return temperature, dict(sorted(by_options.items()))


def _val_logits(model, items, images, pad_id, device, per: int = 32):
    """Temperature-free option logits for each item, in eval mode; `per` questions per forward pass."""
    was_training = model.training
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(items), per):
            chunk = items[b:b + per]
            batch = collate_items([chunk], pad_id)
            assert batch is not None
            logits = _forward(model, batch, torch.stack([images(it["image"]) for it in chunk]), device)
            out.extend(row[:len(it["markers"])].tolist() for row, it in zip(logits.float().cpu(), chunk))
    model.train(was_training)
    model.vision.eval()
    return out


def monitor_records(logits, items, temperature: Optional[Sequence[float]] = None,
                    by_options: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """Per-question records of option logits, in devision-eval's details format (no latency: the questions
    were not asked through decide()). Probabilities use the temperature decide() would (1 without one)."""
    out = []
    for z, it in zip(logits, items):
        t = pick_temperature(temperature, by_options, it["qtype"], len(z)) if temperature else 1.0
        p = torch.softmax(torch.tensor(z, dtype=torch.float64) / t, -1).tolist()
        probs = dict(zip(it["keys"], p))
        pred = max(probs, key=probs.__getitem__)
        gold = it["gold"]
        out.append({"sample_id": it["sample_id"], "qid": it["qid"], "image_id": it["image_id"],
                    "image_used": it["image"], "source": it["source"], "kind": it["kind"], "axis": it["axis"],
                    "type": it["type"], "options": list(it["keys"]), "gold": gold,
                    "gold_answer": max(gold, key=gold.__getitem__), "probabilities": probs, "prediction": pred,
                    "p_prediction": probs[pred], "control": "none", "latency_ms": None, "questions_in_request": None})
    return score_records(out)


def _write_records(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    os.replace(path + ".tmp", path)


def _accuracy(logits, items, qtype: Optional[int] = None) -> float:
    hits = [max(range(len(z)), key=z.__getitem__) == max(range(len(it["target"])), key=it["target"].__getitem__)
            for z, it in zip(logits, items) if qtype is None or it["qtype"] == qtype]
    return sum(hits) / len(hits) if hits else float("nan")


def holdout_by_image(items: List[Dict[str, Any]], n: int):
    """(held-out, rest): whole images, about `n` items, so no picture is both trained on and used to fit
    temperatures. Uses the global `random` state (seeded by `train`)."""
    images = sorted({it["image"] for it in items})
    random.shuffle(images)
    held, count = set(), 0
    for img in images:
        if count >= n:
            break
        held.add(img)
        count += sum(it["image"] == img for it in items)
    return [it for it in items if it["image"] in held], [it for it in items if it["image"] not in held]


def _set_metrics(name: str, logits, items, temperature: Optional[Sequence[float]] = None,
                 by_options: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """`_metrics` as "<name>/<metric>", plus "<name>/<source>/<metric>" when the set mixes sources."""
    out = {"%s/%s" % (name, k): v for k, v in _metrics(logits, items, temperature, by_options).items()}
    sources = sorted({it["source"] for it in items})
    if len(sources) > 1:
        for src in sources:
            idx = [i for i, it in enumerate(items) if it["source"] == src]
            out.update({"%s/%s/%s" % (name, src, k): v for k, v in
                        _metrics([logits[i] for i in idx], [items[i] for i in idx], temperature,
                                 by_options).items()})
        # every source weighs the same: a big source must not decide which checkpoint looks best
        out["%s/macro_accuracy" % name] = sum(out["%s/%s/accuracy" % (name, src)] for src in sources) / len(sources)
    return out


def _metrics(logits, items, temperature: Optional[Sequence[float]] = None,
             by_options: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Accuracy (all / noul / choice), NLL against the gold distribution and ECE on max
    probability, with logits divided by the temperature decide() would use: the bucket's in `by_options`,
    else the type's in `temperature` (1 if no temperatures)."""
    nll, conf, hits = [], [], []
    for z, it in zip(logits, items):
        t = pick_temperature(temperature, by_options, it["qtype"], len(z)) if temperature else 1.0
        logp = torch.log_softmax(torch.tensor(z) / t, -1)
        nll.append(-(torch.tensor(it["target"]) * logp).sum().item())
        conf.append(logp.max().exp().item())
        hits.append(float(logp.argmax().item() == max(range(len(it["target"])), key=it["target"].__getitem__)))
    out = {"accuracy": _accuracy(logits, items), "accuracy_noul": _accuracy(logits, items, QTYPES["noul"]),
           "accuracy_choice": _accuracy(logits, items, QTYPES["choice"]),
           "nll": sum(nll) / len(nll) if nll else float("nan"), "ece": ece_score(np.array(conf), np.array(hits))}
    return {k: v for k, v in out.items() if v == v}  # drop NaN (a type the set has no questions of)


def bucket_metrics(logits, items, temperature: Sequence[float],
                   by_options: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
    """{"all" / bucket: {"n", "temperature" (buckets only), "accuracy", "nll", "ece"}} under these temperatures."""
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, (z, it) in enumerate(zip(logits, items)):
        groups[option_bucket(it["qtype"], len(z))].append(i)
    out = {}
    for name, idx in [("all", list(range(len(items))))] + sorted(groups.items()):
        m = _metrics([logits[i] for i in idx], [items[i] for i in idx], temperature, by_options)
        row = {"n": len(idx), **{k: m[k] for k in ("accuracy", "nll", "ece") if k in m}}
        if name != "all":
            row["temperature"] = pick_temperature(temperature, by_options, items[idx[0]]["qtype"],
                                                  len(logits[idx[0]]))
        out[name] = row
    return out


def calibrate(decider: Decider, samples: Sequence[Sample], data_root, device: str = "auto",
              min_bucket: int = MIN_BUCKET_ITEMS) -> Dict[str, Any]:
    """Refit `decider`'s temperatures (per type and per bucket, `fit_temperatures`) on `samples` without
    training; the weights are untouched. Returns the temperatures and `bucket_metrics` before (the
    checkpoint as it was) and after."""
    dev = _device(device)
    model = decider.model.to(dev).eval()
    items = _items(decider, samples)
    logits = _val_logits(model, items, _Images(data_root, decider), decider.tok.pad_token_id, dev)
    decider.model = model.cpu()
    cfg = decider.cfg
    before = {"temperature": list(cfg.temperature), "temperature_by_options": dict(cfg.temperature_by_options)}
    cfg.temperature, cfg.temperature_by_options = fit_temperatures(logits, items, cfg, min_bucket)
    after = {"temperature": list(cfg.temperature), "temperature_by_options": dict(cfg.temperature_by_options)}
    return {"items": len(items), "min_bucket": min_bucket, "before": before, "after": after,
            "metrics_before": bucket_metrics(logits, items, before["temperature"], before["temperature_by_options"]),
            "metrics_after": bucket_metrics(logits, items, cfg.temperature, cfg.temperature_by_options)}


RESUME = "resume.pt"
# what must match for a resume state to continue a run (logging and saving settings may change)
_RESUME_KEYS = ("epochs", "micro_batch", "group_size", "sigma_start", "sigma_end", "lr_new", "lr_head", "lr_lora",
                "warmup", "lora_r", "lora_alpha", "unfreeze_top", "lr_encoder", "calib_fraction", "seed",
                "grounding_weight", "question_conditioning", "lr_cond", "distill_weight", "distill_teacher",
                "distill_blocks", "vision_unfreeze_top", "lr_vision", "objective", "aux_weight", "aux_slots",
                "aux_head", "region_file", "region_weight", "interaction_file", "interaction_weight", "local_views",
                "local_overlap", "local_source")


def _rng_state() -> Dict[str, Any]:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_state(path: str, state: Dict[str, Any]) -> None:
    """Write `state` to `path` through a temporary file, so a crash mid-write keeps the previous one."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def train(decider: Decider, samples: Sequence[Sample], data_root, out_dir,
          config: Optional[TrainConfig] = None,
          val_samples: Optional[Sequence[Sample]] = None,
          eval_sets: Optional[Dict[str, Sequence[Sample]]] = None) -> Dict[str, Any]:
    """Fine-tune `decider` in place on `samples`, fit temperatures on `val_samples`, save to `out_dir`.

    Without `val_samples`, `calib_fraction` of the training items are held out instead.
    The val items and each of `eval_sets` (e.g. {"pope": ...}) are evaluated at step 0, every
    `eval_every` steps and at each epoch end, then once more with the fitted temperatures.
    Returns {"losses": per-step NLL against the gold distribution (the RLCD policy term is too
    noisy to monitor), "val_accuracy": per epoch, "evals": [{"step", "<set>/<metric>": ...}],
    "final": {"<set>/<metric>": ...} with the fitted temperatures applied as decide() does, "final_per_type":
    the same with only the per-type temperatures, "temperature": [choice, score, noul], "temperature_by_options":
    {bucket: t} (buckets with at least MIN_BUCKET_ITEMS val questions), "calibration": val metrics per bucket
    under per_type / by_options temperatures, "train_items", "calib_items"}.
    """
    c = config or TrainConfig()
    device = _device(c.device)
    distill_blocks = {b for b in c.distill_blocks.split(",") if b}
    if c.objective not in ("rlcd", "ce"):
        raise ValueError("objective must be rlcd or ce, got %r" % c.objective)
    if c.start_check and not (c.monitor_details and out_dir):
        raise ValueError("start_check compares step-0 records: it needs monitor_details and an out directory")
    teacher = None
    if c.distill_weight > 0:
        if not c.distill_teacher or not distill_blocks:
            raise ValueError("distill_weight > 0 needs distill_teacher and distill_blocks")
        # loaded before the seeds below, so the batch order, LoRA init and RLCD noise are those of a run without it
        teacher = Decider.load(c.distill_teacher, device="cpu").model.to(device).eval().requires_grad_(False)
    if c.region_file and not c.question_conditioning:
        raise ValueError("region_file supervises the question conditioning attention: it needs question_conditioning")
    if c.local_views:   # exp50: before the seeds below, so the LoRA init, batch order and noise are unchanged
        if not c.question_conditioning or c.region_file or c.grounding_weight > 0 or c.vision_unfreeze_top \
                or c.distill_weight > 0 or c.aux_weight > 0:
            raise ValueError("local_views needs question_conditioning, a frozen vision tower and none of the region / "
                             "grounding / distillation / number losses")
        if c.local_source not in ("T", "C") or not 0 <= c.local_overlap < 1:
            raise ValueError("local_source must be T or C and local_overlap in [0, 1)")
        cfg = decider.cfg
        if cfg.local_views and (cfg.local_views, cfg.local_overlap, cfg.local_source) != (c.local_views, c.local_overlap, c.local_source):
            raise ValueError("the checkpoint already reads other local views")
        if decider.model.condition is None:
            decider.model.add_question_conditioning()
            cfg.question_conditioning = True
        decider.model.enable_local_views(c.local_views)
        cfg.local_views, cfg.local_overlap, cfg.local_source = c.local_views, c.local_overlap, c.local_source
    aux, aux_ident, aux_inputs = None, None, {}
    if c.aux_weight > 0:   # built and loaded before the seeds below: the LoRA init, batch order and noise are unchanged
        if c.aux_slots < 1:
            raise ValueError("aux_slots must be >= 1")
        aux_inputs = aux_table(decider.tok, samples, c.aux_slots)
        aux, aux_ident = aux_head(text_encoder(decider.model.decision), c.aux_head)
    random.seed(c.seed)
    torch.manual_seed(c.seed)
    model, tok = decider.model, decider.tok

    model.vision.requires_grad_(False)
    vision_params = _vision_params(model.vision, c.vision_unfreeze_top) if c.vision_unfreeze_top else []
    for p in vision_params:
        p.requires_grad_(True)
    if c.question_conditioning and model.condition is None:   # S2: new module, zero output layer
        model.add_question_conditioning()
        decider.cfg.question_conditioning = True
    inter: Dict[Tuple[str, str], Tuple[str, int, int, int, int]] = {}
    if c.interaction_file:   # exp48: no model change, nothing random
        inter = interaction_table(c.interaction_file, samples)
    regions: Dict[str, List[float]] = {}
    if c.region_file:   # exp45: the attention is kept during training (both arms), never in decide
        side = decider.cfg.image_size // int(cast(int, model.vision.config.patch_size))
        regions = region_targets(c.region_file, samples, side, decider.cfg.image_size)
        cast(Any, model.condition).keep_weights = True
    if c.grounding_weight > 0:   # S1: training-only head, removed before saving
        model.grounding = GroundingHead(text_encoder(model.decision).config.hidden_size)
    peft_encoder = get_peft_model(text_encoder(model.decision), LoraConfig(
        r=c.lora_r, lora_alpha=c.lora_alpha, lora_dropout=0.0, target_modules=LORA_TARGETS))
    model.decision.encoder = peft_encoder
    n_layers = text_encoder(model.decision).config.num_hidden_layers
    top = tuple("layers.%d." % i for i in range(n_layers - c.unfreeze_top, n_layers))
    unfrozen = [p for n, p in peft_encoder.named_parameters() if "lora_" not in n and any(t in n for t in top)]
    for p in unfrozen:
        p.requires_grad_(True)
    model.to(device).train()
    model.vision.eval()
    if aux is not None:
        aux.to(device).eval()

    images = _Images(data_root, decider)
    items = _items(decider, samples)
    if val_samples:
        calib, train_items = _items(decider, val_samples), items
    else:
        calib, train_items = holdout_by_image(items, min(400, int(len(items) * c.calib_fraction)))

    sets = {"val": calib, **{name: _items(decider, ss) for name, ss in (eval_sets or {}).items()}}
    if teacher is not None:
        found = {it.get("block") for it in train_items}
        if not distill_blocks <= found:
            raise ValueError("distill_blocks %s not in the training data (blocks: %s)"
                             % (sorted(distill_blocks - found), sorted(b for b in found if b)))

    groups = [
        {"params": list(model.projector.parameters()), "lr": c.lr_new},
        {"params": [p for n, p in model.decision.named_parameters()
                    if not n.startswith("encoder.") and p.requires_grad], "lr": c.lr_head},
        {"params": [p for n, p in peft_encoder.named_parameters() if "lora_" in n], "lr": c.lr_lora},
    ] + ([{"params": unfrozen, "lr": c.lr_encoder}] if unfrozen else []) \
      + ([{"params": vision_params, "lr": c.lr_vision}] if vision_params else [])
    extra = [p for name in ("condition", "grounding") if getattr(model, name, None) is not None
             for p in getattr(model, name).parameters()]
    if extra:
        groups.append({"params": extra, "lr": c.lr_cond})
    group_names = ["projector", "head", "lora"] + (["encoder"] if unfrozen else []) \
        + (["vision"] if vision_params else []) + (["cond"] if extra else [])
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    grouped = any("group" in it for it in train_items)
    first_plan = plan_batches(train_items, c.micro_batch) if grouped else []
    steps = c.epochs * (len(first_plan) if grouped else max(1, -(-len(train_items) // c.micro_batch)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, c.warmup))
                                                  * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    use_amp = device.type in {"cuda": ("cuda",), "mps": ("cuda", "mps"), "off": ()}[c.amp]

    losses: List[float] = []
    val_accuracy: List[float] = []
    evals: List[Dict[str, float]] = []
    order: List[List[int]] = []      # this epoch's micro batches (lists of indices into train_items)
    start_epoch, start_b = 0, 0
    resume_path = os.path.join(str(out_dir), RESUME) if out_dir else ""
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    resumed = None
    if resume_path and os.path.exists(resume_path):
        resumed = torch.load(resume_path, map_location="cpu", weights_only=False)
        defaults = asdict(TrainConfig())   # a state written before a setting existed ran with its default
        changed = {k: (resumed["config"].get(k, defaults[k]), getattr(c, k)) for k in _RESUME_KEYS
                   if resumed["config"].get(k, defaults[k]) != getattr(c, k)}
        if changed or resumed["train_items"] != len(train_items):
            raise ValueError("%s was written by a different run (%s); delete it to start over"
                             % (resume_path, changed or "different training data"))
        with torch.no_grad():
            for n, p in trainable.items():
                p.copy_(resumed["weights"][n].to(device))
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        losses, val_accuracy, evals = resumed["losses"], resumed["val_accuracy"], resumed["evals"]
        order, start_epoch, start_b = resumed["order"], resumed["epoch"], resumed["b"]
        _set_rng_state(resumed["rng"])
        print("resumed from %s at step %d" % (resume_path, len(losses)), flush=True)
    tracker = Tracker(c.swanlab_project, c.run_name, {"train": asdict(c), "model": decider.cfg.to_dict(),
                           "train_items": len(train_items), "val_items": len(calib), "steps": steps,
                           "eval_items": {name: len(its) for name, its in sets.items()}}, record_dir=str(out_dir or ""),
                      resume=resumed is not None)

    def save_state(epoch: int, b: int) -> None:
        _save_state(resume_path, {
            "config": asdict(c), "train_items": len(train_items), "epoch": epoch, "b": b, "order": order,
            "weights": {n: p.detach().cpu() for n, p in trainable.items()},
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "losses": losses, "val_accuracy": val_accuracy, "evals": evals, "rng": _rng_state()})

    def run_eval() -> Dict[str, float]:
        if evals and evals[-1]["step"] == len(losses):
            return evals[-1]
        row: Dict[str, float] = {}
        for name, its in sets.items():
            logits = _val_logits(model, its, images, tok.pad_token_id, device, c.eval_batch)
            row.update(_set_metrics(name, logits, its))
            if c.monitor_details and out_dir:
                _write_records(os.path.join(str(out_dir), "monitor", "step%d" % len(losses),
                                            name + ".details.jsonl"), monitor_records(logits, its))
        tracker.log(row, step=len(losses))
        print("step %d %s" % (len(losses), {k: round(v, 4) for k, v in row.items()}), flush=True)
        evals.append(dict(row, step=len(losses)))
        return evals[-1]

    if resumed is None:
        run_eval()
        if c.start_check:
            step0 = os.path.join(str(out_dir), "monitor", "step0")
            try:
                check = same_start(step0, c.start_check, c.start_tolerance)
            except ValueError as e:
                with open(os.path.join(str(out_dir), START_MISMATCH), "w") as f:
                    f.write("%s vs %s: %s\n" % (step0, c.start_check, e))
                tracker.finish()
                raise StartMismatch("step 0 does not match %s: %s" % (c.start_check, e)) from e
            with open(os.path.join(str(out_dir), "start_check.json"), "w") as f:
                json.dump(dict(check, baseline=c.start_check), f, indent=1)
            print("step 0 matches %s (largest difference %.2g)" % (c.start_check, check["largest"]), flush=True)
    log = TrainLog(tracker, device, steps, c.log_every)
    for epoch in range(start_epoch, c.epochs):
        if not (resumed and epoch == start_epoch):
            if grouped:
                order = first_plan if epoch == 0 and first_plan else plan_batches(train_items, c.micro_batch)
            else:   # no groups: the order (and the RNG use) of every run before round 13
                flat = [i for bt in order for i in bt] or list(range(len(train_items)))
                random.shuffle(flat)
                order = [flat[j:j + c.micro_batch] for j in range(0, len(flat), c.micro_batch)]
        for b in range(start_b if resumed and epoch == start_epoch else 0, len(order)):
            sigma = c.sigma_start + (c.sigma_end - c.sigma_start) * len(losses) / max(1, steps - 1)
            chunk = [train_items[i] for i in order[b]]
            batch = collate_items([chunk], tok.pad_token_id)
            assert batch is not None
            index = None
            if c.local_views:   # exp50: each distinct picture of the micro batch encoded once
                names = sorted({it["image"] for it in chunk})
                pixels = torch.stack([images(n) for n in names])
                index = torch.tensor([names.index(it["image"]) for it in chunk])
            else:
                pixels = torch.stack([images(it["image"]) for it in chunk])
            grounded = [i for i, it in enumerate(chunk) if "grid" in it] if c.grounding_weight > 0 else []
            with torch.autocast(device.type if use_amp else "cpu", dtype=torch.bfloat16, enabled=use_amp):
                if grounded:
                    logits, hidden = _forward(model, batch, pixels, device, return_hidden=True)
                else:
                    logits = _forward(model, batch, pixels, device, index=index)
                logits = logits.float()
            loss, nll = _rlcd_loss(logits, batch["marker_mask"].to(device), batch["target"].to(device),
                              batch["qtype"].to(device), sigma, c.group_size, c.objective)
            interaction = None
            if inter:   # the same forward's option logits; joins the loss checked finite below, one backward / step
                i_loss, i_j = interaction_loss(logits, [inter[(it["sample_id"], it["qid"])] for it in chunk])
                loss = loss + c.interaction_weight * i_loss
                interaction = (i_loss.item(), i_j.item())
            region = None
            if regions:   # the same forward's attention; checked finite with the loss below, one backward / step
                att = cast(Any, model.condition).last_weights
                tgt = torch.tensor([regions[it["sample_id"]] for it in chunk], device=device)
                rkl = region_kl(att, tgt)
                loss = loss + c.region_weight * rkl
                if not math.isfinite(rkl.item()):
                    raise RuntimeError("step %d: region KL is %r; no update made" % (len(losses), rkl.item()))
                region = (rkl.item(), (att.float() * (tgt > REGION_SMOOTH / tgt.size(1) + 1e-9)).sum(-1).mean().item())
            ground = None
            if grounded:
                idx = torch.tensor(grounded, device=device)
                ids, mpos = batch["input_ids"].to(device)[idx], batch["marker_pos"].to(device)[idx]
                head = cast(GroundingHead, getattr(model, "grounding"))
                logp = head(hidden[idx], hidden.size(1) - ids.size(1), question_mask(ids, mpos))
                target = torch.tensor([chunk[i]["grid"] for i in grounded], device=device)
                ground = (target * (target.log() - logp)).sum(-1).mean()
                loss = loss + c.grounding_weight * ground
            distill = None
            kept = [i for i, it in enumerate(chunk) if it.get("block") in distill_blocks] if teacher is not None else []
            if kept:
                idx = torch.tensor(kept, device=device)
                sub = {k: batch[k].to(device)[idx] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}
                with torch.no_grad():
                    t_logits = _forward(teacher, sub, pixels[kept], device).float()
                mask = sub["marker_mask"]
                kl = _distill_kl(logits[idx], t_logits, mask)
                loss = loss + c.distill_weight * kl.sum() / len(chunk)
                distill = (kl.mean().item(), (t_logits.masked_fill(~mask, -1e4).argmax(-1)
                                              == batch["target"].to(device)[idx].argmax(-1)).float().mean().item())
            hit = (logits.masked_fill(~batch["marker_mask"].to(device), -1e4).argmax(-1)
                   == batch["target"].to(device).argmax(-1)).float().mean()
            if not math.isfinite(loss.item()) or (loss.item() == 0.0 and nll.item() == 0.0 and hit.item() < 1.0):
                # a zero loss is only possible when every answer is right: on 2026-10-10 the Apple
                # GPU started ignoring command buffers mid-run and every result came back as zeros. Stop before the
                # resume state is overwritten.
                raise RuntimeError("step %d: training loss is %r (nll %r): the device is returning invalid results; "
                                   "stopping with the last good resume state kept" % (len(losses), loss.item(), nll.item()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            aux_stats = None
            if aux is not None:   # second forward / backward on the same, not yet updated weights; one step below
                ids, tgt = zip(*(aux_inputs[(it["sample_id"], it["qid"])] for it in chunk))
                with torch.autocast(device.type if use_amp else "cpu", dtype=torch.bfloat16, enabled=use_amp):
                    a_logits = aux_logits(model, aux, ids, pixels, device, c.aux_slots, tok.pad_token_id)
                a_loss, a_right = aux_loss(a_logits, torch.tensor(tgt, device=device))
                if not math.isfinite(a_loss.item()):
                    raise RuntimeError("step %d: number loss is %r; no update made" % (len(losses), a_loss.item()))
                (c.aux_weight * a_loss).backward()
                aux_stats = (a_loss.item(), a_right.item())
            grad_norm = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(nll.item())  # .item() syncs the device, so step times are real
            log.add(len(chunk), loss=loss.item(), nll=nll.item(), accuracy=hit.item(),
                    batch_tokens=float(batch["input_ids"].shape[1]),   # padded length; long batches drive memory
                    grad_norm=grad_norm.item(), sigma=sigma,
                    **({"grounding_kl": ground.item()} if ground is not None else {}),
                    **({"distill_kl": distill[0], "teacher_accuracy": distill[1]} if distill is not None else {}),
                    **({"number_loss": aux_stats[0], "number_accuracy": aux_stats[1]} if aux_stats is not None else {}),
                    **({"region_kl": region[0], "region_mass": region[1]} if region is not None else {}),
                    **({"interaction_loss": interaction[0], "interaction_j": interaction[1]} if interaction is not None else {}))
            means = log.flush(len(losses), epoch + 1, {name: g["lr"] for name, g in zip(group_names, optimizer.param_groups)})
            if means:
                print("epoch %d step %d/%d loss %.4f nll %.4f acc %.3f step_s %.3f%s" % (
                    epoch + 1, len(losses), steps, means["train/loss"], means["train/nll"],
                    means["train/accuracy"], means["perf/step_s"],
                    " distill_kl %.4f teacher_acc %.3f" % (means["train/distill_kl"], means["train/teacher_accuracy"])
                    if "train/distill_kl" in means else
                    " number_loss %.4f number_acc %.3f" % (means["train/number_loss"], means["train/number_accuracy"])
                    if "train/number_loss" in means else
                    " region_kl %.4f region_mass %.3f" % (means["train/region_kl"], means["train/region_mass"])
                    if "train/region_kl" in means else
                    " interaction_loss %.4f interaction_j %.3f" % (means["train/interaction_loss"], means["train/interaction_j"])
                    if "train/interaction_loss" in means else ""), flush=True)
            if c.eval_every and len(losses) % c.eval_every == 0:
                run_eval()
            if resume_path and c.save_every and len(losses) % c.save_every == 0:
                save_state(epoch, b + 1)
        val_accuracy.append(run_eval()["val/accuracy"])

    if regions:
        cast(Any, model.condition).keep_weights = False      # the merged pass and the saved model: decide's path
    model.decision.encoder = peft_encoder.merge_and_unload()
    if hasattr(model, "grounding"):   # S1's head is a training device, not part of the model that answers
        del model.grounding
    model.eval()
    set_logits = {name: _val_logits(model, its, images, tok.pad_token_id, device, c.eval_batch) for name, its in sets.items()}
    temperature, by_options = fit_temperatures(set_logits["val"], calib)
    decider.cfg.temperature, decider.cfg.temperature_by_options = temperature, by_options
    print("temperatures (choice, score, noul):", [round(t, 3) for t in temperature])
    print("temperatures by options:", {b: round(t, 3) for b, t in by_options.items()})
    # With the fitted temperatures, i.e. what decide() will return. val is also the fitting set.
    final = {k: v for name, its in sets.items()
             for k, v in _set_metrics(name, set_logits[name], its, temperature, by_options).items()}
    if c.monitor_details and out_dir:
        for name, its in sets.items():
            _write_records(os.path.join(str(out_dir), "monitor", "final", name + ".details.jsonl"),
                           monitor_records(set_logits[name], its, temperature, by_options))
    # the per-type temperatures alone (calibration before buckets), to see what the buckets add
    final_per_type = {k: v for name, its in sets.items()
                      for k, v in _set_metrics(name, set_logits[name], its, temperature).items()}
    print("final %s" % {k: round(v, 4) for k, v in final.items()}, flush=True)
    tracker.log({"calib/temperature_choice": temperature[QTYPES["choice"]],
                 "calib/temperature_noul": temperature[QTYPES["noul"]],
                 **{"calib/temperature/" + b: t for b, t in by_options.items()},
                 **{"final/" + k: v for k, v in final.items()},
                 **{"final_per_type/" + k: v for k, v in final_per_type.items()}}, step=len(losses))
    tracker.finish()

    decider.model = model.cpu().float()
    decider.save(out_dir)
    if resume_path and os.path.exists(resume_path):
        os.remove(resume_path)
    # "final" on the val set is measured on the data the temperatures were fitted on: not held out
    return {"losses": losses, "val_accuracy": val_accuracy, "evals": evals, "final": final,
            "final_per_type": final_per_type,
            "calibration_fit_set": "val",
            "temperature": temperature, "temperature_by_options": by_options,
            "calibration": {"per_type": bucket_metrics(set_logits["val"], calib, temperature),
                            "by_options": bucket_metrics(set_logits["val"], calib, temperature, by_options)},
            "train_items": len(train_items), "calib_items": len(calib), "objective": c.objective,
            **({"aux": {"weight": c.aux_weight, "slots": c.aux_slots, "head": aux_ident}} if aux is not None else {}),
            **({"region": {"weight": c.region_weight, "file": c.region_file, "smooth": REGION_SMOOTH,
                           "file_sha256": __import__("hashlib").sha256(open(c.region_file, "rb").read()).hexdigest()}}
               if regions else {}),
            **({"interaction": {"weight": c.interaction_weight, "file": c.interaction_file, "formula": INTERACTION_FORMULA,
                                "file_sha256": __import__("hashlib").sha256(open(c.interaction_file, "rb").read()).hexdigest()}}
               if inter else {}),
            **({"local": {"views": c.local_views, "overlap": c.local_overlap, "source": c.local_source,
                          "vision_trainable": sum(p.numel() for p in model.vision.parameters() if p.requires_grad),
                          "groups": {n: sum(p.numel() for p in g["params"]) for n, g in zip(group_names, groups)}}}
               if c.local_views else {})}
