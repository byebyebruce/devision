"""Stage 1: align the projector to the text encoder with image-conditioned masked captions.

Input [CLS] <visual tokens> <caption, a share of its tokens replaced by [MASK]> [SEP]; the loss is
cross-entropy on the masked tokens through ModernBERT-large's MLM head (Apache-2.0, used only here,
not saved). The projector trains; with `lora_r` > 0 ModernBERT also adapts through LoRA (merged
before saving), as vision-language encoders do when they align. Deciding straight from a random
projector stalls at nll = ln2 (docs/experiments/2026-09-30-rlcd-plateau.md); this gives it a dense
signal first.

With `decision_data` (exp33, docs/experiments/2026-10-17-exp33-decision-protect-plan.md) every step also takes
`decision_micro_batch` decision questions (samples.py decision format) and adds `decision_weight` times their gold
cross-entropy over the options (raw logits through the frozen decision head, no RLCD noise) to the caption loss,
before one backward / clip / optimizer step: the old decisions supervise the same projector / LoRA that alignment
moves. Decision questions are drawn from their own seeded order, so the caption and mask random stream is the same
as without them; with no decision data or weight 0 the step is exactly the caption-only one.

With `number_mask_sources` (exp36, docs/experiments/2026-10-17-exp36-number-mask-plan.md) captions of those sources
are supervised only on their numbers: after the usual random mask is drawn (so the random stream, caption choice and
every other row are exactly as without it), such a row is rewritten so that every token of every number span (mass /
speed / distance values of both objects, see `number_spans`) is masked and is the only target; labels, quantity
names and units stay visible. A caption whose expected numbers are missing or cut by max_caption_tokens is refused.
`track_sources` logs, per listed source and for the rest, the supervised tokens and their share of the batch loss.

With `micro_batch` (exp39, docs/experiments/2026-10-17-exp39-lowmem-plan.md) each batch of `batch` rows is drawn,
masked and rewritten exactly as without it, then run in forward / backward passes of `micro_batch` rows: each part's
loss is its masked-token cross-entropy SUM divided by the masked tokens of the whole batch, so the summed gradient is
the whole batch's mean; one zero_grad, one clip, one optimizer and one scheduler step per batch; every part is checked
finite before the step. Evaluation keeps its batches of `batch` rows (captions, masks, rotated pictures) and only
splits the forward pass. Not combined with decision supervision.

The checkpoint is a normal devision checkpoint; `devision-train --init` continues from it.
"""
import math
import random
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.modernbert.modeling_modernbert import ModernBertPredictionHead

from laya.common import collate_items

from .rlcd import _device, _forward, _Images, _items
from .samples import Sample, read_decision_jsonl
from .tracking import Tracker, TrainLog
from ..infer import Decider
from ..model import text_encoder


@dataclass
class AlignConfig:
    epochs: int = 3
    batch: int = 16
    lr: float = 1e-3
    lora_r: int = 0                # > 0: also train a LoRA of this rank on ModernBERT
    lora_alpha: int = 32
    lr_lora: float = 1e-4
    warmup: int = 50
    mask_prob: float = 0.5
    max_caption_tokens: int = 40
    mlm_head: str = "answerdotai/ModernBERT-large"  # "" -> random head, trained too (tiny test models)
    eval_every: int = 500          # steps; also evaluated at the end
    seed: int = 0
    device: str = "auto"
    log_every: int = 10
    swanlab_project: str = ""
    run_name: str = ""
    decision_data: str = ""        # JSONL of decision samples supervised alongside the captions (exp33)
    decision_weight: float = 1.0
    decision_micro_batch: int = 8
    number_mask_sources: str = ""  # comma-separated caption sources supervised only on their numbers (exp36)
    track_sources: str = ""        # comma-separated sources whose supervised tokens / loss share are logged
    micro_batch: int = 0           # > 0: run each batch of `batch` rows in forward passes of this many rows (exp39)


class _MLMHead(nn.Module):
    """ModernBERT's MLM head; the decoder is tied to the encoder's input embeddings."""

    def __init__(self, encoder):
        super().__init__()
        self.head = ModernBertPredictionHead(encoder.config)
        self.embeddings = encoder.get_input_embeddings()
        self.bias = nn.Parameter(torch.zeros(encoder.config.vocab_size))

    def load_pretrained(self, repo: str) -> None:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        with safe_open(hf_hub_download(repo, "model.safetensors"), "pt") as f:
            self.head.dense.weight.data.copy_(f.get_tensor("head.dense.weight"))
            self.head.norm.weight.data.copy_(f.get_tensor("head.norm.weight"))
            self.bias.data.copy_(f.get_tensor("decoder.bias"))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h) @ self.embeddings.weight.T + self.bias


def _masked_batch(tok, captions: Sequence[str], c: AlignConfig, rng: random.Random):
    """(input_ids, attention_mask, labels) for [CLS] caption [SEP]; labels are -100 except at masks."""
    seqs, labels = [], []
    for cap in captions:
        ids = tok(cap, add_special_tokens=False)["input_ids"][:c.max_caption_tokens] or [tok.unk_token_id]
        mask = [rng.random() < c.mask_prob for _ in ids]
        if not any(mask):
            mask[rng.randrange(len(ids))] = True
        seqs.append([tok.cls_token_id] + [tok.mask_token_id if m else t for t, m in zip(ids, mask)]
                    + [tok.sep_token_id])
        labels.append([-100] + [t if m else -100 for t, m in zip(ids, mask)] + [-100])
    n, L = len(seqs), max(map(len, seqs))
    input_ids = torch.full((n, L), tok.pad_token_id, dtype=torch.long)
    attention = torch.zeros((n, L), dtype=torch.long)
    target = torch.full((n, L), -100, dtype=torch.long)
    for i, (s, lab) in enumerate(zip(seqs, labels)):
        input_ids[i, :len(s)], attention[i, :len(s)], target[i, :len(lab)] = (
            torch.tensor(s), 1, torch.tensor(lab))
    return input_ids, attention, target


NUMBER_PATTERNS = [("mass", re.compile(r"mass of each particle ([\d,]+(?:\.\d+)?) u")),
                   ("speed", re.compile(r"average particle speed ([\d,]+(?:\.\d+)?) m/s")),
                   ("distance", re.compile(r"(\d[\d,]*(?:\.\d+)?) (?:in|cm|mm) apart"))]
NUMBERS_EXPECTED = {"Sample": 4, "Pair": 2}   # a real-layout caption: 2 samples x (mass, speed), 2 pairs x distance


def number_spans(caption: str) -> List[Dict[str, Any]]:
    """[{field, start, end, text}] of the numbers of a real-layout caption -- the digits (with "," / ".") after
    "mass of each particle", "average particle speed" and before "<unit> apart"; never a "Pair 1" / "Sample A"
    label or a unit (the rule scripts/exp35/mlm_numbers.py read with)."""
    out = [{"field": f, "start": m.start(1), "end": m.end(1), "text": m.group(1)}
           for f, pat in NUMBER_PATTERNS for m in pat.finditer(caption)]
    return sorted(out, key=lambda x: x["start"])


def _number_rows(tok, captions: Sequence[str], rows: Sequence[int], input_ids, target, c: AlignConfig) -> None:
    """Rewrites rows `rows` of a _masked_batch batch in place: every token overlapping a number span masked and the
    only target, every other token visible and not a target."""
    for i in rows:
        cap = captions[i]
        spans = number_spans(cap)
        kind = "Sample" if cap.startswith("Sample") else "Pair" if cap.startswith("Pair") else None
        if kind is None or len(spans) != NUMBERS_EXPECTED[kind]:
            raise ValueError("number mask: %r does not hold the expected numbers" % cap)
        enc = tok(cap, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        hit = [any(a < s["end"] and b > s["start"] and b > a for s in spans) for a, b in offs]
        if len(ids) > c.max_caption_tokens:
            raise ValueError("number mask: %r is longer than max_caption_tokens" % cap)
        for s in spans:
            covered = "".join(cap[a:b] for (a, b), h in zip(offs, hit) if h and a < s["end"] and b > s["start"])
            if covered.strip() != s["text"]:
                raise ValueError("number mask: tokens of %r cover %r" % (s["text"], covered))
        for t, (tid, h) in enumerate(zip(ids, hit)):
            input_ids[i, 1 + t] = tok.mask_token_id if h else tid
            target[i, 1 + t] = tid if h else -100


def _full_picture(model, pixels: torch.Tensor) -> torch.Tensor:
    """The whole picture of a batch: as is, or -- for a model with fixed local views, whose pixels are packed by
    image.local_pixels -- its full-picture channels, the same picture model.visual_inputs projects into the visual
    tokens (PANEL-LEARN-PREP-01; the views only feed the conditioning, which caption alignment does not use)."""
    n = getattr(model, "local_views", 0)
    if not n:
        return pixels
    from ..image import unpack_local
    return unpack_local(pixels, n)[0]


def _mlm_logits(model, head, input_ids, attention, pixels, device) -> torch.Tensor:
    """Vocabulary logits at text positions 1.. (they line up with labels[:, 1:])."""
    with torch.no_grad():
        patches = model.vision_patches(_full_picture(model, pixels.to(device)))
    visual = model.projector(patches)
    h, _ = model.encode(input_ids.to(device), attention.to(device), visual)
    return head(h[:, visual.size(1) + 1:]).float()


def _logits_in_parts(model, head, input_ids, attention, pixels, device, micro: int) -> torch.Tensor:
    """_mlm_logits of a whole batch, run `micro` rows at a time when micro > 0 (same rows, same padding)."""
    if not micro or micro >= input_ids.size(0):
        return _mlm_logits(model, head, input_ids, attention, pixels, device)
    return torch.cat([_mlm_logits(model, head, input_ids[i:i + micro], attention[i:i + micro], pixels[i:i + micro], device)
                      for i in range(0, input_ids.size(0), micro)])


def _evaluate(model, head, tok, samples, images, c: AlignConfig, device) -> Dict[str, float]:
    """Masked-token accuracy / nll on held-out images, with their own captions and with captions
    of other images ("shuffled"). Real beating shuffled means the projector carries image content."""
    out = {}
    was_training = model.training
    model.eval()
    for mode in ("real", "shuffled"):
        rng = random.Random(1)  # same captions and masks for both modes
        hits = total = 0
        nll = 0.0
        with torch.no_grad():
            for b in range(0, len(samples), c.batch):
                chunk = samples[b:b + c.batch]
                caps = [rng.choice(s["captions"]) for s in chunk]
                pix_src = chunk if mode == "real" else chunk[1:] + chunk[:1]
                input_ids, attention, target = _masked_batch(tok, caps, c, rng)
                logits = _logits_in_parts(model, head, input_ids, attention,
                                          torch.stack([images(s["image"]) for s in pix_src]), device, c.micro_batch)
                y = target[:, 1:].to(device)
                m = y != -100
                nll += F.cross_entropy(logits[m], y[m], reduction="sum").item()
                hits += (logits[m].argmax(-1) == y[m]).sum().item()
                total += int(m.sum().item())
        out["val/mlm_acc_" + mode] = hits / max(1, total)
        out["val/mlm_nll_" + mode] = nll / max(1, total)
    out["val/mlm_acc_gap"] = out["val/mlm_acc_real"] - out["val/mlm_acc_shuffled"]  # what the image adds
    model.train(was_training)
    return out


def _gold_ce(logits: torch.Tensor, mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gold cross-entropy over each question's options (the CE term of rlcd._rlcd_loss), mean over questions."""
    return -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()


def _micro_step(model, head, input_ids, attention, target, chunk, images, device, c: AlignConfig, tracked, optimizer,
                params, step: int):
    """One optimizer step over a whole batch run in parts of c.micro_batch rows (see the module docstring); returns
    (mean masked-token loss, accuracy, tracked-source stats, grad norm)."""
    y_all = target[:, 1:]
    n_tok = int((y_all != -100).sum().item())
    pixels = torch.stack([images(s["image"]) for s in chunk])
    optimizer.zero_grad(set_to_none=True)
    loss_sum, hits = 0.0, 0
    src_tok: Dict[str, float] = {}
    src_ce: Dict[str, float] = {}
    for r0 in range(0, len(chunk), c.micro_batch):
        rows = slice(r0, r0 + c.micro_batch)
        logits = _mlm_logits(model, head, input_ids[rows], attention[rows], pixels[rows], device)
        y = y_all[rows].to(device)
        m = y != -100
        part = F.cross_entropy(logits[m], y[m], reduction="sum") / max(1, n_tok)
        if not math.isfinite(part.item()):   # before the step: a broken part never reaches the weights
            raise RuntimeError("step %d: loss is %r: the device is returning invalid results" % (step, part.item()))
        part.backward()
        loss_sum += part.item()
        hits += int((logits[m].argmax(-1) == y[m]).sum().item())
        if tracked:
            with torch.no_grad():
                per = F.cross_entropy(logits.transpose(1, 2), y.clamp(min=0), reduction="none") * m
                for i, s in enumerate(chunk[rows]):
                    name = s.get("source") if s.get("source") in tracked else "rest"
                    src_tok[name] = src_tok.get(name, 0.0) + float(m[i].sum().item())
                    src_ce[name] = src_ce.get(name, 0.0) + float(per[i].sum().item())
        del logits, part
    grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    extra: Dict[str, float] = {}
    if tracked:
        total = sum(src_ce.values()) or 1.0
        for name in tracked + ["rest"]:
            key = name.replace("-", "_")
            extra["tokens_" + key] = src_tok.get(name, 0.0)
            extra["loss_share_" + key] = src_ce.get(name, 0.0) / total
    return loss_sum, hits / max(1, n_tok), extra, float(grad_norm.item())


def align(decider: Decider, samples: Sequence[Sample], data_root, out_dir,
          config: Optional[AlignConfig] = None,
          val_samples: Optional[Sequence[Sample]] = None) -> Dict[str, Any]:
    """Train `decider`'s projector on caption `samples` (see samples.py) and save to `out_dir`.

    Returns {"losses": per-step masked-token loss, "val": one metrics dict per evaluation}.
    """
    c = config or AlignConfig()
    device = _device(c.device)
    rng = random.Random(c.seed)
    torch.manual_seed(c.seed)
    model, tok = decider.model, decider.tok

    head = _MLMHead(text_encoder(model.decision))
    if c.mlm_head:
        head.load_pretrained(c.mlm_head)
    model.requires_grad_(False)
    model.projector.requires_grad_(True)
    params: List[nn.Parameter] = list(model.projector.parameters())
    lora_params: List[nn.Parameter] = []
    peft_encoder = None
    if c.lora_r:
        from peft import LoraConfig, get_peft_model

        from .rlcd import LORA_TARGETS

        peft_encoder = get_peft_model(text_encoder(model.decision), LoraConfig(
            r=c.lora_r, lora_alpha=c.lora_alpha, lora_dropout=0.0, target_modules=LORA_TARGETS))
        model.decision.encoder = peft_encoder
        lora_params = [p for p in peft_encoder.parameters() if p.requires_grad]
    if not c.mlm_head:
        head.head.requires_grad_(True)
        head.bias.requires_grad_(True)
        params += list(head.head.parameters()) + [head.bias]
    else:
        head.requires_grad_(False)
    # eval(): only the projector (and LoRA) train, so dropout in the frozen parts only adds noise.
    model.to(device).eval()
    head.to(device)

    images = _Images(data_root, decider)
    train_samples = [s for s in samples if s["captions"]]
    val = [s for s in (val_samples or []) if s["captions"]]
    steps = c.epochs * max(1, len(train_samples) // c.batch)
    decide = bool(c.decision_data) and c.decision_weight != 0
    dec_items = _items(decider, read_decision_jsonl(c.decision_data)) if decide else []
    dec_rng = random.Random(c.seed + 1)            # its own stream: the caption / mask draws stay unchanged
    dec_order: List[int] = []
    if c.micro_batch and decide:
        raise ValueError("micro_batch is not combined with decision supervision")
    if c.micro_batch < 0 or (c.micro_batch and c.batch % c.micro_batch):
        raise ValueError("micro_batch must divide batch")
    number_sources = {x for x in c.number_mask_sources.split(",") if x}
    tracked = [x for x in c.track_sources.split(",") if x]

    def decision_batch() -> List[Dict[str, Any]]:
        out = []
        while len(out) < c.decision_micro_batch:
            if not dec_order:
                dec_order.extend(dec_rng.sample(range(len(dec_items)), len(dec_items)))
            out.append(dec_items[dec_order.pop()])
        return out

    optimizer = torch.optim.AdamW([{"params": params, "lr": c.lr}]
                                  + ([{"params": lora_params, "lr": c.lr_lora}] if lora_params else []),
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, c.warmup))
                                                  * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    tracker = Tracker(c.swanlab_project, c.run_name, {"align": asdict(c), "model": decider.cfg.to_dict(),
                           "train_images": len(train_samples), "val_images": len(val), "steps": steps,
                           "decision_items": len(dec_items)},
                      record_dir=str(out_dir or ""))
    losses: List[float] = []
    evals: List[Dict[str, float]] = []

    def run_eval():
        if val:
            evals.append(dict(_evaluate(model, head, tok, val, images, c, device), step=len(losses)))
            tracker.log({k: v for k, v in evals[-1].items() if k != "step"}, step=len(losses))
            print("step %d val %s" % (len(losses), {k: round(v, 4) for k, v in evals[-1].items()
                                                    if k != "step"}), flush=True)

    run_eval()
    log = TrainLog(tracker, device, steps, c.log_every)
    for epoch in range(c.epochs):
        order = list(train_samples)
        rng.shuffle(order)
        for b in range(0, len(order) - c.batch + 1, c.batch):
            chunk = order[b:b + c.batch]
            caps = [rng.choice(s["captions"]) for s in chunk]
            input_ids, attention, target = _masked_batch(tok, caps, c, rng)
            if number_sources:   # after the random draw, so the random stream is the same as without it
                _number_rows(tok, caps, [i for i, s in enumerate(chunk) if s.get("source") in number_sources],
                             input_ids, target, c)
            if c.micro_batch:
                mlm_value, acc_value, extra, grad_norm = _micro_step(model, head, input_ids, attention, target, chunk,
                                                                     images, device, c, tracked, optimizer,
                                                                     params + lora_params, len(losses))
                scheduler.step()
                losses.append(mlm_value)
                log.add(len(chunk), loss=losses[-1], accuracy=acc_value, grad_norm=grad_norm, **extra)
                means = log.flush(len(losses), epoch + 1, {"projector": optimizer.param_groups[0]["lr"],
                                                           **({"lora": optimizer.param_groups[1]["lr"]}
                                                              if lora_params else {})})
                if means:
                    print("epoch %d step %d/%d loss %.4f acc %.3f" % (epoch + 1, len(losses), steps,
                                                                     means["train/loss"], means["train/accuracy"]), flush=True)
                if c.eval_every and len(losses) % c.eval_every == 0:
                    run_eval()
                continue
            logits = _mlm_logits(model, head, input_ids, attention,
                                    torch.stack([images(s["image"]) for s in chunk]), device)
            y = target[:, 1:].to(device)
            m = y != -100
            loss = mlm = F.cross_entropy(logits[m], y[m])
            extra: Dict[str, float] = {}
            if tracked:
                with torch.no_grad():
                    per = F.cross_entropy(logits.transpose(1, 2), y.clamp(min=0), reduction="none") * m
                    total = per.sum()
                    for name in tracked + ["rest"]:
                        rows = [i for i, s in enumerate(chunk) if (s.get("source") == name if name != "rest" else s.get("source") not in tracked)]
                        key = name.replace("-", "_")
                        extra["tokens_" + key] = float(m[rows].sum().item()) if rows else 0.0
                        extra["loss_share_" + key] = float((per[rows].sum() / total).item()) if rows else 0.0
            if decide:
                chunk_d = decision_batch()
                batch = collate_items([chunk_d], tok.pad_token_id)
                assert batch is not None
                d_logits = _forward(model, batch, torch.stack([images(it["image"]) for it in chunk_d]), device).float()
                d_mask, d_target = batch["marker_mask"].to(device), batch["target"].to(device)
                dec = _gold_ce(d_logits, d_mask, d_target)
                loss = mlm + c.decision_weight * dec
                extra = {"decision_ce": dec.item(), "decision_accuracy": (d_logits.masked_fill(~d_mask, -1e4).argmax(-1)
                                                                         == d_target.argmax(-1)).float().mean().item()}
            if not math.isfinite(loss.item()):   # before any update: nothing is saved from a broken run (critic EXP38-R1-03)
                raise RuntimeError("step %d: loss is %r: the device is returning invalid results" % (len(losses), loss.item()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params + lora_params, 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(mlm.item())
            log.add(len(chunk), loss=losses[-1], accuracy=(logits[m].argmax(-1) == y[m]).float().mean().item(),
                    grad_norm=grad_norm.item(), **extra)
            means = log.flush(len(losses), epoch + 1, {"projector": optimizer.param_groups[0]["lr"],
                                                       **({"lora": optimizer.param_groups[1]["lr"]}
                                                          if lora_params else {})})
            if means:
                print("epoch %d step %d/%d loss %.4f acc %.3f" % (epoch + 1, len(losses), steps,
                                                                 means["train/loss"], means["train/accuracy"]), flush=True)
            if c.eval_every and len(losses) % c.eval_every == 0:
                run_eval()
    if not evals or evals[-1]["step"] != len(losses):
        run_eval()
    tracker.finish()

    if peft_encoder is not None:
        model.decision.encoder = peft_encoder.merge_and_unload()
    decider.model = model.cpu().float()
    decider.save(out_dir)
    return {"losses": losses, "val": evals, "train_images": len(train_samples), "val_images": len(val)}
