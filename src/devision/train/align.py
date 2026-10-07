"""Stage 1: align the projector to the text encoder with image-conditioned masked captions.

Input [CLS] <visual tokens> <caption, a share of its tokens replaced by [MASK]> [SEP]; the loss is
cross-entropy on the masked tokens through ModernBERT-large's MLM head (Apache-2.0, used only here,
not saved). The projector trains; with `lora_r` > 0 ModernBERT also adapts through LoRA (merged
before saving), as vision-language encoders do when they align. Deciding straight from a random
projector stalls at nll = ln2 (docs/experiments/2026-09-30-rlcd-plateau.md); this gives it a dense
signal first.

The checkpoint is a normal devision checkpoint; `devision-train --init` continues from it.
"""
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.modernbert.modeling_modernbert import ModernBertPredictionHead

from .rlcd import _device, _Images
from .samples import Sample
from .tracking import Tracker, TrainLog
from ..decider import Decider
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


def _mlm_logits(model, head, input_ids, attention, pixels, device) -> torch.Tensor:
    """Vocabulary logits at text positions 1.. (they line up with labels[:, 1:])."""
    with torch.no_grad():
        patches = model.vision(pixel_values=pixels.to(device)).last_hidden_state
    visual = model.projector(patches)
    h, _ = model.encode(input_ids.to(device), attention.to(device), visual)
    return head(h[:, visual.size(1) + 1:]).float()


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
                logits = _mlm_logits(model, head, input_ids, attention,
                                        torch.stack([images(s["image"]) for s in pix_src]), device)
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
    optimizer = torch.optim.AdamW([{"params": params, "lr": c.lr}]
                                  + ([{"params": lora_params, "lr": c.lr_lora}] if lora_params else []),
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, c.warmup))
                                                  * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    tracker = Tracker(c.swanlab_project, c.run_name, {"align": asdict(c), "model": decider.cfg.to_dict(),
                           "train_images": len(train_samples), "val_images": len(val), "steps": steps},
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
            input_ids, attention, target = _masked_batch(tok, [rng.choice(s["captions"]) for s in chunk], c, rng)
            logits = _mlm_logits(model, head, input_ids, attention,
                                    torch.stack([images(s["image"]) for s in chunk]), device)
            y = target[:, 1:].to(device)
            m = y != -100
            loss = F.cross_entropy(logits[m], y[m])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params + lora_params, 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            log.add(len(chunk), loss=losses[-1], accuracy=(logits[m].argmax(-1) == y[m]).float().mean().item(),
                    grad_norm=grad_norm.item())
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
