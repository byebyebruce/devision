"""RLCD fine-tuning of the vision decision model, following Laya's single-device script
(research/scripts/finetune_single_device.py in NandhaKishorM/laya, Apache-2.0).

Frozen: the SigLIP vision tower. Trained: projector, Laya's decision head, LoRA on ModernBERT.
LoRA is merged before saving, so checkpoints load without peft.
"""
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from laya.common import QTYPES, clamp_temperature, collate_items, proper_reward
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image

from .data import Sample
from ..model import Decider, image_tensor, question_item, text_encoder

LORA_TARGETS = ["Wqkv", "Wo", "Wi"]  # ModernBERT attention + MLP projections


@dataclass
class TrainConfig:
    epochs: int = 4
    micro_batch: int = 8
    group_size: int = 4          # RLCD noisy-logit samples per item
    sigma_start: float = 0.4
    sigma_end: float = 0.1
    lr_new: float = 1e-3         # projector (randomly initialised)
    lr_head: float = 1e-4        # Laya decision head
    lr_lora: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    calib_fraction: float = 0.1  # held out to fit temperatures when no val set is given
    seed: int = 0
    device: str = "auto"
    log_every: int = 10


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _items(decider: Decider, samples: Sequence[Sample]) -> List[Dict[str, Any]]:
    items = []
    for s in samples:
        for qid, q in s["questions"].items():
            it = question_item(decider.tok, decider.cfg, "", q)
            probs = s["gold"][qid]["probabilities"]
            keys = ["false", "true"] if q["type"] == "noul" else list(q["criteria"])
            target = [float(probs.get(k, 0.0)) for k in keys]
            total = sum(target)
            it["target"] = [t / total for t in target] if total > 0 else [1 / len(keys)] * len(keys)
            it["image"] = s["image"]
            items.append(it)
    return items


class _Images:
    """Pixel tensors by relative path, decoded once."""

    def __init__(self, root, decider: Decider):
        self.root, self.decider, self.cache = str(root), decider, {}

    def __call__(self, rel: str) -> torch.Tensor:
        if rel not in self.cache:
            with Image.open(os.path.join(self.root, rel)) as img:
                self.cache[rel] = image_tensor(img, self.decider.cfg)
        return self.cache[rel]


def _forward(model, batch, pixels, device):
    with torch.no_grad():
        patches = model.vision(pixel_values=pixels.to(device)).last_hidden_state
    visual = model.projector(patches)
    return model(*(batch[k].to(device) for k in
                   ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")), visual=visual)


def _rlcd_loss(logits, mask, target, qtype, sigma, group_size):
    k = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((group_size,) + logits.shape, device=logits.device) * sigma * mask
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


def _val_logits(model, items, images, pad_id, device):
    """Temperature-free option logits for each item, in eval mode."""
    was_training = model.training
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(items), 32):
            chunk = items[b:b + 32]
            batch = collate_items([chunk], pad_id)
            assert batch is not None
            logits = _forward(model, batch, torch.stack([images(it["image"]) for it in chunk]), device)
            out.extend(row[:len(it["markers"])].tolist() for row, it in zip(logits.float().cpu(), chunk))
    model.train(was_training)
    model.vision.eval()
    return out


def _accuracy(logits, items) -> float:
    hits = [max(range(len(z)), key=z.__getitem__) == max(range(len(it["target"])), key=it["target"].__getitem__)
            for z, it in zip(logits, items)]
    return sum(hits) / len(hits) if hits else float("nan")


def train(decider: Decider, samples: Sequence[Sample], data_root, out_dir,
          config: Optional[TrainConfig] = None,
          val_samples: Optional[Sequence[Sample]] = None) -> Dict[str, Any]:
    """Fine-tune `decider` in place on `samples`, fit temperatures on `val_samples`, save to `out_dir`.

    Without `val_samples`, `calib_fraction` of the training items are held out instead.
    Returns {"losses": per-step NLL against the gold distribution (the RLCD policy term is too
    noisy to monitor), "val_accuracy": per epoch, "temperature": [choice, score, noul],
    "train_items", "calib_items"}.
    """
    c = config or TrainConfig()
    device = _device(c.device)
    random.seed(c.seed)
    torch.manual_seed(c.seed)
    model, tok = decider.model, decider.tok

    model.vision.requires_grad_(False)
    peft_encoder = get_peft_model(text_encoder(model.decision), LoraConfig(
        r=c.lora_r, lora_alpha=c.lora_alpha, lora_dropout=0.0, target_modules=LORA_TARGETS))
    model.decision.encoder = peft_encoder
    model.to(device).train()
    model.vision.eval()

    images = _Images(data_root, decider)
    items = _items(decider, samples)
    if val_samples:
        calib, train_items = _items(decider, val_samples), items
    else:
        order = list(range(len(items)))
        random.shuffle(order)
        n_calib = min(400, int(len(items) * c.calib_fraction))
        calib = [items[i] for i in order[:n_calib]]
        train_items = [items[i] for i in order[n_calib:]]

    groups = [
        {"params": list(model.projector.parameters()), "lr": c.lr_new},
        {"params": [p for n, p in model.decision.named_parameters()
                    if not n.startswith("encoder.") and p.requires_grad], "lr": c.lr_head},
        {"params": [p for p in peft_encoder.parameters() if p.requires_grad], "lr": c.lr_lora},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = c.epochs * max(1, -(-len(train_items) // c.micro_batch))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)
    use_amp = device.type == "cuda"

    losses: List[float] = []
    val_accuracy: List[float] = []
    for epoch in range(c.epochs):
        random.shuffle(train_items)
        sigma = c.sigma_start + (c.sigma_end - c.sigma_start) * epoch / max(1, c.epochs - 1)
        for b in range(0, len(train_items), c.micro_batch):
            chunk = train_items[b:b + c.micro_batch]
            batch = collate_items([chunk], tok.pad_token_id)
            assert batch is not None
            pixels = torch.stack([images(it["image"]) for it in chunk])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = _forward(model, batch, pixels, device).float()
            loss, nll = _rlcd_loss(logits, batch["marker_mask"].to(device), batch["target"].to(device),
                              batch["qtype"].to(device), sigma, c.group_size)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(nll.item())
            if len(losses) % c.log_every == 0:
                print("epoch %d step %d loss %.4f nll %.4f" % (epoch + 1, len(losses), loss.item(),
                                                              nll.item()), flush=True)
        val_accuracy.append(_accuracy(_val_logits(model, calib, images, tok.pad_token_id, device), calib))
        print("epoch %d val accuracy %.3f (%d items)" % (epoch + 1, val_accuracy[-1], len(calib)), flush=True)

    model.decision.encoder = peft_encoder.merge_and_unload()
    model.eval()
    rows: Dict[int, list] = {t: [] for t in QTYPES.values()}
    for z, it in zip(_val_logits(model, calib, images, tok.pad_token_id, device), calib):
        rows[it["qtype"]].append((z, it["target"]))
    decider.cfg.temperature = [_fit_temperature(rows[t]) for t in range(3)]
    print("temperatures (choice, score, noul):", [round(t, 3) for t in decider.cfg.temperature])

    decider.model = model.cpu().float()
    decider.save(out_dir)
    return {"losses": losses, "val_accuracy": val_accuracy, "temperature": decider.cfg.temperature,
            "train_items": len(train_items), "calib_items": len(calib)}
