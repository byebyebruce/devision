"""RLCD fine-tuning of the vision decision model, following Laya's single-device script
(research/scripts/finetune_single_device.py in NandhaKishorM/laya, Apache-2.0).

Frozen: the SigLIP vision tower. Trained: projector, Laya's decision head, LoRA on ModernBERT.
LoRA is merged before saving, so checkpoints load without peft.
"""
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from laya.common import QTYPES, clamp_temperature, collate_items, ece_score, proper_reward
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image

from .samples import Sample
from .tracking import Tracker, TrainLog
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
    swanlab_project: str = ""    # "" disables SwanLab; the CLI defaults to "devision"
    run_name: str = ""


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
            it["source"] = s.get("source", "")
            items.append(it)
    return items


class _Images:
    """Pixel tensor for a relative path. Decoded on every use: a JPEG decode + letterbox costs a few
    ms next to a ~1 s training step, while caching float tensors costs 0.8 MB per image."""

    def __init__(self, root, decider: Decider):
        self.root, self.decider = str(root), decider

    def __call__(self, rel: str) -> torch.Tensor:
        with Image.open(os.path.join(self.root, rel)) as img:
            return image_tensor(img, self.decider.cfg)


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


def _set_metrics(name: str, logits, items, temperature: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """`_metrics` as "<name>/<metric>", plus "<name>/<source>/<metric>" when the set mixes sources."""
    out = {"%s/%s" % (name, k): v for k, v in _metrics(logits, items, temperature).items()}
    sources = sorted({it["source"] for it in items})
    if len(sources) > 1:
        for src in sources:
            idx = [i for i, it in enumerate(items) if it["source"] == src]
            out.update({"%s/%s/%s" % (name, src, k): v for k, v in
                        _metrics([logits[i] for i in idx], [items[i] for i in idx], temperature).items()})
        # every source weighs the same: a big source must not decide which checkpoint looks best
        out["%s/macro_accuracy" % name] = sum(out["%s/%s/accuracy" % (name, src)] for src in sources) / len(sources)
    return out


def _metrics(logits, items, temperature: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Accuracy (all / noul / choice), NLL against the gold distribution and ECE on max
    probability, with logits divided by the per-type temperature as decide() does (1 if none)."""
    nll, conf, hits = [], [], []
    for z, it in zip(logits, items):
        t = temperature[it["qtype"]] if temperature else 1.0
        logp = torch.log_softmax(torch.tensor(z) / t, -1)
        nll.append(-(torch.tensor(it["target"]) * logp).sum().item())
        conf.append(logp.max().exp().item())
        hits.append(float(logp.argmax().item() == max(range(len(it["target"])), key=it["target"].__getitem__)))
    out = {"accuracy": _accuracy(logits, items), "accuracy_noul": _accuracy(logits, items, QTYPES["noul"]),
           "accuracy_choice": _accuracy(logits, items, QTYPES["choice"]),
           "nll": sum(nll) / len(nll) if nll else float("nan"), "ece": ece_score(np.array(conf), np.array(hits))}
    return {k: v for k, v in out.items() if v == v}  # drop NaN (a type the set has no questions of)


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
    "final": {"<set>/<metric>": ...} with temperatures applied, "temperature": [choice, score, noul],
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
    n_layers = text_encoder(model.decision).config.num_hidden_layers
    top = tuple("layers.%d." % i for i in range(n_layers - c.unfreeze_top, n_layers))
    unfrozen = [p for n, p in peft_encoder.named_parameters() if "lora_" not in n and any(t in n for t in top)]
    for p in unfrozen:
        p.requires_grad_(True)
    model.to(device).train()
    model.vision.eval()

    images = _Images(data_root, decider)
    items = _items(decider, samples)
    if val_samples:
        calib, train_items = _items(decider, val_samples), items
    else:
        calib, train_items = holdout_by_image(items, min(400, int(len(items) * c.calib_fraction)))

    sets = {"val": calib, **{name: _items(decider, ss) for name, ss in (eval_sets or {}).items()}}

    groups = [
        {"params": list(model.projector.parameters()), "lr": c.lr_new},
        {"params": [p for n, p in model.decision.named_parameters()
                    if not n.startswith("encoder.") and p.requires_grad], "lr": c.lr_head},
        {"params": [p for n, p in peft_encoder.named_parameters() if "lora_" in n], "lr": c.lr_lora},
    ] + ([{"params": unfrozen, "lr": c.lr_encoder}] if unfrozen else [])
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = c.epochs * max(1, -(-len(train_items) // c.micro_batch))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, c.warmup))
                                                  * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    use_amp = device.type == "cuda"

    losses: List[float] = []
    val_accuracy: List[float] = []
    evals: List[Dict[str, float]] = []
    tracker = Tracker(c.swanlab_project, c.run_name, {"train": asdict(c), "model": decider.cfg.to_dict(),
                           "train_items": len(train_items), "val_items": len(calib), "steps": steps,
                           "eval_items": {name: len(its) for name, its in sets.items()}})

    def run_eval() -> Dict[str, float]:
        if evals and evals[-1]["step"] == len(losses):
            return evals[-1]
        row: Dict[str, float] = {}
        for name, its in sets.items():
            row.update(_set_metrics(name, _val_logits(model, its, images, tok.pad_token_id, device), its))
        tracker.log(row, step=len(losses))
        print("step %d %s" % (len(losses), {k: round(v, 4) for k, v in row.items()}), flush=True)
        evals.append(dict(row, step=len(losses)))
        return evals[-1]

    run_eval()
    log = TrainLog(tracker, device, steps, c.log_every)
    for epoch in range(c.epochs):
        random.shuffle(train_items)
        for b in range(0, len(train_items), c.micro_batch):
            sigma = c.sigma_start + (c.sigma_end - c.sigma_start) * len(losses) / max(1, steps - 1)
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
            grad_norm = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(nll.item())  # .item() syncs the device, so step times are real
            hit = (logits.masked_fill(~batch["marker_mask"].to(device), -1e4).argmax(-1)
                   == batch["target"].to(device).argmax(-1)).float().mean()
            log.add(len(chunk), loss=loss.item(), nll=nll.item(), accuracy=hit.item(),
                    grad_norm=grad_norm.item(), sigma=sigma)
            means = log.flush(len(losses), epoch + 1, {"projector": optimizer.param_groups[0]["lr"],
                                                       "head": optimizer.param_groups[1]["lr"],
                                                       "lora": optimizer.param_groups[2]["lr"],
                                                       **({"encoder": optimizer.param_groups[3]["lr"]}
                                                          if len(optimizer.param_groups) > 3 else {})})
            if means:
                print("epoch %d step %d/%d loss %.4f nll %.4f acc %.3f" % (
                    epoch + 1, len(losses), steps, means["train/loss"], means["train/nll"],
                    means["train/accuracy"]), flush=True)
            if c.eval_every and len(losses) % c.eval_every == 0:
                run_eval()
        val_accuracy.append(run_eval()["val/accuracy"])

    model.decision.encoder = peft_encoder.merge_and_unload()
    model.eval()
    set_logits = {name: _val_logits(model, its, images, tok.pad_token_id, device) for name, its in sets.items()}
    rows: Dict[int, list] = {t: [] for t in QTYPES.values()}
    for z, it in zip(set_logits["val"], calib):
        rows[it["qtype"]].append((z, it["target"]))
    decider.cfg.temperature = [_fit_temperature(rows[t]) for t in range(3)]
    print("temperatures (choice, score, noul):", [round(t, 3) for t in decider.cfg.temperature])
    # With the fitted temperatures, i.e. what decide() will return. val is also the fitting set.
    final = {k: v for name, its in sets.items()
             for k, v in _set_metrics(name, set_logits[name], its, decider.cfg.temperature).items()}
    print("final %s" % {k: round(v, 4) for k, v in final.items()}, flush=True)
    tracker.log({"calib/temperature_choice": decider.cfg.temperature[QTYPES["choice"]],
                 "calib/temperature_noul": decider.cfg.temperature[QTYPES["noul"]],
                 **{"final/" + k: v for k, v in final.items()}}, step=len(losses))
    tracker.finish()

    decider.model = model.cpu().float()
    decider.save(out_dir)
    # "final" on the val set is measured on the data the temperatures were fitted on: not held out
    return {"losses": losses, "val_accuracy": val_accuracy, "evals": evals, "final": final,
            "calibration_fit_set": "val",
            "temperature": decider.cfg.temperature,
            "train_items": len(train_items), "calib_items": len(calib)}
