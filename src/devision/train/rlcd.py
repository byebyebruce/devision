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
import math
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from laya.common import QTYPES, clamp_temperature, collate_items, ece_score, proper_reward
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image

from .samples import Sample
from .tracking import Tracker, TrainLog
from ..model import Decider, ModelConfig, image_tensor, option_bucket, pick_temperature, question_item, text_encoder

LORA_TARGETS = ["Wqkv", "Wo", "Wi"]  # ModernBERT attention + MLP projections
MIN_TYPE_ITEMS = 10      # fewer val questions of a type: its temperature is not fitted
MIN_BUCKET_ITEMS = 50    # fewer val questions in a bucket: no bucket temperature, decide() uses the type's


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
    save_every: int = 1000       # steps between resume states (<out>/resume.pt); 0: none
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
            # text that comes with the image (e.g. a ScienceQA hint), as decide() gets it from the state
            it = question_item(decider.tok, decider.cfg, s.get("state_text", ""), q)
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


def calibrate(decider: Decider, samples: Sequence[Sample], data_root, device: str = "cpu",
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
                "warmup", "lora_r", "lora_alpha", "unfreeze_top", "lr_encoder", "calib_fraction", "seed")


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
    order: List[int] = []            # this epoch's item order (indices into train_items)
    start_epoch, start_b = 0, 0
    resume_path = os.path.join(str(out_dir), RESUME) if out_dir else ""
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    resumed = None
    if resume_path and os.path.exists(resume_path):
        resumed = torch.load(resume_path, map_location="cpu", weights_only=False)
        changed = {k: (resumed["config"].get(k), getattr(c, k)) for k in _RESUME_KEYS
                   if resumed["config"].get(k) != getattr(c, k)}
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
            row.update(_set_metrics(name, _val_logits(model, its, images, tok.pad_token_id, device), its))
        tracker.log(row, step=len(losses))
        print("step %d %s" % (len(losses), {k: round(v, 4) for k, v in row.items()}), flush=True)
        evals.append(dict(row, step=len(losses)))
        return evals[-1]

    if resumed is None:
        run_eval()
    log = TrainLog(tracker, device, steps, c.log_every)
    for epoch in range(start_epoch, c.epochs):
        if not (resumed and epoch == start_epoch):
            order = order or list(range(len(train_items)))
            random.shuffle(order)   # reshuffles last epoch's order, as shuffling the items in place did
        for b in range(start_b if resumed and epoch == start_epoch else 0, len(train_items), c.micro_batch):
            sigma = c.sigma_start + (c.sigma_end - c.sigma_start) * len(losses) / max(1, steps - 1)
            chunk = [train_items[i] for i in order[b:b + c.micro_batch]]
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
                    batch_tokens=float(batch["input_ids"].shape[1]),   # padded length; long batches drive memory
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
            if resume_path and c.save_every and len(losses) % c.save_every == 0:
                save_state(epoch, b + c.micro_batch)
        val_accuracy.append(run_eval()["val/accuracy"])

    model.decision.encoder = peft_encoder.merge_and_unload()
    model.eval()
    set_logits = {name: _val_logits(model, its, images, tok.pad_token_id, device) for name, its in sets.items()}
    temperature, by_options = fit_temperatures(set_logits["val"], calib)
    decider.cfg.temperature, decider.cfg.temperature_by_options = temperature, by_options
    print("temperatures (choice, score, noul):", [round(t, 3) for t in temperature])
    print("temperatures by options:", {b: round(t, 3) for b, t in by_options.items()})
    # With the fitted temperatures, i.e. what decide() will return. val is also the fitting set.
    final = {k: v for name, its in sets.items()
             for k, v in _set_metrics(name, set_logits[name], its, temperature, by_options).items()}
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
            "train_items": len(train_items), "calib_items": len(calib)}
