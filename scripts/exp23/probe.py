"""exp23 digit probe (docs/experiments/2026-10-16-exp23-probe-plan.md): is a printed number's identity present in
the visual features, and where does it get lost? No model weight changes: features are taken from a frozen
checkpoint and small probes are trained on them.

Pictures: a 256 x 256 white picture with one two-digit number (10-99) at a random position, digit ink 6 / 8 / 16 /
48 px (scripts/data/exp21_reading.font_for; 48 px is the positive control). Labels: its tens and its units digit.

Feature levels of one checkpoint (default r15):
  - siglip: the vision tower's last hidden state, 256 patch tokens;
  - projector: the 64 visual tokens fed to ModernBERT;
  - encoder: ModernBERT's last-layer states at the 64 visual positions of "[CLS] <visual> [SEP]" (no question).
Optionally `--base` adds the siglip level of another checkpoint (v12: the original, frozen SigLIP2).

Probes, each trained on 3,000 pictures and tested on 1,000 others per height (fixed seeds), both digits:
  - pooled: a linear classifier on [mean; max] over tokens;
  - attention: four learned queries attend over the tokens, then a linear classifier (can pick out a location).
The level's digit accuracy = the better probe's mean of tens and units accuracy (fixed before running); picking
the better one on the test set makes it, and its copied interval, exploratory (critic EXP23-N01). Labels describe
readability by these probes, not where information is lost (critic EXP23-I01).

    uv run python scripts/exp23/probe.py --checkpoint runs/r15-vision/stage2 --base runs/v12-ext/stage2 \
        --out runs/exp23-probe
"""
import argparse
import hashlib
import json
import os
import random
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from exp21_reading import font_for  # type: ignore  # noqa: E402

from devision.image import to_pixel_values  # noqa: E402
from devision.infer import Decider  # noqa: E402

HEIGHTS = (6, 8, 16, 48)
N_TRAIN, N_TEST = 3000, 1000
LEVELS = ("siglip", "projector", "encoder")
CARRIES, ABSENT = 0.60, 0.20        # digit accuracy (chance 0.10)
CONTROL = 0.90                      # siglip at 48 px must reach this, or the probe itself is not working


def picture(value: int, height: int, rng: random.Random) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """One number at a random position; returns the picture and the ink box (left, top, right, baseline)."""
    img = Image.new("RGB", (256, 256), "white")
    d = ImageDraw.Draw(img)
    font = font_for(height)
    text = str(value)
    x0, y0, x1, y1 = (int(v) for v in d.textbbox((0, 0), text, font=font, anchor="ls"))
    w, up = x1 - x0, -y0
    x = rng.randint(4, 252 - w)
    base = rng.randint(4 + up, 252)
    d.text((x, base), text, fill="black", font=font, anchor="ls")
    box = d.textbbox((x, base), text, font=font, anchor="ls")
    return img, (int(box[0]), int(box[1]), int(box[2]), base)


def digest(img: Image.Image) -> str:
    return hashlib.sha256(img.tobytes()).hexdigest()


def dataset(height: int, n: int, seed: int, exclude: Optional[Set[str]] = None) -> Tuple[List[Image.Image], np.ndarray]:
    """n distinct pictures and labels [n, 2] = (tens, units); pictures whose pixels are in `exclude` (the
    training pictures, when drawing the test set) or already drawn are skipped (critic EXP23-N02)."""
    rng = random.Random(seed * 100 + height)
    imgs, labels, seen = [], [], set(exclude or ())
    while len(imgs) < n:
        v = rng.randint(10, 99)
        img, _ = picture(v, height, rng)
        h = digest(img)
        if h in seen:
            continue
        seen.add(h)
        imgs.append(img)
        labels.append((v // 10, v % 10))
    return imgs, np.array(labels)


@torch.no_grad()
def features(decider: Decider, imgs: List[Image.Image], level: str, batch: int = 32) -> torch.Tensor:
    """[n, tokens, dim] float16 features of `level`, on the CPU."""
    model, device = decider.model, next(decider.model.parameters()).device
    out = []
    for i in range(0, len(imgs), batch):
        px = torch.stack([to_pixel_values(im, decider.cfg.image_size) for im in imgs[i:i + batch]]).to(device)
        patches, visual = model.image_features(px)
        if level == "siglip":
            f = patches
        elif level == "projector":
            f = visual
        elif level == "encoder":
            ids = torch.tensor([[decider.tok.cls_token_id, decider.tok.sep_token_id]] * len(px), device=device)
            h, _ = model.encode(ids, torch.ones_like(ids), visual)
            f = h[:, 1:1 + visual.size(1)]
        else:
            raise ValueError(level)
        out.append(f.float().cpu().half())
    return torch.cat(out)


class AttentionProbe(torch.nn.Module):
    QUERIES = 4

    def __init__(self, dim: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.query = torch.nn.Linear(dim, self.QUERIES)
        self.heads = torch.nn.Linear(self.QUERIES * dim, 20)

    def forward(self, x):
        x = self.norm(x)
        w = torch.softmax(self.query(x), 1)                         # [B, T, Q], over tokens
        return self.heads(torch.einsum("btq,btd->bqd", w, x).flatten(1))


class PooledProbe(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(2 * dim)
        self.heads = torch.nn.Linear(2 * dim, 20)

    def forward(self, x):
        return self.heads(self.norm(torch.cat([x.mean(1), x.amax(1)], -1)))


def train_probe(kind: str, x: torch.Tensor, y: np.ndarray, epochs: int = 60, seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    probe = (AttentionProbe if kind == "attention" else PooledProbe)(x.size(-1))
    opt = torch.optim.AdamW(probe.parameters(), lr=3e-3, weight_decay=1e-2)
    yt = torch.as_tensor(y)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(x), generator=g).split(128):
            logits = probe(x[idx].float())
            loss = (torch.nn.functional.cross_entropy(logits[:, :10], yt[idx, 0])
                    + torch.nn.functional.cross_entropy(logits[:, 10:], yt[idx, 1]))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return probe.eval()


@torch.no_grad()
def correct(probe: torch.nn.Module, x: torch.Tensor, y: np.ndarray) -> np.ndarray:
    """[n, 2] booleans: tens right, units right."""
    logits = torch.cat([probe(x[i:i + 256].float()) for i in range(0, len(x), 256)])
    pred = torch.stack([logits[:, :10].argmax(-1), logits[:, 10:].argmax(-1)], -1).numpy()
    return pred == y


def interval(c: np.ndarray, resamples: int = 2000, seed: int = 0) -> Dict:
    """Mean of tens and units accuracy, 95% interval from resampling test pictures."""
    per = c.mean(1)
    rng = np.random.default_rng(seed)
    boot = per[rng.integers(0, len(per), (resamples, len(per)))].mean(1)
    return {"value": float(per.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "tens": float(c[:, 0].mean()), "units": float(c[:, 1].mean()), "n": int(len(per))}


def level_class(v: float) -> str:
    return "high" if v >= CARRIES else "low" if v <= ABSENT else "middle"


def judge(acc: Dict[str, Dict[str, float]]) -> Dict:
    """acc[level][height] = digit accuracy (the better probe, exploratory). Rules written before running; they
    describe readability by these two small probes, not where information is lost (critic EXP23-I01)."""
    s, p, e = (acc[k]["16"] for k in LEVELS)
    cls = {k: level_class(acc[k]["16"]) for k in LEVELS}
    high = [cls[k] == "high" for k in LEVELS]
    if acc["siglip"]["48"] < CONTROL:
        label = "difficulty control not passed (siglip 48 px below %.2f): no between-level reading this round" % CONTROL
    elif any(high[j] and not high[i] for i in range(3) for j in range(i + 1, 3)):
        label = "non-monotonic readability (a later level reads what an earlier one does not): bottleneck not located"
    elif all(c == "low" for c in cls.values()):
        label = "low readability at all three levels at 16 px"
    elif cls["siglip"] == "high" and p < CARRIES and p <= s - 0.20:
        label = "readability drops after the projector (siglip high, the 64 visual tokens lower)"
    elif cls["siglip"] == "high" and cls["projector"] == "high" and e < CARRIES and e <= p - 0.20:
        label = "readability drops in ModernBERT's visual states (no question in the input)"
    elif all(high):
        label = ("all three levels readable on white two-digit pictures; why the decision training failed "
                 "is still to be separated")
    else:
        label = "uncertain"
    return {"label": label, "classes": cls, "at_16": {"siglip": s, "projector": p, "encoder": e}}


def run(checkpoint: str, out: str, base: str = "", device: str = "auto", heights=HEIGHTS, n_train=N_TRAIN,
        n_test=N_TEST, epochs: int = 60) -> Dict:
    os.makedirs(out, exist_ok=True)
    models = {"": Decider.load(checkpoint, device=device)}
    if base:
        models["base_"] = Decider.load(base, device=device)
    report: Dict = {"checkpoint": checkpoint, "base": base, "heights": list(heights), "n_train": n_train,
                    "n_test": n_test, "levels": {}}
    for h in heights:
        tr_imgs, tr_y = dataset(h, n_train, seed=1)
        te_imgs, te_y = dataset(h, n_test, seed=2, exclude={digest(im) for im in tr_imgs})
        jobs = [("", lv) for lv in LEVELS] + ([("base_", "siglip")] if base else [])
        for prefix, lv in jobs:
            name = prefix + lv
            xtr, xte = features(models[prefix], tr_imgs, lv), features(models[prefix], te_imgs, lv)
            res = {}
            for kind in ("pooled", "attention"):
                res[kind] = interval(correct(train_probe(kind, xtr, tr_y, epochs), xte, te_y))
            best = max(res, key=lambda k: res[k]["value"])
            res["best"] = dict(res[best], probe=best, exploratory=True)   # chosen on the test set (critic EXP23-N01)
            report["levels"].setdefault(name, {})[str(h)] = res
            print("h%-2d %-14s pooled %.3f attention %.3f" % (h, name, res["pooled"]["value"], res["attention"]["value"]),
                  flush=True)
            del xtr, xte
    acc = {lv: {k: v["best"]["value"] for k, v in report["levels"][lv].items()} for lv in LEVELS}
    report["judgement"] = judge(acc) if {"16", "48"} <= set(acc["siglip"]) else {"label": "incomplete"}
    json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=1)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="runs/r15-vision/stage2")
    p.add_argument("--base", default="")
    p.add_argument("--out", default="runs/exp23-probe")
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    r = run(a.checkpoint, a.out, a.base, a.device)
    for name, by_h in r["levels"].items():
        print("%-14s " % name + " | ".join("%s px %.3f [%.3f, %.3f] (%s)" % (
            h, v["best"]["value"], v["best"]["ci95"][0], v["best"]["ci95"][1], v["best"]["probe"]) for h, v in by_h.items()))
    print("judgement:", r["judgement"]["label"])


if __name__ == "__main__":
    main()
