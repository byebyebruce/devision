"""Where does attribute binding fail? (round 13 planning, 2026-10-08)

Procedural scenes with two shape kinds of different colours (e.g. 3 red circles + 2 green squares, random sizes and
places). Three measurements on the same scenes:

  behaviour  the checkpoint through decide(): "What color are the squares?" (4 colours), "How many squares are
             there?" (one..four), "Is there a <colour> <shape>?" where a false statement uses the OTHER kind's colour
             (the salient-object trap). A single-kind control asks the same colour question with only one kind drawn.
  probe      the same colour / count targets read from FROZEN features by a small trained probe: one learned query per
             shape kind attends over the tokens, then a linear layer. Features: (a) SigLIP patch tokens (256 x 768,
             before any compression), (b) the checkpoint's projector output (64 x 1024, what ModernBERT receives).
             Probe train / test on disjoint scenes.

If (a) and (b) are near 100% while behaviour fails, the visual tokens carry the answer and the failure is in how the
text model uses them (fusion / training signal). If (a) is high and (b) low, the 2x2 merge + projector loses it.
If (a) is low, SigLIP at this resolution does not encode it.

    uv run python scripts/diag/binding_probe.py --checkpoint runs/v12-ext/stage2 --out runs/diag/binding_v12.json
"""
import argparse
import json
import random

import torch
import torch.nn as nn
from PIL import Image, ImageDraw

import devision
from devision.image import to_pixel_values

SHAPES = ["circle", "square", "triangle"]
COLORS = {"red": (204, 60, 50), "green": (60, 150, 80), "blue": (55, 95, 200), "yellow": (225, 190, 40)}
COUNTS = ["one", "two", "three", "four"]
BGS = [(244, 232, 220), (236, 238, 230), (240, 240, 245), (250, 248, 240)]


def draw(d, shape, box, fill):
    x0, y0, x1, y1 = box
    if shape == "circle":
        d.ellipse(box, fill=fill)
    elif shape == "square":
        d.rectangle(box, fill=fill)
    else:
        d.polygon([((x0 + x1) / 2, y0), (x1, y1), (x0, y1)], fill=fill)


def scene(rng, kinds):
    """kinds: [(shape, colour, count), ...] -> PIL image; objects of random size, never overlapping."""
    im = Image.new("RGB", (640, 420), rng.choice(BGS))
    d = ImageDraw.Draw(im)
    boxes = []
    for shape, colour, n in kinds:
        size = rng.randint(40, 95)          # one size per kind, so either kind can be the salient one
        for _ in range(n):
            for _ in range(200):
                x, y = rng.randint(10, 630 - size), rng.randint(10, 410 - size)
                b = (x, y, x + size, y + size)
                if all(b[2] + 8 < o[0] or o[2] + 8 < b[0] or b[3] + 8 < o[1] or o[3] + 8 < b[1] for o in boxes):
                    boxes.append(b)
                    draw(d, shape, b, COLORS[colour])
                    break
    return im


def make(rng, n):
    out = []
    for _ in range(n):
        a, b = rng.sample(SHAPES, 2)
        ca, cb = rng.sample(list(COLORS), 2)
        na, nb = rng.randint(1, 4), rng.randint(1, 4)
        out.append({"a": a, "b": b, "ca": ca, "cb": cb, "na": na, "nb": nb, "seed": rng.random()})
    return out


def image_of(s, single=False):
    rng = random.Random(s["seed"])
    kinds = [(s["a"], s["ca"], s["na"])] + ([] if single else [(s["b"], s["cb"], s["nb"])])
    return scene(rng, kinds)


def behaviour(dec, scenes):
    C = lambda q, opts: {"type": "choice", "instructions": q, "criteria": {o: None for o in opts}}
    hits = {"colour": 0, "colour_single": 0, "count": 0, "exists_true": 0, "exists_trap": 0}
    salient = 0
    for i, s in enumerate(scenes):
        qs = {"colour": C("What color are the %ss?" % s["a"], list(COLORS)),
              "count": C("How many %ss are there?" % s["a"], COUNTS),
              "exists_true": {"type": "noul", "instructions": "Is there a %s %s?" % (s["ca"], s["a"])},
              "exists_trap": {"type": "noul", "instructions": "Is there a %s %s?" % (s["cb"], s["a"])}}
        r = dec.decide(image=image_of(s), state="", questions=qs)["answers"]
        hits["colour"] += r["colour"]["choice"] == s["ca"]
        salient += r["colour"]["choice"] == s["cb"]
        hits["count"] += r["count"]["choice"] == COUNTS[s["na"] - 1]
        hits["exists_true"] += r["exists_true"]["noul"] > 0.5
        hits["exists_trap"] += r["exists_trap"]["noul"] < 0.5
        r1 = dec.decide(image=image_of(s, single=True), state="", questions={"colour": qs["colour"]})["answers"]
        hits["colour_single"] += r1["colour"]["choice"] == s["ca"]
        if (i + 1) % 100 == 0:
            print("  behaviour %d / %d" % (i + 1, len(scenes)), flush=True)
    n = len(scenes)
    out = {k: v / n for k, v in hits.items()}
    out["colour_answered_other_kind"] = salient / n
    return out


@torch.inference_mode()
def features(dec, scenes, device):
    size = dec.cfg.image_size
    sig, proj = [], []
    for i in range(0, len(scenes), 32):
        px = torch.stack([to_pixel_values(image_of(s), size) for s in scenes[i:i + 32]]).to(device)
        patches = dec.model.vision(pixel_values=px).last_hidden_state
        sig.append(patches.float().cpu())
        proj.append(dec.model.projector(patches).float().cpu())
    return torch.cat(sig), torch.cat(proj)


class Probe(nn.Module):
    """One learned query per shape kind attends over the tokens; a linear layer reads colour and count."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Embedding(len(SHAPES), 256)
        self.k = nn.Linear(dim, 256)
        self.v = nn.Linear(dim, 256)
        self.colour = nn.Linear(256, len(COLORS))
        self.count = nn.Linear(256, len(COUNTS))

    def forward(self, x, shape):
        x = self.norm(x)
        att = torch.softmax((self.k(x) @ self.q(shape)[:, :, None]).squeeze(-1) / 16, -1)
        h = torch.relu((att[:, :, None] * self.v(x)).sum(1))
        return self.colour(h), self.count(h)


def probe(feats, scenes, n_train, device, epochs=60):
    def targets(idx):
        shape, colour, count = [], [], []
        for i in idx:
            s = scenes[i]
            for k, c, n in ((s["a"], s["ca"], s["na"]), (s["b"], s["cb"], s["nb"])):
                shape.append(SHAPES.index(k)); colour.append(list(COLORS).index(c)); count.append(n - 1)
        rows = torch.tensor([i for i in idx for _ in (0, 1)])
        return rows, torch.tensor(shape), torch.tensor(colour), torch.tensor(count)
    torch.manual_seed(0)
    m = Probe(feats.size(-1)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    tr = targets(range(n_train))
    te = targets(range(n_train, len(scenes)))
    for _ in range(epochs):
        perm = torch.randperm(len(tr[0]))
        for j in range(0, len(perm), 128):
            b = perm[j:j + 128]
            lc, ln = m(feats[tr[0][b]].to(device), tr[1][b].to(device))
            loss = nn.functional.cross_entropy(lc, tr[2][b].to(device)) + nn.functional.cross_entropy(ln, tr[3][b].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        lc, ln = m(feats[te[0]].to(device), te[1].to(device))
        return {"colour": (lc.argmax(-1).cpu() == te[2]).float().mean().item(),
                "count": (ln.argmax(-1).cpu() == te[3]).float().mean().item()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--behaviour", type=int, default=300, help="scenes asked through decide()")
    p.add_argument("--probe-train", type=int, default=3000)
    p.add_argument("--probe-test", type=int, default=600)
    p.add_argument("--out")
    a = p.parse_args()
    dec = devision.load(a.checkpoint, device=a.device)
    device = next(dec.model.parameters()).device
    res = {"checkpoint": a.checkpoint}
    res["behaviour"] = behaviour(dec, make(random.Random(1), a.behaviour))
    print("behaviour", json.dumps(res["behaviour"]), flush=True)
    scenes = make(random.Random(2), a.probe_train + a.probe_test)
    sig, proj = features(dec, scenes, device)
    res["probe_siglip_256"] = probe(sig, scenes, a.probe_train, device)
    print("probe siglip", res["probe_siglip_256"], flush=True)
    res["probe_projector_64"] = probe(proj, scenes, a.probe_train, device)
    print("probe projector", res["probe_projector_64"], flush=True)
    res["chance"] = {"colour": 1 / len(COLORS), "count": 1 / len(COUNTS)}
    if a.out:
        import os
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
