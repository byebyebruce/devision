"""exp49 region number readout (critic EXP49-PREP-02 / 03; rules written before any real readout). On EXP48-X's frozen
region features (scripts/exp49/extract.py), one small character head per layer -- SigLIP last-layer patches and projector
tokens -- with the same structure and budget (only the input projection differs): the region tokens plus learned
row / column embeddings of their position inside the region; 5 learned character queries attend once over them (4 heads,
width 256); a LayerNorm-MLP gives each slot one of 12 classes (0-9, ".", EOS). No question, option, answer set, id,
family, object or field name is an input; the box only selects the tokens (an oracle privilege, stated in the report).
On the projector's 8 x 8 grid a field's box can cover exactly the cells of another question of the same picture (the
mass and speed rows of one object in a 32-px cell): those questions are indistinguishable to the head; they are
counted and the exact score without them is reported beside the full one (the conditions use the full one).

Targets: the gold number without thousands commas, then EOS; slots after EOS are ignored. Loss: per sample the mean CE
over its valid slots (characters and EOS), then the batch mean. A prediction is the characters up to the first EOS; no
EOS, an empty string or not a number ("^[0-9]+(\\.[0-9]+)?$") is invalid; never matched to an option.

Training (fixed before reading anything): fit_O + fit_L (1,152 questions), seed 0, AdamW lr 1e-3, weight decay 0.01,
batch 64 drawn from a seeded permutation per pass, exactly 1,000 updates, the last step kept. tune never trains or
selects anything.

Readout per set (fit_O, fit_L, tune_O = rl_tune, tune_L = rl_tune_L), field-equal: whole-string exact per field and
macro; character accuracy (valid slots, EOS included); both objects right where their targets differ; both twin views
right where the target changes; same answer on different-target pairs; and the donor test -- this question's label
with its value-changed twin's region features: real - donor exact (case-cluster interval, 2,000 resamples, seed 0) and
the share of donor predictions equal to the twin's value. Conditions per layer: validity (fit_L macro >= 0.90 and each
field >= 0.75) and investment (tune macro >= 0.80, each field >= 0.60, both >= 0.50, twin >= 0.50, real - donor >= 0.20
with lower bound > 0), on original (tune_O) and L (tune_L) separately; and the pre-written branch.

    uv run python scripts/exp49/probe.py
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(__file__)
CHARS = "0123456789."
EOS, SLOTS, CLASSES = 11, 5, 12
VALID = re.compile(r"^[0-9]+(\.[0-9]+)?$")
FIELDS = ("mass", "speed", "distance")
SET_NAME = {"fit_O": "fit_O", "fit_L": "fit_L", "rl_tune": "tune_O", "rl_tune_L": "tune_L"}
LR, WD, BATCH, STEPS, SEED, WIDTH, HEADS = 1e-3, 0.01, 64, 1000, 0, 256, 4


def labels(target: str) -> List[int]:
    if len(target) + 1 > SLOTS or any(c not in CHARS for c in target):
        raise ValueError("target %r does not fit %d slots of %r + EOS" % (target, SLOTS, CHARS))
    ids = [CHARS.index(c) for c in target] + [EOS]
    return ids + [-100] * (SLOTS - len(ids))


def decode(ids: List[int]) -> Dict:
    """{"text", "valid"}: the characters before the first EOS; valid only with an EOS and a strict number."""
    if EOS not in ids:
        return {"text": "".join(CHARS[i] for i in ids if i < len(CHARS)), "valid": False}
    text = "".join(CHARS[i] for i in ids[:ids.index(EOS)])
    return {"text": text, "valid": bool(VALID.match(text))}


class Head(nn.Module):
    def __init__(self, d_in: int, max_rel: int = 16):
        super().__init__()
        self.inp = nn.Linear(d_in, WIDTH)
        self.row, self.col = nn.Embedding(max_rel, WIDTH), nn.Embedding(max_rel, WIDTH)
        self.queries = nn.Parameter(torch.randn(SLOTS, WIDTH) * 0.02)
        self.att = nn.MultiheadAttention(WIDTH, HEADS, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, WIDTH), nn.GELU(), nn.Linear(WIDTH, CLASSES))

    def forward(self, x, mask, rel):
        kv = self.inp(x.float()) + self.row(rel[..., 0]) + self.col(rel[..., 1])
        q = self.queries.unsqueeze(0).expand(x.size(0), -1, -1)
        o = self.att(q, kv, kv, key_padding_mask=~mask, need_weights=False)[0]
        return self.out(o)                                                  # [B, SLOTS, CLASSES]


def sample_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per sample the mean CE over its valid slots (EOS included), then the batch mean."""
    valid = y != -100
    ce = nn.functional.cross_entropy(logits.transpose(1, 2), y.clamp(min=0), reduction="none") * valid
    return (ce.sum(1) / valid.sum(1)).mean()


def train_head(x, mask, rel, y, steps: int = STEPS, seed: int = SEED, device: str = "cpu") -> Head:
    torch.manual_seed(seed)
    head = Head(x.shape[-1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    g = torch.Generator().manual_seed(seed)
    order: List[int] = []
    for _ in range(steps):
        if len(order) < BATCH:
            order += torch.randperm(len(y), generator=g).tolist()
        idx, order = order[:BATCH], order[BATCH:]
        logits = head(x[idx].to(device), mask[idx].to(device), rel[idx].to(device))
        loss = sample_loss(logits, y[idx].to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite readout loss")
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head.eval()


def predict(head, x, mask, rel, device: str = "cpu", batch: int = 256) -> List[List[int]]:
    out = []
    with torch.no_grad():
        for s in range(0, len(x), batch):
            out += head(x[s:s + batch].to(device), mask[s:s + batch].to(device), rel[s:s + batch].to(device)).argmax(-1).cpu().tolist()
    return out


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ambiguous(qs: List[Dict], side: int) -> List[bool]:
    """Questions whose region cells on this grid are exactly those of another question of the same picture with a
    different target: the head (which gets no field or object name) cannot tell them apart -- a ceiling of the grid."""
    ex = _load("exp49_extract", "extract.py")
    reg = [tuple(ex.region(q["box"], q["width"], q["height"], side)[0]) for q in qs]
    by = defaultdict(list)
    for i, q in enumerate(qs):
        by[(q["set"], q["picture"], reg[i])].append(i)
    return [any(qs[j]["target"] != qs[i]["target"] for j in by[(qs[i]["set"], qs[i]["picture"], reg[i])]) for i in range(len(qs))]


def readout(qs: List[Dict], pred: List[List[int]], donor: List[List[int]], y: List[List[int]], amb=None) -> Dict:
    """The per-set statistics of the docstring; `donor[i]` is question i's prediction from its twin's features; `amb`
    (optional) marks the questions the grid cannot tell apart, reported apart (counts and exact without them)."""
    r41 = _load("exp41_report", "../exp41/report.py")
    out: Dict = {}
    key_i = {q["key"]: i for i, q in enumerate(qs)}
    for setname, name in SET_NAME.items():
        idx = [i for i, q in enumerate(qs) if q["set"] == setname]
        fam: Dict = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))

        def add(f, k, v):
            fam[f][k][0] += v
            fam[f][k][1] += 1
        dec = {i: decode(pred[i]) for i in idx}
        right = {i: dec[i]["valid"] and dec[i]["text"] == qs[i]["target"] for i in idx}
        chars = [0, 0]
        for i in idx:
            q = qs[i]
            add(q["family"], "exact:" + q["field"], right[i])
            add(q["family"], "valid:" + q["field"], dec[i]["valid"])
            for p, t in zip(pred[i], y[i]):
                if t != -100:
                    chars[0] += p == t
                    chars[1] += 1
            if q["twin_changed"]:
                dd = decode(donor[i])
                add(q["family"], "real_minus_donor:" + q["field"], float(right[i]) - float(dd["valid"] and dd["text"] == q["target"]))
                add(q["family"], "donor_follows:" + q["field"], dd["valid"] and dd["text"] == qs[key_i[q["twin"]]]["target"])
        pairs = defaultdict(list)
        for i in idx:
            pairs[(qs[i]["picture"], qs[i]["field"])].append(i)
        for (_, f), v in pairs.items():
            if len(v) == 2 and qs[v[0]]["target"] != qs[v[1]]["target"]:
                add(qs[v[0]]["family"], "both:" + f, right[v[0]] and right[v[1]])
                add(qs[v[0]]["family"], "same:" + f, dec[v[0]]["text"] == dec[v[1]]["text"])
        for i in idx:
            q = qs[i]
            j = key_i.get(q["twin"]) if q["twin"] else None
            if q["twin_changed"] and j is not None and q["view"] == "v0":
                add(q["family"], "twin:" + q["field"], right[i] and right[j])
        families = sorted(fam)

        def macro(fs, stat):
            vals = []
            for f in FIELDS:
                k = "%s:%s" % (stat, f)
                d = sum(fam[x][k][1] for x in fs if k in fam[x])
                if d:
                    vals.append(sum(fam[x][k][0] for x in fs if k in fam[x]) / d)
            return sum(vals) / len(vals) if vals else float("nan")
        res: Dict = {s: macro(families, s) for s in ("exact", "valid", "both", "twin", "same", "real_minus_donor", "donor_follows")}
        res["chars"] = chars[0] / chars[1] if chars[1] else None
        for s in ("exact", "both", "twin", "real_minus_donor", "donor_follows"):
            for f in FIELDS:
                k = "%s:%s" % (s, f)
                d = sum(fam[x][k][1] for x in families if k in fam[x])
                res[k] = {"value": sum(fam[x][k][0] for x in families if k in fam[x]) / d if d else None, "n": d}
        cl = defaultdict(set)                       # case-linked clusters of this set's families
        parent: Dict[str, str] = {}

        def find(a):
            while parent.setdefault(a, a) != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for i in idx:
            a, b = find("f:" + qs[i]["family"]), find("c:" + qs[i]["case"])
            if a != b:
                parent[max(a, b)] = min(a, b)
        for i in idx:
            cl[find("f:" + qs[i]["family"])].add(qs[i]["family"])
        keys = sorted(cl)
        if amb is not None:
            ua = [i for i in idx if not amb[i]]
            res["ambiguous_questions"] = len(idx) - len(ua)
            res["exact_unambiguous"] = sum(right[i] for i in ua) / len(ua) if ua else None
        res["real_minus_donor_ci"] = r41.interval(lambda fs: macro(fs, "real_minus_donor"), keys,
                                                  lambda ks: [f for k in ks for f in sorted(cl[k])])["ci95"]
        res["clusters"] = len(keys)
        out[name] = res
    return out


def conditions(r: Dict) -> Dict:
    def valid_head(s):
        return r[s]["exact"] >= 0.90 and all((r[s]["exact:" + f]["value"] or 0) >= 0.75 for f in FIELDS)

    def invest(s):
        x = r[s]
        return bool(x["exact"] >= 0.80 and all((x["exact:" + f]["value"] or 0) >= 0.60 for f in FIELDS) and x["both"] >= 0.50
                    and x["twin"] >= 0.50 and x["real_minus_donor"] >= 0.20 and x["real_minus_donor_ci"][0] > 0)
    return {"validity": bool(valid_head("fit_L")), "invest_original": invest("tune_O"), "invest_L": invest("tune_L")}


def branch(c: Dict[str, Dict]) -> str:
    s, p = c["siglip"], c["proj"]
    if not (s["validity"] or p["validity"]):
        return "readout not learned (fit_L validity failed for both heads): no bottleneck located"
    if s["invest_original"] and p["invest_original"]:
        return "both layers readable on the original pictures: next, the path from the question to the content and the options"
    if s["invest_original"] and not p["invest_original"]:
        return "SigLIP readable, projector not: next, local number supervision of the projection / shared visual path"
    if (s["invest_L"] or p["invest_L"]) and not (s["invest_original"] or p["invest_original"]):
        return "only L readable: next, a fixed multi-view / local-detail encoding of the real input"
    return "mixed / not located: report the raw numbers"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="data/exp49/questions.jsonl")
    ap.add_argument("--features", default="runs/exp49")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    info = json.load(open(os.path.join(a.features, "features.json")))
    if info["questions_sha256"] != hashlib.sha256(open(a.questions, "rb").read()).hexdigest() or \
            info["features_sha256"] != hashlib.sha256(open(os.path.join(a.features, "features.npz"), "rb").read()).hexdigest():
        raise SystemExit("features are not bound to these questions / this file")
    qs = [json.loads(x) for x in open(a.questions) if x.strip()]
    z = np.load(os.path.join(a.features, "features.npz"))
    y = torch.tensor([labels(q["target"]) for q in qs])
    train = [i for i, q in enumerate(qs) if q["set"] in ("fit_O", "fit_L")]
    key_i = {q["key"]: i for i, q in enumerate(qs)}
    donor_idx = [key_i[q["twin"]] if q["twin"] else i for i, q in enumerate(qs)]
    res: Dict = {"features": info, "training": {"questions": len(train), "lr": LR, "weight_decay": WD, "batch": BATCH, "steps": STEPS,
                                                "seed": SEED, "width": WIDTH, "heads": HEADS, "slots": SLOTS},
                 "oracle": "the target box selects the region tokens (not available to the product)", "layers": {}}
    for layer in ("siglip", "proj"):
        x, m, r = torch.from_numpy(z[layer]), torch.from_numpy(z[layer + "_mask"]), torch.from_numpy(z[layer + "_rel"])
        random.seed(SEED)
        head = train_head(x[train], m[train], r[train], y[train], device=a.device)
        pred = predict(head, x, m, r, a.device)
        donor = predict(head, x[donor_idx], m[donor_idx], r[donor_idx], a.device)
        ro = readout(qs, pred, donor, y.tolist(), ambiguous(qs, 16 if layer == "siglip" else 8))
        res["layers"][layer] = {"readout": ro, "conditions": conditions(ro)}
    res["branch"] = branch({k: v["conditions"] for k, v in res["layers"].items()})
    os.makedirs(a.features, exist_ok=True)
    json.dump(res, open(os.path.join(a.features, "report.json"), "w"), indent=1, default=str)
    lines = []
    for layer, v in res["layers"].items():
        for s, x in v["readout"].items():
            lines.append("%-6s %-6s exact %.3f (%s) valid %.3f chars %.3f both %.3f twin %.3f same %.3f real-donor %+.3f %s follows %.3f" % (
                layer, s, x["exact"], {f: None if x["exact:" + f]["value"] is None else round(x["exact:" + f]["value"], 3) for f in FIELDS},
                x["valid"], x["chars"] or 0, x["both"], x["twin"], x["same"], x["real_minus_donor"],
                [round(c, 3) for c in x["real_minus_donor_ci"]], x["donor_follows"]) +
                " ambiguous %s exact without them %s" % (x.get("ambiguous_questions"), x.get("exact_unambiguous")))
        lines.append("%s conditions %s" % (layer, v["conditions"]))
    lines.append("branch: %s" % res["branch"])
    open(os.path.join(a.features, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
