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
        self.geom = nn.Linear(4, WIDTH)                                     # continuous geometry relative to the box
        self.queries = nn.Parameter(torch.randn(SLOTS, WIDTH) * 0.02)
        self.att = nn.MultiheadAttention(WIDTH, HEADS, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, WIDTH), nn.GELU(), nn.Linear(WIDTH, CLASSES))

    def forward(self, x, mask, rel, geom):
        kv = self.inp(x.float()) + self.row(rel[..., 0]) + self.col(rel[..., 1]) + self.geom(geom.float())
        q = self.queries.unsqueeze(0).expand(x.size(0), -1, -1)
        o = self.att(q, kv, kv, key_padding_mask=~mask, need_weights=False)[0]
        return self.out(o)                                                  # [B, SLOTS, CLASSES]


def sample_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per sample the mean CE over its valid slots (EOS included), then the batch mean."""
    valid = y != -100
    ce = nn.functional.cross_entropy(logits.transpose(1, 2), y.clamp(min=0), reduction="none") * valid
    return (ce.sum(1) / valid.sum(1)).mean()


def train_head(x, mask, rel, geom, y, steps: int = STEPS, seed: int = SEED, device: str = "cpu") -> Head:
    torch.manual_seed(seed)
    head = Head(x.shape[-1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    g = torch.Generator().manual_seed(seed)
    order: List[int] = []
    for _ in range(steps):
        if len(order) < BATCH:
            order += torch.randperm(len(y), generator=g).tolist()
        idx, order = order[:BATCH], order[BATCH:]
        logits = head(x[idx].to(device), mask[idx].to(device), rel[idx].to(device), geom[idx].to(device))
        loss = sample_loss(logits, y[idx].to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite readout loss")
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head.eval()


def predict(head, x, mask, rel, geom, device: str = "cpu", batch: int = 256) -> List[List[int]]:
    out = []
    with torch.no_grad():
        for s in range(0, len(x), batch):
            out += head(x[s:s + batch].to(device), mask[s:s + batch].to(device), rel[s:s + batch].to(device),
                        geom[s:s + batch].to(device)).argmax(-1).cpu().tolist()
    return out


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ambiguous(qs: List[Dict], side: int) -> List[bool]:
    """Questions whose region CELLS on this grid are exactly those of another question of the same picture with a
    different target -- a difficulty slice defined before the run (critic EXP49-R1-01: with the continuous box geometry
    the head's inputs no longer coincide; reported apart only, the conditions use all questions)."""
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
            sl = [i for i in idx if amb[i]]
            res["cell_collision_slice"] = {"questions": len(sl), "exact": sum(right[i] for i in sl) / len(sl) if sl else None,
                                           "exact_outside": sum(right[i] for i in ua) / len(ua) if ua else None}
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


def status(c: Dict) -> Dict[str, str]:
    """Per layout, what a layer's result may say: positive evidence, a valid head below the investment condition, or a
    readout that was not learned (critic EXP49-R1-03: a low score of an invalid head is never a negative finding)."""
    def one(invest):
        if invest:
            return "positive"
        return "valid, below the condition" if c["validity"] else "not learned: not located"
    return {"original": one(c["invest_original"]), "L": one(c["invest_L"])}


def branch(c: Dict[str, Dict]) -> str:
    s, p = status(c["siglip"]), status(c["proj"])
    if s["original"] == "positive" and p["original"] == "positive":
        return "both layers readable on the original pictures: next, the path from the question to the content and the options"
    if s["original"] == "positive" and p["original"] == "valid, below the condition":
        return "SigLIP readable, projector below the condition with a valid head: next, local number supervision of the projection / shared visual path"
    if s["original"] == "positive" and p["original"].startswith("not learned"):
        return "SigLIP readable on the original pictures; the projector readout was not learned: projector not located"
    l_pos = [k for k, v in (("siglip", s), ("proj", p)) if v["L"] == "positive"]
    if l_pos and all(v["original"] == "valid, below the condition" for k, v in (("siglip", s), ("proj", p)) if k in l_pos):
        return "only L readable (original below the condition with a valid head): next, a fixed multi-view / local-detail encoding"
    if not (c["siglip"]["validity"] or c["proj"]["validity"]):
        return "readout not learned (fit_L validity failed for both heads): no bottleneck located"
    return "mixed / not located: report the raw numbers (siglip %s, projector %s)" % (s, p)


def write_details(path: str, qs: List[Dict], pred: List[List[int]], donor: List[List[int]], donor_idx: List[int], ident: Dict) -> str:
    """Every question's real and donor slot predictions, parses and keys, bound to the features / questions (EXP49-R1-02)."""
    with open(path, "w") as fh:
        fh.write(json.dumps({"identity": ident}) + "\n")
        for i, q in enumerate(qs):
            dr, dd = decode(pred[i]), decode(donor[i])
            fh.write(json.dumps({"key": q["key"], "target": q["target"], "pred": pred[i], "pred_text": dr["text"], "pred_valid": dr["valid"],
                                 "donor_key": qs[donor_idx[i]]["key"] if q["twin_changed"] else None, "donor": donor[i],
                                 "donor_text": dd["text"], "donor_valid": dd["valid"]}) + "\n")
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def read_details(path: str, qs: List[Dict]):
    """(pred, donor) lists in question order from a details file; refused unless it holds every question once, with the
    question file's target and donor (the twin, for value-changed twins)."""
    lines = [json.loads(x) for x in open(path) if x.strip()]
    rows = lines[1:]
    by = {}
    for r in rows:
        if r["key"] in by:
            raise SystemExit("%s: %s twice" % (path, r["key"]))
        by[r["key"]] = r
    if set(by) != {q["key"] for q in qs}:
        raise SystemExit("%s: %d questions missing, %d unknown" % (path, len({q["key"] for q in qs} - set(by)), len(set(by) - {q["key"] for q in qs})))
    for q in qs:
        r = by[q["key"]]
        if r["target"] != q["target"] or r["donor_key"] != (q["twin"] if q["twin_changed"] else None):
            raise SystemExit("%s: %s target or donor differs from the question file" % (path, q["key"]))
    return [by[q["key"]]["pred"] for q in qs], [by[q["key"]]["donor"] for q in qs], lines[0]["identity"]


def recompute(questions: str, out: str) -> Dict:
    """The report from the saved details only (no model, no head): the same readout, conditions and branch."""
    qs = [json.loads(x) for x in open(questions) if x.strip()]
    y = [labels(q["target"]) for q in qs]
    res: Dict = {"layers": {}}
    for layer in ("siglip", "proj"):
        path = os.path.join(out, "details_%s.jsonl" % layer)
        if not os.path.exists(path):
            res["layers"][layer] = "incomplete: no details"
            continue
        pred, donor, _ = read_details(path, qs)
        ro = readout(qs, pred, donor, y, ambiguous(qs, 16 if layer == "siglip" else 8))
        res["layers"][layer] = {"readout": ro, "conditions": conditions(ro)}
    res["branch"] = branch({k: v["conditions"] for k, v in res["layers"].items()}) if all(isinstance(v, dict) for v in res["layers"].values()) \
        else "incomplete: a layer is missing"
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="data/exp49/questions.jsonl")
    ap.add_argument("--features", default="runs/exp49")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--recompute", action="store_true", help="the report from the saved details only (no model)")
    a = ap.parse_args(argv)
    if a.recompute:
        res = recompute(a.questions, a.features)
        print(json.dumps({"branch": res["branch"], "conditions": {k: v["conditions"] for k, v in res["layers"].items() if isinstance(v, dict)}}, indent=1))
        return 0
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
    ident = {"features_sha256": info["features_sha256"], "questions_sha256": info["questions_sha256"], "model": info["model"]}
    res: Dict = {"features": info, "training": {"questions": len(train), "lr": LR, "weight_decay": WD, "batch": BATCH, "steps": STEPS,
                                                "seed": SEED, "width": WIDTH, "heads": HEADS, "slots": SLOTS},
                 "oracle": "the target box selects the region tokens and gives their geometry (not available to the product)", "layers": {}}
    for layer in ("siglip", "proj"):
        x, m = torch.from_numpy(z[layer]), torch.from_numpy(z[layer + "_mask"])
        r, g = torch.from_numpy(z[layer + "_rel"]), torch.from_numpy(z[layer + "_geom"])
        random.seed(SEED)
        head = train_head(x[train], m[train], r[train], g[train], y[train], device=a.device)
        torch.save(head.state_dict(), os.path.join(a.features, "head_%s.pt" % layer))
        pred = predict(head, x, m, r, g, a.device)
        donor = predict(head, x[donor_idx], m[donor_idx], r[donor_idx], g[donor_idx], a.device)
        dsha = write_details(os.path.join(a.features, "details_%s.jsonl" % layer), qs, pred, donor, donor_idx, ident)   # before resampling
        ro = readout(qs, pred, donor, y.tolist(), ambiguous(qs, 16 if layer == "siglip" else 8))
        res["layers"][layer] = {"readout": ro, "conditions": conditions(ro), "status": status(conditions(ro)), "details_sha256": dsha,
                                "head_sha256": hashlib.sha256(open(os.path.join(a.features, "head_%s.pt" % layer), "rb").read()).hexdigest()}
    res["branch"] = branch({k: v["conditions"] for k, v in res["layers"].items()})
    rc = recompute(a.questions, a.features)
    res["recompute_matches"] = all(json.dumps(rc["layers"][k]["readout"], sort_keys=True, default=str) == json.dumps(res["layers"][k]["readout"], sort_keys=True, default=str)
                                   for k in res["layers"]) and rc["branch"] == res["branch"]
    os.makedirs(a.features, exist_ok=True)
    json.dump(res, open(os.path.join(a.features, "report.json"), "w"), indent=1, default=str)
    lines = []
    for layer, v in res["layers"].items():
        for s_, x in v["readout"].items():
            lines.append("%-6s %-6s exact %.3f (%s) valid %.3f chars %.3f both %.3f twin %.3f same %.3f real-donor %+.3f %s follows %.3f" % (
                layer, s_, x["exact"], {f: None if x["exact:" + f]["value"] is None else round(x["exact:" + f]["value"], 3) for f in FIELDS},
                x["valid"], x["chars"] or 0, x["both"], x["twin"], x["same"], x["real_minus_donor"],
                [round(c, 3) for c in x["real_minus_donor_ci"]], x["donor_follows"]) +
                " cell-collision slice %s" % x.get("cell_collision_slice"))
        lines.append("%s conditions %s status %s" % (layer, v["conditions"], v["status"]))
    lines.append("branch: %s; recompute from details matches: %s" % (res["branch"], res["recompute_matches"]))
    open(os.path.join(a.features, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
