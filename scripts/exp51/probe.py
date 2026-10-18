"""exp51 local region number readout (critic EXP51-PREP-01; rules written before any real readout). On the region tokens of
EXP50's real local-view keys / values (scripts/exp51/extract.py; one frozen SigLIP tower; T = views from the original
picture, C = views from the picture reduced to full-letterbox detail), one small character head per condition -- EXP49's
head, loss, decoding, training (fit_O + fit_L, 1,152 questions, seed 0, AdamW 1e-3 / wd 0.01, batch 64 from a seeded
permutation, exactly 1,000 updates, last step kept) and readout (scripts/exp49/probe.py), with the batch padded only to
its own longest region. No question, option, answer set, id, family, object, field or view is an input; the field box
selects the tokens and gives their geometry (an oracle privilege; the head is not a product candidate).

Donor test: question i's label with its value-changed twin's tokens, keeping i's own rel / geometry / mask (the twin's
region is checked to have the same indices, views and geometry). 15 cut-slice questions (EXP50 geometry) are reported
apart only.

Conditions per condition (EXP49's): validity on fit_L (macro >= 0.90, each field >= 0.75) and, for the original pictures,
also fit_O (same thresholds); investment on tune_O and tune_L (macro >= 0.80, each field >= 0.60, both >= 0.50, twin >=
0.50, real - donor >= 0.20 with lower bound > 0). Attribution to original detail: tune_O exact macro T - C >= +0.10 with
its case-cluster lower bound > 0 (2,000 resamples, seed 0) and T's original investment true. Branch (critic EXP51-PREP-01):
  - T original positive and attribution: local content readable under the oracle region; next, direct supervision of
    local content readout, then question selection / decision use;
  - T and C both original positive, attribution not passed: content / selection path may be prepared, not credited to
    original detail; cheaper readable inputs first;
  - valid heads, original below the condition: only this fixed readout condition is negative; no long 12-view training;
  - a head not learned (fit invalid) or a condition incomplete: not located.

    uv run python scripts/exp51/probe.py [--recompute]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch

HERE = os.path.dirname(__file__)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p49 = _load("exp49_probe", "../exp49/probe.py")
r41 = p49._load("exp41_report", "../exp41/report.py")
CONDS = ("T", "C")


class Tokens:
    """One condition's ragged region tokens with the questions' rel / geometry; batches padded to their own longest."""

    def __init__(self, flat: np.ndarray, offsets: np.ndarray, regs: List[Dict]):
        self.flat, self.off = flat, offsets
        self.rel = [np.asarray(r["rel"], np.int64) for r in regs]
        self.geom = [np.asarray(r["geom"], np.float32) for r in regs]
        for i, r in enumerate(regs):
            if offsets[i + 1] - offsets[i] != len(r["index"]):
                raise SystemExit("features and regions disagree at question %d" % i)

    def batch(self, idx: List[int], feats_from: Optional[List[int]] = None):
        """(x, mask, rel, geom) for questions `idx`; `feats_from[k]` (the donor) gives question idx[k]'s token features."""
        src = feats_from or idx
        n = max(len(self.rel[i]) for i in idx)
        x = torch.zeros(len(idx), n, self.flat.shape[1])
        mask = torch.zeros(len(idx), n, dtype=torch.bool)
        rel = torch.zeros(len(idx), n, 2, dtype=torch.long)
        geom = torch.zeros(len(idx), n, 4)
        for k, (i, j) in enumerate(zip(idx, src)):
            m = len(self.rel[i])
            if self.off[j + 1] - self.off[j] != m:
                raise SystemExit("donor %d and question %d have different regions" % (j, i))
            x[k, :m] = torch.from_numpy(self.flat[self.off[j]:self.off[j + 1]].astype(np.float32))
            mask[k, :m] = True
            rel[k, :m] = torch.from_numpy(self.rel[i])
            geom[k, :m] = torch.from_numpy(self.geom[i])
        return x, mask, rel, geom


def train_head(tok: Tokens, train: List[int], y: torch.Tensor, steps: int = p49.STEPS, seed: int = p49.SEED, device: str = "cpu"):
    """EXP49's train_head (same seed use, permutation, loss, optimiser, last step) on per-batch padded tokens."""
    torch.manual_seed(seed)
    head = p49.Head(tok.flat.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=p49.LR, weight_decay=p49.WD)
    g = torch.Generator().manual_seed(seed)
    order: List[int] = []
    for _ in range(steps):
        if len(order) < p49.BATCH:
            order += torch.randperm(len(train), generator=g).tolist()
        pick, order = order[:p49.BATCH], order[p49.BATCH:]
        idx = [train[k] for k in pick]
        x, m, r, gm = tok.batch(idx)
        loss = p49.sample_loss(head(x.to(device), m.to(device), r.to(device), gm.to(device)), y[idx].to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite readout loss")
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head.eval()


def predict(head, tok: Tokens, idx: List[int], feats_from: Optional[List[int]] = None, device: str = "cpu", batch: int = 128) -> List[List[int]]:
    out: List[List[int]] = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            x, m, r, g = tok.batch(b, feats_from[s:s + batch] if feats_from else None)
            out += head(x.to(device), m.to(device), r.to(device), g.to(device)).argmax(-1).cpu().tolist()
    return out


def exact_units(qs: List[Dict], pred: List[List[int]], setname: str):
    """(family -> field -> [right, n], clusters) for one set's whole-string exact, as EXP49's readout counts them."""
    fam: Dict = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    idx = [i for i, q in enumerate(qs) if q["set"] == setname]
    for i in idx:
        d = p49.decode(pred[i])
        fam[qs[i]["family"]][qs[i]["field"]][0] += d["valid"] and d["text"] == qs[i]["target"]
        fam[qs[i]["family"]][qs[i]["field"]][1] += 1
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
    cl: Dict = defaultdict(set)
    for i in idx:
        cl[find("f:" + qs[i]["family"])].add(qs[i]["family"])
    return fam, cl


def macro(fam, fs) -> float:
    vals = []
    for f in p49.FIELDS:
        d = sum(fam[x][f][1] for x in fs if f in fam[x])
        if d:
            vals.append(sum(fam[x][f][0] for x in fs if f in fam[x]) / d)
    return sum(vals) / len(vals) if vals else float("nan")


def t_minus_c(qs, pred_t, pred_c, setname: str) -> Dict:
    ft, cl = exact_units(qs, pred_t, setname)
    fc, cl2 = exact_units(qs, pred_c, setname)
    assert cl == cl2
    keys = sorted(cl)
    expand = lambda ks: [f for k in ks for f in sorted(cl[k])]  # noqa: E731
    iv = r41.interval(lambda fs: macro(ft, fs) - macro(fc, fs), keys, expand)
    return {"value": macro(ft, sorted(ft)) - macro(fc, sorted(fc)), "ci95": iv["ci95"], "clusters": len(keys)}


def validity(r: Dict, s: str) -> bool:
    return bool(r[s]["exact"] >= 0.90 and all((r[s]["exact:" + f]["value"] or 0) >= 0.75 for f in p49.FIELDS))


def status(r: Dict, c: Dict) -> Dict[str, str]:
    """original: positive / valid below / not located (fit_O or fit_L not learned); L as EXP49."""
    ok_o = c["validity"] and validity(r, "fit_O")
    return {"original": "positive" if c["invest_original"] else "valid, below the condition" if ok_o else "not learned: not located",
            "L": "positive" if c["invest_L"] else "valid, below the condition" if c["validity"] else "not learned: not located"}


def branch(st: Dict[str, Dict], attribution) -> str:
    t, c = st["T"]["original"], st["C"]["original"]
    if "not learned" in t or "not learned" in c:
        return "a head was not learned: not located (no new head, steps or thresholds)"
    if t == "positive" and attribution is True:
        return "local content readable under the oracle region, credited to original detail: next, direct supervision of local content readout, then question selection / decision use"
    if t == "positive" and c == "positive":
        return "T and C both readable on the original pictures, attribution not passed: content / selection path may be prepared, not credited to original detail; cheaper readable inputs first"
    if t == "positive":
        return "T readable on the original pictures, C not, but the T - C attribution condition not passed: report the raw numbers; not credited to original detail"
    return "valid heads, original pictures below the condition: negative for this fixed readout only; no long 12-view candidate training"


def cut_slice(qs, pred) -> Dict:
    geo = json.load(open("data/exp50/geometry.json"))
    keys = {(x["set"], x["id"], x["qid"]) for x in geo["cut_slice"]}
    sel = [i for i, q in enumerate(qs) if (q["set"], q["id"], q["qid"]) in keys]
    right = [p49.decode(pred[i])["valid"] and p49.decode(pred[i])["text"] == qs[i]["target"] for i in sel]
    return {"right": int(sum(right)), "of": len(sel)}


def report_from(qs, preds: Dict[str, tuple]) -> Dict:
    """The readout, conditions, T - C and branch from per-condition (pred, donor) -- used by the run and by --recompute."""
    y = [p49.labels(q["target"]) for q in qs]
    res: Dict = {"conditions": {}}
    for cond, (pred, donor) in preds.items():
        ro = p49.readout(qs, pred, donor, y)
        c = p49.conditions(ro)
        res["conditions"][cond] = {"readout": ro, "conditions": c, "fit_O_valid": validity(ro, "fit_O"), "status": status(ro, c),
                                   "cut_slice": cut_slice(qs, pred)}
    if set(preds) == set(CONDS):
        d = {s: t_minus_c(qs, preds["T"][0], preds["C"][0], s) for s in ("fit_O", "fit_L", "rl_tune", "rl_tune_L")}
        res["T_minus_C"] = d
        att = bool(d["rl_tune"]["value"] >= 0.10 and d["rl_tune"]["ci95"][0] > 0 and res["conditions"]["T"]["conditions"]["invest_original"])
        res["attribution"] = att
        res["branch"] = branch({k: v["status"] for k, v in res["conditions"].items()}, att)
    else:
        res["attribution"] = res["branch"] = "incomplete: a condition is missing"
    return res


def recompute(questions: str, out: str) -> Dict:
    qs = [json.loads(x) for x in open(questions) if x.strip()]
    preds = {}
    for cond in CONDS:
        path = os.path.join(out, "details_%s.jsonl" % cond)
        if os.path.exists(path):
            pred, donor, _ = p49.read_details(path, qs)
            preds[cond] = (pred, donor)
    return report_from(qs, preds)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="data/exp49/questions.jsonl")
    ap.add_argument("--regions", default="data/exp51/regions.json.gz")
    ap.add_argument("--features", default="runs/exp51")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args(argv)
    if a.recompute:
        res = recompute(a.questions, a.features)
        print(json.dumps({"attribution": res["attribution"], "branch": res["branch"],
                          "status": {k: v["status"] for k, v in res["conditions"].items()}}, indent=1))
        return 0
    import gzip
    info = json.load(open(os.path.join(a.features, "features.json")))
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
    if info["questions_sha256"] != sha(a.questions) or info["regions_sha256"] != sha(a.regions):
        raise SystemExit("features are not bound to these questions / regions")
    qs = [json.loads(x) for x in open(a.questions) if x.strip()]
    regs = json.load(gzip.open(a.regions, "rt"))
    key_i = {q["key"]: i for i, q in enumerate(qs)}
    donor_idx = [key_i[q["twin"]] if q["twin"] else i for i, q in enumerate(qs)]
    for i, q in enumerate(qs):           # donor regions: same indices, views and geometry (checked again here)
        if q["twin_changed"]:
            j = donor_idx[i]
            if regs[i]["index"] != regs[j]["index"] or regs[i]["view"] != regs[j]["view"] or \
                    np.asarray(regs[i]["geom"], np.float32).tobytes() != np.asarray(regs[j]["geom"], np.float32).tobytes():
                raise SystemExit("donor region differs: %s" % q["key"])
    y = torch.tensor([p49.labels(q["target"]) for q in qs])
    train = [i for i, q in enumerate(qs) if q["set"] in ("fit_O", "fit_L")]
    allq = list(range(len(qs)))
    preds, heads = {}, {}
    for cond in CONDS:
        path = os.path.join(a.features, "features_%s.npz" % cond)
        if sha(path) != info["conditions"][cond]["sha256"]:
            raise SystemExit("%s changed" % path)
        z = np.load(path)
        tok = Tokens(z["x"], z["offsets"], regs)
        head = train_head(tok, train, y, device=a.device)
        hp = os.path.join(a.features, "head_%s.pt" % cond)
        torch.save(head.state_dict(), hp)
        pred = predict(head, tok, allq, device=a.device)
        donor = predict(head, tok, allq, donor_idx, device=a.device)
        ident = {"features_sha256": info["conditions"][cond]["sha256"], "questions_sha256": info["questions_sha256"],
                 "regions_sha256": info["regions_sha256"], "model": info["model"], "condition": cond}
        heads[cond] = {"head_sha256": sha(hp),
                       "details_sha256": p49.write_details(os.path.join(a.features, "details_%s.jsonl" % cond), qs, pred, donor, donor_idx, ident)}
        preds[cond] = (pred, donor)
        del tok, z
    res = report_from(qs, preds)
    res.update(features=info, heads=heads, oracle="the field box selects the region tokens and gives their geometry (not available to the product)",
               training={"questions": len(train), "lr": p49.LR, "weight_decay": p49.WD, "batch": p49.BATCH, "steps": p49.STEPS, "seed": p49.SEED})
    rc = recompute(a.questions, a.features)
    res["recompute_matches"] = json.dumps(rc["conditions"], sort_keys=True, default=str) == json.dumps(
        {k: v for k, v in res["conditions"].items()}, sort_keys=True, default=str) and rc["branch"] == res["branch"]
    json.dump(res, open(os.path.join(a.features, "report.json"), "w"), indent=1, default=str)
    lines = []
    for cond, v in res["conditions"].items():
        for s_, x in v["readout"].items():
            lines.append("%s %-6s exact %.3f (%s) valid %.3f chars %.3f both %.3f twin %.3f same %.3f real-donor %+.3f %s follows %.3f" % (
                cond, s_, x["exact"], {f: None if x["exact:" + f]["value"] is None else round(x["exact:" + f]["value"], 3) for f in p49.FIELDS},
                x["valid"], x["chars"] or 0, x["both"], x["twin"], x["same"], x["real_minus_donor"],
                [round(c, 3) for c in x["real_minus_donor_ci"]], x["donor_follows"]))
        lines.append("%s conditions %s fit_O valid %s status %s cut slice %s" % (cond, v["conditions"], v["fit_O_valid"], v["status"], v["cut_slice"]))
    lines.append("T - C exact %s" % {k: "%+.3f [%+.3f, %+.3f]" % (d["value"], *d["ci95"]) for k, d in res["T_minus_C"].items()})
    lines.append("attribution %s; branch: %s; recompute from details matches: %s" % (res["attribution"], res["branch"], res["recompute_matches"]))
    open(os.path.join(a.features, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
