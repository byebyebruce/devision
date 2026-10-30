"""Offline temperature refit from per-question details (no model forward).

A checkpoint's details hold softmax(z / T_used); z = T_used * log p up to a per-question constant, so any other
temperature is exact: p' = softmax(T_used * log p / T_new). Fits per-type and per-bucket temperatures on the
--fit details with the same rules as rlcd.fit_temperatures (bucket >= 50, type >= 10 questions), then reports
NLL / ECE before and after on every other details file, next to a reference model's.

usage: python scripts/v05/recalib_offline.py --checkpoint runs/r23-v05/train --fit runs/r23-v05/eval/dev_mix.details.jsonl \
          --eval-dir runs/r23-v05/eval --ref-dir runs/r21-soft/eval [--json out.json]
"""
import argparse, glob, json, math, os
from collections import defaultdict
import numpy as np
from devision.model import pick_temperature, option_bucket
from devision.train.evaluate import ece_score

QT = {"choice": 0, "score": 1, "noul": 2}


def load(path):
    return [json.loads(l) for l in open(path)]


def logits(rows, temp, by_opt):
    out = []
    for r in rows:
        opts = r["options"]
        t = pick_temperature(temp, by_opt, QT[r["type"]], len(opts))
        out.append((opts, np.array([t * math.log(max(r["probabilities"][o], 1e-12)) for o in opts])))
    return out


def probs(z, t):
    z = z / t
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def nll_of(rows, zs, temps):
    s = 0.0
    for r, (opts, z), t in zip(rows, zs, temps):
        p = probs(z, t)
        s -= sum(r["gold"].get(o, 0) * math.log(max(pi, 1e-12)) for o, pi in zip(opts, p) if r["gold"].get(o, 0) > 0)
    return s / len(rows)


def fit(rows, zs):
    grid = np.exp(np.linspace(math.log(0.3), math.log(8), 400))
    best = min(grid, key=lambda t: nll_of(rows, zs, [t] * len(rows)))
    return float(best)


def metrics(rows, zs, temp, by_opt):
    ts = [pick_temperature(temp, by_opt, QT[r["type"]], len(o)) for r, (o, _) in zip(rows, zs)]
    conf, corr = [], []
    for r, (opts, z), t in zip(rows, zs, ts):
        p = probs(z, t)
        i = int(p.argmax())
        conf.append(p[i])
        corr.append(float(opts[i] == r["gold_answer"]))
    return nll_of(rows, zs, ts), float(ece_score(np.array(conf), np.array(corr)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fit", required=True, nargs="+")
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.checkpoint, "config.json")))
    t0, b0 = cfg["temperature"], cfg.get("temperature_by_options", {})
    fr = [r for f in a.fit for r in load(f)]
    fz = logits(fr, t0, b0)
    by_type, by_bucket = defaultdict(list), defaultdict(list)
    for i, r in enumerate(fr):
        by_type[QT[r["type"]]].append(i)
        by_bucket[option_bucket(QT[r["type"]], len(r["options"]))].append(i)
    sub = lambda idx: ([fr[i] for i in idx], [fz[i] for i in idx])
    temp = [fit(*sub(by_type[q])) if len(by_type[q]) >= 10 else t0[q] for q in range(3)]
    by_opt = {b: fit(*sub(idx)) for b, idx in by_bucket.items() if len(idx) >= 50}
    print("old", t0, b0)
    print("new", temp, by_opt)
    out = {"old": [t0, b0], "new": [temp, by_opt], "sets": {}}
    fit_names = {os.path.basename(f).split(".")[0] for f in a.fit}
    for p in sorted(glob.glob(os.path.join(a.eval_dir, "*.details.jsonl"))):
        n = os.path.basename(p)[: -len(".details.jsonl")]
        ref = os.path.join(a.ref_dir, n + ".details.jsonl")
        if "." in n or not os.path.exists(ref):
            continue
        rows = load(p)
        zs = logits(rows, t0, b0)
        o, nw = metrics(rows, zs, t0, b0), metrics(rows, zs, temp, by_opt)
        rr = load(ref)
        rc = json.load(open(os.path.join(a.checkpoint.replace("/train", ""), "..", "r21-soft", "train", "config.json"))) if False else None
        rf = json.load(open(os.path.join(a.ref_dir, n + ".json")))
        tag = "(fit)" if n in fit_names else ""
        out["sets"][n] = {"ref": [rf["nll"], rf["ece"]], "old": o, "new": nw}
        w = ("NLL+" if nw[0] > rf["nll"] + 1e-9 else "    ") + (" ECE+" if nw[1] > rf["ece"] + 1e-9 else "")
        print(f"{n:22s}{tag:6s} ref nll {rf['nll']:.3f} ece {rf['ece']:.3f} | old {o[0]:.3f} {o[1]:.3f} | new {nw[0]:.3f} {nw[1]:.3f} {w}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
