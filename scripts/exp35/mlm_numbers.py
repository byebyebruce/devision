"""exp35 short diagnosis (critic EXP35-PREP-01; docs/experiments/2026-10-17-exp35-number-fill-plan.md): does the
real-layout alignment checkpoint fill in the drawn NUMBERS from the picture, or only the caption template? Read
only: no parameter update, no temperature, no download (the MLM head comes from the local Hugging Face cache, run
with HF_HUB_OFFLINE=1).

Data: the 600 development captions of data/real_layout/captions_tune.jsonl (already used in development; never
rl_confirm, the 75 object-property or the 80 ChartQA items), in file order: a family's two views (v0, v1) are
consecutive, so with the original in-batch rotation half of the wrong pictures are the item's own twin (205 with
changed numbers, 95 with the same numbers) and half another family (132 of them another form). The overall
real - wrong difference is therefore the gain from the picture under this fixed wrong picture, mixing numbers,
object attributes, layout and form; whether numbers are bound to their labels is read from the full-number,
different-values and twin readouts, and the number classes are also split by the kind of wrong picture
(critic EXP35-R1-02). START.json (written before the first forward) records the head actually read, the item
order and every wrong picture (EXP35-R1-01).

  1. original: align._evaluate's exact procedure (batch 16, max_caption_tokens 44, mask_prob 0.5, Random(1),
     pictures rotated by one inside each batch) -- the totals must reproduce the checkpoint's final real / shuffled
     accuracy (align_report.json) within a tolerance written in the plan; every masked token is classed as part of
     an asked number (mass / speed / distance value of Sample A / B or Pair 1 / 2) or other text, reported
     separately (n, accuracy, NLL, real - shuffled), and the two classes recombine to the totals;
  2. full numbers: in each caption every token of every number span (both objects) is masked and nothing else, so
     no number can be copied from the other object; real and the same rotated wrong picture, identical text and
     mask. Reported: token accuracy, each number with all of its tokens right, per field; both objects right where
     their values differ; both twin views right where the value changed between the views. The number of masked
     tokens in a span itself can hint at the value, so the real - wrong difference is the reading, not accuracy.

Number spans: the digits (with "," and ".") of "mass of each particle M u", "average particle speed V m/s",
"X <unit> apart" -- never the "1" / "2" of "Pair 1" or the unit. A token is part of a span when its character
offsets overlap it.

    HF_HUB_OFFLINE=1 uv run python scripts/exp35/mlm_numbers.py --checkpoint runs/exp32-align/stage1 \\
        --out runs/exp35-number-fill/report.json [--device mps]
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

DATA = "data/real_layout/captions_tune.jsonl"
HEAD_REPO = "answerdotai/ModernBERT-large"


def head_identity() -> Dict:
    """The MLM head actually read: the cached file (offline), its snapshot revision and sha256, and the three tensors
    _MLMHead.load_pretrained copies (dtype, shape, sha256 of the raw bytes). The decoder is tied to the input
    embeddings of the checkpoint being evaluated (its model.safetensors sha256 is recorded separately)."""
    import numpy as np
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(HEAD_REPO, "model.safetensors")
    tensors = {}
    with safe_open(path, "pt") as f:
        for name in ("head.dense.weight", "head.norm.weight", "decoder.bias"):
            t = f.get_tensor(name)
            raw = t.float().numpy() if str(t.dtype) == "torch.bfloat16" else t.numpy()
            tensors[name] = {"dtype": str(t.dtype), "shape": list(t.shape),
                             "sha256_float32": hashlib.sha256(np.ascontiguousarray(raw, dtype=np.float32).tobytes()).hexdigest()}
    parts = os.path.normpath(path).split(os.sep)
    revision = parts[parts.index("snapshots") + 1] if "snapshots" in parts else None
    return {"repo": HEAD_REPO, "file": path, "snapshot": revision, "sha256": _sha(path), "tensors": tensors,
            "decoder": "tied to the evaluated checkpoint's input embeddings"}


def pairing(samples: Sequence[Dict], batch: int) -> List[Dict]:
    """Per item, in file order: its id, the id whose picture the original evaluation shows instead (rotation by one
    inside each batch of `batch`), and what that wrong picture differs in: the twin with changed numbers, the twin
    with the same numbers, another family of the same form, or another form (critic EXP35-R1-02)."""
    out = []
    for b in range(0, len(samples), batch):
        idx = list(range(b, min(b + batch, len(samples))))
        for i, j in zip(idx, idx[1:] + idx[:1]):
            a, w = samples[i], samples[j]
            if a["family"] == w["family"]:
                kind = "twin, numbers changed" if [x["text"] for x in number_spans(a["captions"][0])] != \
                    [x["text"] for x in number_spans(w["captions"][0])] else "twin, numbers unchanged"
            else:
                kind = "other family, same form" if a.get("source") == w.get("source") else "other family, other form"
            out.append({"id": a["id"], "wrong_picture_id": w["id"], "wrong_picture": kind})
    return out
SPAN_PATTERNS = [("mass", re.compile(r"mass of each particle ([\d,]+(?:\.\d+)?) u")),
                 ("speed", re.compile(r"average particle speed ([\d,]+(?:\.\d+)?) m/s")),
                 ("distance", re.compile(r"(\d[\d,]*(?:\.\d+)?) (?:in|cm|mm) apart"))]
OBJECT = re.compile(r"(Sample [AB]|Pair [12]):")


def number_spans(caption: str) -> List[Dict]:
    """[{field, object, start, end, text}] of the asked numbers, in caption order."""
    heads = [(m.start(), m.group(1)) for m in OBJECT.finditer(caption)]
    out = []
    for field, pat in SPAN_PATTERNS:
        for m in pat.finditer(caption):
            obj = [h for s, h in heads if s <= m.start(1)]
            out.append({"field": field, "object": obj[-1] if obj else "", "start": m.start(1), "end": m.end(1),
                        "text": m.group(1)})
    return sorted(out, key=lambda x: x["start"])


def token_spans(offsets: Sequence[Tuple[int, int]], spans: List[Dict]) -> List[Optional[int]]:
    """For each token, the index of the number span its characters overlap, else None."""
    out: List[Optional[int]] = []
    for a, b in offsets:
        hit = [k for k, s in enumerate(spans) if a < s["end"] and b > s["start"] and b > a]
        out.append(hit[0] if hit else None)
    return out


def encode(tok, caption: str, max_tokens: int):
    enc = tok(caption, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"][:max_tokens], enc["offset_mapping"][:max_tokens]
    return ids, offs


class Acc:
    def __init__(self):
        self.n, self.hit, self.nll = 0, 0, 0.0

    def add(self, hit: bool, nll: float):
        self.n += 1
        self.hit += int(hit)
        self.nll += nll

    def out(self) -> Dict:
        return {"n": self.n, "accuracy": self.hit / self.n if self.n else None, "nll": self.nll / self.n if self.n else None}


def recombine(parts: Sequence[Dict]) -> Tuple[int, float, float]:
    """(n, accuracy, nll) of the union of disjoint parts -- must equal the totals."""
    n = sum(p["n"] for p in parts)
    return n, sum(p["accuracy"] * p["n"] for p in parts if p["n"]) / n, sum(p["nll"] * p["n"] for p in parts if p["n"]) / n


def run(checkpoint: str, out: str, device: str = "mps", data: str = DATA) -> Dict:
    import torch
    import torch.nn.functional as F

    from devision.infer import Decider
    from devision.model import text_encoder
    from devision.train.align import AlignConfig, _masked_batch, _mlm_logits, _MLMHead
    from devision.train.rlcd import _Images
    from devision.train.samples import read_jsonl

    started = time.time()
    c = AlignConfig(batch=16, max_caption_tokens=44, mask_prob=0.5)
    decider = Decider.load(checkpoint, device=device)
    model, tok = decider.model, decider.tok
    head = _MLMHead(text_encoder(model.decision))
    head.load_pretrained(HEAD_REPO)
    dev = next(model.parameters()).device
    model.eval()
    head.to(dev).eval()
    images = _Images("data", decider)
    samples = [s for s in read_jsonl(data) if s["captions"]]
    report: Dict = {"checkpoint": checkpoint, "data": data, "rows": len(samples), "device": str(dev),
                    "sha256": {"model.safetensors": _sha(os.path.join(checkpoint, "model.safetensors")), "data": _sha(data)},
                    "settings": {"batch": c.batch, "max_caption_tokens": c.max_caption_tokens, "mask_prob": c.mask_prob,
                                 "rng": "Random(1) per mode", "wrong_picture": "next item in the batch (rotation by one)"}}
    pairs = pairing(samples, c.batch)
    report["order_and_wrong_pictures"] = pairs
    report["wrong_picture_kinds"] = dict(sorted(Counter(p["wrong_picture"] for p in pairs).items()))
    report["head"] = head_identity()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(report, open(os.path.join(os.path.dirname(out) or ".", "START.json"), "w"), indent=1)   # before any forward
    kind_of = [p["wrong_picture"] for p in pairs]

    # 1. the original evaluation, token classes added
    orig: Dict = {}
    for mode in ("real", "shuffled"):
        rng = random.Random(1)
        acc = defaultdict(Acc)
        with torch.no_grad():
            for b in range(0, len(samples), c.batch):
                chunk = samples[b:b + c.batch]
                caps = [rng.choice(s["captions"]) for s in chunk]
                pix_src = chunk if mode == "real" else chunk[1:] + chunk[:1]
                input_ids, attention, target = _masked_batch(tok, caps, c, rng)
                logits = _mlm_logits(model, head, input_ids, attention, torch.stack([images(s["image"]) for s in pix_src]), dev)
                y = target[:, 1:].to(dev)
                logp = torch.log_softmax(logits, -1)
                for i, cap in enumerate(caps):
                    spans = number_spans(cap)
                    _, offs = encode(tok, cap, c.max_caption_tokens)
                    which = token_spans(offs, spans)
                    for t in range(len(offs)):
                        if int(y[i, t]) == -100:
                            continue
                        hit = int(logits[i, t].argmax()) == int(y[i, t])
                        nll = -float(logp[i, t, y[i, t]])
                        acc["all"].add(hit, nll)
                        k = which[t]
                        acc["number" if k is not None else "other"].add(hit, nll)
                        if k is not None:
                            acc["number:" + spans[k]["field"]].add(hit, nll)
                            acc["number@" + kind_of[b + i]].add(hit, nll)
        orig[mode] = {k: v.out() for k, v in sorted(acc.items())}
    for k in orig["real"]:
        r, s = orig["real"][k]["accuracy"], orig["shuffled"].get(k, {}).get("accuracy")
        orig["real"][k]["real_minus_shuffled"] = (r - s) if r is not None and s is not None else None
    report["original"] = orig
    report["recombines"] = {}
    for mode in ("real", "shuffled"):
        n, a, l = recombine([orig[mode]["number"], orig[mode]["other"]])
        report["recombines"][mode] = abs(a - orig[mode]["all"]["accuracy"]) < 1e-12 and n == orig[mode]["all"]["n"] \
            and abs(l - orig[mode]["all"]["nll"]) < 1e-9
    final = json.load(open(os.path.join(checkpoint, "align_report.json")))["val"][-1]
    report["reproduction"] = {"report_real": final["val/mlm_acc_real"], "report_shuffled": final["val/mlm_acc_shuffled"],
                              "here_real": orig["real"]["all"]["accuracy"], "here_shuffled": orig["shuffled"]["all"]["accuracy"]}

    # 2. every number token masked, nothing else
    full: Dict = {}
    per_number: Dict[str, Dict[Tuple[int, int], bool]] = {}
    for mode in ("real", "wrong"):
        acc = defaultdict(Acc)
        right: Dict[Tuple[int, int], bool] = {}
        with torch.no_grad():
            for b in range(0, len(samples), c.batch):
                idx = list(range(b, min(b + c.batch, len(samples))))
                chunk = [samples[i] for i in idx]
                caps = [s["captions"][0] for s in chunk]
                pix_src = chunk if mode == "real" else chunk[1:] + chunk[:1]
                seqs, labels, whiches = [], [], []
                for cap in caps:
                    ids, offs = encode(tok, cap, c.max_caption_tokens)
                    which = token_spans(offs, number_spans(cap))
                    seqs.append([tok.cls_token_id] + [tok.mask_token_id if w is not None else t for t, w in zip(ids, which)] + [tok.sep_token_id])
                    labels.append([-100] + [t if w is not None else -100 for t, w in zip(ids, which)] + [-100])
                    whiches.append(which)
                L = max(map(len, seqs))
                input_ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
                attention = torch.zeros((len(seqs), L), dtype=torch.long)
                target = torch.full((len(seqs), L), -100, dtype=torch.long)
                for i, (sq, lab) in enumerate(zip(seqs, labels)):
                    input_ids[i, :len(sq)], attention[i, :len(sq)], target[i, :len(lab)] = torch.tensor(sq), 1, torch.tensor(lab)
                logits = _mlm_logits(model, head, input_ids, attention, torch.stack([images(s["image"]) for s in pix_src]), dev)
                y = target[:, 1:].to(dev)
                logp = torch.log_softmax(logits, -1)
                for i, cap in enumerate(caps):
                    spans = number_spans(cap)
                    ok = [True] * len(spans)
                    for t, k in enumerate(whiches[i]):
                        if k is None:
                            continue
                        hit = int(logits[i, t].argmax()) == int(y[i, t])
                        acc["token:" + spans[k]["field"]].add(hit, -float(logp[i, t, y[i, t]]))
                        ok[k] = ok[k] and hit
                    for k, s in enumerate(spans):
                        acc["number:" + s["field"]].add(ok[k], 0.0)
                        acc["number@" + kind_of[idx[i]]].add(ok[k], 0.0)
                        right[(idx[i], k)] = ok[k]
        full[mode] = {k: {"n": v.n, "accuracy": v.out()["accuracy"]} | ({"nll": v.out()["nll"]} if k.startswith("token") else {})
                      for k, v in sorted(acc.items())}
        per_number[mode] = right
    report["full_numbers"] = full
    report["full_numbers_binding"] = binding(samples, per_number)
    report["seconds"] = round(time.time() - started, 1)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)
    return report


def binding(samples: Sequence[Dict], right: Dict[str, Dict[Tuple[int, int], bool]]) -> Dict:
    """Both objects right where their values differ (per picture and field); both twin views right where the value
    changed between the views (per family, field and object)."""
    spans = [number_spans(s["captions"][0]) for s in samples]
    out: Dict = {}
    for mode, r in right.items():
        diff = defaultdict(lambda: [0, 0])
        for i, sp in enumerate(spans):
            by_field = defaultdict(list)
            for k, s in enumerate(sp):
                by_field[s["field"]].append((k, s["text"]))
            for f, ks in by_field.items():
                if len(ks) == 2 and ks[0][1] != ks[1][1]:
                    diff[f][1] += 1
                    diff[f][0] += int(r[(i, ks[0][0])] and r[(i, ks[1][0])])
        twin = defaultdict(lambda: [0, 0])
        fam = defaultdict(list)
        for i, s in enumerate(samples):
            fam[s["family"]].append(i)
        for f, views in fam.items():
            if len(views) != 2:
                continue
            a, b = views
            va = {(s["field"], s["object"]): (k, s["text"]) for k, s in enumerate(spans[a])}
            vb = {(s["field"], s["object"]): (k, s["text"]) for k, s in enumerate(spans[b])}
            for key in va.keys() & vb.keys():
                if va[key][1] != vb[key][1]:
                    twin[key[0]][1] += 1
                    twin[key[0]][0] += int(r[(a, va[key][0])] and r[(b, vb[key][0])])
        out[mode] = {"both_objects_right_when_different": {f: {"right": x, "of": n} for f, (x, n) in sorted(diff.items())},
                     "both_twin_views_right_when_changed": {f: {"right": x, "of": n} for f, (x, n) in sorted(twin.items())}}
    return out


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="runs/exp32-align/stage1")
    ap.add_argument("--out", default="runs/exp35-number-fill/report.json")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args(argv)
    r = run(a.checkpoint, a.out, a.device)
    o = r["original"]
    print("reproduction: report real %.7f shuffled %.7f | here real %.7f shuffled %.7f; recombines %s" % (
        r["reproduction"]["report_real"], r["reproduction"]["report_shuffled"], r["reproduction"]["here_real"],
        r["reproduction"]["here_shuffled"], r["recombines"]))
    for k in o["real"]:
        x, y = o["real"][k], o["shuffled"].get(k, {})
        print("original %-16s n %5d real %.3f shuffled %.3f diff %+.3f" % (k, x["n"], x["accuracy"], y.get("accuracy", math.nan), x["real_minus_shuffled"]))
    for k in r["full_numbers"]["real"]:
        x, y = r["full_numbers"]["real"][k], r["full_numbers"]["wrong"][k]
        print("full numbers %-16s n %5d real %.3f wrong %.3f diff %+.3f" % (k, x["n"], x["accuracy"], y["accuracy"], x["accuracy"] - y["accuracy"]))
    print(json.dumps(r["full_numbers_binding"], indent=1))
    print("wrong pictures: %s; %.0f s" % (r["wrong_picture_kinds"], r["seconds"]))


if __name__ == "__main__":
    sys.exit(main())
