"""exp70 CPU data (critic EXP70-PREP-01; docs/experiments/2026-10-19-exp70-intermediate-plan.md). No model weights (the
truncation check uses EXP68-X's tokenizer / config only).

One question: with EXP68-X frozen, does showing each object's combined quantity K = m * v^2 (computed exactly from the
four numbers already in the input table) change the temperature comparison, compared with the raw table alone?

  raw  the input exactly as EXP68 saw it (the hint + the four-number table).
  K    the same input plus, after the table, a fixed block -- one header line and one line per object in the table's
       order (A then B, never sorted, no comparison, no answer, no unit):
           Computed from the values above (mass times speed squared, no unit):
           Sample A: mass times speed squared = 66,567,600
           Sample B: mass times speed squared = 14,810,400
       K is computed from the row's own state numbers by exact integer arithmetic; gold, option order and answers are
       never read to build it (the label agreement is only checked afterwards).
The question and its options are unchanged. Sets (data/exp70/):
  diag68_K.jsonl   the 480 requests of data/exp68/diag.jsonl with the K block (the raw results are EXP68's registered
                   x_diag68 records, reused).
  fid_raw.jsonl    48 of those requests (8 per class, by sha256("exp70-fid:" + id)), raw, to re-decide as a fidelity
                   check against the registered records.
  train_raw.jsonl / train_K.jsonl   a training slice fixed now by hash ("exp70-train:" + family): 10 temperature families
                   per class of data/exp68/train.jsonl (60 families, both views = 120 rows), the comparison question only;
                   an end-of-training fit check of EXP68-X on its own combinations (not the old facts fit).
Checks: K from the state equals K from the row's gold readings; K's order decides the gold (every row); the options and
question are the source's; no state token is dropped and options stay distinct under EXP68-X's tokenizer.

    uv run python scripts/exp70/build.py --root data
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, cast

HERE = os.path.dirname(__file__)
X_RUN = "runs/exp68-x"
K_HEADER = "Computed from the values above (mass times speed squared, no unit):"
K_QUANTITY = "mass times speed squared"
PER_CLASS_TRAIN, PER_CLASS_FID = 10, 8


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b68 = _load("exp68_build", "../exp68/build.py")
b65, fb = b68.b65, b68.fb
CLASSES = b68.CLASSES


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def k_block(state_text: str) -> str:
    """The K block for a state: K = m * v^2 per object from the state's own table, exact integers, A then B."""
    a, b = b65.state_values(state_text)
    ks = []
    for p in (a, b):
        m, v = p
        if m != int(m) or v != int(v):
            raise SystemExit("non-integer table value %s" % (p,))
        ks.append(int(m) * int(v) * int(v))
    return "%s\nSample A: %s = %s\nSample B: %s = %s" % (K_HEADER, K_QUANTITY, fb.fmt(str(ks[0])), K_QUANTITY, fb.fmt(str(ks[1])))


def with_k(r: Dict) -> Dict:
    return dict(r, state_text=r["state_text"] + "\n" + k_block(r["state_text"]))


def k_values(state_text: str) -> List[int]:
    lines = state_text.split("\n")
    i = lines.index(K_HEADER)
    return [int(x.rsplit(" = ", 1)[1].replace(",", "")) for x in lines[i + 1:i + 3]]


def check(r: Dict, from_gold=None) -> None:
    """K from the state matches the row's own numbers (and its gold readings, when given); K's order is the gold."""
    a, b = b65.state_values(r["state_text"].split("\n" + K_HEADER)[0])
    ka, kb = k_values(r["state_text"])
    if [ka, kb] != [int(a[0]) * int(a[1]) ** 2, int(b[0]) * int(b[1]) ** 2]:
        raise SystemExit("%s: K does not match the table" % r["id"])
    if from_gold is not None and from_gold != (a, b):
        raise SystemExit("%s: the table is not the gold readings" % r["id"])
    q = next(qid for qid, x in r["questions"].items() if x["instructions"] not in fb.TEMPLATES)
    g = next(o for o, p in r["gold"][q]["probabilities"].items() if p == 1.0)
    want = "sample A" if ka > kb else "sample B" if kb > ka else "neither"
    if not (g.startswith("neither") if want == "neither" else g == want):
        raise SystemExit("%s: K's order %s, gold %s" % (r["id"], want, g))


def comparison_only(r: Dict) -> Dict:
    q = next(qid for qid, x in r["questions"].items() if x["instructions"] not in fb.TEMPLATES)
    return dict(r, questions={q: r["questions"][q]}, gold={q: r["gold"][q]})


def build(root: str = "data") -> Dict:
    d68 = os.path.join(root, "exp68")
    m68 = json.load(open(os.path.join(d68, "MANIFEST.json")))
    for f, hh in m68["sha256"].items():
        if sha(os.path.join(d68, f)) != hh:
            raise SystemExit("data/exp68/%s differs from its MANIFEST" % f)
    diag = fb.jsonl(os.path.join(d68, "diag.jsonl"))
    diag_k = [with_k(r) for r in diag]
    for r in diag_k:
        check(r)
    fid = []
    for c in CLASSES:
        fid += sorted((r for r in diag if r["axis"] == c), key=lambda r: h("exp70-fid:" + r["id"]))[:PER_CLASS_FID]
    temp = [r for r in fb.jsonl(os.path.join(d68, "train.jsonl")) if r["kind"] == "temp"]
    fams: Dict[str, List[str]] = {c: sorted({r["family"] for r in temp if r["axis"] == c}, key=lambda f: h("exp70-train:" + f))[:PER_CLASS_TRAIN] for c in CLASSES}
    pick = {f for v in fams.values() for f in v}
    train_raw = [comparison_only(r) for r in temp if r["family"] in pick]
    train_k = [with_k(r) for r in train_raw]
    for r, raw in zip(train_k, train_raw):
        check(r, b65.values_of([x for x in temp if x["id"] == raw["id"]][0]))
    dest = os.path.join(root, "exp70")
    os.makedirs(dest, exist_ok=True)
    files = {"diag68_K.jsonl": diag_k, "fid_raw.jsonl": fid, "train_raw.jsonl": train_raw, "train_K.jsonl": train_k}
    for n, rows in files.items():
        with open(os.path.join(dest, n), "w") as fh:
            fh.write("".join(json.dumps(x) + "\n" for x in rows))
    m = {"requests": {n: sum(len(r["questions"]) for r in rows) for n, rows in files.items()},
         "new_requests": sum(sum(len(r["questions"]) for r in rows) for rows in files.values()),
         "per_class": {n: dict(Counter(r["axis"] for r in rows)) for n, rows in files.items()},
         "k_block": {"header": K_HEADER, "quantity": K_QUANTITY, "order": "Sample A then Sample B (the table's), never sorted",
                     "format": "exp55 fmt (thousands separators), exact integers"},
         "max_k": max(max(k_values(r["state_text"])) for r in diag_k + train_k),
         "sources": {"exp68_manifest": sha(os.path.join(d68, "MANIFEST.json")), **{"exp68_" + f: hh for f, hh in m68["sha256"].items()}},
         "sha256": {n: sha(os.path.join(dest, n)) for n in files}}
    json.dump(m, open(os.path.join(dest, "MANIFEST.json"), "w"), indent=1)
    return m


def truncation(root: str = "data") -> Dict:
    """EXP68-X's tokenizer / config (no weights): no state token dropped, options distinct, per file."""
    from types import SimpleNamespace

    from transformers import AutoTokenizer

    from devision.model import ModelConfig
    st = os.path.join(X_RUN, "stage2")
    cfg = json.load(open(os.path.join(st, "config.json")))
    d = SimpleNamespace(tok=AutoTokenizer.from_pretrained(os.path.join(st, "tokenizer")),
                        cfg=ModelConfig(**{k: v for k, v in cfg.items() if k in ModelConfig.__dataclass_fields__}), model=None)
    out = {n: fb.truncation(cast(Any, d), os.path.join(root, "exp70", n)) for n in ("diag68_K.jsonl", "fid_raw.jsonl", "train_raw.jsonl", "train_K.jsonl")}
    if any(v["state_dropped"] or v["options_not_distinct"] for v in out.values()):
        raise SystemExit("truncation: %s" % out)
    m = json.load(open(os.path.join(root, "exp70", "MANIFEST.json")))
    m["truncation"] = out
    json.dump(m, open(os.path.join(root, "exp70", "MANIFEST.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    a = ap.parse_args()
    print(json.dumps(build(a.root), indent=1))
    print(json.dumps(truncation(a.root), indent=1))
    sys.exit(0)
