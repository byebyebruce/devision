"""PANEL-LEARN-PREP-01 data (docs/experiments/2026-10-19-panel-learn-plan.md). CPU only.

D1 (binding alignment) uses data/exp26/align_train.jsonl / align_val.jsonl as they are (EXP26's panel captions, training
families; audited here). D2 (read -> decision with protection) is a fixed slice written to data/panel_learn/:
  - panel rows: every row of 1,000 exp25 training families (2 value views each, readA / readB / cmp), always including
    the 200 families of the fit set data/exp22/fit.jsonl, "input": "picture";
  - text comparison rows: 600 exp25 "exp21-syn-text" rows whose (erased) picture is not byte-identical to any picture
    of the development / confirmation sets (EXP25's text rows share 86 erased pictures with dev_syn_erased: dropped);
  - replay: 1,600 rows of PROTECT's data/protect2/train_mix.jsonl, as PROTECT marked them -- 1,000 picture rows
    (proportional by source) and 600 text / K rows (proportional temperature : magnets).
Audits (refuse to write on any failure): no training picture byte-identical to a picture of dev16 / dev_syn /
dev_syn_erased / the frozen confirmations (v04_confirm, v04_chartqa, v04_k_confirm) / the full-evaluation sets; no exp25
family of dev16; every row has "input" and "group"; the fit families are all present. Writes MANIFEST.json with counts,
sources, the picture-identity audit and the input files' sha256.

    uv run python scripts/panel_learn/build.py --root data
"""
import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Set

SEED = 4
PANEL_FAMILIES, TEXT_ROWS, REPLAY_PICTURE, REPLAY_TEXT = 1000, 600, 1000, 600
EXP25, FIT, DEV16 = "exp25/train.jsonl", "exp22/fit.jsonl", "exp22/dev16.jsonl"
REPLAY = "protect2/train_mix.jsonl"
HELD = ["exp22/dev16.jsonl", "exp21/dev_syn.jsonl", "exp21/dev_syn_erased.jsonl", "v04_confirm/confirm.jsonl",
        "v04_confirm/confirm_mism.jsonl", "v04_chartqa/chartqa.jsonl", "v04_chartqa/chartqa_mism.jsonl",
        "v04_k_confirm/confirm.jsonl", "v04_k_confirm/confirm_mism.jsonl"]


def rows(path: str) -> List[Dict]:
    return [json.loads(x) for x in open(path) if x.strip()]


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class Pictures:
    def __init__(self, root: str):
        self.root, self.cache = root, {}

    def __call__(self, image: str):
        if image.startswith("text-only"):
            return None
        p = os.path.join(self.root, image)
        if p not in self.cache:
            self.cache[p] = sha(p) if os.path.exists(p) else None
        return self.cache[p]


def held_pictures(root: str, pic: Pictures, extra: List[str]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = defaultdict(set)
    for rel in HELD + extra:
        for r in rows(os.path.join(root, rel)):
            h = pic(r["image"])
            if h:
                out[h].add(rel)
    return out


def full_eval_sets(config: str) -> List[str]:
    import yaml
    ev = yaml.safe_load(open(config))["evaluate"]
    return [(v["path"] if isinstance(v, dict) else v)[len("data/"):] for v in ev["sets"].values()]


def build(root: str, config: str) -> Dict:
    rng = random.Random(SEED)
    pic = Pictures(root)
    held = held_pictures(root, pic, [s for s in full_eval_sets(config) if os.path.exists(os.path.join(root, s))])
    e25 = rows(os.path.join(root, EXP25))
    fit_fams = {r["id"].split(":")[1] for r in rows(os.path.join(root, FIT))}
    dev_fams = {r["id"].split(":")[1] for r in rows(os.path.join(root, DEV16))}
    syn = defaultdict(list)
    for r in e25:
        if r["source"] == "exp21-syn":
            syn[r["family"]].append(r)
    others = sorted(set(syn) - fit_fams)
    rng.shuffle(others)
    fams = sorted(fit_fams) + sorted(others[:PANEL_FAMILIES - len(fit_fams)])
    panel = [dict(r, input="picture", group=r["family"]) for f in fams for r in syn[f]]
    text_ok = [r for r in e25 if r["source"] == "exp21-syn-text" and pic(r["image"]) not in held]
    dropped_text = sum(1 for r in e25 if r["source"] == "exp21-syn-text") - len(text_ok)
    rng.shuffle(text_ok)
    text = [dict(r, input="picture", group=r.get("family") or r["id"]) for r in text_ok[:TEXT_ROWS]]
    rep = rows(os.path.join(root, REPLAY))
    replay = []
    for mode, want in (("picture", REPLAY_PICTURE), ("text", REPLAY_TEXT)):
        pool = defaultdict(list)
        for r in rep:
            if r["input"] == mode:
                pool[r["source"]].append(r)
        total = sum(len(v) for v in pool.values())
        for src in sorted(pool):
            k = round(want * len(pool[src]) / total)
            replay += rng.sample(pool[src], min(k, len(pool[src])))
    train = panel + text + replay
    rng.shuffle(train)
    problems = []
    if not fit_fams <= set(fams):
        problems.append("fit families missing")
    if set(fams) & dev_fams:
        problems.append("dev16 families in training")
    bad = [(r["id"], sorted(held[h])) for r in train for h in [pic(r["image"])] if h and h in held]
    if bad:
        problems.append("training pictures identical to held-out pictures: %s" % bad[:5])
    if any("input" not in r or "group" not in r for r in train):
        problems.append("rows without input / group")
    if problems:
        raise SystemExit("refused:\n  " + "\n  ".join(problems))
    out = os.path.join(root, "panel_learn")
    if os.path.exists(out):
        raise SystemExit("%s exists (fresh only)" % out)
    os.makedirs(out)
    with open(os.path.join(out, "train.jsonl"), "w") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in train))
    man = {"seed": SEED, "rows": len(train), "questions": sum(len(r["questions"]) for r in train),
           "parts": {"panel": {"families": len(fams), "fit_families": len(fit_fams), "rows": len(panel)},
                     "text_comparison": {"rows": len(text), "dropped_for_picture_identity": dropped_text},
                     "replay": {"rows": len(replay), "by": dict(Counter("%s|%s" % (r["input"], r["source"]) for r in replay))}},
           "input": dict(Counter(r["input"] for r in train)),
           "audit": {"held_pictures": len(held), "held_files": HELD, "full_eval_sets": len(full_eval_sets(config)),
                     "training_pictures": len({pic(r["image"]) for r in train} - {None}), "identical_to_held": 0, "dev16_families": 0},
           "inputs": {rel: sha(os.path.join(root, rel)) for rel in (EXP25, FIT, DEV16, REPLAY)},
           "train_sha256": sha(os.path.join(out, "train.jsonl"))}
    json.dump(man, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--config", default="configs/v04-full-eval.yaml")
    a = ap.parse_args(argv)
    print(json.dumps(build(a.root, a.config), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
