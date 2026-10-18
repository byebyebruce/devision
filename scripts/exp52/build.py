"""exp52 number sidecar (critic EXP52-PREP-01), model-free: for every question of data/exp42/train.jsonl (the 576 updates'
4,608 items; every one asks a drawn value) the gold answer's number with the thousands commas removed and the unit
dropped ("1,360 m/s" -> "1360"; scripts/exp49/build.target_of), keyed by row id and qid. A training target only: the
number reader never sees it; its characters must fit the reader's 5 slots (4 + EOS). Writes data/exp52/numbers.jsonl and
MANIFEST.json (sources, counts, target lengths, sha256).

    uv run python scripts/exp52/build.py --root data
"""
import argparse
import hashlib
import importlib.util
import json
import os
from collections import Counter

HERE = os.path.dirname(__file__)
_s = importlib.util.spec_from_file_location("exp49_build", os.path.join(HERE, "..", "exp49", "build.py"))
assert _s is not None and _s.loader is not None
b49 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(b49)


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def build(root: str) -> dict:
    src = os.path.join(root, "exp42", "train.jsonl")
    out = []
    for line in open(src):
        r = json.loads(line)
        for q in r["questions"]:
            gold = next(o for o, p in r["gold"][q]["probabilities"].items() if p == 1.0)
            t = b49.target_of(gold)
            if len(t) > 4:
                raise SystemExit("%s %s: target %r longer than 4 characters" % (r["id"], q, t))
            out.append({"id": r["id"], "qid": q, "target": t})
    dest = os.path.join(root, "exp52")
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, "numbers.jsonl")
    with open(path, "w") as fh:
        fh.write("".join(json.dumps(x) + "\n" for x in out))
    m = {"source": {"train": sha(src)}, "questions": len(out), "unique_keys": len({(x["id"], x["qid"]) for x in out}),
         "target_lengths": dict(sorted(Counter(len(x["target"]) for x in out).items())), "sha256": sha(path)}
    json.dump(m, open(os.path.join(dest, "MANIFEST.json"), "w"), indent=1)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    print(json.dumps(build(ap.parse_args().root), indent=1))
