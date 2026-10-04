"""ScienceQA results split by what a question needs, for two models on laya-vision's ScienceQA set.

    uv run python scripts/sqa_slices.py runs/v6-flip-lr/eval runs/v7-sqa/eval \
        --train data/v4/scienceqa.jsonl [--out runs/v7-sqa/eval/compare/sqa_slices.json]

Slices (fixed by the questions, never by whether a model got them right):
  subject        natural / social / language science (ScienceQA's own label)
  needs_picture  questions whose text, options and hint are the same as another benchmark question's but
                 whose answer differs -- the text alone cannot answer them (393 questions)
  text_seen      the same question text, options and hint occur in the training file (any picture)
For each slice and model: accuracy with the real picture, with a mismatched picture, and their difference
(what the picture adds); and the paired difference B - A with a 95% interval resampling pictures.
Only reads per-question records (<eval dir>/bench_lv_scienceqa{,.mismatched}.details.jsonl).
"""
import argparse
import glob
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple


def read(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def text_key(s: dict) -> Tuple:
    q = s["questions"]["q"]
    return (q["instructions"].strip().lower(), tuple(sorted(o.lower() for o in q.get("criteria") or [])),
            s.get("state_text", "").strip().lower())


def gold(s: dict) -> str:
    g = s["gold"]["q"]["probabilities"]
    return max(g, key=g.get)


def slices(bench: List[dict], train: List[dict], subject: Dict[str, str]) -> Dict[str, set]:
    answers: Dict[Tuple, set] = defaultdict(set)
    for s in bench:
        answers[text_key(s)].add(gold(s))
    seen = {text_key(s) for s in train}
    out: Dict[str, set] = defaultdict(set)
    for s in bench:
        sid = s["id"]
        out["all"].add(sid)
        out["subject=" + subject.get(sid, "?")].add(sid)
        if len(answers[text_key(s)]) > 1:
            out["needs_picture"].add(sid)
        out["text_seen" if text_key(s) in seen else "text_unseen"].add(sid)
    return out


def paired(a: Dict[str, dict], b: Dict[str, dict], ids: set, resamples: int = 2000, seed: int = 0):
    by_pic: Dict[str, List[float]] = defaultdict(list)
    for i in ids:
        if i in a and i in b:
            by_pic[a[i]["image_id"]].append(float(b[i]["correct"]) - float(a[i]["correct"]))
    pics = list(by_pic.values())
    if not pics:
        return None
    sums, counts = [sum(p) for p in pics], [len(p) for p in pics]
    rng = random.Random(seed)
    boot = []
    for _ in range(resamples):
        idx = [rng.randrange(len(pics)) for _ in pics]
        boot.append(sum(sums[i] for i in idx) / sum(counts[i] for i in idx))
    boot.sort()
    return sum(sums) / sum(counts), [boot[int(0.025 * resamples)], boot[min(resamples - 1, int(0.975 * resamples))]]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("eval_a")
    p.add_argument("eval_b")
    p.add_argument("--train", required=True, help="the ScienceQA training file of model B")
    p.add_argument("--bench", default="data/lv_bench/scienceqa.jsonl")
    p.add_argument("--root", default="data")
    p.add_argument("--out")
    a = p.parse_args(argv)
    import pyarrow.parquet as pq
    val = glob.glob(os.path.join(a.root, "raw", "hf", "datasets--derek-thomas--ScienceQA", "snapshots", "*", "data",
                                 "validation-*.parquet"))[0]
    subject = {"lv-scienceqa:val-%d" % i: r["subject"] for i, r in enumerate(pq.read_table(val, columns=["subject"]).to_pylist())}
    groups = slices(read(a.bench), read(a.train), subject)
    recs = {}
    for name, d in (("a", a.eval_a), ("b", a.eval_b)):
        for ctl in ("", ".mismatched"):
            path = os.path.join(d, "bench_lv_scienceqa%s.details.jsonl" % ctl)
            recs[name + ctl] = {r["sample_id"]: r for r in read(path)}
    acc = lambda key, ids: sum(float(recs[key][i]["correct"]) for i in ids if i in recs[key]) / max(1, sum(i in recs[key] for i in ids))
    result = {}
    print("%-24s %5s | %-22s | %-22s | %s" % ("slice", "n", "A real / mism / gain", "B real / mism / gain", "B - A [95%]"))
    for g in sorted(groups, key=lambda g: (g != "all", g)):
        ids = groups[g]
        row = {"n": len(ids)}
        for m in ("a", "b"):
            real, mism = acc(m, ids), acc(m + ".mismatched", ids)
            row[m] = {"real": real, "mismatched": mism, "picture_gain": real - mism}
        d = paired(recs["a"], recs["b"], ids)
        row["b_minus_a"] = d
        result[g] = row
        fmt = lambda r: "%.3f / %.3f / %+.3f" % (r["real"], r["mismatched"], r["picture_gain"])
        print("%-24s %5d | %-22s | %-22s | %s" % (g, len(ids), fmt(row["a"]), fmt(row["b"]),
                                                  "%+.3f [%+.3f, %+.3f]" % (d[0], *d[1]) if d else "-"))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
