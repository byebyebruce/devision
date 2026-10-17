"""exp41 readout (critic EXP41-PREP-01; rules written before any run; an ORACLE diagnosis, not a candidate result).
From devision-eval of exp40-M's frozen endpoint on data/exp41/questions.jsonl (runs/exp41/m.details.jsonl) and, for
the original pictures, M's own frozen rl_tune records (the same questions; not re-run):

  - identity: M's weights / config sha256 now equal runs/exp40-m/FROZEN.json, the exp41 summary names that
    checkpoint with fitted temperatures, every exp41 question answered exactly once, and M's rl_tune holds every
    reading question of the chosen families;
  - per condition (original / keep / zoom / other), form and field: accuracy on same-value and on different-value
    pairs (the two objects of one view and field), both objects right on different-value pairs, and wrong answers
    equal to the other object's gold; for "other" (the asked number is not in the picture) also the share of
    different-value questions answered with the shown (other) object's gold;
  - paired by family (seed 0, 2,000 resamples, a family's views, objects and fields together): different-value
    accuracy and both-objects-right, keep - original, zoom - original, zoom - keep.

Reading, pre-written and conditional (one diagnosis does not establish a cause):
  - keep - original lower bound > 0 and zoom - keep not: supports interference from the other object;
  - zoom - keep lower bound > 0 and keep - original not: supports a readability limit;
  - both: both contribute; neither, with keep and zoom different-value accuracy still < 0.5: next training should
    first target direct number decision reading / its optimisation, not a claim that the architecture cannot;
  - force zoom stays below exp37's validated 16 px (MANIFEST digit_ink_256), so a null zoom result on force does
    not rule out readability; same-value pairs are not informative for "other" (the shown number is the gold).

    uv run python scripts/exp41/report.py
"""
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List

HERE = os.path.dirname(__file__)
M_RUN = "runs/exp40-m"
OUT = "runs/exp41"
DATA = "data/exp41/questions.jsonl"
TUNE = "data/real_layout/questions_tune.jsonl"
CONDITIONS = ("original", "keep", "zoom", "other")


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: str) -> List[Dict]:
    return [json.loads(x) for x in open(path) if x.strip()]


def check_identity(m_run: str = M_RUN, out: str = OUT) -> Dict:
    fr = json.load(open(os.path.join(m_run, "FROZEN.json")))["weights"]
    now = {f: sha(os.path.join(m_run, "stage2", f)) for f in fr}
    if now != fr:
        raise SystemExit("exp40-M's weights / config are not the frozen ones")
    info = json.load(open(os.path.join(out, "m.json")))["run"]
    if os.path.normpath(info.get("checkpoint", "")) != os.path.normpath(os.path.join(m_run, "stage2")) or info.get("temperature") != "fitted":
        raise SystemExit("exp41 was not evaluated on exp40-M's frozen checkpoint with its fitted temperatures")
    return fr


def records(data: List[Dict], got: List[Dict], tune_got: List[Dict]) -> List[Dict]:
    """One record per (condition, orig picture, object, qid) with field, gold, prediction, right, family."""
    want = {(r["id"], q): r for r in data for q in r["questions"]}
    have: Dict = {}
    for g in got:
        k = (g["sample_id"], g["qid"])
        if k in have or k not in want:
            raise SystemExit("exp41 details: repeated or unknown question %s" % (k,))
        have[k] = g
    if set(have) != set(want):
        raise SystemExit("exp41 details: %d questions missing" % len(set(want) - set(have)))
    tune = {(g["sample_id"], g["qid"]): g for g in tune_got}
    out = []
    for (sid, q), r in want.items():
        f = ("mass" if "mass" in r["questions"][q]["instructions"] else "speed" if "speed" in r["questions"][q]["instructions"]
             else "distance")
        base = {"family": r["family"], "kind": r["kind"], "orig_id": r["orig_id"], "object": r["object"], "field": f}
        for cond, g in ((r["condition"], have[(sid, q)]),) + ((("original", tune[(r["orig_id"], q)]),) if r["condition"] == "keep" else ()):
            if g is None or g["gold_answer"] != next(o for o, p in r["gold"][q]["probabilities"].items() if p == 1.0):
                raise SystemExit("gold mismatch at %s %s" % (sid, q))
            out.append(dict(base, condition=cond, gold=g["gold_answer"], prediction=g["prediction"], right=bool(g["correct"])))
    return out


def units(recs: List[Dict]) -> Dict:
    """family -> statistic -> [sum, count]."""
    pairs = defaultdict(list)
    for r in recs:
        pairs[(r["condition"], r["orig_id"], r["field"])].append(r)
    fam: Dict = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for (cond, _, field), v in pairs.items():
        if len(v) != 2:
            raise SystemExit("not two objects for %s %s" % (cond, field))
        a, b = v
        strat = "same" if a["gold"] == b["gold"] else "different"
        for x, y in ((a, b), (b, a)):
            for key in ("%s:%s" % (cond, strat), "%s:%s:%s" % (cond, strat, field), "%s:%s:%s" % (cond, strat, x["kind"])):
                u = fam[x["family"]][key]
                u[0] += x["right"]
                u[1] += 1
            if strat == "different":
                u = fam[x["family"]]["%s:shown_other" % cond]
                u[0] += x["prediction"] == y["gold"]
                u[1] += 1
                if not x["right"]:
                    u = fam[x["family"]]["%s:wrong_is_other" % cond]
                    u[0] += x["prediction"] == y["gold"]
                    u[1] += 1
        if strat == "different":
            u = fam[a["family"]]["%s:both" % cond]
            u[0] += a["right"] and b["right"]
            u[1] += 1
    return fam


def value(fam, families, stat) -> float:
    n = sum(fam[f][stat][0] for f in families if stat in fam[f])
    d = sum(fam[f][stat][1] for f in families if stat in fam[f])
    return n / d if d else float("nan")


def interval(fn, families, resamples: int = 2000, seed: int = 0) -> Dict:
    rng = random.Random(seed)
    boots = sorted(x for x in (fn([rng.choice(families) for _ in families]) for _ in range(resamples)) if x == x)
    return {"value": fn(list(families)), "ci95": [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]]}


def build(m_run: str = M_RUN, out: str = OUT, data_path: str = DATA) -> Dict:
    fr = check_identity(m_run, out)
    data = load(data_path)
    recs = records(data, load(os.path.join(out, "m.details.jsonl")), load(os.path.join(m_run, "eval", "rl_tune.details.jsonl")))
    fam = units(recs)
    families = sorted(fam)
    res: Dict = {"identity": {"m": fr, "data_sha256": sha(data_path)}, "families": len(families), "conditions": {}}
    for c in CONDITIONS:
        res["conditions"][c] = {k: {"value": value(fam, families, "%s:%s" % (c, k)),
                                    "n": sum(fam[f]["%s:%s" % (c, k)][1] for f in families if "%s:%s" % (c, k) in fam[f])}
                                for k in ["same", "different", "both", "shown_other", "wrong_is_other"]
                                + ["%s:%s" % (s, x) for s in ("same", "different") for x in ("mass", "speed", "distance", "temp", "force")]}
    res["differences"] = {}
    for a, b in (("keep", "original"), ("zoom", "original"), ("zoom", "keep")):
        for k in ("different", "both"):
            res["differences"]["%s-%s:%s" % (a, b, k)] = interval(
                lambda fs, a=a, b=b, k=k: value(fam, fs, "%s:%s" % (a, k)) - value(fam, fs, "%s:%s" % (b, k)), families)
    d = res["differences"]
    keep_up, zoom_up = d["keep-original:different"]["ci95"][0] > 0, d["zoom-keep:different"]["ci95"][0] > 0
    low = all(res["conditions"][c]["different"]["value"] < 0.5 for c in ("keep", "zoom"))
    res["reading"] = ("both the other object's interference and readability" if keep_up and zoom_up else
                      "supports interference from the other object" if keep_up else
                      "supports a readability limit" if zoom_up else
                      "neither; still low -- next: direct number decision reading" if low else "neither, not low")
    return res


def main() -> int:
    res = build()
    json.dump(res, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    for c, v in res["conditions"].items():
        print("%-8s same %.3f (%d) different %.3f (%d) both %.3f (%d) shown-other %.3f wrong=other %.3f (%d)" % (
            c, v["same"]["value"], v["same"]["n"], v["different"]["value"], v["different"]["n"], v["both"]["value"], v["both"]["n"],
            v["shown_other"]["value"], v["wrong_is_other"]["value"], v["wrong_is_other"]["n"]))
    for k, v in res["differences"].items():
        print("%s %+.3f %s" % (k, v["value"], [round(x, 3) for x in v["ci95"]]))
    print("reading (conditional): %s" % res["reading"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
