"""Left/right on both sides of a mirror: is a question answered right on the picture AND on its mirror?

    uv run python scripts/pair_stats.py runs/v6-flip-lr/eval/test_relation.details.jsonl \
        runs/v6-flip-lr/eval/test_relation_flip.details.jsonl [--axis lr]

The mirrored set (scripts/data/v6_flip.py) asks every left/right question again on the picture flipped left
to right, with the answer swapped; its sample ids are "flip:" + the original id. A model that answers from
names or option order gets at most one of the two right, so the share of pairs with BOTH right is the
grounding measure (two independent coin flips: 25%). Reported per kind (position / relation) x question
type (choice / noul), with a 95% interval from resampling original pictures. Only reads per-question records.

Against a base model (round 9 plan, review R9-V9-04):

    uv run python scripts/pair_stats.py NEW/test_relation.details.jsonl NEW/test_relation_flip.details.jsonl \
        --base BASE/test_relation.details.jsonl BASE/test_relation_flip.details.jsonl \
        [--mismatched NEW/test_relation.mismatched.details.jsonl BASE/test_relation.mismatched.details.jsonl]

adds, per group, the paired difference of the both-right share (new - base) over the pairs both models have,
and with --mismatched the change of the picture gain on the original questions of those pairs:
(new real - new mismatched) - (base real - base mismatched). Both intervals resample original pictures, the
same pictures for both models.
"""
import argparse
import json
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def read(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def pairs(original: List[dict], mirrored: List[dict], axis: str = "lr") -> List[Tuple[dict, dict]]:
    """(record on the picture, record on its mirror) for every question asked on both."""
    by_id = {(r["sample_id"], r["qid"]): r for r in original if not axis or r.get("axis", axis) == axis}
    out = []
    for m in mirrored:
        sid = m["sample_id"]
        if sid.startswith("flip:") and (sid[5:], m["qid"]) in by_id:
            out.append((by_id[(sid[5:], m["qid"])], m))
    return out


def summary(ps: List[Tuple[dict, dict]], resamples: int = 2000, seed: int = 0) -> Dict[str, Any]:
    by_pic: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    for o, m in ps:
        by_pic[o["image_id"]].append((float(o["correct"]), float(m["correct"]), float(o["correct"] and m["correct"])))
    pics = list(by_pic.values())
    sums = [[sum(x[i] for x in p) for i in range(3)] for p in pics]
    counts = [len(p) for p in pics]
    n = sum(counts)
    rng = random.Random(seed)
    boot = []
    for _ in range(resamples):
        idx = [rng.randrange(len(pics)) for _ in pics]
        boot.append(sum(sums[i][2] for i in idx) / max(1, sum(counts[i] for i in idx)))
    boot.sort()
    return {"pairs": n, "pictures": len(pics), "original": sum(s[0] for s in sums) / n,
            "mirrored": sum(s[1] for s in sums) / n, "both": sum(s[2] for s in sums) / n,
            "both_ci95": [boot[int(0.025 * resamples)], boot[min(resamples - 1, int(0.975 * resamples))]]}


def _interval(values: List[float], resamples: int) -> List[float]:
    values.sort()
    return [values[int(0.025 * resamples)], values[min(resamples - 1, int(0.975 * resamples))]]


def paired_diff(new: List[Tuple[dict, dict]], base: List[Tuple[dict, dict]],
                resamples: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """Both-right share new - base over the pairs both models answered, resampling original pictures."""
    key = lambda o: (o["sample_id"], o["qid"])
    base_by = {key(o): float(o["correct"] and m["correct"]) for o, m in base}
    by_pic: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for o, m in new:
        if key(o) in base_by:
            by_pic[o["image_id"]].append((float(o["correct"] and m["correct"]), base_by[key(o)]))
    pics = [(sum(a for a, _ in p), sum(b for _, b in p), len(p)) for p in by_pic.values()]
    n = sum(c for _, _, c in pics)
    if not n:
        return {"pairs": 0}
    rng = random.Random(seed)
    boot = []
    for _ in range(resamples):
        idx = [pics[rng.randrange(len(pics))] for _ in pics]
        boot.append((sum(a for a, _, _ in idx) - sum(b for _, b, _ in idx)) / max(1, sum(c for _, _, c in idx)))
    new_both, base_both = sum(a for a, _, _ in pics) / n, sum(b for _, b, _ in pics) / n
    return {"pairs": n, "pictures": len(pics), "new_both": new_both, "base_both": base_both,
            "diff": new_both - base_both, "diff_ci95": _interval(boot, resamples)}


def gain_change(questions: List[dict], new_real: List[dict], new_mis: List[dict], base_real: List[dict],
                base_mis: List[dict], resamples: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """(new real - new mismatched) - (base real - base mismatched) on `questions` (original-picture records),
    over the questions all four runs answered, resampling pictures."""
    key = lambda r: (r["sample_id"], r["qid"])
    runs = [{key(r): float(r["correct"]) for r in rs} for rs in (new_real, new_mis, base_real, base_mis)]
    by_pic: Dict[str, List[List[float]]] = defaultdict(list)
    for q in questions:
        if all(key(q) in r for r in runs):
            by_pic[q["image_id"]].append([r[key(q)] for r in runs])
    pics = [([sum(x[i] for x in p) for i in range(4)], len(p)) for p in by_pic.values()]
    n = sum(c for _, c in pics)
    if not n:
        return {"questions": 0}
    did = lambda ps: (lambda s, c: ((s[0] - s[1]) - (s[2] - s[3])) / max(1, c))(
        [sum(p[0][i] for p in ps) for i in range(4)], sum(p[1] for p in ps))
    rng = random.Random(seed)
    boot = [did([pics[rng.randrange(len(pics))] for _ in pics]) for _ in range(resamples)]
    acc = [sum(p[0][i] for p in pics) / n for i in range(4)]
    return {"questions": n, "pictures": len(pics), "new_real": acc[0], "new_mismatched": acc[1],
            "base_real": acc[2], "base_mismatched": acc[3], "new_gain": acc[0] - acc[1],
            "base_gain": acc[2] - acc[3], "gain_change": did(pics), "gain_change_ci95": _interval(boot, resamples)}


def _group(o: dict) -> str:
    return "%s/%s" % (o.get("kind", "-"), o.get("type", "-"))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("original")
    p.add_argument("mirrored")
    p.add_argument("--axis", default="lr", help="only original records with this axis ('' for all)")
    p.add_argument("--base", nargs=2, metavar=("ORIGINAL", "MIRRORED"), help="the base model's two files")
    p.add_argument("--mismatched", nargs=2, metavar=("NEW", "BASE"),
                   help="mismatched-picture records of the original set, new and base model (needs --base)")
    p.add_argument("--out", help="write the result as JSON")
    a = p.parse_args(argv)
    if a.mismatched and not a.base:
        p.error("--mismatched needs --base")
    ps = pairs(read(a.original), read(a.mirrored), a.axis)
    if not ps:
        raise SystemExit("no question is in both files")
    groups: Dict[str, List[Tuple[dict, dict]]] = defaultdict(list)
    for o, m in ps:
        groups[_group(o)].append((o, m))
    result: Dict[str, Any] = {"all": summary(ps), "groups": {g: summary(v) for g, v in sorted(groups.items())}}
    if a.base:
        base = pairs(read(a.base[0]), read(a.base[1]), a.axis)
        base_groups: Dict[str, List[Tuple[dict, dict]]] = defaultdict(list)
        for o, m in base:
            base_groups[_group(o)].append((o, m))
        result["vs_base"] = {"all": paired_diff(ps, base),
                             "groups": {g: paired_diff(v, base_groups[g]) for g, v in sorted(groups.items())}}
        if a.mismatched:
            new_real, base_real = [o for o, _ in ps], [o for o, _ in base]
            new_mis, base_mis = read(a.mismatched[0]), read(a.mismatched[1])
            result["picture_gain_change"] = {
                "all": gain_change(new_real, new_real, new_mis, base_real, base_mis),
                "groups": {g: gain_change([o for o, _ in v], new_real, new_mis, base_real, base_mis)
                           for g, v in sorted(groups.items())}}
    fmt = lambda s: "pairs %5d  original %.3f  mirrored %.3f  both %.3f [%.3f, %.3f]" % (
        s["pairs"], s["original"], s["mirrored"], s["both"], *s["both_ci95"])
    print("all            ", fmt(result["all"]))
    for g, s in result["groups"].items():
        print("%-15s" % g, fmt(s))
    dfmt = lambda d: "pairs %5d  base %.3f  new %.3f  diff %+.3f [%+.3f, %+.3f]" % (
        d["pairs"], d["base_both"], d["new_both"], d["diff"], *d["diff_ci95"]) if d["pairs"] else "no shared pairs"
    gfmt = lambda d: "questions %5d  gain base %+.3f  new %+.3f  change %+.3f [%+.3f, %+.3f]" % (
        d["questions"], d["base_gain"], d["new_gain"], d["gain_change"], *d["gain_change_ci95"]
    ) if d["questions"] else "no shared questions"
    if "vs_base" in result:
        print("both right, new - base:")
        print("all            ", dfmt(result["vs_base"]["all"]))
        for g, d in result["vs_base"]["groups"].items():
            print("%-15s" % g, dfmt(d))
    if "picture_gain_change" in result:
        print("picture gain (real - mismatched), new - base:")
        print("all            ", gfmt(result["picture_gain_change"]["all"]))
        for g, d in result["picture_gain_change"]["groups"].items():
            print("%-15s" % g, gfmt(d))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
