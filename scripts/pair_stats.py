"""Left/right on both sides of a mirror: is a question answered right on the picture AND on its mirror?

    uv run python scripts/pair_stats.py runs/v6-flip-lr/eval/test_relation.details.jsonl \
        runs/v6-flip-lr/eval/test_relation_flip.details.jsonl [--axis lr]

The mirrored set (scripts/data/v6_flip.py) asks every left/right question again on the picture flipped left
to right, with the answer swapped; its sample ids are "flip:" + the original id. A model that answers from
names or option order gets at most one of the two right, so the share of pairs with BOTH right is the
grounding measure (two independent coin flips: 25%). Reported per kind (position / relation) x question
type (choice / noul), with a 95% interval from resampling original pictures. Only reads per-question records.
"""
import argparse
import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple


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


def summary(ps: List[Tuple[dict, dict]], resamples: int = 2000, seed: int = 0) -> Dict[str, float]:
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


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("original")
    p.add_argument("mirrored")
    p.add_argument("--axis", default="lr", help="only original records with this axis ('' for all)")
    p.add_argument("--out", help="write the result as JSON")
    a = p.parse_args(argv)
    ps = pairs(read(a.original), read(a.mirrored), a.axis)
    if not ps:
        raise SystemExit("no question is in both files")
    groups: Dict[str, List[Tuple[dict, dict]]] = defaultdict(list)
    for o, m in ps:
        groups["%s/%s" % (o.get("kind", "-"), o.get("type", "-"))].append((o, m))
    result = {"all": summary(ps), "groups": {g: summary(v) for g, v in sorted(groups.items())}}
    fmt = lambda s: "pairs %5d  original %.3f  mirrored %.3f  both %.3f [%.3f, %.3f]" % (
        s["pairs"], s["original"], s["mirrored"], s["both"], *s["both_ci95"])
    print("all            ", fmt(result["all"]))
    for g, s in result["groups"].items():
        print("%-15s" % g, fmt(s))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
