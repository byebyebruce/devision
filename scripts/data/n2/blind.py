"""data-n2 quality gate: questions answerable without the picture (plan docs/experiments/2026-10-21-v04-plan.md).

    uv run python scripts/data/n2/blind.py --out data/n2/blind_drop.json runs/n2blind/v7w.mismatched.details.jsonl ...

Each argument is the per-question output of `devision-eval --control mismatched` (v0.3 answering with another
picture) on a candidate training block. A question is dropped when the model, looking at the wrong picture, still
puts at least --threshold on the right answer: the text alone gives it away, so it teaches nothing about looking.
Left alone: counting questions (a wrong picture often truly has none of the thing) and yes / no questions whose
mismatched answer "no" is the truth for that other picture are not judged this way -- only choice questions with
more than two options are filtered. Writes the ids and per-source counts.
"""
import argparse
import json
from collections import Counter


def blind_ids(details, threshold: float):
    out, seen = [], Counter()
    for d in details:
        if d["type"] != "choice" or len(d["options"]) <= 2 or d.get("kind") == "count":
            continue
        seen[d["source"]] += 1
        g = d["gold"]
        best = max(g, key=lambda o: g[o])
        if g[best] >= 1.0 and d["probabilities"].get(best, 0.0) >= threshold:
            out.append(d["sample_id"])
    return out, seen


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("details", nargs="+")
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    ids, judged, by = [], Counter(), Counter()
    for path in a.details:
        rows = [json.loads(l) for l in open(path) if l.strip()]
        src = {d["sample_id"]: d["source"] for d in rows}
        got, seen = blind_ids(rows, a.threshold)
        ids += got
        judged.update(seen)
        by.update(src[i] for i in got)
    with open(a.out, "w") as f:
        json.dump({"threshold": a.threshold, "ids": sorted(set(ids)), "judged": dict(judged), "dropped": dict(by)}, f,
                  indent=1)
    print(json.dumps({"judged": dict(judged), "dropped": dict(by)}, indent=1))


if __name__ == "__main__":
    main()
