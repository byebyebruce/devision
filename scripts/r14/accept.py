"""Round 14 acceptance against v12 (docs/experiments/2026-10-12-r14-count.md, written before training): per set,
r14 - v12 per question (picture-resampled 95% intervals) for accuracy and calibrated NLL, then the three rules.
Every required set must have details on both sides, else no verdict (critic R14-R1-01); a set is held when its
lower bound is > -2 points, whether or not it is also proven worse (R14-R1-02).

    uv run python scripts/r14/accept.py [--cand runs/r14-count/eval] [--base runs/v12-ext/eval]
"""
import argparse
import json
import os

from devision.train.compare import load_records, paired_metric

PRIMARY = ("test_count_fresh", "bench_countbench")
PROTECTED = ("test_vsr", "test_v7w", "bench_pope", "test_vqa_yesno", "test_vqa_choice", "test_gqa", "test_exist",
             "test_position", "test_relation", "test_size", "test_position_flip", "test_relation_flip",
             "test_count_fresh", "test_bind_gqa", "bench_lv_vqav2_yesno", "bench_lv_aokvqa", "bench_lv_scienceqa",
             "bench_lv_pope")
REPORT = ("bench_countbench", "bench_whatsup", "dev_mix")


REQUIRED = tuple(dict.fromkeys(PRIMARY + PROTECTED + REPORT))
FLOOR = -0.02   # rule 2: a protected set's lower bound must be strictly above -2 points (unrounded)


def missing(cand, base):
    """Required sets without per-question details on either side; a verdict needs all of them."""
    return [(s, side) for s in REQUIRED for side, d in (("v12", base), ("candidate", cand))
            if not os.path.exists(os.path.join(d, s + ".details.jsonl"))]


def protection(lo, hi):
    """Two separate facts about one set: held (lo > -2 points, the rule) and proven worse (hi < 0, reported).
    A small proven drop inside the floor, e.g. [-1.5, -0.5], is both and passes."""
    return {"held": lo > FLOOR, "proven_worse": hi < 0}


def decide(sets):
    """sets: name -> {"accuracy": {"ci95": [lo, hi]}, "nll": {"difference": d, "ci95": [lo, hi]}} for every
    required set. Returns the three rules and the verdict."""
    absent = [s for s in REQUIRED if s not in sets]
    if absent:
        raise ValueError("no verdict without every required set; missing %s" % absent)
    prot = {s: protection(*sets[s]["accuracy"]["ci95"]) for s in PROTECTED}
    primary = any(sets[s]["accuracy"]["ci95"][0] > 0 for s in PRIMARY)
    not_held = [s for s in PROTECTED if not prot[s]["held"]]
    warnings = [s for s in REQUIRED if sets[s]["nll"]["difference"] > 0.02 and sets[s]["nll"]["ci95"][0] > 0]
    return {"primary": primary, "protection": not not_held, "not_held": not_held,
            "proven_worse": [s for s in PROTECTED if prot[s]["proven_worse"]], "warnings": warnings,
            "candidate": primary and not not_held and not warnings, "per_set": prot}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cand", default="runs/r14-count/eval")
    p.add_argument("--base", default="runs/v12-ext/eval")
    p.add_argument("--out", help="verdict JSON (default <cand>/compare/accept_vs_v12.json); give another path when --base"
                                 " is not v12, so the comparison does not overwrite the acceptance")
    a = p.parse_args(argv)
    gone = missing(a.cand, a.base)
    if gone:
        raise SystemExit("no verdict: per-question details missing for %s" % ", ".join("%s (%s)" % g for g in gone))
    out = {}
    for s in REQUIRED:
        A = load_records(os.path.join(a.base, s + ".details.jsonl"))
        B = load_records(os.path.join(a.cand, s + ".details.jsonl"))
        out[s] = {"accuracy": paired_metric(A, B, "accuracy"), "nll": paired_metric(A, B, "nll")}
    v = decide(out)
    for s in REQUIRED:
        acc, nll = out[s]["accuracy"], out[s]["nll"]
        lo, hi = acc["ci95"]
        label = ""
        if s in PROTECTED:
            label = ("held" if v["per_set"][s]["held"] else "not held") + (", proven worse" if v["per_set"][s]["proven_worse"] else "")
        print("%-22s %.3f -> %.3f  %+.1f [%+.1f, %+.1f]  %-24s nll %+.3f [%+.3f, %+.3f]%s" % (
            s, acc["a"], acc["b"], 100 * acc["difference"], 100 * lo, 100 * hi, label,
            nll["difference"], nll["ci95"][0], nll["ci95"][1], "  WARNING" if s in v["warnings"] else ""))
        out[s].update(v["per_set"].get(s, {}), nll_warning=s in v["warnings"])
    print("\nprimary (a counting test up, lo > 0): %s" % v["primary"])
    print("protection (every test / bench lo > -2): %s  not held %s; proven worse (reported) %s" % (
        v["protection"], v["not_held"], v["proven_worse"]))
    print("probability warnings: %s" % v["warnings"])
    print("verdict: %s" % ("candidate for v0.3" if v["candidate"] else "v12 stays by the rules; owner decides"))
    path = a.out or os.path.join(a.cand, "compare", "accept_vs_v12.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"sets": out, **{k: v[k] for k in ("primary", "protection", "not_held", "proven_worse", "warnings",
                                                   "candidate")}},
              open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
