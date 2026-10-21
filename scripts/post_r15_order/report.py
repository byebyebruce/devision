"""Frozen post-r15 screening rules, based only on completed temperature-1 monitor records."""
import argparse
import copy
import json
from pathlib import Path

from devision.train.compare import _occurrences, _valid_probabilities, load_records, paired_metric, same_start

SETS = ("val", "ext", "science", "exist", "flip", "count_wide", "count_narrow",
        "count_wide_reverse", "count_narrow_reverse")
COUNTS = dict(zip(SETS, (3892, 3408, 577, 736, 1544, 300, 300, 300, 300)))
EXT = ("clevr", "clevr_math", "figureqa", "iconqa", "mapqa", "objects365", "pixmo_count",
       "snli_ve", "spatialsense", "superclevr", "tallyqa", "vision_flan", "visonlyqa")
GUARDS = ("vsr", "aokvqa", "science", "exist", "flip", "ext")
RULES = {
    "resamples": 10000, "seed": 1518, "start_tolerance": 1e-5,
    "order_difference_le": -0.05, "order_hi_lt": 0.0,
    "count_protection_lo_gt": -0.02, "guard_difference_le": -0.03,
    "guard_hi_lt": 0.0, "nll_difference_gt": 0.02, "nll_lo_gt": 0.0,
    "early_difference_le": -0.05, "early_hi_lt": 0.0,
}


def complete(directory):
    for name, n in COUNTS.items():
        p = Path(directory) / (name + ".details.jsonl")
        if not p.exists():
            return False
        if sum(1 for _ in p.open()) != n:
            return False
    return True


def normalise(rs):
    """Keep all duplicate copies, canonicalise images, and break probability ties by option key."""
    aliases_path = Path("data/v5/image_identity.json")
    aliases = json.loads(aliases_path.read_text()) if aliases_path.exists() else {}
    out = []
    for (sid, qid, occurrence), original in _occurrences(rs).items():
        r = copy.deepcopy(original)
        if not _valid_probabilities(r):
            raise ValueError("invalid probabilities: %s" % sid)
        r["sample_id"] = json.dumps([sid, occurrence], ensure_ascii=False)
        picture = r["image_id"].removesuffix(":flip")
        r["image_id"] = aliases.get(picture, picture)
        r["prediction"] = min(r["probabilities"], key=lambda k: (-r["probabilities"][k], k))
        r["correct"] = r["prediction"] == r["gold_answer"]
        out.append(r)
    return out


def panel(directory, name):
    source = name if name in ("vsr", "aokvqa") else None
    raw = load_records(str(Path(directory) / (("val" if source else name) + ".details.jsonl")))
    if source:
        raw = [r for r in raw if r["source"] == source]
    if not raw:
        raise ValueError("empty panel: %s" % name)
    return normalise(raw)


def order_rows(original, reverse, metric):
    a = {(r["sample_id"], r["qid"]): r for r in original}
    b = {(r["sample_id"], r["qid"]): r for r in reverse}
    if len(a) != len(original) or len(b) != len(reverse) or a.keys() != b.keys():
        raise ValueError("different/duplicate questions in order views")
    out = []
    for key, r in a.items():
        s = b[key]
        if r["gold"] != s["gold"] or r["image_id"] != s["image_id"] or set(r["options"]) != set(s["options"]):
            raise ValueError("order view changes gold, picture, or options")
        value = r["prediction"] != s["prediction"] if metric == "flip" else r["correct"] and s["correct"]
        out.append(dict(r, correct=float(value)))
    return out


def compare(a, b, metric="accuracy", resamples=None):
    # Reuse the source-stratified, image-cluster macro implementation for soft-gold NLL too.
    if metric == "macro_nll":
        a, b = [dict(r, correct=r["nll"]) for r in a], [dict(r, correct=r["nll"]) for r in b]
    actual = "macro_accuracy" if metric == "macro_nll" else metric
    result = paired_metric(a, b, actual, sources=EXT if actual == "macro_accuracy" else None,
                           resamples=resamples or RULES["resamples"], seed=RULES["seed"])
    result["metric"] = metric
    return result


def pair(a_dir, b_dir, resamples=None):
    out = {}
    for name in GUARDS + ("count_wide", "count_narrow", "count_wide_reverse", "count_narrow_reverse"):
        a, b = panel(a_dir, name), panel(b_dir, name)
        accuracy = "balanced_accuracy" if name == "vsr" else "macro_accuracy" if name == "ext" else "accuracy"
        nll = "macro_nll" if name == "ext" else "nll"
        out[name] = {"accuracy": compare(a, b, accuracy, resamples), "nll": compare(a, b, nll, resamples)}
        if name in ("vsr", "exist"):
            out[name]["recall"] = compare(a, b, "recall", resamples)
            out[name]["fpr"] = compare(a, b, "fpr", resamples)
        if name == "vsr":
            out[name]["raw_accuracy"] = compare(a, b, "accuracy", resamples)
            out[name]["yes_fraction"] = {"a": sum(r["prediction"] == "true" for r in a) / len(a),
                                         "b": sum(r["prediction"] == "true" for r in b) / len(b)}
    for name in ("count_wide", "count_narrow"):
        for metric in ("flip", "joint"):
            a = order_rows(panel(a_dir, name), panel(a_dir, name + "_reverse"), metric)
            b = order_rows(panel(b_dir, name), panel(b_dir, name + "_reverse"), metric)
            out[name][metric] = compare(a, b, "accuracy", resamples)
    return out


def clear_losses(comparison, early=False):
    threshold = RULES["early_difference_le" if early else "guard_difference_le"]
    panels = GUARDS + (("count_wide", "count_narrow", "count_wide_reverse", "count_narrow_reverse") if early else ())
    return [p for p in panels if comparison[p]["accuracy"]["difference"] <= threshold
            and comparison[p]["accuracy"]["ci95"][1] < 0]


def decide(b_minus_a, b_minus_start):
    ba, bs = b_minus_a["count_wide"]["flip"], b_minus_start["count_wide"]["flip"]
    primary = (ba["difference"] <= RULES["order_difference_le"] and ba["ci95"][1] < 0
               and bs["ci95"][1] < 0)
    counts = {"wide": b_minus_start["count_wide"]["accuracy"],
              "wide_reverse": b_minus_start["count_wide_reverse"]["accuracy"],
              "wide_joint": b_minus_start["count_wide"]["joint"],
              "narrow": b_minus_start["count_narrow"]["accuracy"]}
    not_held = [p for p, r in counts.items() if r["ci95"][0] <= RULES["count_protection_lo_gt"]]
    losses = clear_losses(b_minus_start)
    warnings = [p for p, row in b_minus_start.items() if row["nll"]["difference"] > RULES["nll_difference_gt"]
                and row["nll"]["ci95"][0] > 0]
    return {"primary": primary, "count_not_held": not_held, "clear_losses": losses, "nll_warnings": warnings,
            "screen_pass": primary and not not_held and not losses and not warnings,
            "scope": "Small-experiment screen, not release approval or independent heldout confirmation."}


def final_report(a, b, out):
    for d in (a / "step0", a / "step2000", b / "step0", b / "step2000"):
        if not complete(d):
            raise ValueError("incomplete fixed monitor: %s" % d)
    result = {"rules": RULES, "same_start": same_start(str(a / "step0"), str(b / "step0"), 1e-5)}
    for key, left, right in (("A_minus_r15", a / "step0", a / "step2000"),
                             ("B_minus_r15", b / "step0", b / "step2000"),
                             ("B_minus_A", a / "step2000", b / "step2000")):
        result[key] = pair(left, right)
    result.update(decide(result["B_minus_A"], result["B_minus_r15"]))
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", default="runs/post-r15-order-a/stage2/monitor")
    p.add_argument("--b", default="runs/post-r15-order-b/stage2/monitor")
    p.add_argument("--out", default="runs/post-r15-order/report.json")
    args = p.parse_args()
    result = final_report(Path(args.a), Path(args.b), args.out)
    print(json.dumps({k: result[k] for k in ("primary", "count_not_held", "clear_losses", "nll_warnings", "screen_pass")}, indent=2))
    return 0 if result["screen_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
