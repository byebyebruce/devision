"""v12 retrain (docs/experiments/2026-10-11-v12-retrain-plan.md, sections 3, 9.2, 10.2 and 11): the training mix,
the calibration set and the monitor panels, written to data/r13b/ with a MANIFEST.

    uv run python scripts/r13b/build.py --root data

Training mix (question exposures; each question counted in one block, decided by its task, not its source):

  binding     24,000   GQA binding 8,000 (whole pictures), VisMin 8,000 (whole original/edit pairs),
                       counterfactual synthetic 8,000 (whole scene families, synth-r13p75)
  existence   16,000   v12 pool "Is there ...?" questions (COCO-exist, Objects365), 8,000 yes / 8,000 no, the same
                       number of yes and no drawn inside every object category
  count       16,000   0-10 counting from real photos, every question rewritten to the full candidate set 0..10
                       (shuffled per question): old replay (TallyQA, Objects365, PixMo-Count, VQAv2 choice) 8,000
                       and new (PixMo two-kind, TallyQA complex) 8,000; inside each half a third per answer bucket
                       0-2 / 3-5 / 6-10
  science      8,000   ScienceQA / AI2D / TQA of the v12 pool in full (7,875), topped up by repeating questions of
                       templates whose answer depends on the picture (same text, options and hint, different gold)
  general     16,000   A-OKVQA, VQAv2 yes/no, size, both relation sets, GQA-v2 in full; the rest shared equally by
                       VQAv2 choice (non-counting), Visual7W, VSR, COCO position, Objects365 left/right
  ext         12,000   the ten v12-ext sources, 1,200 each (whole pictures), original format

Replay rows (the v12 pool) carry their question template as `group`; here a replay row's group is its picture,
so micro batches are not filled with one template (the r13 mix kept the template).

Calibration (`val`): data/v5/dev_mix.jsonl without the pictures of the existence and science panels, plus
data/v11/dev_count_existing.jsonl rewritten to 0..10 (580 questions with 11 options, so the choice:11+ bucket gets
a temperature). Panels: existence (the COCO-exist part of dev_mix), science (its ScienceQA / AI2D / TQA part),
count_wide (data/v11/dev_count_transfer.jsonl rewritten to 0..10); the others are used as they are.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "r13"))
import identity  # noqa: E402

WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}
WORDS["none"] = 0
EXT = ("clevr", "snli_ve", "figureqa", "iconqa", "vision_flan", "visonlyqa", "clevr_math", "superclevr", "mapqa",
       "spatialsense")
OLD_COUNT = ("tallyqa", "objects365", "pixmo_count", "vqav2-choice")
NEW_COUNT = ("pixmo_two_kind", "tallyqa_complex")
SCIENCE = ("scienceqa", "ai2d", "tqa")
GENERAL_FULL = ("aokvqa", "vqav2-yesno", "coco-size", "coco-relation", "vg-relation", "gqa-v2")
GENERAL_SHARED = ("vqav2-choice", "v7w", "vsr", "coco-position", "objects365")


def rows(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write(path, samples):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


def nq(ss):
    return sum(len(s["questions"]) for s in ss)


def top(gold):
    return max(gold["probabilities"], key=gold["probabilities"].get)


def number(text):
    t = str(text).strip().lower()
    return int(t) if t.isdigit() else WORDS.get(t)


def is_count(q, gold):
    """A 0-10 counting question: 'How many', every option a number, the gold's mass on numbers in 0..10."""
    if q["type"] != "choice" or not re.match(r"(?i)how many\b", q["instructions"]):
        return False
    nums = [number(o) for o in q["criteria"]]
    if None in nums:
        return False
    mass = {number(o): p for o, p in gold["probabilities"].items() if p > 0}
    return None not in mass and all(0 <= n <= 10 for n in mass)


def wide(s, rng):
    """The sample (one question per row) with every counting question rewritten to the candidates 0..10."""
    qs, gs = {}, {}
    for k, q in s["questions"].items():
        g = s["gold"][k]
        if not is_count(q, g):
            qs[k], gs[k] = q, g
            continue
        probs = defaultdict(float)
        for o, p in g["probabilities"].items():
            probs[number(o)] += p
        keys = [str(n) for n in range(11)]
        rng.shuffle(keys)
        qs[k] = dict(q, criteria={o: None for o in keys})
        gs[k] = {"probabilities": {o: float(probs.get(int(o), 0.0)) for o in keys}}
    out = dict(s, questions=qs, gold=gs, candidates="0..10")
    out.pop("answer_position", None)
    return out


def bucket(s):
    n = number(top(next(iter(s["gold"].values()))))
    return "0-2" if n <= 2 else "3-5" if n <= 5 else "6-10"


def take(samples, key, target, rng):
    """Whole groups (by `key`) in a seeded random order until `target` questions."""
    groups = defaultdict(list)
    for s in samples:
        groups[key(s)].append(s)
    names = sorted(groups)
    rng.shuffle(names)
    out = []
    for g in names:
        if nq(out) >= target:
            break
        out += groups[g]
    return out


def sample_rows(samples, n, rng):
    samples = sorted(samples, key=lambda s: s["id"])
    return rng.sample(samples, min(n, len(samples)))


def existence_category(q):
    m = re.match(r"^is there (?:an? |any )?(.+?)(?: in (?:the|this) (?:image|picture|photo))?\?$", q["instructions"].strip().lower())
    return m.group(1) if m else q["instructions"].strip().lower()


def as_picture_group(s):
    """A replay row's unit is its picture (the pool's `group` is the question template)."""
    return dict(s, template=s.get("group"), group=s["image_id"])


def build_existence(pool, rng, target):
    by = defaultdict(lambda: {"true": [], "false": []})
    for s in pool:
        if s.get("source") in ("coco-exist", "objects365"):
            k, q = next(iter(s["questions"].items()))
            if q["type"] == "noul":
                by[existence_category(q)][top(s["gold"][k])].append(s)
    slots = [c for c in sorted(by) for _ in range(min(len(by[c]["true"]), len(by[c]["false"])))]
    chosen = Counter(rng.sample(slots, target // 2))
    out = []
    for c, n in sorted(chosen.items()):
        out += sample_rows(by[c]["true"], n, rng) + sample_rows(by[c]["false"], n, rng)
    stats = {"categories_available": len(by), "categories_with_both": sum(1 for c in by.values() if c["true"] and c["false"]),
             "categories_used": len(chosen), "pairs_available": len(slots)}
    return out, stats


def build_count(pool, new_sources, rng, half):
    old = [s for s in pool if s.get("source") in OLD_COUNT and is_count(*next(iter(zip(s["questions"].values(), s["gold"].values()))))]
    new = [s for s in new_sources if is_count(*next(iter(zip(s["questions"].values(), s["gold"].values()))))]
    out, stats = [], {}
    for name, part in (("old", old), ("new", new)):
        by = defaultdict(list)
        for s in part:
            by[bucket(s)].append(s)
        per = {"0-2": half // 3, "3-5": half // 3, "6-10": half - 2 * (half // 3)}
        for b, n in per.items():
            if len(by[b]) < n:
                raise SystemExit("count %s %s: %d available < %d" % (name, b, len(by[b]), n))
            got = sample_rows(by[b], n, rng)
            out += got
            stats["%s/%s" % (name, b)] = {"questions": n, "available": len(by[b]),
                                          "by_source": dict(Counter(s["source"] for s in got))}
    return out, stats


def picture_dependent_templates(science):
    answers = defaultdict(set)
    for s in science:
        k, q = next(iter(s["questions"].items()))
        answers[(q["instructions"], tuple(sorted(q.get("criteria") or {})), s.get("state_text", ""))].add(top(s["gold"][k]))
    return {key for key, a in answers.items() if len(a) > 1}


def option_rules(samples):
    """Accuracy of three option-only rules on numeric choice questions (smallest, lower median, most common)."""
    n = hits = Counter()
    n, hits, gold_counts = 0, Counter(), Counter()
    items = []
    for s in samples:
        for k, q in s["questions"].items():
            if q["type"] != "choice":
                continue
            vals = [number(o) for o in q["criteria"]]
            if None in vals:
                continue
            g = number(top(s["gold"][k]))
            items.append((sorted(vals), g))
            gold_counts[g] += 1
    if not items:
        return {}
    common = gold_counts.most_common(1)[0][0]
    for opts, g in items:
        n += 1
        hits["smallest"] += opts[0] == g
        hits["lower_median"] += opts[(len(opts) - 1) // 2] == g
        hits["most_common_answer"] += (common in opts and common == g)
    return {"questions": n, **{k: round(v / n, 4) for k, v in hits.items()},
            "chance": round(sum(1 / len(o) for o, _ in items) / n, 4), "most_common_answer_value": common}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="data/r13b")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    R = lambda rel: rows(os.path.join(a.root, rel))
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    pool = R("ext/train_mix.jsonl")
    assert all(len(s["questions"]) == 1 for s in pool), "the v12 pool has one question per row"

    blocks = {}
    # binding
    blocks["binding"] = (take(R("r13/gqa/train.jsonl"), lambda s: s["group"], 8000, rng)
                         + take(R("r13/vismin/train.jsonl"), lambda s: s["group"], 8000, rng)
                         + take(R("r13/synth-r13p75/train.jsonl"), lambda s: s["group"], 8000, rng))
    # existence
    blocks["existence"], ex_stats = build_existence(pool, rng, 16000)
    # count
    count, count_stats = build_count(pool, R("r13/pixmo_two_kind/train.jsonl") + R("r13/tallyqa_complex/train.jsonl"), rng, 8000)
    count_narrow_rules = option_rules(count)
    blocks["count"] = [wide(s, rng) for s in count]
    # science
    sci = sorted((s for s in pool if s.get("source") in SCIENCE), key=lambda s: s["id"])
    dep = picture_dependent_templates(sci)
    dep_rows = [s for s in sci if (lambda k, q: (q["instructions"], tuple(sorted(q.get("criteria") or {})), s.get("state_text", "")))(*next(iter(s["questions"].items()))) in dep]
    extra = [dict(s, id=s["id"] + ":rep1") for s in sample_rows(dep_rows, 8000 - nq(sci), rng)]
    blocks["science"] = sci + extra
    # general (counting questions of these sources are in the count block or stay as their task's format)
    used = {s["id"] for s in blocks["existence"]} | {s["id"] for s in count}
    def general_ok(s):
        k, q = next(iter(s["questions"].items()))
        if s["id"] in used or (s.get("source") in OLD_COUNT and is_count(q, s["gold"][k])):
            return False
        return not (s.get("source") == "objects365" and q["type"] == "noul")
    gen = [s for s in pool if s.get("source") in GENERAL_FULL and general_ok(s)]
    rest = 16000 - nq(gen)
    per = rest // len(GENERAL_SHARED)
    for i, src in enumerate(GENERAL_SHARED):
        cand = [s for s in pool if s.get("source") == src and general_ok(s)]
        gen += sample_rows(cand, per + (rest - per * len(GENERAL_SHARED) if i == 0 else 0), rng)
    blocks["general"] = gen
    # ext
    blocks["ext"] = [s for src in EXT for s in take([s for s in pool if s.get("source") == src], lambda s: s["image_id"], 1200, rng)]

    mix = []
    for name, ss in blocks.items():
        for s in ss:
            s = as_picture_group(s) if name != "binding" else s
            mix.append(dict(s, block=name))
    ids = [s["id"] for s in mix]
    dup = [i for i, c in Counter(ids).items() if c > 1]
    assert not dup, "duplicate ids across blocks: %s" % dup[:5]
    rng.shuffle(mix)
    write(os.path.join(a.out, "train.jsonl"), mix)

    # calibration set and panels
    dev = R("v5/dev_mix.jsonl")
    exist_imgs = {s["image_id"] for s in dev if s.get("source") == "coco-exist"}
    sci_imgs = {s["image_id"] for s in dev if s.get("source") in SCIENCE}
    panel_exist = [s for s in dev if s.get("source") == "coco-exist"]
    panel_science = [s for s in dev if s.get("source") in SCIENCE]
    crng = random.Random(a.seed + 1)
    calib_wide = [wide(s, crng) for s in R("v11/dev_count_existing.jsonl")]
    val = [s for s in dev if s["image_id"] not in exist_imgs | sci_imgs] + calib_wide
    count_wide = [wide(s, crng) for s in R("v11/dev_count_transfer.jsonl")]
    for name, ss in (("val", val), ("panel_exist", panel_exist), ("panel_science", panel_science),
                     ("count_wide", count_wide)):
        write(os.path.join(a.out, name + ".jsonl"), ss)

    # isolation: no training picture in any panel / calibration / test set (canonical ids and file paths)
    ident = identity.load(a.root)
    held = {}
    for rel in ("r13b/val.jsonl", "r13b/panel_exist.jsonl", "r13b/panel_science.jsonl", "r13b/count_wide.jsonl",
                "ext/dev_ext.jsonl", "r13/gqa/dev_bind_gqa.jsonl", "r13/gqa/test_bind_gqa.jsonl",
                "r13/screen/dev_bind_cf_watch.jsonl", "v11/dev_count_transfer.jsonl", "v11/test_count_fresh.jsonl",
                "v2/test_exist.jsonl", "v2/test_gqa.jsonl", "v2/test_position.jsonl", "v2/test_relation.jsonl",
                "v2/test_size.jsonl", "v2/test_vqa_choice.jsonl", "v2/test_vqa_yesno.jsonl", "v2/bench_pope.jsonl",
                "v5/test_vsr.jsonl", "v5/test_v7w.jsonl"):
        path = os.path.join(a.root, rel)
        if os.path.exists(path):
            for s in rows(path):
                held.setdefault(ident.canonical(s["image_id"]), rel)
                held.setdefault("file:" + s["image"], rel)
    clash = Counter()
    for s in mix:
        for key in (ident.canonical(s["image_id"]), "file:" + s["image"]):
            if key in held:
                clash[held[key]] += 1
    lv_ids = {ident.canonical(x["image_id"]) for f in ("aokvqa", "scienceqa", "vqav2_yesno", "pope")
              for x in rows(os.path.join(a.root, "lv_bench", f + ".jsonl"))}
    lv = sum(1 for s in mix if ident.canonical(s["image_id"]) in lv_ids)

    def describe(ss):
        yes = sum(1 for s in ss for k, q in s["questions"].items() if q["type"] == "noul" and top(s["gold"][k]) == "true")
        noul = sum(1 for s in ss for q in s["questions"].values() if q["type"] == "noul")
        return {"questions": nq(ss), "rows": len(ss), "pictures": len({s["image_id"] for s in ss}),
                "by_source": dict(Counter(s.get("source") for s in ss for _ in s["questions"]).most_common()),
                "noul_yes_share": round(yes / noul, 4) if noul else None,
                "options": dict(sorted(Counter(len(q.get("criteria") or {}) or 2 for s in ss for q in s["questions"].values()).items()))}

    manifest = {
        "seed": a.seed, "questions": nq(mix), "rows": len(mix),
        "blocks": {k: describe(v) for k, v in blocks.items()},
        "existence": ex_stats,
        "count": {"buckets": count_stats, "option_only_rules_before_rewrite": count_narrow_rules,
                  "option_only_rules_after_rewrite": option_rules(blocks["count"]),
                  "ext_block_counting_kept_original": option_rules(blocks["ext"])},
        "science": {"unique_questions": nq(sci), "repeated": len(extra), "max_repeats": 2 if extra else 1,
                    "picture_dependent_templates": len(dep), "picture_dependent_questions": nq(dep_rows)},
        "val": describe(val), "panel_exist": describe(panel_exist), "panel_science": describe(panel_science),
        "count_wide": describe(count_wide),
        "isolation": {"training_pictures_in_held_sets": dict(clash), "training_pictures_in_lv_bench": lv},
        "sha256": {n: hashlib.sha256(open(os.path.join(a.out, n + ".jsonl"), "rb").read()).hexdigest()
                   for n in ("train", "val", "panel_exist", "panel_science", "count_wide")},
    }
    json.dump(manifest, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1)
    print(json.dumps({k: manifest[k] for k in ("questions", "isolation")}, indent=1))
    print(json.dumps({k: v["questions"] for k, v in manifest["blocks"].items()}))


if __name__ == "__main__":
    main()
