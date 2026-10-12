"""exp14 quick check (owner 2026-10-12): does PixMo-Points counting supervision -- new to v12 -- help counting at an
equal budget, and at what cost to old abilities? Two 16,000-question mixes from v12's pool, trained 2,000 steps from v12:

  base    replay only: existence 3,000 (yes = no inside each object category), counting 3,000 (original format),
          science 2,000, general 3,000, v12-ext 5,000 (500 per source)
  pixmo   the same blocks cut to 75% (a nested subset, 12,000) + 4,000 PixMo-Points questions on pictures v12 never
          saw: counting 3,000 (gold 4-10; half with candidates 0..10, half with a run of 3-5 consecutive numbers in
          which the gold sits at a uniformly random place, so no option-only rule beats chance) and existence 1,000
          ("Are there any <phrase> in the image?", yes = count > 0, no = count 0, which the annotators marked absent)

PixMo pictures: those of data/r13/pixmo_two_kind (already isolated from every dev / test / bench picture in round 13),
split 90/10 by SHA-256 into training and a dev panel (pixmo_count_wide / pixmo_count_narrow: the same 400 questions in
both formats; pixmo_exist: 300, balanced).

    uv run python scripts/exp14/build.py --root data
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "r13b"))
import build as r13b  # noqa: E402

PHRASE = re.compile(r"^[a-z][a-z \-']{1,60}$")


def dev_picture(sha):
    return int(hashlib.sha256(sha.encode()).hexdigest(), 16) % 10 == 0


def narrow(gold, rng):
    k = rng.choice([3, 4, 5])
    pos = rng.randrange(k)
    lo = max(0, min(gold - pos, 10 - (k - 1)))
    return [str(n) for n in range(lo, lo + k)]


def count_q(sid, image_id, image, phrase, gold, fmt, rng):
    opts = [str(n) for n in range(11)] if fmt == "wide" else narrow(gold, rng)
    rng.shuffle(opts)
    return {"id": "%s:count:%s" % (sid, fmt), "source": "pixmo_points", "kind": "count", "image_id": image_id,
            "image": image, "candidates": fmt, "questions": {"q": {"type": "choice",
            "instructions": "How many %s are there?" % phrase, "criteria": {o: None for o in opts}}},
            "gold": {"q": {"probabilities": {o: float(o == str(gold)) for o in opts}}}, "group": image_id}


def exist_q(sid, image_id, image, phrase, yes):
    return {"id": "%s:exist" % sid, "source": "pixmo_points", "kind": "exist", "image_id": image_id, "image": image,
            "questions": {"q": {"type": "noul", "instructions": "Are there any %s in the image?" % phrase}},
            "gold": {"q": {"probabilities": {"true": float(yes), "false": float(not yes)}}}, "group": image_id}


def pixmo(root, rng):
    imgs = {}
    for l in open(os.path.join(root, "r13/pixmo_two_kind/train.jsonl")):
        s = json.loads(l)
        imgs[os.path.basename(s["image"]).split(".")[0]] = (s["image_id"], s["image"])
    rows = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(root, "raw/hf/datasets--allenai--pixmo-points/snapshots/*/data/*.parquet"))):
        t = pq.read_table(f, columns=["image_sha256", "count", "label", "collection_method"]).to_pydict()
        for sha, c, lab, m in zip(t["image_sha256"], t["count"], t["label"], t["collection_method"]):
            lab = (lab or "").strip().lower()
            if m == "counting" and sha in imgs and PHRASE.match(lab) and len(lab.split()) <= 6:
                rows[sha].append((lab, int(c)))
    split = {"train": defaultdict(list), "dev": defaultdict(list)}
    for sha in sorted(rows):
        image_id, image = imgs[sha]
        part = split["dev" if dev_picture(sha) else "train"]
        seen = set()
        for lab, c in rows[sha]:
            if lab in seen:
                continue
            seen.add(lab)
            sid = "exp14-pixmo:%s:%s" % (sha[:16], re.sub(r"\W+", "-", lab))
            if 4 <= c <= 10:
                part["count"].append((sid, image_id, image, lab, c))
            if c == 0 or c >= 4:
                part["exist_" + ("yes" if c else "no")].append((sid, image_id, image, lab, c > 0))
    return split


def pick(items, n, rng, per_picture=2):
    items = sorted(items)
    rng.shuffle(items)
    used, out = Counter(), []
    for it in items:
        if used[it[1]] < per_picture:
            out.append(it)
            used[it[1]] += 1
        if len(out) == n:
            break
    if len(out) < n:
        raise SystemExit("only %d of %d" % (len(out), n))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="data/exp14")
    a = p.parse_args()
    rng = random.Random(14)
    os.makedirs(a.out, exist_ok=True)

    split = pixmo(a.root, rng)
    tr, dv = split["train"], split["dev"]
    counts = pick(tr["count"], 3000, rng)
    train_px = [count_q(*c, fmt="wide" if i % 2 == 0 else "narrow", rng=rng) for i, c in enumerate(counts)]
    train_px += [exist_q(*e[:4], e[4]) for e in pick(tr["exist_yes"], 500, rng, 1) + pick(tr["exist_no"], 500, rng, 1)]
    dev_counts = pick(dv["count"], 400, rng, 1)
    panel_wide = [count_q(*c, fmt="wide", rng=rng) for c in dev_counts]
    panel_narrow = [count_q(*c, fmt="narrow", rng=rng) for c in dev_counts]
    panel_exist = [exist_q(*e[:4], e[4]) for e in pick(dv["exist_yes"], 150, rng, 1) + pick(dv["exist_no"], 150, rng, 1)]

    pool = r13b.rows(os.path.join(a.root, "ext/train_mix.jsonl"))
    exist, _ = r13b.build_existence(pool, rng, 3000)
    is_count = lambda s: s.get("source") in r13b.OLD_COUNT and r13b.is_count(*next(iter(zip(s["questions"].values(), s["gold"].values()))))
    count = r13b.sample_rows([s for s in pool if is_count(s)], 3000, rng)
    science = r13b.sample_rows([s for s in pool if s.get("source") in r13b.SCIENCE], 2000, rng)
    used = {s["id"] for s in exist + count}
    general_pool = [s for s in pool if s.get("source") in r13b.GENERAL_FULL + r13b.GENERAL_SHARED and s["id"] not in used
                    and not is_count(s) and not (s.get("source") == "objects365" and next(iter(s["questions"].values()))["type"] == "noul")]
    general = r13b.sample_rows(general_pool, 3000, rng)
    ext = [s for src in r13b.EXT for s in r13b.sample_rows([s for s in pool if s.get("source") == src], 500, rng)]
    blocks = {"existence": exist, "count": count, "science": science, "general": general, "ext": ext}
    base = [dict(r13b.as_picture_group(s), block=k) for k, ss in blocks.items() for s in ss]
    cut = [dict(r13b.as_picture_group(s), block=k) for k, ss in blocks.items() for s in ss[: len(ss) * 3 // 4]]
    mixes = {"train_base": base, "train_pixmo": cut + [dict(s, block="pixmo") for s in train_px]}
    for name, ss in mixes.items():
        ss = list(ss)
        random.Random(1).shuffle(ss)
        r13b.write(os.path.join(a.out, name + ".jsonl"), ss)
    for name, ss in (("pixmo_count_wide", panel_wide), ("pixmo_count_narrow", panel_narrow), ("pixmo_exist", panel_exist)):
        r13b.write(os.path.join(a.out, name + ".jsonl"), ss)
    train_imgs = {s["image_id"] for s in train_px}
    assert not train_imgs & {s["image_id"] for s in panel_exist + panel_wide}, "a dev picture in training"
    manifest = {"train_base": len(base), "train_pixmo": len(mixes["train_pixmo"]),
                "pixmo_train": {"count": 3000, "exist": 1000, "pictures": len(train_imgs),
                                "gold": dict(sorted(Counter(c[4] for c in counts).items()))},
                "pixmo_pool": {k: len(v) for k, v in tr.items()}, "pixmo_dev_pool": {k: len(v) for k, v in dv.items()},
                "option_rules": {"train_narrow": r13b.option_rules([s for s in train_px if s.get("candidates") == "narrow"]),
                                 "panel_narrow": r13b.option_rules(panel_narrow), "panel_wide": r13b.option_rules(panel_wide)},
                "blocks_base": {k: len(v) for k, v in blocks.items()}}
    json.dump(manifest, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
