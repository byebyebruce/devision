"""Training mix for the weak points, from v12's pool plus the public-data pack (a starting point: the quotas are not
tuned; the exp14 quick check is the first evidence on PixMo -- docs/experiments/2026-10-12-exp14-pixmo.md).

    uv run python scripts/data/pub/pub_mix.py --root data [--quota cosyn=4000 --quota ext=24000 ...]

Blocks (question exposures, each question in one block; a block whose source is missing on this machine is skipped
and named in the MANIFEST):
  binding    24,000  round-13 GQA binding / VisMin / counterfactual synthetic (8,000 each, whole groups)
  existence  16,000  v12 pool "Is there ...", yes = no inside every object category
  count      16,000  v12 pool counting in its ORIGINAL format 8,000 (the v12 retrain stopped when every counting
                     question was rewritten to 0..10) + PixMo-Points 8,000 (half 0..10, half runs of 3-5)
  vizwiz      all    VizWiz yes/no and counting with 10-person soft gold
  science     8,000  v12 pool ScienceQA / AI2D / TQA (all 7,875 + picture-dependent repeats)
  cosyn       4,000  CoSyn closed questions (diagram / chemical / circuit / math)
  general    16,000  as the v12 retrain
  ext        24,000  the ten v12-ext sources, 2,400 each (twice the retrain's: it lost ~2 points at 1,200)
  abo         4,000  product colour / kind
  onejev      4,000  OneJev static-picture questions
Writes data/pub/train_mix.jsonl, data/pub/dev_pub.jsonl (the new sources' dev splits, for --eval) and MANIFEST.json.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "r13b"))
import build as r13b  # noqa: E402

QUOTAS = {"binding": 24000, "existence": 16000, "count_v12": 8000, "pixmo_points": 8000, "vizwiz": 10 ** 9,
          "science": 8000, "cosyn": 4000, "general": 16000, "ext": 24000, "abo": 4000, "onejev": 4000}
PUB = ("pixmo_points", "vizwiz", "cosyn", "abo", "onejev")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="data/pub")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quota", action="append", default=[], help="block=N (overrides a default)")
    a = p.parse_args()
    q = dict(QUOTAS)
    for kv in a.quota:
        k, v = kv.split("=")
        if k not in q:
            raise SystemExit("unknown block %s (one of %s)" % (k, list(q)))
        q[k] = int(v)
    rng = random.Random(a.seed)
    R = lambda rel: r13b.rows(os.path.join(a.root, rel))
    have = lambda rel: os.path.exists(os.path.join(a.root, rel))
    blocks, skipped = {}, []
    pool = R("ext/train_mix.jsonl")
    if all(have("r13/%s/train.jsonl" % n) for n in ("gqa", "vismin", "synth-r13p75")):
        per = q["binding"] // 3
        blocks["binding"] = sum((r13b.take(R("r13/%s/train.jsonl" % n), lambda s: s["group"], per, rng)
                                 for n in ("gqa", "vismin", "synth-r13p75")), [])
    else:
        skipped.append("binding (data/r13 sources missing)")
    blocks["existence"], _ = r13b.build_existence(pool, rng, q["existence"])
    first = lambda s: next(iter(zip(s["questions"].values(), s["gold"].values())))
    count_pool = [s for s in pool if s.get("source") in r13b.OLD_COUNT and r13b.is_count(*first(s))]
    blocks["count_v12"] = r13b.sample_rows(count_pool, q["count_v12"], rng)
    sci = sorted((s for s in pool if s.get("source") in r13b.SCIENCE), key=lambda s: s["id"])
    if q["science"] > len(sci):     # every science question once, then repeats (ids marked) up to the quota
        blocks["science"] = sci + [dict(s, id=s["id"] + ":rep1") for s in r13b.sample_rows(sci, q["science"] - len(sci), rng)]
    else:
        blocks["science"] = r13b.sample_rows(sci, q["science"], rng)
    used = {s["id"] for s in blocks["existence"] + blocks["count_v12"]}
    gen_ok = lambda s: s["id"] not in used and not (s.get("source") in r13b.OLD_COUNT and r13b.is_count(*first(s))) \
        and not (s.get("source") == "objects365" and first(s)[0]["type"] == "noul")
    gen = [s for s in pool if s.get("source") in r13b.GENERAL_FULL and gen_ok(s)]
    rest = max(0, q["general"] - len(gen))
    for src in r13b.GENERAL_SHARED:
        gen += r13b.sample_rows([s for s in pool if s.get("source") == src and gen_ok(s)], rest // len(r13b.GENERAL_SHARED), rng)
    blocks["general"] = gen
    per = q["ext"] // len(r13b.EXT)
    blocks["ext"] = [s for src in r13b.EXT for s in r13b.take([s for s in pool if s.get("source") == src],
                                                               lambda s: s["image_id"], per, rng)]
    dev = []
    for src in PUB:
        if not have("ext/%s/train.jsonl" % src):
            skipped.append("%s (not built)" % src)
            continue
        rows = R("ext/%s/train.jsonl" % src)
        blocks[src] = rows if q[src] >= len(rows) else r13b.take(rows, lambda s: s["image_id"], q[src], rng)
        if have("ext/%s/dev.jsonl" % src):
            dev += R("ext/%s/dev.jsonl" % src)
    mix = []
    for name, ss in blocks.items():
        for s in ss:
            s = dict(s, block=name)
            if name != "binding":
                s = r13b.as_picture_group(s)
            mix.append(s)
    dup = [i for i, c in Counter(s["id"] for s in mix).items() if c > 1]
    assert not dup, "duplicate ids: %s" % dup[:5]
    rng.shuffle(mix)
    os.makedirs(a.out, exist_ok=True)
    r13b.write(os.path.join(a.out, "train_mix.jsonl"), mix)
    r13b.write(os.path.join(a.out, "dev_pub.jsonl"), dev)
    manifest = {"questions": r13b.nq(mix), "steps_at_micro_batch_8": r13b.nq(mix) // 8, "quotas": q, "skipped": skipped,
                "blocks": {k: r13b.nq(v) for k, v in blocks.items()}, "dev_pub": len(dev),
                "count_option_rules": {"count_v12 (original format)": r13b.option_rules(blocks["count_v12"]),
                                       "pixmo_points": r13b.option_rules(blocks.get("pixmo_points", []))}}
    json.dump(manifest, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
