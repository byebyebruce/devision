"""Round 14b (r14b-mirror) training mix: r14-count's recipe with a reserved left/right mirror-pair and VSR replay
(docs/experiments/2026-10-13-r14b-mirror.md). r14 diluted them (VSR 53 questions, no mirror pairs) and lost
test_relation_flip / test_position_flip / test_vsr.

    uv run python scripts/r14b/build.py --root data      # -> data/r14b/train.jsonl + MANIFEST.json

  as r14            every block identical to r14's, except v12-ext cut to 2,100 per source (-3,000)
  mirror            1,000 v6 left/right pairs (original + mirrored picture, 2,000 questions): 600 relation, 400 position
  vsr               1,000 v12 pool VSR questions not already in the general replay
No picture of the mirror / VSR blocks is in an evaluation set of configs/r14b-mirror.yaml (checked).
"""
import argparse
import importlib.util
import os
import random

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("r14_build", os.path.join(HERE, "..", "r14", "build.py"))
r14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r14)

r13b = r14.r13b
CONFIG = os.path.join(HERE, "..", "..", "configs", "r14b-mirror.yaml")


def eval_pictures(root):
    cfg = yaml.safe_load(open(CONFIG))
    paths = [v if isinstance(v, str) else v["path"] for v in cfg["evaluate"]["sets"].values()]
    paths += list(cfg["stages"][0]["eval"].values()) + [cfg["stages"][0]["val"]]
    ids, files = set(), set()
    for p in paths:
        for s in r13b.rows(os.path.join(root, os.path.relpath(p, "data"))):
            ids.add(s["image_id"].removesuffix(":flip"))
            files.add(os.path.basename(s["image"]))
    return ids, files


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    p.add_argument("--out", default="data/r14b")
    a = p.parse_args()
    pool, _, blocks, info = r14.build_blocks(a.root)   # r14's blocks unchanged
    rng = random.Random(1414)
    blocks["ext"] = [s for src in r13b.EXT for s in r13b.take([s for s in blocks["ext"] if s.get("source") == src],
                                                               lambda s: s["image_id"], 2100, rng)]

    pairs = {}
    for s in r13b.rows(os.path.join(a.root, "v6/train_flip_mix.jsonl")):
        if s.get("axis") == "lr":
            pairs.setdefault(s["id"].removeprefix("flip:"), []).append(s)
    assert all(len(v) == 2 for v in pairs.values()), "an unpaired mirror question"
    mirror = []
    for src, n in (("coco-relation", 600), ("coco-position", 400)):
        keys = sorted(k for k, v in pairs.items() if v[0]["source"] == src)
        for k in rng.sample(keys, n):
            mirror += sorted(pairs[k], key=lambda s: s["id"])
    blocks["mirror"] = mirror
    used = {s["id"] for ss in blocks.values() for s in ss}
    blocks["vsr"] = r13b.sample_rows([s for s in pool if s.get("source") == "vsr" and s["id"] not in used], 1000, rng)

    ids, files = eval_pictures(a.root)
    for b in ("mirror", "vsr"):
        bad = [s["id"] for s in blocks[b] if s["image_id"].removesuffix(":flip") in ids or os.path.basename(s["image"]) in files]
        assert not bad, "%s: %d pictures in an evaluation set, e.g. %s" % (b, len(bad), bad[:3])
    r14.write_mix(a.out, blocks, info, replay=r14.REPLAY + ("mirror", "vsr"))


if __name__ == "__main__":
    main()
