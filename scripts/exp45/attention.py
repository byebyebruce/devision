"""exp45 attention observation (critic EXP45-PREP-02; mechanism only, never a gate): with a frozen S or G checkpoint, the
question-conditioning attention (heads and visual-token queries averaged, over the 256 SigLIP patches) of every reading
question of a set, through the same item construction and forward as training (rlcd._items / _forward), 8 questions
per pass, no gradient. Per question: the attention mass on the asked field's box (the sidecar's cells with any
coverage), on the other object's same-field box, the box's share of the patches (the uniform baseline), whether the
most attended patch is in the box, and that patch. Boxes come from data/exp45/regions_readout.jsonl and are never an
input. Writes <out>/<name>.json and <name>.details.jsonl.

    uv run python scripts/exp45/attention.py runs/exp45-g/stage2 data/real_layout/questions_tune.jsonl rl_tune runs/exp45-g/eval attn_rl_tune
"""
import argparse
import hashlib
import json
import os
import sys
from typing import cast

import torch
from laya.common import collate_items

from devision.infer import Decider, best_device
from devision.train.rlcd import _forward, _Images, _items, box_grid
from devision.train.samples import read_decision_jsonl


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("data")
    p.add_argument("set", help="the set name in regions_readout.jsonl")
    p.add_argument("out")
    p.add_argument("name")
    p.add_argument("--regions", default="data/exp45/regions_readout.jsonl")
    p.add_argument("--data-root", default="data")
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    device = best_device() if a.device == "auto" else a.device
    d = Decider.load(a.checkpoint, device=device)
    if d.model.condition is None:
        raise SystemExit("%s has no question conditioning module" % a.checkpoint)
    d.model.condition.keep_weights = True
    regs = {}
    for line in open(a.regions):
        r = json.loads(line)
        if r["set"] == a.set:
            regs[(r["id"], r["qid"])] = r
    by_field = {(r["picture"], r["field"], r["object"]): r for r in regs.values()}
    samples = [s for s in read_decision_jsonl(a.data)]
    samples = [dict(s, questions={q: v for q, v in s["questions"].items() if (s["id"], q) in regs},
                    gold={q: g for q, g in s["gold"].items() if (s["id"], q) in regs}) for s in samples]
    samples = [s for s in samples if s["questions"]]
    items = _items(d, samples)
    images = _Images(a.data_root, d)
    side = d.cfg.image_size // int(cast(int, d.model.vision.config.patch_size))
    recs = []
    with torch.no_grad():
        for i in range(0, len(items), 8):
            chunk = items[i:i + 8]
            batch = collate_items([chunk], d.tok.pad_token_id)
            assert batch is not None
            pixels = torch.stack([images(it["image"]) for it in chunk])
            _forward(d.model, batch, pixels, device)
            w = cast(torch.Tensor, d.model.condition.last_weights).float().cpu()
            for it, row in zip(chunk, w.tolist()):
                r = regs[(it["sample_id"], it["qid"])]
                other = by_field[(r["picture"], r["field"], "B" if r["object"] == "A" else "A" if r["object"] == "B"
                                  else "2" if r["object"] == "1" else "1")]
                cells = [c > 0 for c in box_grid([r["box"]], r["width"], r["height"], d.cfg.image_size, grid=side, smooth=0.0)]
                ocells = [c > 0 for c in box_grid([other["box"]], other["width"], other["height"], d.cfg.image_size, grid=side, smooth=0.0)]
                top = max(range(len(row)), key=row.__getitem__)
                recs.append({"sample_id": it["sample_id"], "qid": it["qid"], "picture": r["picture"], "field": r["field"],
                             "object": r["object"], "side": r["side"], "layout": r["layout"],
                             "mass_target": sum(v for v, c in zip(row, cells) if c),
                             "mass_other": sum(v for v, c in zip(row, ocells) if c),
                             "uniform_share": sum(cells) / len(cells), "top_in_target": bool(cells[top]), "top_patch": top})
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, a.name + ".details.jsonl"), "w") as fh:
        for x in recs:
            fh.write(json.dumps(x) + "\n")
    info = {"checkpoint": a.checkpoint, "data": a.data, "data_sha256": hashlib.sha256(open(a.data, "rb").read()).hexdigest(),
            "regions_sha256": hashlib.sha256(open(a.regions, "rb").read()).hexdigest(), "set": a.set, "device": device,
            "questions": len(recs)}
    json.dump({"run": info}, open(os.path.join(a.out, a.name + ".json"), "w"), indent=1)
    print("%s: %d questions, mean mass on the box %.3f (uniform %.3f)" % (
        a.name, len(recs), sum(x["mass_target"] for x in recs) / len(recs), sum(x["uniform_share"] for x in recs) / len(recs)))


if __name__ == "__main__":
    sys.exit(main())
