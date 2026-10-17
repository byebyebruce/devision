"""exp40 batch plan on the real code path, no model (critic EXP40-PREP-01): each arm's file through
samples.read_decision_jsonl -> rlcd._items (exp37-L's own tokenizer and config) -> rlcd.plan_batches(micro 8) after
random.seed(1), as rlcd.train seeds it (the run's log gives the actual update count). Checks and writes to
data/exp40/PLAN.json: every question becomes an item (none dropped or merged), no (group, qid) unit is larger than
8 or split across batches, real-layout unit sizes (8 for a reading field, 4 for the comparison), the number of
updates; and that the two arms give the same items apart from the picture and the same batch plan.

    uv run python scripts/exp40/plan_check.py --root data
"""
import argparse
import json
import os
import random
from collections import Counter
from types import SimpleNamespace

from transformers import AutoTokenizer

from devision.model import ModelConfig
from devision.train.rlcd import _items, plan_batches
from devision.train.samples import read_decision_jsonl

INIT = "runs/exp37-l/stage1"


def stub():
    cfg = json.load(open(os.path.join(INIT, "config.json")))
    return SimpleNamespace(tok=AutoTokenizer.from_pretrained(os.path.join(INIT, "tokenizer")),
                           cfg=ModelConfig(**{k: v for k, v in cfg.items() if k in ModelConfig.__dataclass_fields__}),
                           model=None)


def check(path: str, decider) -> dict:
    samples = read_decision_jsonl(path)
    items = _items(decider, samples)
    assert len(items) == sum(len(s["questions"]) for s in samples)
    random.seed(1)
    plan = plan_batches(items, 8)
    flat = sorted(i for b in plan for i in b)
    assert flat == list(range(len(items))), "an item is missing or repeated in the plan"
    units: dict = {}
    for i, it in enumerate(items):
        if "group" in it:
            units.setdefault((it["group"], it["qid"]), []).append(i)
    where = {i: k for k, b in enumerate(plan) for i in b}
    split = [u for u, v in units.items() if len({where[i] for i in v}) > 1]
    rl_sizes = Counter((it["qid"], len(units[(it["group"], it["qid"])])) for it in items if it["source"].startswith("real-layout"))
    return {"items": len(items), "updates": len(plan), "batch_sizes": dict(Counter(map(len, plan))),
            "units": len(units), "max_unit": max(map(len, units.values())), "split_units": len(split),
            "real_layout_unit_sizes": {"%s:%d" % k: v for k, v in sorted(rl_sizes.items())},
            "_plan": plan, "_items": items}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    a = p.parse_args(argv)
    d = stub()
    res = {arm: check(os.path.join(a.root, "exp40", "train_%s.jsonl" % arm), d) for arm in ("O", "M")}
    o, m = res["O"], res["M"]
    keys = ("ids", "markers", "target", "qid", "group", "sample_id", "source", "keys")
    same_items = all(all(x[k] == y[k] for k in keys) for x, y in zip(o["_items"], m["_items"]))
    image_differs = sum(x["image"] != y["image"] for x, y in zip(o["_items"], m["_items"]))
    out: dict = {arm: {k: v for k, v in r.items() if not k.startswith("_")} for arm, r in res.items()}
    out["arms_same_items_apart_from_image"] = same_items
    out["items_with_a_different_image"] = image_differs
    out["arms_same_plan"] = o["_plan"] == m["_plan"]
    for arm in ("O", "M"):
        assert out[arm]["max_unit"] <= 8 and out[arm]["split_units"] == 0
    assert same_items and out["arms_same_plan"]
    json.dump(out, open(os.path.join(a.root, "exp40", "PLAN.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
