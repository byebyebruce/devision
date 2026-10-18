"""exp50 batch plan on the real code path, no model (critic EXP50-PREP-03): scripts/exp48/plan_check.py's check with
EXP48-X's tokenizer and config (the arms' start) -- data/exp42/train.jsonl -> rlcd._items -> random.seed(1) ->
plan_batches(8); every batch two complete four-corner quads (one original, one L), no quad split, 576 updates -- and the
plan must be EXP48's batch for batch (local views are enabled before the seeds and change no item). Also the distinct
pictures per batch (each encoded once with its 16 views). Writes data/exp50/PLAN.json.

    HF_HUB_OFFLINE=1 uv run python scripts/exp50/plan_check.py --root data
"""
import argparse
import hashlib
import importlib.util
import json
import os
from collections import Counter

_spec = importlib.util.spec_from_file_location("exp40_plan", os.path.join(os.path.dirname(__file__), "..", "exp40", "plan_check.py"))
assert _spec is not None and _spec.loader is not None
p40 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p40)
setattr(p40, "INIT", "runs/exp48-x/stage2")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    a = ap.parse_args(argv)
    r = p40.check(os.path.join(a.root, "exp42", "train.jsonl"), p40.stub())
    items, plan = r.pop("_items"), r.pop("_plan")
    quads = {(x["id"], x["qid"]): x for x in (json.loads(l) for l in open(os.path.join(a.root, "exp48", "quads.jsonl")))}
    where, per_batch, pictures = {}, [], []
    for bi, b in enumerate(plan):
        qs = Counter()
        for i in b:
            q = quads[(items[i]["sample_id"], items[i]["qid"])]
            qs[(q["quad"], q["layout"])] += 1
            where.setdefault(q["quad"], set()).add(bi)
        per_batch.append(sorted(qs.items()))
        pictures.append(len({items[i]["image"] for i in b}))
    r.update(init=p40.INIT, quads=len(where), quads_split=sum(len(v) > 1 for v in where.values()),
             batches_two_whole_quads_O_and_L=all(len(b) == 2 and all(n == 4 for _, n in b) and {lay for (_, lay), _ in b} == {"O", "L"}
                                                 for b in per_batch),
             pictures_per_batch=dict(Counter(pictures)),
             plan_sha256=hashlib.sha256(json.dumps([[items[i]["sample_id"] + "|" + items[i]["qid"] for i in b] for b in plan]).encode()).hexdigest())
    old = json.load(open(os.path.join(a.root, "exp48", "PLAN.json")))
    same = {k: json.loads(json.dumps(r[k])) == old[k] for k in ("items", "updates", "batch_sizes", "units", "max_unit", "split_units", "quads",
                                         "quads_split", "batches_two_whole_quads_O_and_L")}
    r["same_as_exp48"] = same
    assert r["updates"] == 576 and r["quads"] == 1152 and r["quads_split"] == 0 and r["batches_two_whole_quads_O_and_L"] and all(same.values())
    os.makedirs(os.path.join(a.root, "exp50"), exist_ok=True)
    json.dump(r, open(os.path.join(a.root, "exp50", "PLAN.json"), "w"), indent=1)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
