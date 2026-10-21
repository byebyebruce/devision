"""exp48 batch plan on the real code path, no model (critic EXP48-PREP-03): data/exp42/train.jsonl through read_decision_jsonl
-> rlcd._items (EXP45-G's tokenizer and config) -> random.seed(1) -> plan_batches(8) (scripts/exp40/plan_check.py's check),
then every batch must hold exactly two complete four-corner quads of data/exp48/quads.jsonl -- one on the original and one
on the L picture -- and every quad must lie in one batch. Writes data/exp48/PLAN.json.

    HF_HUB_OFFLINE=1 uv run python scripts/exp48/plan_check.py --root data
"""
import argparse
import importlib.util
import json
import os
from collections import Counter

_spec = importlib.util.spec_from_file_location("exp40_plan", os.path.join(os.path.dirname(__file__), "..", "exp40", "plan_check.py"))
assert _spec is not None and _spec.loader is not None
p40 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p40)
setattr(p40, "INIT", "runs/exp45-g/stage2")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    a = ap.parse_args(argv)
    r = p40.check(os.path.join(a.root, "exp42", "train.jsonl"), p40.stub())
    items, plan = r.pop("_items"), r.pop("_plan")
    quads = {(x["id"], x["qid"]): x for x in (json.loads(l) for l in open(os.path.join(a.root, "exp48", "quads.jsonl")))}
    where, per_batch = {}, []
    for bi, b in enumerate(plan):
        qs = Counter()
        for i in b:
            q = quads[(items[i]["sample_id"], items[i]["qid"])]
            qs[(q["quad"], q["layout"])] += 1
            where.setdefault(q["quad"], set()).add(bi)
        per_batch.append(sorted(qs.items()))
    r.update(init=p40.INIT, quads=len(where), quads_split=sum(len(v) > 1 for v in where.values()),
             batches_two_whole_quads_O_and_L=all(len(b) == 2 and all(n == 4 for _, n in b) and {lay for (_, lay), _ in b} == {"O", "L"}
                                                 for b in per_batch))
    assert r["updates"] == 576 and r["quads"] == 1152 and r["quads_split"] == 0 and r["batches_two_whole_quads_O_and_L"]
    json.dump(r, open(os.path.join(a.root, "exp48", "PLAN.json"), "w"), indent=1)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
