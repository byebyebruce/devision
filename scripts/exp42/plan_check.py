"""exp42 batch plan on the real code path, no model (critic EXP42-PREP-01): data/exp42/train.jsonl through
read_decision_jsonl -> rlcd._items (exp40-M's own tokenizer and config) -> random.seed(1) -> plan_batches(8), with
scripts/exp40/plan_check.py's check (unchanged). Both arms read the same file and the plan is drawn before the first
update, so it is the same for both objectives. Writes data/exp42/PLAN.json: 4,608 items, 576 updates of 8, every
(group, qid) unit exactly 8 and whole, each batch one unit.

    uv run python scripts/exp42/plan_check.py --root data
"""
import argparse
import importlib.util
import json
import os

_spec = importlib.util.spec_from_file_location("exp40_plan", os.path.join(os.path.dirname(__file__), "..", "exp40", "plan_check.py"))
assert _spec is not None and _spec.loader is not None
p40 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p40)
setattr(p40, "INIT", "runs/exp40-m/stage2")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    a = p.parse_args(argv)
    r = p40.check(os.path.join(a.root, "exp42", "train.jsonl"), p40.stub())
    items, plan = r.pop("_items"), r.pop("_plan")
    unit_of = lambda i: (items[i]["group"], items[i]["qid"])  # noqa: E731
    r["batches_one_unit"] = all(len({unit_of(i) for i in b}) == 1 for b in plan)
    r["init"] = p40.INIT
    assert r["items"] == 4608 and r["updates"] == 576 and r["batch_sizes"] == {8: 576} and r["max_unit"] == 8 \
        and r["split_units"] == 0 and r["batches_one_unit"]
    json.dump(r, open(os.path.join(a.root, "exp42", "PLAN.json"), "w"), indent=1)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
