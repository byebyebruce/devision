"""PANEL-D2-R1-01: D2's real training plan, from the actual train.jsonl with rlcd's own planner (plan_mixed, units
(group, qid) whole, one input mode per micro batch), the way rlcd expands rows -- one item per question -- without a
model or tokenizer. Prints updates per epoch, unit sizes, items / updates per input and source, the monitor items per
round, and the time arithmetic of the 95-minute D2 budget at the measured / assumed per-update and per-item costs.

    uv run python scripts/panel_learn/plan.py [--epochs 1] [--seed 1] [--micro 8]
"""
import argparse
import json
import random
from collections import Counter

from devision.train.rlcd import plan_mixed

TRAIN = "data/panel_learn/train.jsonl"
MONITORS = {"val(fit)": "data/panel_learn/fit.jsonl", "dev16": "data/panel_learn/dev16.jsonl", "erased": "data/panel_learn/erased.jsonl",
            "k_text": "data/protect_p/slice_K_text.jsonl", "rs_temp": "data/panel_learn/read_states_sqa_temp.jsonl",
            "rs_force": "data/panel_learn/read_states_sqa_force.jsonl", "rs_temp_mism": "data/panel_learn/read_states_sqa_temp_mism.jsonl",
            "rs_force_mism": "data/panel_learn/read_states_sqa_force_mism.jsonl"}
STEP_S = (2.6, 3.0)          # PROTECT-K, the same mixed_input path (measured 2.6 s per update); panel rows assumed up to 3.0
PICTURE_ITEM_S = 0.11        # forward-only, picture item (13 views), about a third of a training update per item
TEXT_ITEM_S = 0.02
PROBE_S = 5.0 * 60           # exist (572 requests) + science (269) through devision-eval, with two model loads
LOAD_S = 90


def items(path):
    out = []
    for r in map(json.loads, open(path)):
        for qid in r["questions"]:
            out.append({"input": r["input"], "group": r["group"], "qid": qid, "source": r.get("source", "?")})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--micro", type=int, default=8)
    a = ap.parse_args(argv)
    its = items(TRAIN)
    random.seed(a.seed)
    plans = [plan_mixed(its, a.micro) for _ in range(a.epochs)]
    units = Counter((it["group"], it["qid"], it["input"]) for it in its)
    by_input = Counter(it["input"] for it in its)
    by_source = Counter(it["source"] for it in its)
    upd_source = Counter()
    for bt in plans[0]:
        for s in {its[i]["source"] for i in bt}:
            upd_source[s] += 1
    mon = {n: Counter(it["input"] for it in items(p)) for n, p in MONITORS.items()}
    pic = sum(c["picture"] for c in mon.values())
    txt = sum(c["text"] for c in mon.values())
    round_s = pic * PICTURE_ITEM_S + txt * TEXT_ITEM_S
    updates = sum(len(p) for p in plans)
    train_s = [updates * s for s in STEP_S]
    total = [t + 3 * round_s + PROBE_S + LOAD_S for t in train_s]   # monitor rounds: step 0, epoch end, final
    out = {"epochs": a.epochs, "items": len(its), "by_input": dict(by_input), "updates_per_epoch": [len(p) for p in plans],
           "updates": updates, "largest_unit": max(units.values()), "units": len(units),
           "items_by_source": dict(by_source.most_common()), "updates_with_source_epoch1": dict(upd_source.most_common()),
           "monitor_items": {n: dict(c) for n, c in mon.items()}, "monitor_round_min": round(round_s / 60, 1),
           "train_min": [round(t / 60, 1) for t in train_s], "total_min": [round(t / 60, 1) for t in total], "budget_min": 95}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
