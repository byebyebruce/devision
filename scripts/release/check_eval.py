"""Before a release reuses evaluation results (docs/experiments/2026-10-17-v04-delivery-plan.md, "发布证据准备"): a
result file existing is not enough. For every set of an evaluation YAML (and each planned control), check that the
summary was made by the YAML's checkpoint with its fitted temperatures and the planned control, on the current
bytes of the data file, with the role the YAML assigns, and that the per-question details are complete (one record
per question of the data file -- choice questions only for the reversed control -- no duplicates). Prints one line per problem; exit 1 if any.

    uv run python scripts/release/check_eval.py configs/v04-full-eval.yaml
"""
import hashlib
import json
import os
import sys
from typing import List

import yaml

from devision.train.pipeline import build_plan


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def questions(path: str, choice_only: bool = False) -> set:
    """(sample id, qid) of the data file; the reversed-options control asks only the choice questions."""
    keys = set()
    for line in open(path):
        if line.strip():
            s = json.loads(line)
            keys |= {(s["id"], q) for q, spec in s["questions"].items() if not choice_only or spec["type"] == "choice"}
    return keys


def check_one(summary: str, data: str, checkpoint: str, control: str, role: str, want: set) -> List[str]:
    if not os.path.exists(summary):
        return ["%s: missing" % summary]
    run = json.load(open(summary)).get("run", {})
    problems = []
    for field, expect in (("checkpoint", checkpoint), ("data_sha256", sha256(data)), ("temperature", "fitted"),
                          ("control", control), ("role", role)):
        if run.get(field) != expect:
            problems.append("%s: run.%s is %r, expected %r" % (summary, field, run.get(field), expect))
    details = summary[: -len(".json")] + ".details.jsonl"
    if not os.path.exists(details):
        return problems + ["%s: missing" % details]
    got = [(r["sample_id"], r["qid"]) for r in map(json.loads, filter(str.strip, open(details)))]
    if len(got) != len(set(got)):
        problems.append("%s: %d duplicate records" % (details, len(got) - len(set(got))))
    if set(got) != want:
        problems.append("%s: %d of %d questions missing, %d extra" % (details, len(want - set(got)), len(want), len(set(got) - want)))
    return problems


def check(config: str) -> List[str]:
    plan = build_plan(yaml.safe_load(open(config)))
    sets = yaml.safe_load(open(config))["evaluate"]["sets"]
    problems = []
    for name, out in plan.eval_outputs.items():
        spec = sets[name]
        data = spec["path"] if isinstance(spec, dict) else spec
        want = questions(data)
        problems += check_one(out, data, plan.checkpoint, "none", plan.roles[name], want)
        for control, ctl in plan.control_outputs.get(name, {}).items():
            problems += check_one(ctl, data, plan.checkpoint, control, plan.roles[name],
                                  questions(data, choice_only=True) if control == "reversed" else want)
    return problems


if __name__ == "__main__":
    found = check(sys.argv[1])
    for p in found:
        print(p)
    print("%d problems" % len(found))
    sys.exit(1 if found else 0)
