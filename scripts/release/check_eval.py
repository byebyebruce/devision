"""Evaluation results a release may use (docs/experiments/2026-10-17-v04-release-draft.md; critic V04-REL-R1-01 /
02): a result file existing is not evidence. Four steps, in this order, around one `devision-pipeline` run:

  freeze     before the candidate's first final evaluation: write <eval dir>/IDENTITY.json -- sha256 of every file
             of the checkpoint (weights, config with its temperatures, tokenizer, ...) and the temperature table;
             refuses to overwrite a different identity.
  preflight  for every planned output (each set and each planned control): ok / missing / invalid. ok = recorded by
             `seal` under the current identity with the same file hashes and passing the content checks; anything
             else that exists is invalid (made before the freeze, by other weights, or wrong content).
             --quarantine moves invalid files to <eval dir>/_invalid/<time>/ so the pipeline evaluates them again
             (the pipeline skips a set whenever its summary exists).
  seal       after the pipeline run: the checkpoint still hashes to IDENTITY.json; every output written after the
             freeze that passes the content checks is recorded with its file hashes in <eval dir>/RESULTS.json.
  check      everything planned exists, is sealed under the current identity, unchanged, and passes the checks.

Content checks (one output): the summary's run fields (checkpoint, data sha256, fitted temperature, control, role)
and its temperature table equal the frozen ones; the details hold each planned question exactly once (reversed:
choice questions only) with the data file's type and gold, the options of the data file (reversed: in reverse
order), the picture of the data file (mismatched: another picture), finite probabilities over exactly the options
summing to 1, the arg-max as prediction and `correct` matching the gold; the summary's n and accuracy equal the
details'. `--legacy` (historical results without an identity, e.g. v0.3's) runs the content checks only and says
that identity is not proven.

    uv run python scripts/release/check_eval.py freeze|preflight [--quarantine]|seal|check configs/v04-full-eval.yaml
    uv run python scripts/release/check_eval.py check --legacy configs/r15-vision.yaml
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import yaml

from devision.train.pipeline import build_plan


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_hashes(directory: str) -> Dict[str, str]:
    out = {}
    for base, _, files in os.walk(directory):
        for f in files:
            p = os.path.join(base, f)
            out[os.path.relpath(p, directory)] = sha256(p)
    return dict(sorted(out.items()))


def temperature_table(checkpoint: str) -> Dict:
    """What devision-eval records as `temperature` for this checkpoint's fitted temperatures."""
    from devision.infer import config_path
    c = json.load(open(config_path(checkpoint)))
    t = c["temperature"]
    return {"choice": t[0], "noul": t[2], "by_options": c.get("temperature_by_options") or {}}


class Plan:
    def __init__(self, config: str):
        cfg = yaml.safe_load(open(config))
        self.plan = build_plan(cfg)
        if not self.plan.checkpoint or not self.plan.eval_dir:
            raise SystemExit("%s evaluates no checkpoint" % config)
        self.checkpoint: str = self.plan.checkpoint
        self.eval_dir: str = self.plan.eval_dir
        self.outputs: List[Tuple[str, str, str, str]] = []      # (summary, data, control, role)
        for name, out in self.plan.eval_outputs.items():
            spec = cfg["evaluate"]["sets"][name]
            data = spec["path"] if isinstance(spec, dict) else spec
            self.outputs.append((out, data, "none", self.plan.roles[name]))
            for control, ctl in self.plan.control_outputs.get(name, {}).items():
                self.outputs.append((ctl, data, control, self.plan.roles[name]))

    def path(self, name: str) -> str:
        return os.path.join(self.eval_dir, name)


def details_of(summary: str) -> str:
    return summary[: -len(".json")] + ".details.jsonl"


def load_data(path: str) -> Dict[Tuple[str, str], Dict]:
    out = {}
    for line in open(path):
        if line.strip():
            s = json.loads(line)
            for qid, q in s["questions"].items():
                out[(s["id"], qid)] = {"type": q["type"], "options": list(q["criteria"]) if q["type"] == "choice" else ["false", "true"],
                                       "image": s["image"],
                                       "gold": s["gold"][qid]["probabilities"]}
    return out


def content(summary: str, data: str, checkpoint: str, control: str, role: str,
            temperature: Optional[Dict], cache: Dict[str, Dict]) -> List[str]:
    """The content checks of one output; [] when it passes."""
    if not os.path.exists(summary):
        return ["%s: missing" % summary]
    details = details_of(summary)
    if not os.path.exists(details):
        return ["%s: missing" % details]
    s = json.load(open(summary))
    run = s.get("run", {})
    problems = []
    if data not in cache:
        cache[data] = {"sha": sha256(data), "q": load_data(data)}
    for field, expect in (("checkpoint", checkpoint), ("data_sha256", cache[data]["sha"]), ("temperature", "fitted"),
                          ("control", control), ("role", role)):
        if run.get(field) != expect:
            problems.append("%s: run.%s is %r, expected %r" % (summary, field, run.get(field), expect))
    if temperature is not None and s.get("temperature") != temperature:
        problems.append("%s: temperature table %r differs from the frozen %r" % (summary, s.get("temperature"), temperature))
    want = cache[data]["q"]
    if control == "reversed":
        want = {k: v for k, v in want.items() if v["type"] == "choice"}
    recs = [json.loads(x) for x in open(details) if x.strip()]
    keys = [(r["sample_id"], r["qid"]) for r in recs]
    if len(keys) != len(set(keys)):
        problems.append("%s: %d duplicate records" % (details, len(keys) - len(set(keys))))
    if set(keys) != set(want):
        problems.append("%s: %d of %d questions missing, %d extra" % (
            details, len(set(want) - set(keys)), len(want), len(set(keys) - set(want))))
    bad: Dict[str, int] = {}
    for r in recs:
        q = want.get((r["sample_id"], r["qid"]))
        if q is None:
            continue
        options = q["options"][::-1] if control == "reversed" else q["options"]
        p = r.get("probabilities") or {}
        vals = list(p.values())
        checks = {
            "type": r.get("type") == q["type"],
            "gold": r.get("gold") == q["gold"],
            "options": r.get("options") == options,
            "picture": (r.get("image_used") != q["image"]) if control == "mismatched" else (r.get("image_used") == q["image"]),
            "probabilities": set(p) == set(q["options"]) and all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)
            and abs(sum(vals) - 1) < 1e-3,
            "prediction": bool(p) and r.get("prediction") == max(p, key=lambda k: p[k]),
            "correct": r.get("correct") == (q["gold"].get(r.get("prediction"), 0.0) == max(q["gold"].values())),
        }
        for k, ok in checks.items():
            if not ok:
                bad[k] = bad.get(k, 0) + 1
    problems += ["%s: %d records with wrong %s" % (details, n, k) for k, n in sorted(bad.items())]
    if s.get("n") != len(recs):
        problems.append("%s: n %r but %d records" % (summary, s.get("n"), len(recs)))
    elif recs and abs(s.get("accuracy_all", -1) - sum(bool(r.get("correct")) for r in recs) / len(recs)) > 1e-9:
        problems.append("%s: accuracy_all %r differs from the details" % (summary, s.get("accuracy_all")))
    return problems


def identity_problems(p: Plan) -> Tuple[Optional[Dict], List[str]]:
    path = p.path("IDENTITY.json")
    if not os.path.exists(path):
        return None, ["%s: missing (run freeze before the first final evaluation)" % path]
    ident = json.load(open(path))
    if ident.get("checkpoint") != p.checkpoint:
        return ident, ["identity is for %s, the config evaluates %s" % (ident.get("checkpoint"), p.checkpoint)]
    now = tree_hashes(p.checkpoint)
    changed = sorted(k for k in set(now) | set(ident["files"]) if now.get(k) != ident["files"].get(k))
    return ident, (["checkpoint files differ from IDENTITY.json: %s" % ", ".join(changed[:5])] if changed else [])


def sealed(p: Plan) -> Dict[str, Dict[str, str]]:
    path = p.path("RESULTS.json")
    return json.load(open(path)) if os.path.exists(path) else {}


def status(p: Plan, ident: Optional[Dict], ident_bad: List[str], cache: Dict) -> List[Tuple[str, str, List[str]]]:
    """(summary, ok | missing | invalid, reasons) for every planned output."""
    record = sealed(p)
    out = []
    for summary, data, control, role in p.outputs:
        files = [summary, details_of(summary)]
        if not any(os.path.exists(f) for f in files):
            out.append((summary, "missing", []))
            continue
        why = list(ident_bad)
        rec = record.get(os.path.basename(summary))
        if rec is None:
            why.append("not sealed under the frozen identity")
        elif any(not os.path.exists(f) or sha256(f) != rec.get(os.path.basename(f)) for f in files):
            why.append("changed since it was sealed")
        why += content(summary, data, p.checkpoint, control, role, ident["temperature"] if ident else None, cache)
        out.append((summary, "invalid" if why else "ok", why))
    return out


def cmd_freeze(p: Plan) -> int:
    path = p.path("IDENTITY.json")
    ident = {"checkpoint": p.checkpoint, "files": tree_hashes(p.checkpoint), "temperature": temperature_table(p.checkpoint),
             "frozen": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "frozen_epoch": time.time()}
    if os.path.exists(path):
        old = json.load(open(path))
        if old["files"] != ident["files"] or old["temperature"] != ident["temperature"] or old["checkpoint"] != p.checkpoint:
            print("refusing: %s holds a different identity" % path)
            return 1
        print("identity already frozen and unchanged: %s" % path)
        return 0
    os.makedirs(p.eval_dir, exist_ok=True)
    json.dump(ident, open(path, "w"), indent=1)
    print("froze %s: model.safetensors %s" % (path, ident["files"].get("model.safetensors")))
    return 0


def cmd_preflight(p: Plan, quarantine: bool) -> int:
    ident, bad = identity_problems(p)
    if ident is None:
        print(bad[0])
        return 1
    rows = status(p, ident, bad, {})
    counts = {k: sum(r[1] == k for r in rows) for k in ("ok", "missing", "invalid")}
    for summary, st, why in rows:
        if st == "invalid":
            print("invalid %s: %s" % (summary, "; ".join(why[:3])))
    print("ok %(ok)d, missing %(missing)d (the pipeline evaluates them), invalid %(invalid)d" % counts)
    if bad:
        print("the checkpoint does not match its frozen identity: stop, do not evaluate")
        return 1
    if counts["invalid"] and quarantine:
        dest = p.path(os.path.join("_invalid", time.strftime("%Y%m%d-%H%M%S")))
        os.makedirs(dest)
        for summary, st, _ in rows:
            if st == "invalid":
                for f in (summary, details_of(summary)):
                    if os.path.exists(f):
                        shutil.move(f, os.path.join(dest, os.path.basename(f)))
        print("moved the invalid outputs to %s" % dest)
        return 0
    return 1 if counts["invalid"] else 0


def cmd_seal(p: Plan) -> int:
    ident, bad = identity_problems(p)
    if ident is None or bad:
        print("; ".join(bad))
        return 1
    record, cache, problems = sealed(p), {}, []
    for summary, data, control, role in p.outputs:
        files = [summary, details_of(summary)]
        if not all(os.path.exists(f) for f in files) or os.path.basename(summary) in record:
            continue
        if min(os.path.getmtime(f) for f in files) < ident["frozen_epoch"]:
            problems.append("%s: written before the freeze, not sealed" % summary)
            continue
        why = content(summary, data, p.checkpoint, control, role, ident["temperature"], cache)
        if why:
            problems += why
            continue
        record[os.path.basename(summary)] = {os.path.basename(f): sha256(f) for f in files}
    json.dump(record, open(p.path("RESULTS.json"), "w"), indent=1, sort_keys=True)
    for x in problems:
        print(x)
    print("sealed %d outputs, %d problems" % (len(record), len(problems)))
    return 1 if problems else 0


def cmd_check(p: Plan, legacy: bool) -> int:
    cache: Dict = {}
    if legacy:
        problems = [x for summary, data, control, role in p.outputs
                    for x in content(summary, data, p.checkpoint, control, role, None, cache)]
        for x in problems:
            print(x)
        print("%d problems (content only; identity not proven -- historical results have no frozen identity)" % len(problems))
        return 1 if problems else 0
    ident, bad = identity_problems(p)
    rows = status(p, ident, bad, cache) if ident else []
    problems = list(bad) + ["%s: %s %s" % (s, st, "; ".join(w[:3])) for s, st, w in rows if st != "ok"]
    for x in problems:
        print(x)
    print("%d problems%s" % (len(problems), "" if ident else " (no identity)"))
    return 1 if problems or ident is None else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("freeze", "preflight", "seal", "check"))
    ap.add_argument("config")
    ap.add_argument("--quarantine", action="store_true")
    ap.add_argument("--legacy", action="store_true")
    a = ap.parse_args(argv)
    p = Plan(a.config)
    if a.command == "freeze":
        return cmd_freeze(p)
    if a.command == "preflight":
        return cmd_preflight(p, a.quarantine)
    if a.command == "seal":
        return cmd_seal(p)
    return cmd_check(p, a.legacy)


if __name__ == "__main__":
    sys.exit(main())
