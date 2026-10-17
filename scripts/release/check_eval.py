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
details'. A question id that a data file holds more than once (data/r13b/val.jsonl: 242 ids twice, with different
options) is matched copy by copy on its options, so every copy must be there once. The mismatched control must show
exactly the picture evaluate.picture_pairs(samples, seed) assigns (seed recorded in the summary, 0 expected), and
every probability lies in [0, 1]. `--legacy` (historical results without an identity, e.g. v0.3's) runs the
content checks only and says that identity is not proven.

`--extras <yaml>` adds outputs outside the evaluation YAML to the same frozen identity (critic V04-REL-R2-04): a
list of {name, data, temperature: fitted | raw, control, role}, written to <eval dir>/extra_<name>.json;
`run-extras` evaluates the missing ones with devision-eval (raw = --no-temperature). freeze / preflight / seal /
check then cover them like the planned sets; calibration.py and delivery.py read only outputs that pass `check`.

    uv run python scripts/release/check_eval.py freeze|preflight [--quarantine]|seal|check configs/v04-full-eval.yaml
    uv run python scripts/release/check_eval.py run-extras --extras configs/v04-extra-evidence.yaml configs/v04-full-eval.yaml
    uv run python scripts/release/check_eval.py check --legacy configs/r15-vision.yaml
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import subprocess
import time
from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional, Tuple

import yaml

from devision.train.evaluate import picture_pairs
from devision.train.pipeline import build_plan
from devision.train.samples import read_decision_jsonl

RAW_TEMPERATURE = {"choice": 1.0, "noul": 1.0, "by_options": {}}   # what devision-eval --no-temperature records


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


class Output(NamedTuple):
    summary: str
    data: str
    control: str
    role: str
    temperature: str = "fitted"            # fitted | raw


class Plan:
    def __init__(self, config: str, extras: Optional[str] = None):
        cfg = yaml.safe_load(open(config))
        self.plan = build_plan(cfg)
        if not self.plan.checkpoint or not self.plan.eval_dir:
            raise SystemExit("%s evaluates no checkpoint" % config)
        self.checkpoint: str = self.plan.checkpoint
        self.eval_dir: str = self.plan.eval_dir
        self.outputs: List[Output] = []
        for name, out in self.plan.eval_outputs.items():
            spec = cfg["evaluate"]["sets"][name]
            data = spec["path"] if isinstance(spec, dict) else spec
            self.outputs.append(Output(out, data, "none", self.plan.roles[name]))
            for control, ctl in self.plan.control_outputs.get(name, {}).items():
                self.outputs.append(Output(ctl, data, control, self.plan.roles[name]))
        self.extras: Dict[str, Output] = {}
        for e in (yaml.safe_load(open(extras)) or []) if extras else []:
            if e.get("temperature", "fitted") not in ("fitted", "raw") or e.get("control", "none") not in ("none", "mismatched", "reversed"):
                raise SystemExit("extra %r: temperature must be fitted | raw, control none | mismatched | reversed" % e.get("name"))
            o = Output(os.path.join(self.eval_dir, "extra_%s.json" % e["name"]), e["data"], e.get("control", "none"), e["role"],
                       e.get("temperature", "fitted"))
            self.extras[e["name"]] = o
            self.outputs.append(o)

    def path(self, name: str) -> str:
        return os.path.join(self.eval_dir, name)


def details_of(summary: str) -> str:
    return summary[: -len(".json")] + ".details.jsonl"


def load_data(path: str) -> Dict[Tuple[str, str], List[Dict]]:
    """(sample id, qid) -> every copy of that question in file order (an id may legally appear more than once)."""
    out: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for line in open(path):
        if line.strip():
            s = json.loads(line)
            for qid, q in s["questions"].items():
                out[(s["id"], qid)].append({"type": q["type"], "image": s["image"], "image_id": s["image_id"],
                                            "options": list(q["criteria"]) if q["type"] == "choice" else ["false", "true"],
                                            "gold": s["gold"][qid]["probabilities"]})
    return dict(out)


def match_copies(recs: List[Dict], want: Dict[Tuple[str, str], List[Dict]], control: str) -> Tuple[List[Tuple[Dict, Dict]], int, int]:
    """Pairs every record with one copy of its question (copies of one id matched on their option lists);
    returns (pairs, unmatched data copies, unmatched records)."""
    pool = {k: list(v) for k, v in want.items()}
    pairs, extra = [], 0
    for r in recs:
        copies = pool.get((r["sample_id"], r["qid"]), [])
        shown = r.get("options")
        hit = next((q for q in copies if (q["options"][::-1] if control == "reversed" else q["options"]) == shown), None)
        if hit is None and copies and len(want[(r["sample_id"], r["qid"])]) == 1:
            hit = copies[0]                  # a single copy: pair it, the option check reports the difference
        if hit is None:
            extra += 1
            continue
        copies.remove(hit)
        pairs.append((r, hit))
    return pairs, sum(len(v) for v in pool.values()), extra


def content(o: Output, checkpoint: str, temperature: Optional[Dict], cache: Dict[str, Dict]) -> List[str]:
    """The content checks of one output; [] when it passes. temperature: the frozen fitted table (None: legacy)."""
    summary, data, control, role = o.summary, o.data, o.control, o.role
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
    mode = "fitted" if o.temperature == "fitted" else "none (raw)"
    for field, expect in (("checkpoint", checkpoint), ("data_sha256", cache[data]["sha"]), ("temperature", mode),
                          ("control", control), ("role", role)):
        if run.get(field) != expect:
            problems.append("%s: run.%s is %r, expected %r" % (summary, field, run.get(field), expect))
    table = RAW_TEMPERATURE if o.temperature == "raw" else temperature
    if table is not None and s.get("temperature") != table:
        problems.append("%s: temperature table %r differs from the expected %r" % (summary, s.get("temperature"), table))
    shown: Dict[str, str] = {}
    if control == "mismatched":
        if run.get("seed") != 0:
            problems.append("%s: mismatched seed %r, expected 0" % (summary, run.get("seed")))
        if "pairs" not in cache[data]:
            cache[data]["pairs"] = picture_pairs(read_decision_jsonl(data), seed=0)
        shown = cache[data]["pairs"]
    want = cache[data]["q"]
    if control == "reversed":
        want = {k: [q for q in v if q["type"] == "choice"] for k, v in want.items()}
        want = {k: v for k, v in want.items() if v}
    recs = [json.loads(x) for x in open(details) if x.strip()]
    pairs, missing, extra = match_copies(recs, want, control)
    if missing or extra:
        problems.append("%s: %d of %d questions missing, %d extra or duplicate records" % (
            details, missing, sum(len(v) for v in want.values()), extra))
    bad: Dict[str, int] = {}
    for r, q in pairs:
        options = q["options"][::-1] if control == "reversed" else q["options"]
        p = r.get("probabilities") or {}
        vals = list(p.values())
        picture = shown.get(q["image_id"], q["image"]) if control == "mismatched" else q["image"]
        checks = {
            "type": r.get("type") == q["type"],
            "gold": r.get("gold") == q["gold"],
            "options": r.get("options") == options,
            "picture": "image_used" in r and r["image_used"] == picture and (control != "mismatched" or picture != q["image"]),
            "probabilities": set(p) == set(q["options"]) and len(vals) > 0
            and all(isinstance(v, (int, float)) and math.isfinite(v) and 0.0 <= v <= 1.0 for v in vals)
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
    for o in p.outputs:
        summary = o.summary
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
        why += content(o, p.checkpoint, ident["temperature"] if ident else None, cache)
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
        record, revoked = sealed(p), {}
        for summary, st, why in rows:
            if st == "invalid":
                for f in (summary, details_of(summary)):
                    if os.path.exists(f):
                        shutil.move(f, os.path.join(dest, os.path.basename(f)))
                if os.path.basename(summary) in record:      # its old seal goes with it (critic V04-REL-R2-01)
                    revoked[os.path.basename(summary)] = record.pop(os.path.basename(summary))
        json.dump({"moved": [r[0] for r in rows if r[1] == "invalid"], "reasons": {r[0]: r[2] for r in rows if r[1] == "invalid"},
                   "revoked_seals": revoked}, open(os.path.join(dest, "QUARANTINE.json"), "w"), indent=1)
        json.dump(record, open(p.path("RESULTS.json"), "w"), indent=1, sort_keys=True)
        print("moved the invalid outputs to %s (%d seals revoked)" % (dest, len(revoked)))
        return 0
    return 1 if counts["invalid"] else 0


def cmd_seal(p: Plan) -> int:
    ident, bad = identity_problems(p)
    if ident is None or bad:
        print("; ".join(bad))
        return 1
    record, cache, problems = sealed(p), {}, []
    for o in p.outputs:
        summary = o.summary
        files = [summary, details_of(summary)]
        if not all(os.path.exists(f) for f in files) or os.path.basename(summary) in record:
            continue
        if min(os.path.getmtime(f) for f in files) < ident["frozen_epoch"]:
            problems.append("%s: written before the freeze, not sealed" % summary)
            continue
        why = content(o, p.checkpoint, ident["temperature"], cache)
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
        problems = [x for o in p.outputs for x in content(o, p.checkpoint, None, cache)]
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


def verified(config: str, extras: Optional[str], names: List[str]) -> Dict[str, str]:
    """name -> details path of extras (or planned sets by summary basename) that pass `check` under the frozen
    identity; raises SystemExit otherwise. calibration.py / delivery.py read their inputs only through this."""
    p = Plan(config, extras)
    ident, bad = identity_problems(p)
    if ident is None or bad:
        raise SystemExit("identity: %s" % "; ".join(bad))
    by_name = dict(p.extras)
    for o in p.outputs:
        by_name.setdefault(os.path.basename(o.summary)[: -len(".json")], o)
    rows = {s: (st, why) for s, st, why in status(p, ident, bad, {})}
    out = {}
    for n in names:
        if n not in by_name:
            raise SystemExit("%s is not planned in %s / %s" % (n, config, extras))
        st, why = rows[by_name[n].summary]
        if st != "ok":
            raise SystemExit("%s is %s: %s" % (n, st, "; ".join(why[:3])))
        out[n] = details_of(by_name[n].summary)
    return out


def cmd_run_extras(p: Plan) -> int:
    ident, bad = identity_problems(p)
    if ident is None or bad:
        print("; ".join(bad) or "no identity")
        return 1
    for name, o in p.extras.items():
        if os.path.exists(o.summary):
            continue
        cmd = ["uv", "run", "devision-eval", "--checkpoint", p.checkpoint, "--data", o.data, "--data-root", "data",
               "--out", o.summary, "--role", o.role] + (["--control", o.control] if o.control != "none" else []) \
            + (["--no-temperature"] if o.temperature == "raw" else [])
        print("== extra %s: %s" % (name, " ".join(cmd)), flush=True)
        if subprocess.call(cmd) != 0:
            return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("freeze", "preflight", "seal", "check", "run-extras"))
    ap.add_argument("config")
    ap.add_argument("--extras")
    ap.add_argument("--quarantine", action="store_true")
    ap.add_argument("--legacy", action="store_true")
    a = ap.parse_args(argv)
    p = Plan(a.config, a.extras)
    if a.command == "run-extras":
        return cmd_run_extras(p)
    if a.command == "freeze":
        return cmd_freeze(p)
    if a.command == "preflight":
        return cmd_preflight(p, a.quarantine)
    if a.command == "seal":
        return cmd_seal(p)
    return cmd_check(p, a.legacy)


if __name__ == "__main__":
    sys.exit(main())
