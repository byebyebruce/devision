"""Binding metrics and the screening gate on the counterfactual sets (dev_bind_cf / test_bind_cf), round 13 plan
revision 2 (as corrected after critic R13-I04 / R13-I05).

    uv run python scripts/r13/bind_report.py data/r13/synth-v2/dev_bind_cf.jsonl \
        runs/r13/v12-synth-v2/dev_bind_cf.details.jsonl runs/exp13-s0/eval/dev_bind_cf.details.jsonl \
        [--out runs/exp13-s0/eval/compare/bind_cf.json]

Input checks (any failure stops the report): every (sample, question) of the data file has exactly one
prediction row -- no missing, duplicate or extra rows -- each prediction is one of the question's options, the
row's gold equals the data file's, and when the summary JSON next to the details names a data SHA-256 it must be
this data file's. Partial files are refused, never scored.

Scored task set (the SAME for the model, the name-blind baselines and the gate): colour and count questions,
which every picture asks about both kinds.
  binding   share of (picture, task) units in which BOTH kinds are answered right
  strata    per-question accuracy in each salience stratum, next to the two name-blind policies (answer every
            question with the most salient kind's value / the least salient kind's value)
  gate      binding >= 0.8 and every stratum >= 0.7. A name-blind policy is right on one stratum by construction
            (1.0) but scores <= ~0.55 on another and cannot give both kinds right on most pictures, so it fails.
Also reported: accuracy of exists_true / exists_trap / exists_absent / compare, the share of wrong colour/count
answers that are the salient kind's value, and counts (questions, pictures, families) with the data SHA-256.
Differences B - A (first details file = baseline) resample whole scene families, 2,000 draws, seed 0, for the
fixed-unit metrics (binding, strata, tasks); both files must cover the same units, so the denominators are
identical. wrong_is_salient is reported per model only: its denominator is that model's wrong answers.
"""
import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

SCORED = ("colour", "count")
GATE = {"binding": 0.8, "stratum": 0.7}
STRATA = ("target_salient", "close", "target_minor")


class ReportError(ValueError):
    pass


def rows(path: str) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in open(path) if line.strip()]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def answer(gold: Dict[str, Any]) -> str:
    p = gold["probabilities"]
    return max(p, key=p.get)


def predictions(samples, details: List[Dict[str, Any]], data_sha: str, summary: Optional[Dict[str, Any]] = None
                ) -> Dict[Tuple[str, str], str]:
    """(sample id, qid) -> predicted option, after checking the details match the data one to one."""
    want = {(s["id"], q): s for s in samples for q in s["questions"]}
    got: Dict[Tuple[str, str], str] = {}
    for r in details:
        key = (r["sample_id"], r["qid"])
        if key in got:
            raise ReportError("duplicate prediction for %s" % (key,))
        if key not in want:
            raise ReportError("prediction for a question not in the data: %s" % (key,))
        s = want[key]
        options = ["false", "true"] if s["questions"][key[1]]["type"] == "noul" else list(s["questions"][key[1]]["criteria"])
        if r["prediction"] not in options:
            raise ReportError("prediction %r for %s is not an option" % (r["prediction"], key))
        if "gold" in r and answer({"probabilities": r["gold"]}) != answer(s["gold"][key[1]]):
            raise ReportError("gold of %s differs from the data file (another data version?)" % (key,))
        got[key] = r["prediction"]
    missing = set(want) - set(got)
    if missing:
        raise ReportError("%d of %d questions have no prediction, e.g. %s" % (len(missing), len(want), sorted(missing)[0]))
    named = ((summary or {}).get("run") or {}).get("data_sha256")
    if named and named != data_sha:
        raise ReportError("the evaluation ran on data %s, not this file (%s)" % (named[:12], data_sha[:12]))
    return got


def family_counts(samples, pred, policy: Optional[str] = None):
    """Per family: name -> [hits, n]. `policy` ("salient" / "minor") replaces the model's answers on the scored
    tasks with that name-blind policy (used for the baselines and the gate's self-check)."""
    fam: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for s in samples:
        f = fam[s["group"]]
        per_task: Dict[str, List[bool]] = defaultdict(list)
        for qid, m in s["meta"].items():
            gold = answer(s["gold"][qid])
            if policy and m["task"] not in SCORED:   # the policies are defined on the scored tasks only
                continue
            said = m[policy] if policy else pred[(s["id"], qid)]
            ok = said == gold
            f["task:" + m["task"]][0] += ok
            f["task:" + m["task"]][1] += 1
            if m["task"] in SCORED:
                per_task[m["task"]].append(ok)
                f["stratum:" + m["stratum"]][0] += ok
                f["stratum:" + m["stratum"]][1] += 1
                if m["salient"] != gold and not ok:
                    f["wrong_is_salient"][0] += said == m["salient"]
                    f["wrong_is_salient"][1] += 1
        for task in SCORED:
            if len(per_task[task]) != len(s["scene"]):
                raise ReportError("%s: %s asked about %d of %d kinds" % (s["id"], task, len(per_task[task]), len(s["scene"])))
            f["binding"][0] += all(per_task[task])
            f["binding"][1] += 1
    return fam


def totals(fam) -> Dict[str, Any]:
    t: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for f in fam.values():
        for k, (h, n) in f.items():
            t[k][0] += h
            t[k][1] += n
    return {k: {"accuracy": round(h / n, 4), "n": n} for k, (h, n) in sorted(t.items()) if n}


def gate(summary: Dict[str, Any]) -> Dict[str, Any]:
    strata = {s: summary.get("stratum:" + s, {}).get("accuracy", 0.0) for s in STRATA}
    ok = summary.get("binding", {}).get("accuracy", 0.0) >= GATE["binding"] and all(v >= GATE["stratum"] for v in strata.values())
    return {"passes": ok, "binding": summary.get("binding", {}).get("accuracy"), "strata": strata, "thresholds": GATE}


def paired(fa, fb, key: str, draws: int = 2000, seed: int = 0) -> Optional[Dict[str, Any]]:
    if set(fa) != set(fb):
        raise ReportError("the two evaluations cover different scene families")
    vals = []
    for g in sorted(fa):
        (ha, na), (hb, nb) = fa[g][key], fb[g][key]
        if na != nb:
            raise ReportError("family %s: %s has %d units in one file and %d in the other" % (g, key, na, nb))
        if na:
            vals.append((ha, hb, na))
    if not vals:
        return None
    n = sum(v[2] for v in vals)
    diff = (sum(v[1] for v in vals) - sum(v[0] for v in vals)) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(draws):
        smp = [vals[rng.randrange(len(vals))] for _ in vals]
        boots.append((sum(v[1] for v in smp) - sum(v[0] for v in smp)) / sum(v[2] for v in smp))
    boots.sort()
    return {"difference": round(diff, 4), "ci95": [round(boots[int(0.025 * draws)], 4), round(boots[int(0.975 * draws) - 1], 4)],
            "families": len(vals), "units": n}


def report(data_path: str, details_paths: List[str]) -> Dict[str, Any]:
    samples = rows(data_path)
    data_sha = sha256(data_path)
    out: Dict[str, Any] = {"data": data_path, "data_sha256": data_sha, "pictures": len(samples),
                           "families": len({s["group"] for s in samples}),
                           "questions": sum(len(s["questions"]) for s in samples), "models": []}
    for policy in ("salient", "minor"):
        summary = totals(family_counts(samples, {}, policy))
        out["policy_" + policy] = {"summary": summary, "gate": gate(summary)}
    fams = []
    for path in details_paths:
        side = path.replace(".details.jsonl", ".json")
        summary_json = json.load(open(side)) if os.path.exists(side) else None
        fam = family_counts(samples, predictions(samples, rows(path), data_sha, summary_json))
        fams.append(fam)
        entry: Dict[str, Any] = {"details": path, "summary": totals(fam)}
        entry["gate"] = gate(entry["summary"])
        if len(fams) > 1:
            # fixed-unit metrics only: wrong_is_salient's denominator is the model's own wrong answers, so it differs
            # between models by design (critic R13-I06); it stays in each model's summary, not in the paired list
            keys = ["binding"] + ["stratum:" + s for s in STRATA] + \
                ["task:" + t for t in ("colour", "count", "exists_true", "exists_trap", "exists_absent", "compare")]
            entry["vs_baseline"] = {k: paired(fams[0], fam, k) for k in keys}
        out["models"].append(entry)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("samples")
    p.add_argument("details", nargs="+", help="baseline first")
    p.add_argument("--out")
    a = p.parse_args()
    r = report(a.samples, a.details)
    print("data %s (sha256 %s): %d pictures, %d families, %d questions" % (
        r["data"], r["data_sha256"][:12], r["pictures"], r["families"], r["questions"]))
    for policy in ("salient", "minor"):
        g = r["policy_" + policy]["gate"]
        print("name-blind %-7s binding %.3f strata %s gate %s" % (policy, g["binding"], g["strata"], g["passes"]))
    for m in r["models"]:
        s, g = m["summary"], m["gate"]
        acc = lambda k: s.get(k, {}).get("accuracy", float("nan"))
        print("%s\n  binding %.3f | colour %.3f count %.3f | exists true %.3f trap %.3f absent %.3f | compare %.3f | wrong=salient %.3f"
              % (m["details"], acc("binding"), acc("task:colour"), acc("task:count"), acc("task:exists_true"),
                 acc("task:exists_trap"), acc("task:exists_absent"), acc("task:compare"), acc("wrong_is_salient")))
        print("  strata %s -> gate %s" % ({k: round(v, 3) for k, v in g["strata"].items()}, "PASS" if g["passes"] else "fail"))
        for k, v in (m.get("vs_baseline") or {}).items():
            if v:
                print("  vs baseline %-22s %+.3f [%+.3f, %+.3f]" % (k, v["difference"], *v["ci95"]))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(r, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
