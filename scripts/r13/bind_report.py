"""Binding metrics on the counterfactual sets (dev_bind_cf / test_bind_cf), round 13 plan revision 2.

    uv run python scripts/r13/bind_report.py data/r13/synth/dev_bind_cf.jsonl \
        runs/v12-ext/eval/dev_bind_cf.details.jsonl runs/r13-s0/eval/dev_bind_cf.details.jsonl \
        [--out runs/r13-s0/eval/compare/bind_cf.json]

Per model (the first details file is the baseline, usually v12; the others are compared with it):
  binding     share of (picture, task) pairs answered right for EVERY kind in the picture, tasks colour, count
              and exists_trap ("Is there a <other kind's colour> <shape>?"): the "same picture, every kind right"
              score the plan's gates use
  per task    accuracy of colour, count, exists_true, exists_trap, compare
  strata      accuracy over colour + count + exists_trap questions in each salience stratum, next to the two
              name-blind baselines (answer read off the most salient kind / the least salient kind) for
              colour + count in that stratum
  salient     share of colour / count answers that equal the most salient kind's value when that is wrong
Differences B - A with 95% intervals resample whole scene families (`group`), 2,000 draws, seed 0.
"""
import argparse
import json
import random
from collections import defaultdict

TASKS = ("colour", "count", "exists_trap")


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def correct(details):
    out = {}
    for r in details:
        out[(r["sample_id"], r["qid"])] = bool(r["correct"])
        out[("pred", r["sample_id"], r["qid"])] = r["prediction"]
    return out


def units(samples, hit):
    """Per family: list of (binding_hits, binding_n, per-task hits/n, per-stratum hits/n, salient hits/n)."""
    fam = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for s in samples:
        g = s["group"]
        by_task = defaultdict(list)
        for qid, m in s["meta"].items():
            key = (s["id"], qid)
            if key not in hit:
                continue
            ok = hit[key]
            by_task[m["task"]].append(ok)
            fam[g]["task:" + m["task"]][0] += ok
            fam[g]["task:" + m["task"]][1] += 1
            if m["task"] in TASKS:
                fam[g]["stratum:" + m["stratum"]][0] += ok
                fam[g]["stratum:" + m["stratum"]][1] += 1
            if m["task"] in ("colour", "count"):
                gold = max(s["gold"][qid]["probabilities"], key=s["gold"][qid]["probabilities"].get)
                for base in ("salient", "minor"):
                    fam[g]["%s:%s" % (base, m["stratum"])][0] += m[base] == gold
                    fam[g]["%s:%s" % (base, m["stratum"])][1] += 1
                if m["salient"] != gold:
                    fam[g]["answered_salient"][0] += hit[("pred", s["id"], qid)] == m["salient"]
                    fam[g]["answered_salient"][1] += 1
        for t in TASKS:
            if by_task[t]:
                fam[g]["binding"][0] += all(by_task[t])
                fam[g]["binding"][1] += 1
    return fam


def summarise(fam):
    tot = defaultdict(lambda: [0, 0])
    for f in fam.values():
        for k, (h, n) in f.items():
            tot[k][0] += h
            tot[k][1] += n
    return {k: round(h / n, 4) for k, (h, n) in sorted(tot.items()) if n}


def paired(fa, fb, key, draws=2000, seed=0):
    groups = [g for g in fa if fa[g][key][1] and fb[g][key][1]]
    vals = [(fa[g][key][0], fb[g][key][0], fa[g][key][1]) for g in groups]
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
            "families": len(groups)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("samples")
    p.add_argument("details", nargs="+", help="baseline first")
    p.add_argument("--out")
    a = p.parse_args()
    samples = rows(a.samples)
    fams = [units(samples, correct(rows(d))) for d in a.details]
    report = {"samples": a.samples, "models": []}
    for path, fam in zip(a.details, fams):
        entry = {"details": path, "summary": summarise(fam)}
        if fam is not fams[0]:
            keys = ["binding", "answered_salient"] + ["task:" + t for t in ("colour", "count", "exists_true", "exists_trap", "compare")] \
                + ["stratum:" + s for s in ("target_salient", "close", "target_minor")]
            entry["vs_baseline"] = {k: paired(fams[0], fam, k) for k in keys}
        report["models"].append(entry)
        s = entry["summary"]
        print("%s\n  binding %.3f | colour %.3f count %.3f trap %.3f true %.3f | answered salient %.3f" % (
            path, s.get("binding", 0), s.get("task:colour", 0), s.get("task:count", 0), s.get("task:exists_trap", 0),
            s.get("task:exists_true", 0), s.get("answered_salient", 0)))
        for st in ("target_salient", "close", "target_minor"):
            print("  %-15s model %.3f | name-blind salient %.3f minor %.3f" % (
                st, s.get("stratum:" + st, 0), s.get("salient:" + st, 0), s.get("minor:" + st, 0)))
        for k, v in (entry.get("vs_baseline") or {}).items():
            if v:
                print("  vs baseline %-22s %+.3f [%+.3f, %+.3f]" % (k, v["difference"], *v["ci95"]))
    if a.out:
        import os
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
