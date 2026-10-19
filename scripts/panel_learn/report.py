"""EXP74 report, rules written before running (docs/experiments/2026-10-19-panel-learn-plan.md); read-only.

  d1  scripts/exp24/gate.py on runs/exp74-panel-d1/d1/align_report.json: masked-token accuracy with the right picture
      minus another picture on 1,000 held-out panel pictures >= 0.30 -> run D2 (exit 0), else stop (exit 10).
  d2  from D2's last epoch-end monitor and its exist / science probes:
      valid      -- the erased control stays invalid-for-reading (read < 0.50 and compare < 0.55, delivery.py's rule);
      fit        -- readA / readB and compare on the 200 training families (exp21 readout classify: pass = point >= 0.80
                    and lower >= 0.75; fail = point < 0.60 and upper < 0.65; else near);
      dev16      -- the same rule on the unseen families (development monitor; voided by an invalid control);
      kept       -- K text slice >= 0.95; the frozen reading step's states on the real questions: temperature >= 39/41,
                    magnets >= 0.458 (sealed 0.508 - 0.05); exist and science within 3 points of the sealed candidate;
      outcome    -- invalid (control) / A fit learned + kept / B fit learned, protection broken / C fit not learned
                    (fail) / D near threshold. No outcome starts anything.
"""
import glob
import importlib.util
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp21"))
from readout import classify, syn_index, syn_stat  # type: ignore  # noqa: E402

D1 = "runs/exp74-panel-d1/d1/align_report.json"
D2 = "runs/exp74-panel-d2"
SEALED = "runs/v04-k/eval"
LINES = {"k_text": 0.95, "rs_temp": 39 / 41, "rs_force": 0.458}


def gate():
    spec = importlib.util.spec_from_file_location("gate", os.path.join(os.path.dirname(__file__), "..", "exp24", "gate.py"))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def jsonl(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def accuracy(recs):
    return sum(bool(r["correct"]) for r in recs) / len(recs) if recs else None


MONITOR = {"val": "data/panel_learn/fit.jsonl", "dev16": "data/panel_learn/dev16.jsonl", "erased": "data/panel_learn/erased.jsonl",
           "k_text": "data/protect_p/slice_K_text.jsonl", "rs_temp": "data/panel_learn/read_states_sqa_temp.jsonl",
           "rs_force": "data/panel_learn/read_states_sqa_force.jsonl", "rs_temp_mism": "data/panel_learn/read_states_sqa_temp_mism.jsonl",
           "rs_force_mism": "data/panel_learn/read_states_sqa_force_mism.jsonl"}
PROBES = {"exist": "data/r13b/panel_exist.jsonl", "science": "data/r13b/panel_science.jsonl"}
TRAIN = "data/panel_learn/train.jsonl"
CONFIG = "configs/exp74-panel-d2.yaml"


def sha(path: str) -> str:
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def expected(path: str):
    """(sample id, qid) -> (options, gold) of a data file; a repeated key is refused."""
    out = {}
    for r in jsonl(path):
        for qid, q in r["questions"].items():
            k = (r["id"], qid)
            if k in out:
                raise ValueError("%s repeats %s" % (path, k))
            opts = ["false", "true"] if q["type"] == "noul" else list(q["criteria"])
            out[k] = (opts, r["gold"][qid]["probabilities"])
    return out


def content_problems(name: str, recs, want) -> list:
    keys = [(r["sample_id"], r["qid"]) for r in recs]
    out = []
    if len(keys) != len(set(keys)):
        out.append("%s: a question appears twice" % name)
    if set(keys) != set(want):
        out.append("%s: %d questions recorded, %d planned (%d missing, %d extra)" % (
            name, len(set(keys)), len(want), len(set(want) - set(keys)), len(set(keys) - set(want))))
    for r in recs:
        w = want.get((r["sample_id"], r["qid"]))
        if w is None:
            continue
        opts, gold = w
        if list(r.get("options") or []) != opts or r.get("gold") != gold or r.get("gold_answer") != max(gold, key=lambda o: gold[o]) \
                or bool(r.get("correct")) != (r.get("prediction") == r.get("gold_answer")):
            out.append("%s: %s/%s options / gold / correct differ from the data" % (name, r["sample_id"], r["qid"]))
            break
    return out


def planned_updates(config: str = CONFIG) -> int:
    """What rlcd plans for this config's data and epochs (scripts/panel_learn/plan.py, rlcd.plan_mixed)."""
    import random
    import yaml
    from devision.train.rlcd import plan_mixed
    st = yaml.safe_load(open(config))["stages"][0]
    its = [{"input": r["input"], "group": r["group"], "qid": q} for r in jsonl(st["data"]) for q in r["questions"]]
    random.seed(st["params"]["seed"])
    return sum(len(plan_mixed(its, st["params"]["micro_batch"])) for _ in range(st["params"]["epochs"]))


def d2_report(d2: str = D2, sealed: str = SEALED, config: str = CONFIG, planned: int = 0):
    """Evidence first (critic PANEL-D2-R1-02): invalid when an identity or content check fails, incomplete when the run
    did not reach its planned end or a monitor / probe is missing; A / B / C / D only from complete, valid evidence.
    Whatever finished is still reported."""
    invalid: list = []
    incomplete: list = []
    out: dict = {"parts": {}}
    run = os.path.join(d2, "run.json")
    if os.path.exists(run):
        cfg = json.load(open(run))["config"]
        want = __import__("yaml").safe_load(open(config))
        if cfg.get("stages") != want.get("stages"):
            invalid.append("run.json's stages are not %s" % config)
    else:
        incomplete.append("no run.json (D2 did not start)")
    man = json.load(open("data/panel_learn/MANIFEST.json"))
    if sha(TRAIN) != man["train_sha256"]:
        invalid.append("the training data is not the manifest's")
    rep_path = os.path.join(d2, "d2", "train_report.json")
    rep = json.load(open(rep_path)) if os.path.exists(rep_path) else None
    planned = planned or planned_updates(config)
    end = None
    if rep is None:
        incomplete.append("training did not finish (no train_report.json)")
    else:
        n = len(rep["losses"])
        if n != planned:
            invalid.append("the run made %d updates, %d planned" % (n, planned))
        elif not rep["evals"] or rep["evals"][-1]["step"] != n:
            incomplete.append("no evaluation at the last update")
        else:
            end = os.path.join(d2, "d2", "monitor", "step%d" % n)
        if not os.path.isdir(os.path.join(d2, "d2", "monitor", "final")):
            incomplete.append("no final monitor")
    rec = {}
    if end is not None:
        for name, data in MONITOR.items():
            p = os.path.join(end, name + ".details.jsonl")
            if not os.path.exists(p):
                incomplete.append("monitor %s missing at the end step" % name)
                continue
            rec[name] = jsonl(p)
            invalid += content_problems(name, rec[name], expected(data))
    probes = {}
    model = os.path.join(d2, "d2", "model.safetensors")
    for name, data in PROBES.items():
        p, det, s = os.path.join(d2, "eval", name + ".json"), os.path.join(d2, "eval", name + ".details.jsonl"), os.path.join(sealed, name + ".json")
        probes[name] = {"sealed": json.load(open(s))["accuracy_all"], "d2": None}
        if not (os.path.exists(p) and os.path.exists(det)):
            incomplete.append("probe %s missing" % name)
            continue
        summ = json.load(open(p))
        r = summ.get("run", {})
        if os.path.normpath(r.get("checkpoint", "")) != os.path.normpath(os.path.join(d2, "d2")) or r.get("data_sha256") != sha(data) \
                or not os.path.exists(model) or os.path.getmtime(p) < os.path.getmtime(model):
            invalid.append("probe %s is not of this D2 endpoint / data" % name)
            continue
        invalid += content_problems("probe " + name, jsonl(det), expected(data))
        probes[name]["d2"] = summ["accuracy_all"]
    out["probes"] = probes
    if rec.keys() >= {"erased", "val", "dev16"}:
        e = syn_index(rec["erased"])
        erased = {w: syn_stat(e, None, w) for w in ("read", "cmp")}
        valid = erased["read"]["value"] < 0.50 and erased["cmp"]["value"] < 0.55
        f, d = syn_index(rec["val"]), syn_index(rec["dev16"])
        fit = {w: syn_stat(f, 16, w) for w in ("read", "cmp")}
        dev = {w: syn_stat(d, 16, w) for w in ("read", "cmp")}
        for st in (fit, dev):
            for w in ("read", "cmp"):
                st[w + "_class"] = classify(st[w]) if valid else "void: erased control invalid"
        out["parts"].update(erased=dict(erased, valid=valid), fit=fit, dev16=dev)
    acc = {n: accuracy(rec[n]) for n in ("k_text", "rs_temp", "rs_force", "rs_temp_mism", "rs_force_mism") if n in rec}
    out["parts"]["accuracy"] = acc
    out["monitor"] = end
    out["invalid"], out["incomplete"] = invalid, incomplete
    if invalid:
        out["outcome"] = "invalid: " + "; ".join(invalid[:3])
        return out
    if incomplete:
        out["outcome"] = "incomplete: " + "; ".join(incomplete[:3])
        return out
    kept = {n: (acc[n] or 0) >= LINES[n] for n in LINES}
    kept.update({n: (v["d2"] or 0) >= v["sealed"] - 0.03 for n, v in probes.items()})
    out["kept"] = kept
    fit = out["parts"]["fit"]
    classes = [fit["read_class"], fit["cmp_class"]]
    if not out["parts"]["erased"]["valid"]:
        out["outcome"] = "invalid: the erased control reads digits"
    elif all(c == "pass" for c in classes):
        out["outcome"] = "A: fit learned, protection kept" if all(kept.values()) else "B: fit learned, protection broken"
    elif any(c == "fail" for c in classes):
        out["outcome"] = "C: fit not learned"
    else:
        out["outcome"] = "D: near threshold"
    return out


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv[:1] == ["d1"]:
        out = gate().decide(json.load(open(argv[1] if len(argv) > 1 else D1)))
        print(json.dumps(out, indent=1))
        return 0 if out["run_phase_b"] else 10
    out = d2_report(argv[1] if len(argv) > 1 else D2, argv[2] if len(argv) > 2 else SEALED)
    json.dump(out, open(os.path.join(argv[1] if len(argv) > 1 else D2, "report.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
