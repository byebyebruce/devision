"""Candidate B endpoint screen (critic V04-B-R1-02; the delivery plan's numbers, unchanged). From devision-eval runs
of the final checkpoint (its fitted temperatures) written by scripts/v04b/chain.sh, every step recorded in the
status file. REQUIRED, each completed or the outcome is "incomplete":
  - ScienceQA temperature / force on true and fixed wrong pictures -> exp21's transfer readout against v0.3's exp27
    records (68 : 114): accuracy change >= +0.10 with lower bound > 0 and picture-gain change lower bound > 0;
  - the 7 protection panels against v0.3: a cost when <= -3 points with the upper bound < 0 (ext 13-source macro,
    flip with its originals);
  - dev16 reading and comparison by exp26's rule (pass = point >= 0.80 and lower bound >= 0.75), counted only with
    a valid erased-digit control (read < 0.50 and compare < 0.55);
  - the real-layout tune decision questions: reading accuracy by field, both objects right where values differ,
    both twin views right where the gold moves (exp32's readouts), against v0.3's exp32 records.
The screen passes only when the transfer target is met, no protection cost, the erased control is valid and dev16
reading and comparison both pass. Report only: exp31 by level with the predicted equal / Pair 1 / Pair 2 shares,
exp34 semantics tune. Bound to the final weights (model / config sha256) and the start (exp39-H) given to the chain.

    uv run python scripts/v04b/report.py runs/v04b-fit <start model sha256>
"""
import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "exp21"))
from readout import classify, syn_index, syn_stat, transfer  # type: ignore  # noqa: E402

from devision.train.compare import load_records, paired_metric  # noqa: E402

_spec = importlib.util.spec_from_file_location("exp32_report", os.path.join(HERE, "..", "exp32", "report.py"))
assert _spec is not None and _spec.loader is not None
r32 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r32)

V03 = "runs/exp27-transfer/eval/v03"
V03_RL = "runs/exp32-fit/eval_v03"
GUARD = ("exist", "science", "ext", "count_wide", "count_narrow", "flip", "vsr")
SQA = ("sqa_temp", "sqa_force", "sqa_temp_mism", "sqa_force_mism")
REQUIRED = SQA + GUARD + ("dev16", "syn_erased", "rl_tune")


def sha(path: str):
    return hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None


def states(path: str) -> dict:
    out = {}
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                k, v = line.split()
                out[k] = v
    return out


def recs(d: str, name: str):
    return r32.recs(d, name)


def binding(rs, data_path):
    return {"by_field": r32.by_kind(rs, data_path), **r32.views(rs), "both_objects": both_objects(rs, data_path)}


def both_objects(rs, data_path):
    """Per picture and field (mass / speed / distance): both objects' reading questions right, where their values
    differ (the two gold answers differ)."""
    kind = {}
    for s in map(json.loads, open(data_path)):
        for qid, q in s["questions"].items():
            ins = q["instructions"]
            f = "mass" if "mass" in ins else "speed" if "speed" in ins else "distance" if "distance" in ins else None
            if f:
                kind[(s["id"], qid)] = f
    by = defaultdict(list)
    for r in rs:
        f = kind.get((r["sample_id"], r["qid"]))
        if f:
            by[(r["sample_id"], f)].append(r)
    out = defaultdict(lambda: [0, 0])
    for (sid, f), v in by.items():
        if len(v) == 2 and v[0]["gold_answer"] != v[1]["gold_answer"]:
            out[f][1] += 1
            out[f][0] += int(v[0]["correct"] and v[1]["correct"])
    return {f: {"right": a, "of": n, "rate": a / n if n else None} for f, (a, n) in sorted(out.items())}


def build(run: str, start_sha: str, status_path: str = "") -> dict:
    ev = os.path.join(run, "eval")
    st = states(status_path or os.path.join(run, "..", "v04b-status.txt"))
    ckpt = os.path.join(run, "stage2")
    out: dict = {"identity": {"model.safetensors": sha(os.path.join(ckpt, "model.safetensors")),
                              "config.json": sha(os.path.join(ckpt, "config.json")), "start_model_sha256": start_sha},
                 "status": st}
    missing = [n for n in REQUIRED if st.get("eval:" + n) != "completed"]
    if st.get("train") != "completed" or missing or out["identity"]["model.safetensors"] is None:
        out["outcome"] = "incomplete: %s" % ", ".join((["train"] if st.get("train") != "completed" else []) + missing)
        return out
    for n in REQUIRED:
        run_info = json.load(open(os.path.join(ev, n + ".json")))["run"]
        if run_info.get("checkpoint") != ckpt or run_info.get("temperature") != "fitted":
            raise SystemExit("%s was not evaluated on %s with its fitted temperatures" % (n, ckpt))
    t = transfer({k[4:]: recs(V03, k) for k in SQA}, {k[4:]: recs(ev, k) for k in SQA})
    out["transfer"] = t
    target = t["accuracy_change"]["value"] >= 0.10 and t["accuracy_change"]["ci95"][0] > 0 and t["gain_change"]["ci95"][0] > 0
    guard, costs = {}, []
    for p in GUARD:
        a, b = recs(V03, p), recs(ev, p)
        m = paired_metric(a, b, "macro_accuracy", sources=r32.EXT_SOURCES) if p == "ext" else paired_metric(a, b, "accuracy")
        guard[p] = {"difference": m["difference"], "ci95": m["ci95"]}
        if m["difference"] * 100 <= -3 and m["ci95"][1] < 0:
            costs.append(p)
    out["protection"] = {"panels": guard, "costs": costs}
    d, e = syn_index(recs(ev, "dev16")), syn_index(recs(ev, "syn_erased"))
    dev = {w: syn_stat(d, 16, w) for w in ("read", "cmp")}
    era = {w: syn_stat(e, None, w) for w in ("read", "cmp")}
    valid = era["read"]["value"] < 0.50 and era["cmp"]["value"] < 0.55
    dev["read_class"] = classify(dev["read"]) if valid else "void: erased control invalid"
    dev["cmp_class"] = classify(dev["cmp"]) if valid else "void: erased control invalid"
    out["dev16"], out["erased"] = dev, dict(era, valid=valid)
    tune = "data/real_layout/questions_tune.jsonl"
    out["real_layout_tune"] = {"b": binding(recs(ev, "rl_tune"), tune), "v03": binding(recs(V03_RL, "rl_tune"), tune),
                               "change": paired_metric(r32.by_family(recs(V03_RL, "rl_tune")), r32.by_family(recs(ev, "rl_tune")), "accuracy")}
    checks = {"transfer_target": target, "no_protection_cost": not costs, "erased_control_valid": valid,
              "dev16_read_pass": dev["read_class"] == "pass", "dev16_cmp_pass": dev["cmp_class"] == "pass"}
    out["checks"] = checks
    out["outcome"] = "candidate screen passed" if all(checks.values()) else \
        "candidate screen not passed: %s" % ", ".join(k for k, v in checks.items() if not v)
    ro = {}
    for name in ("exp31", "semantics_tune"):
        if st.get("eval:" + name) == "completed":
            rs = recs(ev, name)
            ro[name] = {"accuracy": sum(r["correct"] for r in rs) / len(rs), "n": len(rs)}
    out["report_only"] = ro
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    out = build(argv[0], argv[1])
    json.dump(out, open(os.path.join(argv[0], "screen.json"), "w"), indent=1, default=str)
    print("outcome: " + out["outcome"])
    print("identity: %s" % out["identity"])
    if "checks" in out:
        t = out["transfer"]
        print("transfer accuracy %+.3f %s gain %+.3f %s" % (t["accuracy_change"]["value"], t["accuracy_change"]["ci95"],
                                                            t["gain_change"]["value"], t["gain_change"]["ci95"]))
        print("protection %s costs %s" % ({k: round(100 * v["difference"], 2) for k, v in out["protection"]["panels"].items()},
                                          out["protection"]["costs"] or "none"))
        print("dev16 read %.3f (%s) cmp %.3f (%s); erased valid %s" % (
            out["dev16"]["read"]["value"], out["dev16"]["read_class"], out["dev16"]["cmp"]["value"], out["dev16"]["cmp_class"],
            out["erased"]["valid"]))
        print("real-layout tune: %s" % {k: out["real_layout_tune"]["b"][k] for k in ("by_field", "both_objects", "follows_values")})
        print("checks %s; report only %s" % (out["checks"], out["report_only"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
