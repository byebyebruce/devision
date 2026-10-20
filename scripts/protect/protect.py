"""PROTECT: the one-arm mixed replay recovery (critic EXP73-RESULT-01 / PROTECT-PREP-01 / PROTECT-R1-01;
docs/experiments/2026-10-19-protect-replay-plan.md). From EXP72-K's frozen end (cbd1e7ac / 4f9ccf48), EXP72-K's recipe on
data/protect2/train_mix.jsonl (3,800 new-task questions on text, 7,380 replay questions of v0.3's training set on their
pictures; rlcd mixed_input), 1,398 updates, last step; the new end re-bound with EXP71's three reading components (a new
combination identity). Raw temperature 1 in every readout. An engineering recovery, not a mechanism attribution;
candidate_eligible is always false.

  start       model-free: identities (EXP72's run and EXP72-K's frozen end, data/protect2 + every exposure file its MANIFEST
              lists (the isolation evidence), data/protect_p, the readout files, EXP71's bundle_on, the seven panels and
              v0.3's records, EXP58's picture manifest for them, EXP73's run (described against), the YAML, this code) ->
              runs/protect/START.json.
  fid         MPS: EXP72-K's end on the first 48 rows (by id) of its slice -- must reproduce EXP72's registered k_slice records.
  train       MPS: devision-pipeline configs/protect-k.yaml; progress from its log; the first window reports how many
              text / picture updates it held (from the plan, which equals the MANIFEST's).
  freeze      tensors only: 1,398 updates; the export against EXP72-K's end; FROZEN.json; the new end re-bound with EXP71's
              selector / head / corrector (runs/protect/bundle, components checked equal to EXP71's).
  decide_b    MPS: the new end on the training slice, the new diagnostic and G (K states) -- B, required.
  decide_c    MPS: the new end and EXP72-K's end on P_K (EXP56's automatic tables, K where defined) -- C, required.
  guard       MPS: the new combination (Reading.load(new end, bundle)) on the seven protection panels -- A, required.
  decide_fit  MPS, last: the old facts fit with K (report only).
  report      model-free: A against v0.3 (EXP58 / EXP73's cost rule), B against EXP72's K-arm lines, C against the old
              reading gate; the pre-written branch.

    uv run python scripts/protect/protect.py start --deadline EPOCH | fid | train | freeze | decide_b | decide_c | guard | decide_fit | report
"""
import argparse
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(__file__)
OUT, RUN, YAML = "runs/protect", "runs/protect-k", "configs/protect-k.yaml"
START_RUN = "runs/exp72-k"
START_SHA = {"model.safetensors": "cbd1e7ac8ade5fbb036239a3b526b7610ed30256cfac34b80cc3f171938c3782",
             "config.json": "4f9ccf4811713fac17c98f186ea35929311e367107d6a3f2934235d31f060e23"}
E72 = "runs/exp72"
E72_RUN = {"START.json": "d61b3080f073d7412f55297aa46440fc28d286c62a864535560e3368960dbc37",
           "report.json": "7b719fdf3ed42f406b6a50703538196763d945fbda611216558b0bfc41ec1885",
           "registry.json": "f986172927aa2dfadded4ff395f1336d9a617d8ce3e76a80936adad127662d39"}
E73 = "runs/exp73"
E73_RUN = {"START.json": "a7f63a00ef059188587bf6b33b6d944563035a9d20166fbe152e9c14dfc283b7",
           "report.json": "0617dcd396ccee87ad2c52f3289a9573d021c339bd587093b18818aefe6a4625",
           "registry.json": "0d1b08f34b89e184ec73fbbf1f6fbe904da238e9f495cf3ea791f15e144c65cd"}
D2 = {"replay.jsonl": "d86c86f8a949dd3f1eeedaef74ce1500ce5770b1172799f080ee01afe796a7ee",
      "train_mix.jsonl": "7ae73f112caac88518e2808c50cf5e4b5936772be314462870463e2f88675850",
      "MANIFEST.json": "918fec577eac5626b22f74b61dffd200ac7313db8b3ee1c0d87e9eaa2e7fcf54"}
DP = {"P_K.jsonl": "ca6254d672bb6b8d17333fe440f145e43eaf178561ecdcce5f39ebc0cf46883e",
      "slice_K_text.jsonl": "0e855bbd9ca793da42219ed2011c52379da8e0e570fa588d8522be1a02c3ba11"}
UPDATES, FID_ROWS = 1398, 48
B_SETS = {"slice": "data/exp72/slice_K.jsonl", "diag": "data/exp70/diag68_K.jsonl", "G": "data/exp72/G_K.jsonl", "fit": "data/exp72/fit_old_K.jsonl"}
P_SET = "data/protect_p/P_K.jsonl"
B_REQUIRED = ("slice", "diag", "G")
FIT_LINE, CONFLICT_LINE, G_LINE, P_CMP_LINE, P_ALL_LINE = 0.90, 0.80, 0.98, 0.65, 0.866
FILES = ["scripts/protect/protect.py", "scripts/protect/run.sh", "scripts/protect/build.py", "scripts/protect/build_p.py", YAML,
         "scripts/exp72/exp72.py", "scripts/exp73/exp73.py", "scripts/exp71/exp71.py", "scripts/exp70/exp70.py", "scripts/exp70/build.py",
         "scripts/exp63/exp63.py", "scripts/exp57/exp57.py", "scripts/exp54/exp54.py", "scripts/exp58/live.py", "scripts/exp32/report.py",
         "src/devision/train/rlcd.py", "src/devision/train/pipeline.py", "src/devision/infer.py", "src/devision/model.py",
         "src/devision/reading.py", "src/devision/image.py", "src/devision/train/evaluate.py", "src/devision/train/compare.py",
         "scripts/exp53/watch.py", "scripts/v04b/freeze.py"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e72 = _load("exp72", "../exp72/exp72.py")
e73 = _load("exp73", "../exp73/exp73.py")
pb = _load("protect_build", "build.py")
e70, e68, e63, e57, e54, fz = e72.e70, e72.e68, e72.e63, e72.e57, e72.e54, e72.fz
e71, e58 = e73.e71, e73.e58
sha, write_json, register, jsonl = e72.sha, e72.write_json, e72.register, e72.jsonl
GUARD = e73.GUARD
CLASSES, CONFLICT = e72.CLASSES, e72.CONFLICT


# ------------------------------------------------------------------ identities

def bound_data() -> Dict:
    """data/protect2 and data/protect_p as reviewed, and the isolation evidence: every input and exposure file the
    MANIFEST lists still has the sha256 it recorded (critic PROTECT NOTE-01)."""
    bad = [f for f, h in D2.items() if sha(os.path.join("data/protect2", f)) != h]
    bad += [f for f, h in DP.items() if sha(os.path.join("data/protect_p", f)) != h]
    if bad:
        raise SystemExit("data/protect2 / data/protect_p differ from the reviewed files: %s" % bad)
    man = json.load(open("data/protect2/MANIFEST.json"))
    changed = [p for p, h in man["inputs"].items() if sha(p) != h]
    changed += [p for p, v in man["exposure_files"].items() if not os.path.exists(p) or sha(p) != v["sha256"]]
    if changed:
        raise SystemExit("isolation evidence changed since data/protect2 was built: %s" % changed[:5])
    if man["plan"]["updates"] != UPDATES or man["plan"]["mirrored_pairs_in_one_batch"] != man["plan"]["mirrored_pairs"]:
        raise SystemExit("data/protect2's plan is not the reviewed one")
    pm = json.load(open("data/protect_p/MANIFEST.json"))
    if any(sha(p) != h for p, h in pm["inputs"].items()) or pm["sha256"] != DP:
        raise SystemExit("data/protect_p was built on other inputs")
    return {"protect2": dict(D2), "protect_p": dict(DP), "exposure_files": len(man["exposure_files"]),
            "plan_sha256": man["plan"]["plan_sha256"], "P_K_rows": pm["P_K_rows"]}


def replay_pictures(data_root: str = "data") -> int:
    rows = jsonl("data/protect2/replay.jsonl")
    missing = [r["image"] for r in rows if not os.path.exists(os.path.join(data_root, r["image"]))]
    if missing:
        raise SystemExit("%d replay pictures missing, e.g. %s" % (len(missing), missing[0]))
    return len({r["image"] for r in rows})


def check_yaml() -> None:
    import yaml
    c = yaml.safe_load(open(YAML))
    k = yaml.safe_load(open(e72.YAML["k"]))
    st, ks = c["stages"][0], k["stages"][0]
    want = {**{p: v for p, v in ks["params"].items() if p != "text_only"}, "mixed_input": True}
    if c["name"] != "protect-k" or len(c["stages"]) != 1 or c.get("evaluate") or st["init"] != "runs/exp72-k/stage2" \
            or st["data"] != "data/protect2/train_mix.jsonl" or st["val"] != "data/protect_p/slice_K_text.jsonl" or st["params"] != want \
            or {x: c[x] for x in c if x not in ("name", "stages")} != {x: k[x] for x in k if x not in ("name", "stages")}:
        raise SystemExit("%s is not EXP72-K's recipe with mixed_input on data/protect2" % YAML)


def identities() -> Dict:
    from devision.reading import tree_sha
    if fz.check(START_RUN, [])["weights"] != START_SHA:
        raise SystemExit("EXP72-K's frozen end is not the bound one")
    return {"exp72": e71.bound_files(E72, E72_RUN, "EXP72"), "start": dict(START_SHA), "exp73": e71.bound_files(E73, E73_RUN, "EXP73"),
            "tokenizer": tree_sha(os.path.join(START_RUN, "stage2", "tokenizer")), "data": bound_data(),
            "sets": {**{n: sha(p) for n, p in B_SETS.items()}, "P": sha(P_SET)}, "exp71": e73.bound_e71(),
            "exp58": e71.bound_files(e71.E58R, e71.E58_RUN, "EXP58"), "panels": {n: sha(p) for n, p in GUARD.items()},
            "v03": {n: sha(os.path.join(e58.V03, n + ".details.jsonl")) for n in GUARD},
            "exp49_questions": sha(e54.QUESTIONS), "exp54_inputs_P": sha(os.path.join(e54.OUT, "inputs_P.jsonl"))}


def stage_start(deadline: int, out: str = OUT, data_root: str = "data") -> Dict:
    path = os.path.join(out, "START.json")
    if os.path.exists(path) or os.path.exists(RUN):
        raise SystemExit("%s or %s exists (fresh only)" % (path, RUN))
    check_yaml()
    ident = identities()
    n_pics = replay_pictures(data_root)
    pics = e73.pictures(data_root)
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "pictures.json"), pics)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, **ident, "replay_pictures": n_pics,
         "pictures": {"pictures.json": sha(os.path.join(out, "pictures.json")), "count": len(pics)},
         "requests": {**{n: len(jsonl(p)) for n, p in B_SETS.items()}, "P": len(jsonl(P_SET)), "guard": e73.request_counts(), "fid": FID_ROWS},
         "updates": UPDATES, "temperature": "none (raw), in memory", "candidate_eligible": False,
         "purpose": "one-arm engineering recovery from EXP72-K (PROTECT-PREP-01); not a mechanism attribution; not a candidate",
         "started": time.strftime("%F %T"), "deadline": deadline, "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
    write_json(path, s)
    return s


def check_start(out: str = OUT, data_root: str = "data", code: bool = True) -> Dict:
    st = json.load(open(os.path.join(out, "START.json")))
    if code:
        bad = [f for f, h in st["files"].items() if sha(f) != h]
        if bad:
            raise SystemExit("code differs from START: %s" % bad)
    for k, v in identities().items():
        if st[k] != v:
            raise SystemExit("%s differs from START" % k)
    if sha(os.path.join(out, "pictures.json")) != st["pictures"]["pictures.json"] or e73.pictures("") != json.load(open(os.path.join(out, "pictures.json"))):
        raise SystemExit("the panel picture manifest differs from START")
    if data_root:
        e73.pictures(data_root)
    return st


# ------------------------------------------------------------------ fid / train / freeze

def rows_inputs(path: str, only: Optional[List[str]] = None):
    rows = {r["id"]: r for r in jsonl(path) if only is None or r["id"] in only}
    return rows, {rid: {"state": r["state_text"]} for rid, r in rows.items()}


def fid_ids() -> List[str]:
    return sorted(r["id"] for r in jsonl(B_SETS["slice"]))[:FID_ROWS]


def stage_fid(out: str = OUT, decider=None) -> Dict:
    """EXP72-K's end before any training on the first 48 slice rows: same predictions, probabilities within 1e-4 of EXP72's
    registered k_slice records."""
    from devision.infer import Decider, best_device
    check_start(out)
    rows, inputs = rows_inputs(B_SETS["slice"])
    ref = {(r["sample_id"], r["qid"]): r for r in e57.read_set(E72, "k_slice", rows, START_SHA, inputs)}
    if decider is None:
        if fz.weights(START_RUN) != START_SHA:
            raise SystemExit("EXP72-K's weights are not START's")
        decider = Decider.load(os.path.join(START_RUN, "stage2"), device=best_device())
    decider.cfg.temperature, decider.cfg.temperature_by_options = [1.0, 1.0, 1.0], {}
    frows, finputs = rows_inputs(B_SETS["slice"], fid_ids())
    recs = e54.decide_condition(decider, frows, finputs)
    e57.write_set(out, "fid", recs, {"set": "fid", "model_of": "exp72-k", "start_sha256": sha(os.path.join(out, "START.json")),
                                     "inputs_sha256": e57.hash_inputs(finputs), "model": START_SHA, "temperature": "none (raw)", "image": None})
    diffs = [max(abs(r["probabilities"][o] - ref[(r["sample_id"], r["qid"])]["probabilities"][o]) for o in r["probabilities"]) for r in recs]
    same = [r["prediction"] == ref[(r["sample_id"], r["qid"])]["prediction"] for r in recs]
    fid = {"n": len(recs), "mismatches": [i for i, s in enumerate(same) if not s], "max_abs_probability_difference": max(diffs),
           "reproduced": len(recs) == FID_ROWS and all(same) and max(diffs) <= 1e-4}
    write_json(os.path.join(out, "fidelity.json"), fid)
    register(out, "fidelity", {"fidelity.json": sha(os.path.join(out, "fidelity.json"))})
    if not fid["reproduced"]:
        raise SystemExit("EXP72-K's end does not reproduce EXP72's registered records -- no training")
    return fid


def plan_inputs() -> List[str]:
    """The input of each update of the plan training draws (random.seed(1); rlcd.plan_mixed on train_mix in file order),
    checked against the MANIFEST's plan hash."""
    from devision.train import rlcd
    mix = jsonl("data/protect2/train_mix.jsonl")
    items = [dict({"qid": q, "input": r["input"]}, **({"group": r["group"]} if r.get("group") else {})) for r in mix for q in r["questions"]]
    random.seed(1)
    plan = rlcd.plan_mixed(items, 8)
    import hashlib
    if hashlib.sha256(json.dumps(plan).encode()).hexdigest() != json.load(open("data/protect2/MANIFEST.json"))["plan"]["plan_sha256"]:
        raise SystemExit("the plan differs from data/protect2's MANIFEST")
    return [items[b[0]]["input"] for b in plan]


def stage_train(out: str = OUT, poll: float = 5.0, cmd: Optional[List[str]] = None) -> int:
    check_start(out)
    fid = json.load(open(os.path.join(out, "fidelity.json")))
    if e54.check_registered(out, "fidelity").get("fidelity.json") != sha(os.path.join(out, "fidelity.json")) or not fid["reproduced"]:
        raise SystemExit("the start's fidelity is not established -- no training")
    kinds = plan_inputs()
    log = os.path.join(RUN, "logs", "stage2.log")
    proc = subprocess.Popen(cmd or ["uv", "run", "devision-pipeline", YAML])
    seen, t0, told = -1, time.time(), False
    while True:
        rc = proc.poll()
        if os.path.exists(log):
            n = e57.progress_of(open(log, errors="replace").read().splitlines())
            if n is not None and n != seen:
                e54.progress(out, "train", n, UPDATES)
                seen = n
                if not told and n > 0:
                    told = True
                    head = kinds[:n]
                    print("first window: %d updates in %.0f s (text %d / picture %d by the plan); ETA %s" % (
                        n, time.time() - t0, head.count("text"), head.count("picture"),
                        time.strftime("%T", time.localtime(t0 + (time.time() - t0) * UPDATES / n))), flush=True)
        if rc is not None:
            return rc
        time.sleep(poll)


def stage_freeze(out: str = OUT) -> Dict:
    from safetensors.torch import load_file

    from devision.reading import write_bundle
    st = check_start(out)
    rep = json.load(open(os.path.join(RUN, "stage2", "train_report.json")))
    if len(rep["losses"]) != UPDATES:
        raise SystemExit("ran %d updates" % len(rep["losses"]))
    if fz.weights(START_RUN) != START_SHA:
        raise SystemExit("EXP72-K's end changed")
    res = e57.export_check(load_file(os.path.join(START_RUN, "stage2", "model.safetensors")), load_file(os.path.join(RUN, "stage2", "model.safetensors")))
    res.update(updates=len(rep["losses"]), baseline=dict(START_SHA))
    write_json(os.path.join(RUN, "export_check.json"), res)
    if not res["ok"]:
        raise SystemExit("export check failed: %s" % res)
    res["frozen"] = fz.freeze(RUN)["weights"]
    b71 = st["exp71"]["bundle_on"]
    dims = json.load(open(os.path.join(e73.BUNDLE, "reading.json")))["dims"]
    ident = write_bundle(os.path.join(out, "bundle"), os.path.join(RUN, "stage2"), os.path.join(e73.BUNDLE, "selector.pt"),
                         os.path.join(e73.BUNDLE, "head.pt"), {"components_of": "EXP71 bundle_on", "exp71_reading_json": b71["reading.json"]},
                         dims=dims, corrector_pt=os.path.join(e73.BUNDLE, "corrector.pt"))
    if any(ident["sha256"][c] != b71[c] for c in ("selector.pt", "head.pt", "corrector.pt")):
        raise SystemExit("the re-bound components are not EXP71's")
    register(out, "bundle", {"bundle/reading.json": sha(os.path.join(out, "bundle", "reading.json"))})
    res["bundle"] = sha(os.path.join(out, "bundle", "reading.json"))
    return res


# ------------------------------------------------------------------ readouts

def new_weights() -> Dict[str, str]:
    return fz.check(RUN, [])["weights"]


def bundle_bound(out: str) -> str:
    f = os.path.join(out, "bundle", "reading.json")
    if e54.check_registered(out, "bundle").get("bundle/reading.json") != sha(f):
        raise SystemExit("the re-bound bundle is not the registered one")
    return sha(f)


def decide_sets(out: str, plan: List[tuple], label: str, deciders: Optional[Dict] = None) -> Dict:
    """plan: [(model label, run, weights, [(set name, path)])]; each model loaded after its weights / tokenizer checks."""
    from devision.infer import Decider, best_device
    from devision.reading import tree_sha
    st = check_start(out)
    s_sha = sha(os.path.join(out, "START.json"))
    total = sum(len(jsonl(p)) for _, _, _, sets in plan for _, p in sets)
    done, t0 = 0, time.time()
    for who, run, weights, sets in plan:
        if deciders is None:
            if fz.weights(run) != weights or tree_sha(os.path.join(run, "stage2", "tokenizer")) != st["tokenizer"]:
                raise SystemExit("%s: weights / tokenizer not the bound ones" % who)
            d = Decider.load(os.path.join(run, "stage2"), device=best_device())
        else:
            d = deciders[who]
        d.cfg.temperature, d.cfg.temperature_by_options = [1.0, 1.0, 1.0], {}
        for name, path in sets:
            rows, inputs = rows_inputs(path)
            recs = e54.decide_condition(d, rows, inputs, on_request=lambda k, b=done: e54.progress(out, label, b + k, total))
            done += len(rows)
            e57.write_set(out, name, recs, {"set": name, "model_of": who, "start_sha256": s_sha, "inputs_sha256": e57.hash_inputs(inputs),
                                            "model": weights, "temperature": "none (raw)", "image": None})
            print("%s done after %.1f s" % (name, time.time() - t0), flush=True)
        del d
    return {"seconds": time.time() - t0}


def stage_decide_b(out: str = OUT, deciders=None) -> Dict:
    return decide_sets(out, [("new", RUN, new_weights(), [("B_" + n, B_SETS[n]) for n in B_REQUIRED])], "decide_b", deciders)


def stage_decide_c(out: str = OUT, deciders=None) -> Dict:
    return decide_sets(out, [("new", RUN, new_weights(), [("C_P_new", P_SET)]), ("start", START_RUN, START_SHA, [("C_P_start", P_SET)])],
                       "decide_c", deciders)


def stage_decide_fit(out: str = OUT, deciders=None) -> Dict:
    return decide_sets(out, [("new", RUN, new_weights(), [("B_fit", B_SETS["fit"])])], "decide_fit", deciders)


def stage_guard(out: str = OUT, data_root: str = "data", reading=None) -> Dict:
    from devision.infer import best_device
    from devision.reading import Reading
    st = check_start(out, data_root)
    weights, bundle = new_weights(), bundle_bound(out)
    if reading is None:
        reading = Reading.load(os.path.join(RUN, "stage2"), os.path.join(out, "bundle"), device=best_device())
        if reading.corrector is None:
            raise SystemExit("the re-bound bundle loaded without its corrector")
    reading.decider.cfg.temperature, reading.decider.cfg.temperature_by_options = [1.0, 1.0, 1.0], {}
    pics = json.load(open(os.path.join(out, "pictures.json")))
    s_sha = sha(os.path.join(out, "START.json"))
    total = sum(v["requests"] for v in st["requests"]["guard"].values())
    done, t0 = 0, time.time()
    for name, path in GUARD.items():
        recs, reads = e73.run_set(reading, jsonl(path), data_root, pics, on_request=lambda k, b=done: e54.progress(out, "guard", b + k, total))
        done += st["requests"]["guard"][name]["requests"]
        e57.write_set(out, "A_" + name, recs, {"set": "A_" + name, "start_sha256": s_sha, "data_sha256": st["panels"][name], "model": weights,
                                              "bundle": bundle, "pictures": st["pictures"]["pictures.json"], "temperature": "none (raw)",
                                              "path": "devision.reading.Reading.decide(image=actual picture), new end + EXP71's components"})
        f = "reads_%s.jsonl" % name
        with open(os.path.join(out, f), "w") as fh:
            fh.write("".join(json.dumps(x) + "\n" for x in reads))
        register(out, "reads_" + name, {f: sha(os.path.join(out, f))})
        print("%s done after %.1f s" % (name, time.time() - t0), flush=True)
    return {"seconds": time.time() - t0}


# ------------------------------------------------------------------ report

def a_records(out: str, name: str, st: Dict, weights: Dict, bundle: str) -> List[Dict]:
    """A panel after every check: registered, identity bound, every question of the source once, each on its own picture."""
    f = "details_A_%s.jsonl" % name
    if e54.check_registered(out, "decide_A_" + name).get(f) != sha(os.path.join(out, f)):
        raise SystemExit("%s not registered" % f)
    lines = jsonl(os.path.join(out, f))
    want = {"set": "A_" + name, "start_sha256": sha(os.path.join(out, "START.json")), "data_sha256": st["panels"][name], "model": weights,
            "bundle": bundle, "pictures": st["pictures"]["pictures.json"], "temperature": "none (raw)"}
    if any(lines[0]["identity"].get(k) != v for k, v in want.items()):
        raise SystemExit("%s: identity does not bind" % f)
    src = jsonl(GUARD[name])
    got = e57.r41.check_details("protect/%s" % name, src, lines[1:])
    image_of = {(r["id"], qid): r["image"] for r in src for qid in r["questions"]}
    if any(r["image_used"] != image_of[(r["sample_id"], r["qid"])] for r in lines[1:]):
        raise SystemExit("%s: a question was answered on another picture than its row's" % f)
    rf = "reads_%s.jsonl" % name
    if e54.check_registered(out, "reads_" + name).get(rf) != sha(os.path.join(out, rf)):
        raise SystemExit("%s not registered" % rf)
    return list(got.values())


def panel_metric(a: List[Dict], b: List[Dict], name: str) -> Dict:
    from devision.train.compare import paired_metric
    r32 = e73._load("exp32_report", "../exp32/report.py")
    if name == "flip":
        a = [dict(r, image_id=r["image_id"].removesuffix(":flip")) for r in a]
        b = [dict(r, image_id=r["image_id"].removesuffix(":flip")) for r in b]
    return paired_metric(a, b, "macro_accuracy", sources=r32.EXT_SOURCES) if name == "ext" else paired_metric(a, b, "accuracy")


def gates(res: Dict) -> Dict:
    """Each required gate True / False, or "incomplete" when its evidence is missing (never a pass)."""
    g: Dict[str, Any] = {}
    a = res.get("A", {})
    g["A"] = "incomplete" if set(a) != set(GUARD) else not any(v["cost"] for v in a.values())
    b = res.get("B", {})
    if not all(k in b for k in ("fit_slice", "diag", "G_reading")):
        g["B"] = "incomplete"
    else:
        g["B_detail"] = {"fit": all((b["fit_slice"].get(c) or 0) >= FIT_LINE for c in CLASSES),
                         "conflict": all((b["diag"]["accuracy"].get(c) or 0) >= CONFLICT_LINE for c in CONFLICT),
                         "G": (b["G_reading"]["all"] or 0) >= G_LINE}
        g["B"] = all(g["B_detail"].values())
    c = res.get("C", {})
    if not all(k in c for k in ("comparison", "reading", "P_use")):
        g["C"] = "incomplete"
    else:
        g["C_detail"] = {"comparison": (c["comparison"]["accuracy"] or 0) >= P_CMP_LINE, "P_all": (c["reading"]["all"] or 0) >= P_ALL_LINE,
                         "P_use": c["P_use"] is True}
        g["C"] = all(g["C_detail"].values())
    return g


def branch(states: Dict[str, str], g: Dict) -> str:
    bad = [k for k, v in states.items() if v != "completed" and not k.startswith("report_only:")]
    if bad or "incomplete" in (g.get("A"), g.get("B"), g.get("C")):
        kind = "invalid" if any(v.startswith("invalid") or v.startswith("missing / invalid") for v in states.values()) else "incomplete"
        return "%s: %s -- no verdict (the computed metrics are reported as they are)" % (kind, bad or [k for k in "ABC" if g.get(k) == "incomplete"])
    if g["A"] is False:
        return "recovery failed: a pre-written protection cost against v0.3 remains -- nothing further; not a candidate"
    if g["B"] is False or g["C"] is False:
        failed = [k for k, v in {**g.get("B_detail", {}), **g.get("C_detail", {})}.items() if v is False]
        return "protection recovered but the new ability is not kept (%s): old ability traded for new; not a candidate; no added budget" % failed
    return ("engineering recovery: no protection cost against v0.3 and the new ability kept on given tables (B) and EXP56's "
            "automatic tables (C) -- not a mechanism attribution; next, each separately permitted: the new combination's live "
            "transfer on dev_sqa, independent confirmation, calibration, full evaluation, install / release checks; not a candidate")


def report(out: str = OUT, finalize: str = "") -> Dict:
    res: Dict[str, Any] = {"states": {}, "A": {}, "B": {}, "C": {}, "candidate_eligible": False}
    if finalize:
        res["finalized"] = {"exit_codes": finalize, "note": "missing items are not low scores"}
    if not os.path.exists(os.path.join(out, "START.json")):
        res["branch"] = "incomplete: no START (nothing ran)"
        return write_report(out, res)
    try:
        st = check_start(out, data_root="", code=False)
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["branch"] = "invalid: the run's identity does not verify (%s) -- no verdict" % e
        return write_report(out, res)
    S = res["states"]
    try:
        fid = json.load(open(os.path.join(out, "fidelity.json")))
        if e54.check_registered(out, "fidelity").get("fidelity.json") != sha(os.path.join(out, "fidelity.json")):
            raise SystemExit("fidelity.json not registered")
        S["fid"] = "completed" if fid["reproduced"] else "invalid: the start's fidelity not reproduced"
    except (SystemExit, OSError, ValueError, KeyError) as e:
        S["fid"] = e68.read_state(out, "fidelity", e)
    weights = bundle = None
    try:
        weights = new_weights()
        bundle = bundle_bound(out)
        S["frozen"] = "completed"
    except (SystemExit, OSError, ValueError, KeyError) as e:
        S["frozen"] = "missing: %s" % e
    recs: Dict[str, List[Dict]] = {}
    if weights is not None:
        for name, path, model in ([("B_" + n, B_SETS[n], weights) for n in B_REQUIRED + ("fit",)]
                                  + [("C_P_new", P_SET, weights), ("C_P_start", P_SET, START_SHA)]):
            key = ("report_only:" + name) if name == "B_fit" else name
            rows, inputs = rows_inputs(path)
            try:
                recs[name] = e57.read_set(out, name, rows, model, inputs)
                S[key] = "completed"
            except (SystemExit, OSError, ValueError, KeyError) as e:
                S[key] = e68.read_state(out, name, e)
    qs53 = {q["key"]: q for q in jsonl(e54.QUESTIONS) if q["set"] == "rl_tune"}
    pin = e54.load_inputs(e54.OUT, "P")
    tune = {r["id"]: r for r in jsonl(e54.TUNE)}
    p_right = {k: {(o, qn): v for o, qn, v in pin[q["id"]]["table"]}[e54.TEMPLATES[tune[q["id"]]["questions"][q["qid"]]["instructions"]]] == q["target"]
               for k, q in qs53.items()}
    b = res["B"]
    if "B_slice" in recs:
        b["fit_slice"] = e72.by_class(recs["B_slice"], rows_inputs(B_SETS["slice"])[0])
    if "B_diag" in recs:
        b["diag"] = e68.diag68_summary(recs["B_diag"], rows_inputs(B_SETS["diag"])[0])
    if "B_G" in recs:
        try:
            b["G_reading"] = e72.g_reading(recs["B_G"], rows_inputs(B_SETS["G"])[0], qs53, p_right)
        except SystemExit as e:
            S["B_G"] = "invalid: %s" % e
    if "B_fit" in recs:
        frows = rows_inputs(B_SETS["fit"])[0]
        b["fit_old"] = {kind: e63.summary(e63.comparisons(recs["B_fit"], frows, kind), kind) for kind in ("temp", "force")}
    c = res["C"]
    prows = rows_inputs(P_SET)[0]
    if "C_P_new" in recs:
        c["comparison"] = e63.summary(e63.comparisons(recs["C_P_new"], prows, "temp"), "temp")
        c["reading"] = e57.reading_summary(recs["C_P_new"], prows, qs53, p_right)
        c["P_use"] = e54.gates({"conditions": {"P": c["reading"]}, "differences": {}})["P_use"]
        c["rows_without_K"] = {k: v for k, v in st["data"]["P_K_rows"].items() if k.startswith("no K")}
    if "C_P_new" in recs and "C_P_start" in recs:
        c["new_minus_start"] = e57.paired_difference(e63.comparisons(recs["C_P_new"], prows, "temp"), e63.comparisons(recs["C_P_start"], prows, "temp"))
        c["start_comparison"] = e63.summary(e63.comparisons(recs["C_P_start"], prows, "temp"), "temp")
    st73 = json.load(open(os.path.join(E73, "START.json")))
    for name in GUARD:
        if weights is None or bundle is None:
            S["A_" + name] = "missing"
            continue
        if not os.path.exists(os.path.join(out, "details_A_%s.jsonl" % name)):
            S["A_" + name] = "missing"
            continue
        try:
            new = a_records(out, name, st, weights, bundle)
            m = panel_metric(e58.v03_records(name, st), new, name)
        except (SystemExit, OSError, ValueError, KeyError) as e:
            S["A_" + name] = "invalid: %s" % e
            continue
        S["A_" + name] = "completed"
        res["A"][name] = {"difference": m["difference"], "ci95": m["ci95"], "cost": e73.cost(m), "routed": sum(r["routed"] for r in new)}
        try:
            m73 = panel_metric(e73.g_records(E73, name, st73), new, name)
            res["A"][name]["vs_exp73"] = {"difference": m73["difference"], "ci95": m73["ci95"]}
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["A"][name]["vs_exp73"] = "not computed: %s" % e
        if name == "science":
            routed, ordinary = [r for r in new if r["routed"]], [r for r in new if not r["routed"]]
            res["A"][name]["slices"] = {"routed": {"n": len(routed), "accuracy": sum(r["correct"] for r in routed) / len(routed) if routed else None},
                                        "ordinary": {"n": len(ordinary), "accuracy": sum(r["correct"] for r in ordinary) / len(ordinary) if ordinary else None}}
    res["gates"] = gates(res)
    res["branch"] = branch(S, res["gates"])
    res["limits"] = ("C uses EXP56's automatic tables, not the new combination's live reader (EXP71's components with the corrector): "
                     "passing C is not the new live combination meeting its transfer target; EXP71's evidence is not reused for the new end")
    return write_report(out, res)


def write_report(out: str, res: Dict) -> Dict:
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "report.json"), res)
    text = json.dumps({k: res[k] for k in ("finalized", "states", "gates", "branch", "candidate_eligible") if k in res}, indent=1, default=str)
    lines = [text] + ["A %s %+.4f [%+.4f, %+.4f]%s" % (n, v["difference"], *v["ci95"], " COST" if v["cost"] else "") for n, v in res.get("A", {}).items()]
    open(os.path.join(out, "report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["start", "fid", "train", "freeze", "decide_b", "decide_c", "guard", "decide_fit", "report"])
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--finalize", default="")
    a = ap.parse_args(argv)
    if a.stage == "start":
        print(json.dumps({k: v for k, v in stage_start(a.deadline).items() if k != "files"}, indent=1, default=str))
    elif a.stage == "fid":
        print(json.dumps(stage_fid(), indent=1))
    elif a.stage == "train":
        return stage_train()
    elif a.stage == "freeze":
        print(json.dumps(stage_freeze(), indent=1))
    elif a.stage in ("decide_b", "decide_c", "decide_fit"):
        print(json.dumps({"decide_b": stage_decide_b, "decide_c": stage_decide_c, "decide_fit": stage_decide_fit}[a.stage](), indent=1))
    elif a.stage == "guard":
        print(json.dumps(stage_guard(), indent=1))
    else:
        try:
            report(finalize=a.finalize)
        except Exception as e:
            os.makedirs(OUT, exist_ok=True)
            open(os.path.join(OUT, "report.txt"), "w").write("REPORT FAILED (invalid, no verdict): %s\n" % e)
            print("REPORT FAILED: %s" % e, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
