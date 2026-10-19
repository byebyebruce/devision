"""LIVE-K: the live K combination's transfer check (critic LIVE-K-RUNNER-PREP-02; docs/experiments/2026-10-19-live-k-plan.md,
runner section). Fixed: PROTECT's end 185b9496 / 4f9ccf48 (PROTECT-RESULT-01: complete, A / B / C true), EXP71's selector /
head / corrector, the K rule (devision.reading.K_RULE) -- a new combination with its own bundle (runs/livek/bundle_k) and
identity; PROTECT's no-K combination (runs/protect/bundle) is kept as it is. No training. Raw temperature 1.
candidate_eligible is always false.

  start          model-free: identities (PROTECT's run + its report's states and gates, its frozen end and no-K bundle,
                 EXP71's bundle_on, the K bundle written here from the same components with k_rule, tune_P / P_K, the dev_sqa
                 true / fixed-wrong files, the seven panels, v0.3's records, PROTECT's registered C_P_new and A records of the
                 six reused panels with the evidence that they hold no reading-routed question, every picture opened, this
                 code) -> runs/livek/START.json, before any forward.
  fid_text       MPS: the end on P_K's 600 given states (text) -- must equal PROTECT's registered C_P_new (same predictions,
                 probabilities within 1e-5); else stop (nothing after it runs).
  live_p         MPS: the K combination on tune_P's 600 pictures (each row's text without its table; the table and K from the
                 picture actually shown); reads, tables, each question's state and K registered.
  transfer       MPS: the K combination on dev_sqa temperature / magnets, true and fixed wrong pictures (the wrong picture
                 read itself), EXP58's request grouping.
  guard_science  MPS: the K combination on the whole science panel (269 requests; its 9 temperature reading questions are
                 the K-affected range).
  report         model-free: the identity and fidelity first (missing / invalid before any score); live P against C's
                 lines; dev_sqa against v0.3 (68:114; accuracy change >= +10 points with lower bound > 0, picture-gain
                 change lower bound > 0), per domain described; protection: science recomputed, the six other panels
                 PROTECT-A's registered records reused (cited by file and identity) -- the pre-written branch.

    uv run python scripts/livek/livek.py counts | start --deadline EPOCH | fid_text | live_p | transfer | guard_science | report
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(__file__)
OUT = "runs/livek"
PROTECT_OUT, PROTECT_RUN = "runs/protect", "runs/protect-k"
PROTECT = {"START.json": "0808bb2a47ecd5fce6ba013838d8bf048af1e8ffea8844dc7587b9e81c180a53",
           "report.json": "b75f78023f970b6a4dba2e076ba180e0261f57257a2283f2653650cf9bf85093",
           "registry.json": "5ddb30f47f0235b1efc92af264677c600bbae5381c0b7efd1037e69a10316881"}
NEW_SHA = {"model.safetensors": "185b9496fc9ea64439eb996ab8b20c3a5d16dd490feff5b954b0dba5b1096395",
           "config.json": "4f9ccf4811713fac17c98f186ea35929311e367107d6a3f2934235d31f060e23"}
TUNE_P, P_K = "data/facts/tune_P.jsonl", "data/protect_p/P_K.jsonl"
FID_TOL = 1e-5
REUSED = ("exist", "ext", "count_wide", "count_narrow", "flip", "vsr")
FILES = ["scripts/livek/livek.py", "scripts/livek/run.sh", "scripts/protect/protect.py", "scripts/exp73/exp73.py", "scripts/exp71/exp71.py",
         "scripts/exp58/live.py", "scripts/exp21/readout.py", "scripts/exp63/exp63.py", "scripts/exp57/exp57.py", "scripts/exp54/exp54.py",
         "scripts/exp32/report.py", "src/devision/reading.py", "src/devision/infer.py", "src/devision/model.py", "src/devision/image.py",
         "src/devision/train/evaluate.py", "src/devision/train/compare.py", "scripts/exp53/watch.py"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pr = _load("protect", "../protect/protect.py")
e73, e72, e71, e58, e63, e57, e54, fz = pr.e73, pr.e72, pr.e71, pr.e58, pr.e63, pr.e57, pr.e54, pr.fz
sha, write_json, register, jsonl = pr.sha, pr.write_json, pr.register, pr.jsonl
GUARD, TRANSFER = e58.GUARD, e58.TRANSFER


# ------------------------------------------------------------------ identities

def bound_protect() -> Dict:
    """PROTECT's run as reviewed (PROTECT-RESULT-01): pinned files, every state completed, gates A / B / C true; its end
    and the no-K bundle it registered."""
    run = e71.bound_files(PROTECT_OUT, PROTECT, "PROTECT")
    rep = json.load(open(os.path.join(PROTECT_OUT, "report.json")))
    if any(v != "completed" for v in rep["states"].values()) or not all(rep["gates"].get(k) is True for k in "ABC"):
        raise SystemExit("PROTECT is not complete with A / B / C true")
    if fz.check(PROTECT_RUN, [])["weights"] != NEW_SHA:
        raise SystemExit("PROTECT's frozen end is not 185b9496 / 4f9ccf48")
    nok = os.path.join(PROTECT_OUT, "bundle", "reading.json")
    if e54.check_registered(PROTECT_OUT, "bundle").get("bundle/reading.json") != sha(nok):
        raise SystemExit("PROTECT's no-K bundle is not the registered one")
    return {"run": run, "end": dict(NEW_SHA), "no_k_bundle": sha(nok)}


def reused_records() -> Dict:
    """The six panels whose PROTECT-A records are reused: their registered files and the equivalence evidence -- no
    question routed to reading (devision.reading.domain_of), so the K rule never reaches them and Reading.decide takes the
    same ordinary path with the same end, temperature and grouping."""
    from devision.reading import domain_of
    out = {}
    for n in REUSED:
        routed = sum(domain_of(q.get("instructions", "")) is not None for r in jsonl(GUARD[n]) for q in r["questions"].values())
        if routed:
            raise SystemExit("%s holds %d reading-routed questions: its records cannot be reused" % (n, routed))
        f = "details_A_%s.jsonl" % n
        h = e54.check_registered(PROTECT_OUT, "decide_A_" + n).get(f)
        if h != sha(os.path.join(PROTECT_OUT, f)):
            raise SystemExit("PROTECT's %s is not the registered one" % f)
        out[n] = {"file": os.path.join(PROTECT_OUT, f), "sha256": h, "routed_questions": 0}
    return out


def live_rows() -> List[Dict]:
    """tune_P's 600 rows as live requests: each row's own text without the table (the stem before the reading header),
    its picture and questions."""
    from devision.reading import HEADER
    out = []
    for r in jsonl(TUNE_P):
        stem = r["state_text"].split("\n" + HEADER)[0] if ("\n" + HEADER) in r["state_text"] else r["state_text"].split(HEADER)[0].rstrip("\n")
        out.append(dict(r, state_text=stem))
    return out


def needed_pictures() -> List[str]:
    paths = {r["image"] for r in jsonl(TUNE_P)} | {r["image"] for p in TRANSFER.values() for r in jsonl(p)} | {r["image"] for r in jsonl(GUARD["science"])}
    return sorted(paths)


def pictures(data_root: str = "data", manifest: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Every picture this run opens with its sha256; those in EXP58's bound manifest must agree with it."""
    if manifest is None:
        manifest = {p: sha(os.path.join(data_root, p)) for p in needed_pictures()}
    m58 = json.load(open(os.path.join(e71.E58R, "pictures.json")))
    bad = [p for p, h in manifest.items() if p in m58 and m58[p] != h]
    if bad:
        raise SystemExit("%d pictures differ from EXP58's manifest, e.g. %s" % (len(bad), bad[0]))
    if data_root:
        bad = [p for p, h in manifest.items() if sha(os.path.join(data_root, p)) != h]
        if bad:
            raise SystemExit("%d pictures differ from START, e.g. %s" % (len(bad), bad[0]))
    return manifest


def request_counts() -> Dict:
    return {"fid_text": len(jsonl(P_K)), "live_p": len(e58.requests(live_rows())),
            "transfer": {n: len(e58.requests(jsonl(p))) for n, p in TRANSFER.items()}, "science": len(e58.requests(jsonl(GUARD["science"])))}


def identities() -> Dict:
    return {"protect": bound_protect(), "exp71": e73.bound_e71(), "reused": reused_records(),
            "data": {"tune_P": sha(TUNE_P), "P_K": sha(P_K), **{n: sha(p) for n, p in TRANSFER.items()}, **{"panel_" + n: sha(p) for n, p in GUARD.items()}},
            "v03": {n: sha(os.path.join(e58.V03, n + ".details.jsonl")) for n in list(TRANSFER) + list(GUARD)},
            "c_p_new": e54.check_registered(PROTECT_OUT, "decide_C_P_new").get("details_C_P_new.jsonl"),
            "exp49_questions": sha(e54.QUESTIONS), "exp54_inputs_P": sha(os.path.join(e54.OUT, "inputs_P.jsonl"))}


def write_k_bundle(out: str, st_exp71: Dict) -> Dict:
    from devision.reading import write_bundle
    b71 = st_exp71["bundle_on"]
    dims = json.load(open(os.path.join(e73.BUNDLE, "reading.json")))["dims"]
    ident = write_bundle(os.path.join(out, "bundle_k"), os.path.join(PROTECT_RUN, "stage2"), os.path.join(e73.BUNDLE, "selector.pt"),
                         os.path.join(e73.BUNDLE, "head.pt"), {"components_of": "EXP71 bundle_on", "exp71_reading_json": b71["reading.json"],
                                                                "decision_of": "PROTECT end"}, dims=dims,
                         corrector_pt=os.path.join(e73.BUNDLE, "corrector.pt"), k_rule=True)
    if any(ident["sha256"][c] != b71[c] for c in ("selector.pt", "head.pt", "corrector.pt")) or "k_rule" not in ident:
        raise SystemExit("the K bundle's components / rule are not the intended ones")
    return {"reading.json": sha(os.path.join(out, "bundle_k", "reading.json")), "rules_sha256": ident["rules_sha256"]}


def stage_start(deadline: int, out: str = OUT, data_root: str = "data") -> Dict:
    path = os.path.join(out, "START.json")
    if os.path.exists(path) or os.path.exists(os.path.join(out, "bundle_k")):
        raise SystemExit("%s exists (fresh only)" % out)
    ident = identities()
    pics = pictures(data_root)
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "pictures.json"), pics)
    kb = write_k_bundle(out, ident["exp71"])
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, **ident, "k_bundle": kb,
         "pictures": {"pictures.json": sha(os.path.join(out, "pictures.json")), "count": len(pics)}, "requests": request_counts(),
         "temperature": "none (raw), in memory", "candidate_eligible": False,
         "purpose": "LIVE-K transfer check of PROTECT's end with EXP71's components and the K rule (LIVE-K-RUNNER-PREP-02); not a candidate",
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
    kj = os.path.join(out, "bundle_k", "reading.json")
    if sha(kj) != st["k_bundle"]["reading.json"]:
        raise SystemExit("the K bundle differs from START")
    if sha(os.path.join(out, "pictures.json")) != st["pictures"]["pictures.json"]:
        raise SystemExit("the picture manifest differs from START")
    pictures(data_root, json.load(open(os.path.join(out, "pictures.json"))))
    return st


# ------------------------------------------------------------------ stages

def stage_fid_text(out: str = OUT, decider=None) -> Dict:
    """The end on P_K's 600 given states equals PROTECT's registered C_P_new (predictions, probabilities within 1e-5)."""
    from devision.infer import Decider, best_device
    check_start(out)
    rows, inputs = pr.rows_inputs(P_K)
    ref = {(r["sample_id"], r["qid"]): r for r in e57.read_set(PROTECT_OUT, "C_P_new", rows, NEW_SHA, inputs)}
    if decider is None:
        if fz.weights(PROTECT_RUN) != NEW_SHA:
            raise SystemExit("PROTECT's end is not START's")
        decider = Decider.load(os.path.join(PROTECT_RUN, "stage2"), device=best_device())
    decider.cfg.temperature, decider.cfg.temperature_by_options = [1.0, 1.0, 1.0], {}
    recs = e54.decide_condition(decider, rows, inputs, on_request=lambda k: e54.progress(out, "fid_text", k, len(rows)))
    e57.write_set(out, "fid_text", recs, {"set": "fid_text", "start_sha256": sha(os.path.join(out, "START.json")), "inputs_sha256": e57.hash_inputs(inputs),
                                          "model": NEW_SHA, "temperature": "none (raw)", "image": None})
    keys = [(r["sample_id"], r["qid"]) for r in recs]
    diff = max(abs(r["probabilities"][o] - ref[k]["probabilities"][o]) for r, k in zip(recs, keys) for o in r["probabilities"])
    mism = [k for r, k in zip(recs, keys) if r["prediction"] != ref[k]["prediction"]]
    fid = {"n": len(recs), "registered": len(ref), "mismatches": mism[:20], "max_abs_probability_difference": diff,
           "reproduced": len(recs) == len(ref) and set(keys) == set(ref) and not mism and diff <= FID_TOL}
    write_json(os.path.join(out, "fidelity.json"), fid)
    register(out, "fidelity", {"fidelity.json": sha(os.path.join(out, "fidelity.json"))})
    if not fid["reproduced"]:
        raise SystemExit("the end does not reproduce PROTECT's C_P_new -- nothing after this runs")
    return fid


def load_k_reading(out: str):
    from devision.infer import best_device
    from devision.reading import Reading
    reading = Reading.load(os.path.join(PROTECT_RUN, "stage2"), os.path.join(out, "bundle_k"), device=best_device(), full_rules=True)
    if reading.corrector is None or not reading.k_rule:
        raise SystemExit("the K bundle loaded without its corrector or K rule")
    reading.decider.cfg.temperature, reading.decider.cfg.temperature_by_options = [1.0, 1.0, 1.0], {}
    return reading


def run_set(reading, samples: List[Dict], data_root: str, pics: Dict[str, str], on_request=None):
    """EXP58's grouping through e58.ask (the actual picture as image=), each question's actual state kept
    (Reading's diagnostics `states`), the request's reads / table / K block registered with it."""
    from devision.reading import domain_of
    from devision.train.evaluate import _probabilities, score_records
    asked, reads = {}, []
    for n, (image, text, items) in enumerate(e58.requests(samples)):
        questions = {"%d:%s" % (i, qid): q for i, qid, q in items}
        diag: Dict = {}
        res = e58.ask(reading, image, text, questions, data_root, pics, diag)
        reads.append({"image": image, "text": text, "questions": [[samples[i]["id"], qid] for i, qid, _ in items], "reads": diag.get("reads"),
                      "table": diag.get("table"), "k_block": diag.get("k_block"), "picture": diag.get("picture"),
                      "states": {"%s/%s" % (samples[i]["id"], qid): (diag.get("states") or {}).get(key) for key, (i, qid, _) in zip(questions, items)}})
        for key, (i, qid, q) in zip(questions, items):
            asked[(i, qid)] = (q, image, len(items), _probabilities(res["answers"][key]))
        if on_request:
            on_request(n + 1)
    recs = []
    for i, s in enumerate(samples):
        for qid in s["questions"]:
            q, image, size, probs = asked[(i, qid)]
            gold = s["gold"][qid]["probabilities"]
            recs.append({"sample_id": s["id"], "qid": qid, "image_id": s["image_id"], "image_used": image, "source": s.get("source", ""),
                         "kind": s.get("kind", ""), "axis": s.get("axis", ""), "type": q["type"],
                         "options": list(q["criteria"]) if q["type"] == "choice" else ["false", "true"], "gold": gold,
                         "gold_answer": max(gold, key=gold.__getitem__), "probabilities": probs, "prediction": max(probs, key=probs.__getitem__),
                         "control": "none", "questions_in_request": size, "routed": domain_of(q.get("instructions", "")) is not None})
    return score_records(recs), reads


def register_set(out: str, name: str, recs: List[Dict], reads: List[Dict], st: Dict, data_sha: str) -> None:
    ident = {"set": name, "start_sha256": sha(os.path.join(out, "START.json")), "data_sha256": data_sha, "model": NEW_SHA,
             "bundle": st["k_bundle"]["reading.json"], "pictures": st["pictures"]["pictures.json"], "temperature": "none (raw)",
             "path": "devision.reading.Reading.decide(image=actual picture), PROTECT end + EXP71's components + K rule"}
    e57.write_set(out, name, recs, ident)
    f = "reads_%s.jsonl" % name
    with open(os.path.join(out, f), "w") as fh:
        fh.write("".join(json.dumps(x) + "\n" for x in reads))
    register(out, "reads_" + name, {f: sha(os.path.join(out, f))})


def fidelity_ok(out: str) -> bool:
    f = os.path.join(out, "fidelity.json")
    return os.path.exists(f) and e54.check_registered(out, "fidelity").get("fidelity.json") == sha(f) and json.load(open(f))["reproduced"]


def stage_live(out: str, which: str, data_root: str = "data", reading=None) -> Dict:
    """live_p / transfer / guard_science through the K combination, after the text fidelity held."""
    st = check_start(out, data_root)
    if not fidelity_ok(out):
        raise SystemExit("the text fidelity is not established -- no live stage")
    reading = reading or load_k_reading(out)
    pics = json.load(open(os.path.join(out, "pictures.json")))
    plan = ({"live_P": (live_rows(), st["data"]["tune_P"])} if which == "live_p" else
            {n: (jsonl(p), st["data"][n]) for n, p in TRANSFER.items()} if which == "transfer" else
            {"science": (jsonl(GUARD["science"]), st["data"]["panel_science"])})
    total = sum(len(e58.requests(s)) for s, _ in plan.values())
    done, t0 = 0, time.time()
    for name, (samples, dsha) in plan.items():
        recs, reads = run_set(reading, samples, data_root, pics, on_request=lambda k, b=done: e54.progress(out, which, b + k, total))
        done += len(reads)
        register_set(out, name, recs, reads, st, dsha)
        print("%s done after %.1f s" % (name, time.time() - t0), flush=True)
    return {"seconds": time.time() - t0}


# ------------------------------------------------------------------ report

def set_records(out: str, name: str, src: List[Dict], st: Dict, data_sha: str) -> List[Dict]:
    """A registered set after every check: identity bound to START / the K bundle, every source question once, each on its
    own row's picture, and its reading evidence complete and consistent (check_reads)."""
    f = "details_%s.jsonl" % name
    if e54.check_registered(out, "decide_" + name).get(f) != sha(os.path.join(out, f)):
        raise SystemExit("%s not registered" % f)
    lines = jsonl(os.path.join(out, f))
    want = {"set": name, "start_sha256": sha(os.path.join(out, "START.json")), "data_sha256": data_sha, "model": NEW_SHA,
            "bundle": st["k_bundle"]["reading.json"], "pictures": st["pictures"]["pictures.json"], "temperature": "none (raw)"}
    if any(lines[0]["identity"].get(k) != v for k, v in want.items()):
        raise SystemExit("%s: identity does not bind" % f)
    got = e57.r41.check_details("livek/%s" % name, src, lines[1:])
    image_of = {(r["id"], q): r["image"] for r in src for q in r["questions"]}
    if any(r["image_used"] != image_of[(r["sample_id"], r["qid"])] for r in lines[1:]):
        raise SystemExit("%s: a question was answered on another picture than its row's" % f)
    rf = "reads_%s.jsonl" % name
    if e54.check_registered(out, "reads_" + name).get(rf) != sha(os.path.join(out, rf)):
        raise SystemExit("%s not registered" % rf)
    check_reads(rf, src, jsonl(os.path.join(out, rf)))
    return list(got.values())


def check_reads(rf: str, src: List[Dict], lines: List[Dict]) -> None:
    """The reading evidence of a set, complete and consistent (critic LIVE-K-RUNNER-R1-01): exactly the requests EXP58's
    grouping makes of the source, in order, each once -- picture, source text and question members; for a request with
    read questions its reads name each of the request's reading templates once, in order, every read value is the table's
    value for that template (an "unreadable" read is a valid failure and stays in the denominator), K is computed from that
    table, and each read question's state is the request text + that table (+ K for a temperature question when K is
    defined); a request without read questions carries no table. Anything else is refused."""
    from devision.reading import TEMPLATES, domain_of, k_block, reads_for, serialise
    want = e58.requests(src)
    if len(lines) != len(want):
        raise SystemExit("%s: %d requests registered, the source makes %d" % (rf, len(lines), len(want)))
    by_id = {r["id"]: r for r in src}
    for n, ((image, text, items), x) in enumerate(zip(want, lines)):
        members = [[src[i]["id"], qid] for i, qid, _ in items]
        if x.get("image") != image or x.get("text") != text or x.get("questions") != members:
            raise SystemExit("%s: request %d is not the source's (picture / text / questions)" % (rf, n))
        qs = {"%s/%s" % (sid, qid): by_id[sid]["questions"][qid] for sid, qid in members}
        routed = {k: domain_of(q.get("instructions", "")) for k, q in qs.items() if domain_of(q.get("instructions", "")) is not None}
        states = x.get("states") or {}
        if not routed:
            if x.get("table") or x.get("reads") or any(v is not None for v in states.values()):
                raise SystemExit("%s: request %d has no read question but carries reads" % (rf, n))
            continue
        tmpls = reads_for(qs)
        reads, table = x.get("reads") or [], x.get("table") or []
        if [r.get("question") for r in reads] != tmpls:
            raise SystemExit("%s: request %d's reads are not its reading templates once each" % (rf, n))
        if table != [[*TEMPLATES[t], r.get("value")] for t, r in zip(tmpls, reads)]:
            raise SystemExit("%s: request %d's table is not its reads' values" % (rf, n))
        plain = serialise(text, table)
        kb = k_block(table)
        hot = kb is not None and any(d == "temperature" for d in routed.values())
        if x.get("k_block") != (kb if any(d == "temperature" for d in routed.values()) else None):
            raise SystemExit("%s: request %d's K block is not its table's" % (rf, n))
        for k, d in routed.items():
            if states.get(k) != (plain + "\n" + str(kb) if hot and d == "temperature" else plain):
                raise SystemExit("%s: request %d, %s: the state is not the text + its table (+ K)" % (rf, n, k))


def bare(value: Optional[str]) -> Optional[str]:
    """A table value without thousands separators ("1,270" -> "1270"; "unreadable" / None unchanged)."""
    return value.replace(",", "") if isinstance(value, str) else value


def live_p_metrics(out: str, recs: List[Dict], st: Dict) -> Dict:
    """C's lines on the live tables: temperature comparison, reading 'all' and P_use (exp57.reading_summary with read_right
    from the live tables), plus the live tables against EXP56's (described) and how many temperature requests had K."""
    rows = {r["id"]: r for r in jsonl(TUNE_P)}
    reads = {x["image"]: x for x in jsonl(os.path.join(out, "reads_live_P.jsonl"))}
    qs53 = {q["key"]: q for q in jsonl(e54.QUESTIONS) if q["set"] == "rl_tune"}
    tune = {r["id"]: r for r in jsonl(e54.TUNE)}
    pin = e54.load_inputs(e54.OUT, "P")
    live_val, p_right, same = {}, {}, []
    for k, q in qs53.items():
        tmpl = e54.TEMPLATES[tune[q["id"]]["questions"][q["qid"]]["instructions"]]
        table = {(o, qn): v for o, qn, v in reads[rows[q["id"]]["image"]]["table"]}
        live_val[k] = bare(table.get(tmpl))            # the live table has thousands separators; targets / EXP56's table do not
        p_right[k] = live_val[k] == bare(q["target"])
        same.append(live_val[k] == bare({(o, qn): v for o, qn, v in pin[q["id"]]["table"]}[tmpl]))
    rd = e57.reading_summary(recs, rows, qs53, p_right)
    cmp_ = e63.summary(e63.comparisons(recs, rows, "temp"), "temp")
    temp_reqs = [x for x in reads.values() if rows[x["questions"][0][0]]["kind"] == "temp"]
    return {"comparison": cmp_, "reading": rd, "P_use": e54.gates({"conditions": {"P": rd}, "differences": {}})["P_use"],
            "live_values_right": sum(p_right.values()) / len(p_right), "live_equals_exp56_table": sum(same) / len(same),
            "temperature_requests": len(temp_reqs), "with_K": sum(x["k_block"] is not None for x in temp_reqs)}


def branch(res: Dict) -> str:
    S = res["states"]
    bad = [k for k, v in S.items() if v != "completed"]
    if res.get("fidelity") is not None and not res["fidelity"].get("reproduced"):
        return "stopped: the end does not reproduce PROTECT's C_P_new on given P_K states -- no live result"
    if bad:
        kind = "invalid" if any(v.startswith("invalid") or v.startswith("missing / invalid") for v in S.values()) else "incomplete"
        return "%s: %s -- no verdict (missing / invalid before any score)" % (kind, bad)
    g = res["gates"]
    if not g["protection"]:
        return "protection cost against v0.3 with the K combination (%s): not continued" % res["costs"]
    if not g["transfer"]:
        return "the K combination does not meet the v0.3 transfer target on dev_sqa (68:114); not continued"
    if not g["live_P"]:
        return "transfer met but the live P lines fail (%s): not continued" % [k for k, v in g["live_P_detail"].items() if not v]
    return ("the K combination meets the v0.3 transfer target on dev_sqa with its live P lines and no protection cost (science "
            "recomputed, six panels PROTECT-A reused): a development result -- independent confirmation, calibration and full "
            "release checks remain, each separately permitted; not a candidate")


def report(out: str = OUT, finalize: str = "") -> Dict:
    res: Dict[str, Any] = {"states": {}, "candidate_eligible": False}
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
        if not os.path.exists(os.path.join(out, "fidelity.json")):
            raise SystemExit("missing")
        res["fidelity"] = json.load(open(os.path.join(out, "fidelity.json")))
        rows, inputs = pr.rows_inputs(P_K)
        e57.read_set(out, "fid_text", rows, NEW_SHA, inputs)
        if not fidelity_ok(out):
            raise SystemExit("fidelity.json not registered / not reproduced")
        S["fid_text"] = "completed"
    except (SystemExit, OSError, ValueError, KeyError) as e:
        S["fid_text"] = "missing" if str(e) == "missing" else "invalid: %s" % e
    recs: Dict[str, List[Dict]] = {}
    for name, src, key in ([("live_P", live_rows(), "tune_P")] + [(n, jsonl(p), n) for n, p in TRANSFER.items()]
                           + [("science", jsonl(GUARD["science"]), "panel_science")]):
        if not os.path.exists(os.path.join(out, "details_%s.jsonl" % name)):
            S[name] = "missing"
            continue
        try:
            recs[name] = set_records(out, name, src, st, st["data"][key])
            S[name] = "completed"
        except (SystemExit, OSError, ValueError, KeyError) as e:
            S[name] = "invalid: %s" % e
    gates: Dict[str, Any] = {}
    if "live_P" in recs:
        res["live_P"] = lp = live_p_metrics(out, recs["live_P"], st)
        gates["live_P_detail"] = {"comparison": (lp["comparison"]["accuracy"] or 0) >= pr.P_CMP_LINE,
                                  "P_all": (lp["reading"]["all"] or 0) >= pr.P_ALL_LINE, "P_use": lp["P_use"] is True}
        gates["live_P"] = all(gates["live_P_detail"].values())
    if all(n in recs for n in TRANSFER):
        sys.path.insert(0, os.path.join(HERE, "..", "exp21"))
        from readout import transfer  # type: ignore
        t = transfer({k[4:]: e58.v03_records(k, st) for k in TRANSFER}, {k[4:]: recs[k] for k in TRANSFER})
        res["transfer"] = t
        res["accuracy"] = {k: sum(r["correct"] for r in recs[k]) / len(recs[k]) for k in TRANSFER}
        gates["transfer"] = bool(t["accuracy_change"]["value"] >= 0.10 and t["accuracy_change"]["ci95"][0] > 0 and t["gain_change"]["ci95"][0] > 0)
    res["protection"], res["costs"] = {}, []
    if "science" in recs:
        m = pr.panel_metric(e58.v03_records("science", st), recs["science"], "science")
        res["protection"]["science"] = {"difference": m["difference"], "ci95": m["ci95"], "cost": e73.cost(m), "source": "recomputed here",
                                        "routed": sum(r["routed"] for r in recs["science"])}
    st_p = json.load(open(os.path.join(PROTECT_OUT, "START.json")))
    for n in REUSED:
        try:
            old = pr.a_records(PROTECT_OUT, n, st_p, NEW_SHA, st["protect"]["no_k_bundle"])
            m = pr.panel_metric(e58.v03_records(n, st), old, n)
            res["protection"][n] = {"difference": m["difference"], "ci95": m["ci95"], "cost": e73.cost(m),
                                    "source": "PROTECT-A reused: %s (sha256 %s, no reading-routed question; same end, raw temperature, "
                                              "EXP58 grouping, Reading.decide ordinary path)" % (st["reused"][n]["file"], st["reused"][n]["sha256"])}
            S["reused_" + n] = "completed"
        except (SystemExit, OSError, ValueError, KeyError) as e:
            S["reused_" + n] = "invalid: %s" % e
    res["costs"] = [n for n, v in res["protection"].items() if v["cost"]]
    if len(res["protection"]) == len(GUARD):
        gates["protection"] = not res["costs"]
    res["gates"] = gates
    res["branch"] = branch(res) if res.get("fidelity") is not None or S else "incomplete"
    res["limits"] = ("dev_sqa is a repeatedly used development set; the six reused panels are PROTECT-A's no-K records (valid: no "
                     "reading-routed question); the K block is computed outside the model (automatic reading + deterministic "
                     "arithmetic + model decision); not a candidate")
    return write_report(out, res)


def write_report(out: str, res: Dict) -> Dict:
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "report.json"), res)
    text = json.dumps({k: res[k] for k in ("finalized", "states", "gates", "costs", "accuracy", "branch", "candidate_eligible") if k in res},
                      indent=1, default=str)
    t = res.get("transfer")
    if isinstance(t, dict):
        text += "\ntransfer vs v0.3: accuracy %+.3f %s, gain %+.3f %s" % (t["accuracy_change"]["value"], t["accuracy_change"]["ci95"],
                                                                        t["gain_change"]["value"], t["gain_change"]["ci95"])
    open(os.path.join(out, "report.txt"), "w").write(text + "\n")
    print(text)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["counts", "start", "fid_text", "live_p", "transfer", "guard_science", "report"])
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--finalize", default="")
    a = ap.parse_args(argv)
    if a.stage == "counts":
        print(json.dumps(request_counts(), indent=1))
    elif a.stage == "start":
        print(json.dumps({k: v for k, v in stage_start(a.deadline).items() if k != "files"}, indent=1, default=str))
    elif a.stage == "fid_text":
        print(json.dumps(stage_fid_text(), indent=1))
    elif a.stage in ("live_p", "transfer", "guard_science"):
        print(json.dumps(stage_live(OUT, a.stage), indent=1))
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
