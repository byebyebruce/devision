"""exp71: EXP69's speed corrector in the live reading, on / off with everything else fixed (critic EXP71-PREP-01;
docs/experiments/2026-10-19-exp71-live-corrector-plan.md). Fixed: EXP57's decision end (962503fb / 09a5b447 -- the
exploration identity of EXP58: EXP57 failed its fit gate, report e5a8565d, disclosed; chosen because it is the live
path's and EXP69's encoder, not because it meets any release line), EXP62's selector c92b7b7e, EXP64's head 0e275bdf, the
local views 12 / 0.25 / T. The only difference between the conditions: off = devision.reading.Reading without a
corrector; on = the same with EXP69-RUN-02's corrector 199b5367 (speed templates only). No EXP65 / 68 decision weights,
no K block, no hand-written magnet facts. candidate_eligible is always false.

  start     model-free: identities (EXP57 frozen + its exploration record, EXP62 / EXP64 / EXP69-RUN-02 runs and files and
            EXP69's registered reads, EXP58's run (the earlier reader, for a descriptive comparison), the dev_sqa true and
            fixed wrong-picture files, v0.3's records, EXP60's / EXP58's / EXP64's scoring sources, the picture manifest
            (EXP58's entries)); two reading bundles (off: c92 + head; on: + corrector) -> START.json.
  live      stage A (MPS): Reading.load(on bundle) -- the verified entry -- and the off Reading on the same decider /
            selector / head (the off bundle's files checked equal); per condition the 600 rl_tune pictures and EXP60's 24
            transcribed pictures, each request the row's questions with the actual picture as image=; reads, tables,
            states and answers registered per condition.
  fidelity  model-free: the live reads reproduce EXP69-RUN-02's registered reads (off = c92: rl_tune_c92 / dev24_c92; on =
            corr: rl_tune_corr / dev24_corr; decoded, valid, value) and every state is its reads' table serialised. A
            failure stops here (stage B1 does not start).
  transfer  stage B1 (MPS): both conditions on dev_sqa temp / force and their fixed wrong-picture files (the wrong
            picture is read itself), EXP58's request grouping; answers and reads registered per set and condition.
  report    model-free: exp21's 68:114 transfer per condition against v0.3, the on - off differences (accuracy and
            picture gain; temperature apart), the off reader against EXP58's earlier reader (descriptive); the branch.

    uv run python scripts/exp71/exp71.py start --deadline EPOCH | live | fidelity | transfer | report [--finalize CODES]
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
OUT, E69, E58R, E62, E64 = "runs/exp71", "runs/exp69-r2", "runs/exp58", "runs/exp62", "runs/exp64"
E69_RUN = {"START.json": "7920aa1e4a536ab57b45b1e3bfed53c3e3f620b644be04ae62e58ccfccf3a8ba",
           "report.json": "6ecd0a4aeb177ab38b5596fc86748704850d5ed854084b0fc89bf07907acd829",
           "registry.json": "9ba1909d6d4088f7c9d8b765670aea82381de664105bf91727c1941b001e5d50",
           "corrector.pt": "199b5367c23fa35c1dcc9217259fffafe4c6e75cf0b8a1b459b6382a6c165a10"}
E58_RUN = {"START.json": "505516bd7bbd9055a83212118ab1e22632e86ba2e60fb8d4b6e15cc3a090b53e",
           "report.json": "d3ada400479e9f27c4a566d819e2094565dec46900087d37b637d1add8dee267",
           "registry.json": "ef0327d06da589388939c780b13fc230c01076aa1cae54e2bc67812a79bcd11e",
           "pictures.json": "4e29b380faad88f5c90230dc0591b899ea76fd5fc50a3f4e8ce8c3b7d559100c"}
SELECTOR, HEAD = os.path.join(E62, "selector.pt"), os.path.join(E64, "head.pt")
CORRECTOR = os.path.join(E69, "corrector.pt")
CONDS = ("off", "on")
EXP69_READS = {"off": {"rl_tune": "rl_tune_c92", "dev24": "dev24_c92"}, "on": {"rl_tune": "rl_tune_corr", "dev24": "dev24_corr"}}
FILES = ["scripts/exp71/exp71.py", "scripts/exp71/run.sh", "src/devision/reading.py", "src/devision/infer.py", "src/devision/model.py",
         "src/devision/image.py", "src/devision/train/evaluate.py", "scripts/exp58/live.py", "scripts/exp69/exp69.py", "scripts/exp21/readout.py",
         "scripts/exp53/watch.py"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e69 = _load("exp69", "../exp69/exp69.py")
e58 = _load("exp58_live", "../exp58/live.py")
e64, e59, e54, e57 = e69.e64, e69.e59, e69.e54, e58.e57
sha, write_json, register, jsonl = e58.sha, e58.write_json, e58.register, e58.jsonl
TRANSFER = e58.TRANSFER
C92, HEAD_SHA, CORR_SHA = e69.C92, e69.HEAD_SHA, E69_RUN["corrector.pt"]


# ------------------------------------------------------------------ identities

def bound_files(run: str, want: Dict[str, str], label: str) -> Dict[str, str]:
    bad = [f for f, h in want.items() if sha(os.path.join(run, f)) != h]
    if bad:
        raise SystemExit("%s's %s changed" % (label, bad))
    reg = os.path.join(run, "registry.json")
    for files in json.load(open(reg)).values():
        for f, h in files.items():
            if sha(os.path.join(run, f)) != h:
                raise SystemExit("%s's registered %s changed" % (label, f))
    return dict(want)


def pictures(data_root: str = "data") -> Dict[str, str]:
    """Every picture any stage opens -- the 600 rl_tune pictures, EXP60's 24, the dev_sqa true and fixed wrong pictures --
    with EXP58's bound manifest entries (with data_root, the bytes are checked against them)."""
    m58 = json.load(open(os.path.join(E58R, "pictures.json")))
    need = {r["image"] for r in jsonl(e54.TUNE)} | {it["row"]["image"] for it in e59.items()} | {r["image"] for p in TRANSFER.values() for r in jsonl(p)}
    missing = sorted(need - set(m58))
    if missing:
        raise SystemExit("pictures outside EXP58's manifest: %s" % missing[:3])
    pics = {p: m58[p] for p in sorted(need)}
    if data_root:
        bad = [p for p, h in pics.items() if sha(os.path.join(data_root, p)) != h]
        if bad:
            raise SystemExit("%d pictures differ from EXP58's manifest, e.g. %s" % (len(bad), bad[0]))
    return pics


def identities() -> Dict:
    if sha(SELECTOR) != C92 or sha(HEAD) != HEAD_SHA or sha(CORRECTOR) != CORR_SHA:
        raise SystemExit("c92 / EXP64's head / EXP69's corrector changed")
    return {"mode": e58.mode_record(e58.EXPLORE), "model": dict(e58.NEW_SHA or {}), "exp62": e64.bound_e62(), "exp64": e69.bound_e64(),
            "exp69": bound_files(E69, E69_RUN, "EXP69-RUN-02"), "exp58": bound_files(E58R, E58_RUN, "EXP58"), "sources": e69.bound_sources(),
            "data": {n: sha(p) for n, p in TRANSFER.items()}, "tune": sha(e54.TUNE),
            "v03": {n: sha(os.path.join(e58.V03, n + ".details.jsonl")) for n in TRANSFER}}


def request_counts() -> Dict:
    return {"A": {"rl_tune": len(jsonl(e54.TUNE)), "dev24": len(e59.items())},
            "B1": {n: len(e58.requests(jsonl(p))) for n, p in TRANSFER.items()}}


def stage_start(deadline: int, out: str = OUT, data_root: str = "data") -> Dict:
    from devision.reading import write_bundle
    path = os.path.join(out, "START.json")
    if os.path.exists(path):
        raise SystemExit("%s exists (fresh only)" % path)
    ident = identities()
    pics = pictures(data_root)
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "pictures.json"), pics)
    src = {"selector": {"path": SELECTOR, "sha256": C92}, "head": {"path": HEAD, "sha256": HEAD_SHA}}
    b_off = write_bundle(os.path.join(out, "bundle_off"), os.path.join(e58.NEW_RUN, "stage2"), SELECTOR, HEAD, src)
    b_on = write_bundle(os.path.join(out, "bundle_on"), os.path.join(e58.NEW_RUN, "stage2"), SELECTOR, HEAD,
                        {**src, "corrector": {"path": CORRECTOR, "sha256": CORR_SHA}}, corrector_pt=CORRECTOR)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    s = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
         "files": {f: sha(f) for f in FILES}, **ident,
         "bundles": {"off": {"reading.json": sha(os.path.join(out, "bundle_off", "reading.json")), **b_off["sha256"]},
                     "on": {"reading.json": sha(os.path.join(out, "bundle_on", "reading.json")), **b_on["sha256"]}},
         "pictures": {"pictures.json": sha(os.path.join(out, "pictures.json")), "count": len(pics)},
         "requests": request_counts(), "temperature": "none (raw), in memory", "max_questions": e58.MAX_QUESTIONS,
         "candidate_eligible": False, "started": time.strftime("%F %T"), "deadline": deadline, "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
    write_json(path, s)
    return s


def check_start(out: str, data_root: str = "data", code: bool = True) -> Dict:
    st = json.load(open(os.path.join(out, "START.json")))
    if code:
        bad = [f for f, h in st["files"].items() if sha(f) != h]
        if bad:
            raise SystemExit("code differs from START: %s" % bad)
    now = identities()
    for k, v in now.items():
        if st[k] != v:
            raise SystemExit("%s differs from START" % k)
    if e58.fz.weights(e58.NEW_RUN) != st["model"]:
        raise SystemExit("EXP57's weights / config are not START's")
    for c in CONDS:
        b = os.path.join(out, "bundle_" + c)
        if sha(os.path.join(b, "reading.json")) != st["bundles"][c]["reading.json"]:
            raise SystemExit("bundle_%s changed" % c)
        for f in ("selector.pt", "head.pt", "corrector.pt"):                    # the copied component files themselves
            if f in st["bundles"][c] and sha(os.path.join(b, f)) != st["bundles"][c][f]:
                raise SystemExit("bundle_%s/%s changed" % (c, f))
        rj = json.load(open(os.path.join(b, "reading.json")))["sha256"]
        if any(rj.get(k) != v for k, v in st["bundles"][c].items() if k != "reading.json") or (("corrector.pt" in rj) != (c == "on")):
            raise SystemExit("bundle_%s does not bind START's files" % c)
    if st["bundles"]["off"]["selector.pt"] != C92 or st["bundles"]["on"]["corrector.pt"] != CORR_SHA \
            or any(st["bundles"]["off"][k] != st["bundles"]["on"][k] for k in ("selector.pt", "head.pt", "model.safetensors", "config.json", "tokenizer")):
        raise SystemExit("the two bundles differ in something else than the corrector")
    if sha(os.path.join(out, "pictures.json")) != st["pictures"]["pictures.json"] or pictures("") != json.load(open(os.path.join(out, "pictures.json"))):
        raise SystemExit("the picture manifest differs from START")
    if data_root:
        pictures(data_root)
    return st


def load_readings(out: str, data_root: str = "data", device: str = "auto"):
    """on: Reading.load(EXP57, bundle_on) (the verified entry); off: the same decider / selector / head without a
    corrector, after checking bundle_off binds exactly those files."""
    from devision.infer import best_device
    from devision.reading import Reading
    st = check_start(out, data_root)                                            # before any load
    on = Reading.load(os.path.join(e58.NEW_RUN, "stage2"), os.path.join(out, "bundle_on"), device=best_device() if device == "auto" else device)
    on.decider.cfg.temperature, on.decider.cfg.temperature_by_options = [1.0, 1.0, 1.0], {}
    off = Reading(on.decider, on.selector, on.head, json.load(open(os.path.join(out, "bundle_off", "reading.json"))))
    if off.corrector is not None or on.corrector is None:
        raise SystemExit("the conditions are not off / on")
    return {"off": off, "on": on}, st, json.load(open(os.path.join(out, "pictures.json")))


# ------------------------------------------------------------------ stage A / fidelity

def a_rows() -> Dict[str, Dict]:
    """Stage A's requests: the 600 rl_tune rows and EXP60's 24 transcribed rows (one question each), by id."""
    rows = {r["id"]: dict(r, a_set="rl_tune") for r in jsonl(e54.TUNE)}
    for it in e59.items():
        rows[it["row"]["id"]] = dict(it["row"], a_set="dev24", questions={it["qid"]: it["row"]["questions"][it["qid"]]})
    return rows


def stage_live(out: str = OUT, data_root: str = "data", readings=None) -> Dict:
    from devision.train.evaluate import _probabilities
    if readings is None:
        readings, st, pics = load_readings(out, data_root)
    else:
        st, pics = check_start(out, data_root), json.load(open(os.path.join(out, "pictures.json")))
    rows = a_rows()
    s_sha = sha(os.path.join(out, "START.json"))
    total, done, t0 = 2 * len(rows), 0, time.time()
    for c in CONDS:
        reads, recs = [], []
        for rid in sorted(rows):
            r = rows[rid]
            diag: Dict = {}
            res = e58.ask(readings[c], r["image"], r.get("state_text", ""), r["questions"], data_root, pics, diag)
            reads.append({"id": rid, "set": r["a_set"], "image": r["image"], "reads": diag["reads"], "table": diag["table"], "state": diag["state"]})
            for qid in r["questions"]:
                recs.append({"sample_id": rid, "qid": qid, "set": r["a_set"], "probabilities": _probabilities(res["answers"][qid])})
            done += 1
            e54.progress(out, "live", done, total)
        f = "A_%s.jsonl" % c
        ident = {"start_sha256": s_sha, "condition": c, "bundle": st["bundles"][c]["reading.json"], "model": st["model"], "temperature": "none (raw)"}
        with open(os.path.join(out, f), "w") as fh:
            fh.write(json.dumps({"identity": ident}) + "\n" + "".join(json.dumps(x) + "\n" for x in reads))
        fa = "A_%s_answers.jsonl" % c
        with open(os.path.join(out, fa), "w") as fh:
            fh.write(json.dumps({"identity": ident}) + "\n" + "".join(json.dumps(x) + "\n" for x in recs))
        register(out, "A_" + c, {f: sha(os.path.join(out, f)), fa: sha(os.path.join(out, fa))})
    return {"seconds": time.time() - t0}


def a_reads(out: str, c: str, st: Dict) -> Dict[str, Dict]:
    """A condition's stage A file after its checks (registered, identity, every request once)."""
    f = "A_%s.jsonl" % c
    if e54.check_registered(out, "A_" + c).get(f) != sha(os.path.join(out, f)):
        raise SystemExit("%s not registered" % f)
    lines = jsonl(os.path.join(out, f))
    want = {"start_sha256": sha(os.path.join(out, "START.json")), "condition": c, "bundle": st["bundles"][c]["reading.json"], "model": st["model"]}
    if any(lines[0]["identity"].get(k) != v for k, v in want.items()):
        raise SystemExit("%s: identity does not bind" % f)
    got = {x["id"]: x for x in lines[1:]}
    if len(got) != len(lines) - 1 or set(got) != set(a_rows()):
        raise SystemExit("%s: requests missing / repeated" % f)
    return got


def verify_fidelity(out: str, st: Dict) -> Dict:
    """Recomputed from the registered evidence every time: per condition, every live read equals EXP69-RUN-02's
    registered read (decoded, valid, value) for rl_tune (600 pictures) and dev24 (24), the table is the reads' values
    and the state is the row's text with that table serialised."""
    from devision.reading import TEMPLATES, serialise
    rows = a_rows()
    res: Dict[str, Any] = {}
    key = lambda x: (x["decoded"], x["valid"], x["value"])  # noqa: E731
    for c in CONDS:
        got = a_reads(out, c, st)
        same, n, states = 0, 0, 0
        for s_, name in EXP69_READS[c].items():
            f = os.path.join(E69, name + ".jsonl")
            if e54.check_registered(E69, "reads_" + name).get(name + ".jsonl") != sha(f):
                raise SystemExit("EXP69's %s is not registered" % name)
            for x in jsonl(f)[1:]:
                rid = x["id"]
                live = {r["question"]: r for r in got[rid]["reads"]}
                for r in x["reads"]:
                    n += 1
                    same += r["question"] in live and key(live[r["question"]]) == key(r)
        for rid, x in got.items():
            table = [[*TEMPLATES[r["question"]], r["value"]] for r in x["reads"]]
            states += x["table"] == table and x["state"] == serialise(rows[rid].get("state_text") or "", table)
        res[c] = {"reads": n, "same_reads": same, "requests": len(got), "consistent_states": states}
        res[c]["reproduced"] = bool(n == 1800 + 72 and same == n and states == len(rows))
    res["reproduced"] = all(res[c]["reproduced"] for c in CONDS)
    return res


def stage_fidelity(out: str = OUT) -> Dict:
    st = check_start(out, data_root="")
    res = verify_fidelity(out, st)
    write_json(os.path.join(out, "fidelity.json"), res)
    register(out, "fidelity", {"fidelity.json": sha(os.path.join(out, "fidelity.json"))})
    if not res["reproduced"]:
        raise SystemExit("stage A not reproduced: %s" % {c: res[c] for c in CONDS})
    return res


# ------------------------------------------------------------------ stage B1

def run_set(reading, samples: List[Dict], data_root: str, pics: Dict[str, str], on_request=None):
    """EXP58's run_set with each request's reads / table / state kept (the picture used is the row's own -- for a
    *_mism file its fixed wrong picture, read itself)."""
    from devision.reading import domain_of
    from devision.train.evaluate import _probabilities, score_records
    asked, reads = {}, []
    for n, (image, text, items) in enumerate(e58.requests(samples)):
        questions = {"%d:%s" % (i, qid): q for i, qid, q in items}
        diag: Dict = {}
        res = e58.ask(reading, image, text, questions, data_root, pics, diag)
        reads.append({"image": image, "questions": [[samples[i]["id"], qid] for i, qid, _ in items], "reads": diag.get("reads"),
                      "table": diag.get("table"), "state": diag.get("state")})
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
                         "kind": s.get("kind", ""), "axis": s.get("axis", ""), "type": q["type"], "options": list(q["criteria"]),
                         "gold": gold, "gold_answer": max(gold, key=gold.__getitem__), "probabilities": probs,
                         "prediction": max(probs, key=probs.__getitem__), "control": "none", "questions_in_request": size,
                         "routed": domain_of(q.get("instructions", "")) is not None})
    return score_records(recs), reads


def stage_transfer(out: str = OUT, data_root: str = "data", readings=None) -> Dict:
    st0 = check_start(out, data_root="")
    if not verify_fidelity(out, st0)["reproduced"]:
        raise SystemExit("stage A not reproduced: stage B1 does not start")
    if readings is None:
        readings, st, pics = load_readings(out, data_root)
    else:
        st, pics = st0, json.load(open(os.path.join(out, "pictures.json")))
    s_sha = sha(os.path.join(out, "START.json"))
    total = 2 * sum(st["requests"]["B1"].values())
    done, t0 = 0, time.time()
    for c in CONDS:
        for name, path in TRANSFER.items():
            recs, reads = run_set(readings[c], jsonl(path), data_root, pics, on_request=lambda k, b=done: e54.progress(out, "transfer", b + k, total))
            done += st["requests"]["B1"][name]
            set_name = "%s_%s" % (name, c)
            ident = {"set": set_name, "start_sha256": s_sha, "data_sha256": st["data"][name], "model": st["model"], "bundle": st["bundles"][c]["reading.json"],
                     "pictures": st["pictures"]["pictures.json"], "temperature": "none (raw)", "path": "devision.reading.Reading.decide(image=actual picture)"}
            e57.write_set(out, set_name, recs, ident)
            f = "reads_%s.jsonl" % set_name
            with open(os.path.join(out, f), "w") as fh:
                fh.write("".join(json.dumps(x) + "\n" for x in reads))
            register(out, "reads_" + set_name, {f: sha(os.path.join(out, f))})
    return {"seconds": time.time() - t0}


# ------------------------------------------------------------------ report

def b_records(out: str, name: str, c: str, st: Dict) -> List[Dict]:
    """A B1 set after every check: registered, identity bound, every question of the source once; each record's picture
    is the source row's own (the wrong picture for *_mism), and each request's reads come from that picture."""
    set_name = "%s_%s" % (name, c)
    f = "details_%s.jsonl" % set_name
    if e54.check_registered(out, "decide_" + set_name).get(f) != sha(os.path.join(out, f)):
        raise SystemExit("%s not registered" % f)
    lines = jsonl(os.path.join(out, f))
    want = {"set": set_name, "start_sha256": sha(os.path.join(out, "START.json")), "data_sha256": st["data"][name], "model": st["model"],
            "bundle": st["bundles"][c]["reading.json"], "pictures": st["pictures"]["pictures.json"], "temperature": "none (raw)"}
    if any(lines[0]["identity"].get(k) != v for k, v in want.items()):
        raise SystemExit("%s: identity does not bind" % f)
    src = jsonl(TRANSFER[name])
    got = e57.r41.check_details("exp71/%s" % set_name, src, lines[1:])
    image_of = {r["id"]: r["image"] for r in src}
    if any(r["image_used"] != image_of[r["sample_id"]] for r in lines[1:]):
        raise SystemExit("%s: a question was answered on another picture than its row's" % f)
    rf = "reads_%s.jsonl" % set_name
    if e54.check_registered(out, "reads_" + set_name).get(rf) != sha(os.path.join(out, rf)):
        raise SystemExit("%s not registered" % rf)
    for x in jsonl(os.path.join(out, rf)):
        if any(image_of[sid] != x["image"] for sid, _ in x["questions"]) or not x["reads"]:
            raise SystemExit("%s: reads not of the row's own picture" % rf)
    return list(got.values())


def exp58_records(name: str) -> List[Dict]:
    """EXP58's earlier reader (EXP53 selector + EXP51 head, same EXP57 end) on the same set, after EXP58's checks."""
    st58 = json.load(open(os.path.join(E58R, "START.json")))
    return e58.set_records(E58R, name, st58)


def branch(res: Dict) -> str:
    if not (res.get("fidelity") or {}).get("reproduced"):
        return "stage A not reproduced (or missing): the live corrector path is not established; stage B1 not judged"
    if any(res["transfer"].get(c) is None for c in CONDS) or res.get("on_minus_off") is None:
        return "incomplete: stage B1 missing for a condition"
    d = res["on_minus_off"]
    gain = d["accuracy_change"]["ci95"][0] > 0
    temp = d["temp"]["accuracy_change"]["ci95"][0] > 0
    if res["target"]["on"]:
        return ("with the corrector the v0.3 transfer target (68:114) is met on dev_sqa%s: a development result -- protection B2, "
                "independent confirmation, calibration and install checks remain; not a candidate" % (" and on - off is positive" if gain else ""))
    if gain or temp:
        return "the corrector gives a measurable decision gain on dev_sqa (%s), but the v0.3 transfer target is not met" % ("overall" if gain else "temperature only")
    return "no measurable decision gain from the corrector on dev_sqa; the v0.3 transfer target is not met"


def report(out: str = OUT, finalize: str = "") -> Dict:
    res: Dict[str, Any] = {"states": {}, "transfer": {}, "candidate_eligible": False}
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
    res["mode"] = st["mode"]
    try:
        res["fidelity"] = verify_fidelity(out, st)
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["fidelity"] = None
        res["states"]["fidelity"] = "missing / invalid: %s" % e
    sys.path.insert(0, os.path.join(HERE, "..", "exp21"))
    from readout import transfer  # type: ignore
    recs: Dict[str, Dict[str, List[Dict]]] = {c: {} for c in CONDS}
    for c in CONDS:
        for name in TRANSFER:
            try:
                recs[c][name] = b_records(out, name, c, st)
                res["states"]["%s_%s" % (name, c)] = "completed"
            except (SystemExit, OSError, ValueError, KeyError) as e:
                res["states"]["%s_%s" % (name, c)] = "missing / invalid: %s" % e
    res["target"] = {}
    for c in CONDS:
        if all(n in recs[c] for n in TRANSFER):
            t = transfer({k[4:]: e58.v03_records(k, st) for k in TRANSFER}, {k[4:]: recs[c][k] for k in TRANSFER})
            res["transfer"][c] = t
            res["target"][c] = bool(t["accuracy_change"]["value"] >= 0.10 and t["accuracy_change"]["ci95"][0] > 0 and t["gain_change"]["ci95"][0] > 0)
            res.setdefault("accuracy", {})[c] = {k: sum(r["correct"] for r in recs[c][k]) / len(recs[c][k]) for k in TRANSFER}
        else:
            res["transfer"][c] = None
    if all(res["transfer"].get(c) for c in CONDS):
        res["on_minus_off"] = transfer({k[4:]: recs["off"][k] for k in TRANSFER}, {k[4:]: recs["on"][k] for k in TRANSFER})
        try:
            res["off_minus_exp58_reader"] = transfer({k[4:]: exp58_records(k) for k in TRANSFER}, {k[4:]: recs["off"][k] for k in TRANSFER})
        except (SystemExit, OSError, ValueError, KeyError) as e:
            res["off_minus_exp58_reader"] = "invalid: %s" % e
    res["branch"] = branch(res)
    res["limits"] = ("magnets read the distance only (no size / pole facts); EXP57 failed its fit gate (exploration identity); "
                     "dev_sqa is a repeatedly used development set; not a candidate")
    return write_report(out, res)


def write_report(out: str, res: Dict) -> Dict:
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "report.json"), res)
    keys = ("finalized", "states", "fidelity", "target", "accuracy", "branch", "candidate_eligible")
    text = json.dumps({k: res[k] for k in keys if k in res}, indent=1, default=str)
    d = res.get("on_minus_off")
    if isinstance(d, dict):
        text += "\non - off: accuracy %+.3f %s, gain %+.3f %s; temperature accuracy %+.3f %s" % (
            d["accuracy_change"]["value"], d["accuracy_change"]["ci95"], d["gain_change"]["value"], d["gain_change"]["ci95"],
            d["temp"]["accuracy_change"]["value"], d["temp"]["accuracy_change"]["ci95"])
    open(os.path.join(out, "report.txt"), "w").write(text + "\n")
    print(text)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["counts", "start", "live", "fidelity", "transfer", "report"])
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--finalize", default="")
    a = ap.parse_args(argv)
    if a.stage == "counts":
        print(json.dumps(request_counts(), indent=1))
    elif a.stage == "start":
        print(json.dumps({k: v for k, v in stage_start(a.deadline).items() if k not in ("files",)}, indent=1, default=str))
    elif a.stage == "live":
        print(json.dumps(stage_live(), indent=1))
    elif a.stage == "fidelity":
        print(json.dumps(stage_fidelity(), indent=1))
    elif a.stage == "transfer":
        print(json.dumps(stage_transfer(), indent=1))
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
