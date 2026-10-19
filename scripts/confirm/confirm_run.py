"""The synthetic confirmation run (critic V04-K-CONFIRM-PREP-02; prep03 5.3; scripts/release/confirm.py). Runs only after
the final combination is frozen and this runner and a model permission are granted; nothing here is a permission.

  start      model-free: the final combination's freeze verified (confirm.freeze_problems: complete IDENTITY of this
             checkpoint, frozen code = this implementation, K, LIVE-K lineage), v0.3's baseline (runs/r15-vision/stage2,
             pinned files), this code -> START.json; one start only.
  generate   CPU: confirm.generate -- the set once (120 pictures, 480 questions, fixed wrong pictures), registered with
             every picture's hash and the exclusion record.
  cand_true / cand_mism / cand_rev   MPS: the frozen combination through devision.load (reading on, its own frozen
             temperatures) -- true pictures, the fixed wrong pictures of confirm_mism.jsonl (never re-drawn), reversed options.
  v03_true / v03_mism                 MPS: v0.3 (plain Decider, its own temperatures) on the true and the same wrong pictures.
  report     model-free: every set registered, bound and complete (each question of its rows once, on its row's picture);
             the primary outcome (confirm.outcome on the formal 120 ids); readings per domain / class, the reversed control
             and v0.3's wrong pictures reported only.

600 requests, 2,400 questions (120 pictures / 480 questions per condition), serial. Hard limit 20 minutes, the last 120 s
for the report (scripts/confirm/run.sh). candidate_eligible is not decided here.

    CONFIRM_IDENTITY=<eval>/IDENTITY.json CONFIRM_CHECKPOINT=<frozen candidate> bash scripts/confirm/run.sh
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT, DATA = "runs/v04-k-confirm", "data/v04_k_confirm"
V03 = "runs/r15-vision/stage2"
V03_SHA = {"model.safetensors": "110fea95889947c373258c88a396ad3eec574971815ee437b9c7cc3df8a5d2d5",
           "config.json": "3a10bacd56e7f48aa07ad981821a592fff10c5eb317c39a9591ad425c175c324"}
CONDITIONS = {"cand_true": ("candidate", "confirm.jsonl", "none"), "cand_mism": ("candidate", "confirm_mism.jsonl", "none"),
              "cand_rev": ("candidate", "confirm.jsonl", "reversed"), "v03_true": ("v0.3", "confirm.jsonl", "none"),
              "v03_mism": ("v0.3", "confirm_mism.jsonl", "none")}
PER_CONDITION = {"requests": 120, "questions": 480}
FILES = ["scripts/confirm/confirm_run.py", "scripts/confirm/run.sh", "scripts/release/confirm.py", "scripts/release/rl_calib.py",
         "scripts/release/check_eval.py", "scripts/release/rebind.py", "scripts/release/route_calib.py", "scripts/data/real_layout.py",
         "src/devision/__init__.py", "src/devision/reading.py", "src/devision/infer.py", "src/devision/model.py", "src/devision/image.py",
         "src/devision/train/evaluate.py", "scripts/exp53/watch.py"]
PINS: Optional[Dict[str, str]] = None           # None: the release's LIVE-K pins; fixtures set theirs


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cf = _load("confirm_tool", "scripts/release/confirm.py")


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: str, obj: Any) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)


def jsonl(path: str) -> List[Dict]:
    return [json.loads(x) for x in open(path) if x.strip()]


def register(out: str, stage: str, files: Dict[str, str]) -> None:
    path = os.path.join(out, "registry.json")
    reg = json.load(open(path)) if os.path.exists(path) else {}
    if stage in reg:
        raise SystemExit("stage %s already registered (one run only)" % stage)
    reg[stage] = files
    write_json(path, reg)


def registered(out: str, stage: str) -> Dict[str, str]:
    path = os.path.join(out, "registry.json")
    return (json.load(open(path)) if os.path.exists(path) else {}).get(stage, {})


def progress(out: str, stage: str, done: int, total: int) -> None:
    with open(os.path.join(out, "progress-%s.txt" % stage), "a") as fh:
        fh.write("%.3f %d %d\n" % (time.time(), done, total))


def begin(out: str, stage: str, outputs: List[str]) -> None:
    """The stage's one start, taken atomically BEFORE anything is loaded, computed or written (critic CONF-RUN-R1-02): a
    stage already started, registered or with an output present is refused; a start without a registration is reported
    as incomplete."""
    if registered(out, stage) or any(os.path.exists(p) for p in outputs):
        raise SystemExit("stage %s already ran (one run only)" % stage)
    try:
        fd = os.open(os.path.join(out, "stage-%s.started" % stage), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit("stage %s already started (one run only)" % stage) from None
    os.write(fd, time.strftime("%F %T").encode())
    os.close(fd)


def started(out: str, stage: str) -> bool:
    return os.path.exists(os.path.join(out, "stage-%s.started" % stage))


def weights(run_dir: str) -> Dict[str, str]:
    return {f: sha(os.path.join(run_dir, f)) for f in ("model.safetensors", "config.json")}


# ------------------------------------------------------------------ start / generate

def stage_start(identity: str, checkpoint: str, deadline: int, out: str = OUT) -> Dict:
    if os.path.exists(out) or os.path.exists(DATA):
        raise SystemExit("%s or %s exists (fresh only)" % (out, DATA))
    bad = cf.freeze_problems(identity, checkpoint, PINS)
    if bad:
        raise SystemExit("the final combination is not frozen -- no confirmation:\n  " + "\n  ".join(bad))
    if weights(V03) != V03_SHA:
        raise SystemExit("v0.3's baseline files are not the pinned ones")
    os.makedirs(out)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()  # noqa: E731
    st = {"git_head": git("rev-parse", "HEAD"), "dirty_files": [f for f in FILES if git("status", "--porcelain", "--", f)],
          "files": {f: sha(os.path.join(ROOT, f)) for f in FILES}, "identity": {"path": os.path.abspath(identity), "sha256": sha(identity)},
          "checkpoint": os.path.abspath(checkpoint), "candidate_weights": weights(checkpoint), "v03": {"path": V03, **V03_SHA},
          "conditions": {k: list(v) for k, v in CONDITIONS.items()}, "per_condition": PER_CONDITION,
          "started": time.strftime("%F %T"), "deadline": deadline, "deadline_text": time.strftime("%F %T", time.localtime(deadline))}
    write_json(os.path.join(out, "START.json"), st)
    register(out, "start", {"START.json": sha(os.path.join(out, "START.json"))})
    return st


def check_start(out: str = OUT, code: bool = True) -> Dict:
    st = json.load(open(os.path.join(out, "START.json")))
    if registered(out, "start").get("START.json") != sha(os.path.join(out, "START.json")):
        raise SystemExit("START.json is not the registered one")
    if code:
        bad = [f for f, h in st["files"].items() if sha(os.path.join(ROOT, f)) != h]
        if bad:
            raise SystemExit("code differs from START: %s" % bad)
    if sha(st["identity"]["path"]) != st["identity"]["sha256"]:
        raise SystemExit("the frozen identity changed")
    bad = cf.freeze_problems(st["identity"]["path"], st["checkpoint"], PINS)
    if bad:
        raise SystemExit("the final combination no longer verifies: %s" % bad)
    if weights(V03) != V03_SHA:
        raise SystemExit("v0.3's baseline changed")
    return st


def generate_set(st: Dict, used) -> Dict:
    return cf.generate(DATA, st["identity"]["path"], st["checkpoint"], pins=PINS, used=used)


def excluded_snapshot():
    return cf.excluded("data")


def stage_generate(out: str = OUT) -> Dict:
    """The set once, against an exclusion snapshot taken and written first (critic CONF-RUN-R1-04): the exact canonical
    cases selection excluded, registered by hash, so a later check never re-scans a data directory that may have changed."""
    st = check_start(out)
    snap = os.path.join(out, "excluded_cases.json")
    begin(out, "generate", [snap, DATA])
    used = excluded_snapshot()
    with open(snap, "w") as fh:
        json.dump(sorted(repr(k) for k in used), fh)
    man = generate_set(st, used)
    rows = jsonl(os.path.join(DATA, "confirm.jsonl"))
    if len(rows) != cf.PICTURES or sum(len(r["questions"]) for r in rows) != PER_CONDITION["questions"]:
        raise SystemExit("the generated set is not 120 pictures / 480 questions")
    if {r["case"] for r in rows} & set(json.load(open(snap))):
        raise SystemExit("a generated case is in the exclusion snapshot")
    pics = sorted({r["image"] for r in rows})
    register(out, "generate", {"MANIFEST.json": sha(os.path.join(DATA, "MANIFEST.json")), "confirm.jsonl": sha(os.path.join(DATA, "confirm.jsonl")),
                               "confirm_mism.jsonl": sha(os.path.join(DATA, "confirm_mism.jsonl")), "excluded_cases.json": sha(snap),
                               "pictures": hashlib.sha256(json.dumps({p: sha(os.path.join("data", p)) for p in pics}, sort_keys=True).encode()).hexdigest()})
    return {k: man[k] for k in ("pictures", "classes", "questions")}


def pictures_digest(rows: List[Dict]) -> str:
    return hashlib.sha256(json.dumps({p: sha(os.path.join("data", p)) for p in sorted({r["image"] for r in rows})}, sort_keys=True).encode()).hexdigest()


def check_set(out: str) -> None:
    reg = registered(out, "generate")
    if not reg:
        raise SystemExit("the set is not generated")
    for f in ("MANIFEST.json", "confirm.jsonl", "confirm_mism.jsonl"):
        if sha(os.path.join(DATA, f)) != reg[f]:
            raise SystemExit("%s changed since it was generated" % f)
    if pictures_digest(jsonl(os.path.join(DATA, "confirm.jsonl"))) != reg["pictures"]:
        raise SystemExit("a confirmation picture changed")
    snap = os.path.join(out, "excluded_cases.json")
    if not os.path.exists(snap) or sha(snap) != reg.get("excluded_cases.json"):
        raise SystemExit("the exclusion snapshot is not the registered one")
    if {r["case"] for r in jsonl(os.path.join(DATA, "confirm.jsonl"))} & set(json.load(open(snap))):
        raise SystemExit("a confirmation case is in the exclusion snapshot")


# ------------------------------------------------------------------ model stages

def load_model(who: str, st: Dict, device: str):
    import devision
    if who == "candidate":
        if weights(st["checkpoint"]) != st["candidate_weights"]:
            raise SystemExit("the candidate's weights changed")
        m = devision.load(st["checkpoint"], device=device, reading="on")
        if m.reading_status.get("mode") != "on" or not m.reading_status.get("k_rule"):
            raise SystemExit("the candidate did not load as the K reading combination")
        return m
    if weights(V03) != V03_SHA:
        raise SystemExit("v0.3 changed")
    return devision.load(V03, device=device, reading="off")


def temperature_of(cfg) -> Dict:
    """The temperatures a model applies, in devision-eval's / check_eval's table form."""
    out = {"choice": cfg.temperature[0], "noul": cfg.temperature[2], "by_options": dict(cfg.temperature_by_options)}
    if getattr(cfg, "temperature_by_route", None):
        out["by_route"] = dict(cfg.temperature_by_route)
    return out


def stage_model(name: str, out: str = OUT, device: str = "auto", model=None) -> Dict:
    """One condition through the formal evaluation path with the model's own (frozen) temperatures."""
    from devision.train.evaluate import evaluate_records, runtime_code
    st = check_start(out)
    check_set(out)
    who, file, control = CONDITIONS[name]
    f = os.path.join(out, "details_%s.jsonl" % name)
    begin(out, name, [f])                                        # before any load or forward
    rows = jsonl(os.path.join(DATA, file))
    model = model or load_model(who, st, device)
    n, real = [0], model.decide

    def counted(*a, **k):
        res = real(*a, **k)
        n[0] += 1
        progress(out, name, n[0], PER_CONDITION["requests"])
        return res
    model.decide = counted                                       # progress for the watch's stall rule
    recs = evaluate_records(model, rows, "data", control=control)
    model.decide = real
    ident = {"set": name, "start_sha256": sha(os.path.join(out, "START.json")), "rows": registered(out, "generate")[file],
             "model": who, "weights": st["candidate_weights"] if who == "candidate" else V03_SHA, "control": control,
             "reading": getattr(model, "reading_status", {"mode": "none"}), "runtime_code": runtime_code(),
             "temperature": temperature_of(model.cfg)}
    with open(f, "w") as fh:
        fh.write(json.dumps({"identity": ident}) + "\n" + "".join(json.dumps(r) + "\n" for r in recs))
    register(out, name, {"details_%s.jsonl" % name: sha(f)})
    return {"records": len(recs)}


# ------------------------------------------------------------------ report

def expected_temperature(who: str, st: Dict) -> Dict:
    ce = _load("confirm_run_ce", "scripts/release/check_eval.py")
    return ce.temperature_table(st["checkpoint"] if who == "candidate" else V03)


_PIXELS: Dict[str, str] = {}


def pixels(image: str) -> str:
    """The decoded pixels' sha256 of a picture under data/ (what evaluate_records records as picture_pixels_sha256)."""
    if image not in _PIXELS:
        from PIL import Image

        from devision.infer import picture_identity
        ident = picture_identity(Image.open(os.path.join("data", image)).convert("RGB"))
        assert ident is not None
        _PIXELS[image] = ident["pixels_sha256"]
    return _PIXELS[image]


def records(out: str, name: str, st: Dict) -> List[Dict]:
    """A condition's records after every check (critic CONF-RUN-R1-01): registered and bound to START / the set / the
    model weights / control / the model's own frozen temperatures / this implementation's runtime code; the candidate
    loaded as the frozen K reading combination (mode on, K, the frozen reading.json), v0.3 as the plain model; each
    question of its rows exactly once, on its row's picture with those pixels; the source's options (reversed for the
    reversed control) and gold; legal probabilities, prediction = arg-max, correct recomputed; the route each question
    had to take, from its own text (candidate: read for the reading domains, ordinary otherwise; v0.3: ordinary). A read
    that failed (unreadable) is the model's failure and stays in the denominator."""
    import math

    from devision.reading import domain_of
    f = os.path.join(out, "details_%s.jsonl" % name)
    if not registered(out, name):
        raise SystemExit("started, not completed" if started(out, name) else "missing")
    if registered(out, name).get("details_%s.jsonl" % name) != sha(f):
        raise SystemExit("details_%s.jsonl changed" % name)
    lines = jsonl(f)
    who, file, control = CONDITIONS[name]
    ident = lines[0]["identity"]
    want = {"set": name, "start_sha256": sha(os.path.join(out, "START.json")), "rows": registered(out, "generate")[file], "control": control,
            "weights": st["candidate_weights"] if who == "candidate" else V03_SHA, "temperature": expected_temperature(who, st)}
    if any(ident.get(k) != v for k, v in want.items()):
        raise SystemExit("%s: identity does not bind (%s)" % (name, [k for k, v in want.items() if ident.get(k) != v]))
    rd = ident.get("reading") or {}
    if who == "candidate":
        frozen = json.load(open(st["identity"]["path"]))["reading"]["reading_json_sha256"]
        if rd.get("mode") != "on" or not rd.get("k_rule") or rd.get("reading_json_sha256") != frozen:
            raise SystemExit("%s: not the frozen K reading combination (%s)" % (name, rd))
    elif rd.get("mode") not in ("none", "off"):
        raise SystemExit("%s: v0.3 did not run as the plain model" % name)
    rc = _load("confirm_run_rc", "scripts/release/route_calib.py")
    bad = rc.runtime_problems(ident.get("runtime_code") or {}, ROOT)
    if bad:
        raise SystemExit("%s: not run by this implementation: %s" % (name, bad[:2]))
    rows = {r["id"]: r for r in jsonl(os.path.join(DATA, file))}
    expect = {(rid, q) for rid, r in rows.items() for q in r["questions"]}
    got = lines[1:]
    keys = [(x["sample_id"], x["qid"]) for x in got]
    if len(keys) != len(set(keys)) or set(keys) != expect:
        raise SystemExit("%s: not every question once (%d records, %d expected)" % (name, len(keys), len(expect)))
    for x in got:
        r = rows[x["sample_id"]]
        q = r["questions"][x["qid"]]
        opts = list(q["criteria"])
        gold = r["gold"][x["qid"]]["probabilities"]
        p = x.get("probabilities") or {}
        dom = domain_of(q.get("instructions", ""))
        problems = []
        if x["image_used"] != r["image"] or x.get("picture_pixels_sha256") != pixels(r["image"]):
            problems.append("picture")
        if x.get("options") != (opts[::-1] if control == "reversed" else opts) or x.get("gold") != gold \
                or x.get("gold_answer") != max(gold, key=lambda o: gold[o]):
            problems.append("options / gold")
        if set(p) != set(opts) or not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in p.values()) \
                or abs(sum(p.values()) - 1) > 1e-4:
            problems.append("probabilities")
        elif x.get("prediction") != max(p, key=lambda o: p[o]) or type(x.get("correct")) is not bool or x["correct"] != (x["prediction"] == x["gold_answer"]):
            problems.append("prediction / correct")
        want_route = "read" if (who == "candidate" and dom is not None) else "ordinary"
        if x.get("reading_domain") != dom or x.get("route") != want_route:
            problems.append("route %r / domain %r, expected %r / %r" % (x.get("route"), x.get("reading_domain"), want_route, dom))
        if problems:
            raise SystemExit("%s: %s/%s: %s" % (name, x["sample_id"], x["qid"], problems))
    return got


def picture_interval(per_picture: Dict[str, float]) -> Dict:
    """Mean over pictures with the 95 % percentile interval of 2,000 picture resamples (seed 0)."""
    import random
    ids = sorted(per_picture)
    if not ids:
        return {"pictures": 0, "value": None, "ci95": None}
    rng = random.Random(0)
    means = sorted(sum(per_picture[ids[rng.randrange(len(ids))]] for _ in ids) / len(ids) for _ in range(2000))
    return {"pictures": len(ids), "value": sum(per_picture.values()) / len(ids), "ci95": [means[50], means[1949]]}


def clustered(recs: List[Dict], rows: Dict[str, Dict], reading: bool) -> Dict:
    """Reading questions (or the comparison) per domain: each picture's share right, then intervals over pictures --
    questions of one picture are resampled together, never as independent questions (critic CONF-RUN-R1-03). Report only."""
    per: Dict[str, Dict[str, List[bool]]] = defaultdict(lambda: defaultdict(list))
    for x in recs:
        if (x["qid"] != "q0") == reading:
            r = rows[base_id(x["sample_id"])]
            per[r["kind"]][base_id(x["sample_id"])].append(bool(x["correct"]))
    return {dom: picture_interval({pid: sum(v) / len(v) for pid, v in pics.items()}) for dom, pics in sorted(per.items())}


def flips(true: List[Dict], rev: List[Dict], rows: Dict[str, Dict]) -> Dict:
    """The reversed-options control: per question, whether the chosen answer's content changed (the same option text,
    whatever its position), paired by question; per domain, the share of flips over pictures with a picture interval.
    Report only."""
    a = {(x["sample_id"], x["qid"]): x["prediction"] for x in true}
    per: Dict[str, Dict[str, List[bool]]] = defaultdict(lambda: defaultdict(list))
    for x in rev:
        k = (x["sample_id"], x["qid"])
        per[rows[x["sample_id"]]["kind"]][x["sample_id"]].append(a[k] != x["prediction"])
    return {dom: picture_interval({pid: sum(v) / len(v) for pid, v in pics.items()}) for dom, pics in sorted(per.items())}


def base_id(sample_id: str) -> str:
    return sample_id[:-len(":mism")] if sample_id.endswith(":mism") else sample_id


def report(out: str = OUT, finalize: str = "") -> Dict:
    res: Dict[str, Any] = {"states": {}}
    if finalize:
        res["finalized"] = {"exit_codes": finalize, "note": "missing items are not low scores"}
    if not os.path.exists(os.path.join(out, "START.json")):
        res["branch"] = "incomplete: no START (nothing ran)"
        return write_report(out, res)
    try:
        st = check_start(out, code=False)
    except (SystemExit, OSError, ValueError, KeyError) as e:
        res["branch"] = "invalid: the run's identity does not verify (%s) -- no verdict" % e
        return write_report(out, res)
    S = res["states"]
    try:
        check_set(out)
        S["generate"] = "completed"
    except (SystemExit, OSError, ValueError, KeyError) as e:
        S["generate"] = "missing" if str(e) == "the set is not generated" else "invalid: %s" % e
    recs: Dict[str, List[Dict]] = {}
    for name in CONDITIONS:
        if S["generate"] != "completed":
            S[name] = "missing"
            continue
        try:
            recs[name] = records(out, name, st)
            S[name] = "completed"
        except (SystemExit, OSError, ValueError, KeyError) as e:
            S[name] = str(e) if str(e) in ("missing", "started, not completed") else "invalid: %s" % e
    bad = [k for k, v in S.items() if v != "completed"]
    if bad:
        kind = "invalid" if any(v.startswith("invalid") for v in S.values()) else "incomplete"
        res["branch"] = "%s: %s -- no confirmation verdict" % (kind, bad)
        return write_report(out, res)
    ids = cf.picture_ids(os.path.join(DATA, "confirm.jsonl"))
    cmp_ = {name: {base_id(x["sample_id"]): x["correct"] for x in recs[name] if x["qid"] == "q0"} for name in recs}
    res["primary"] = cf.outcome(cmp_["cand_true"], cmp_["cand_mism"], cmp_["v03_true"], ids)
    rows = {r["id"]: r for r in jsonl(os.path.join(DATA, "confirm.jsonl"))}
    res["readings_by_picture"] = {name: clustered(recs[name], rows, reading=True) for name in recs}
    res["reversed_flips"] = flips(recs["cand_true"], recs["cand_rev"], rows)
    report_only: Dict[str, Any] = {}
    for name, rs in recs.items():
        per = defaultdict(list)
        for x in rs:
            r = rows[base_id(x["sample_id"])]
            per[("comparison" if x["qid"] == "q0" else "reading", r["kind"], r["axis"])].append(x["correct"])
        report_only[name] = {"%s|%s|%s" % k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in sorted(per.items())}
    res["report_only"] = report_only
    res["branch"] = res["primary"]["branch"] + " (synthetic confirmation; not a confirmation on unseen real pictures)"
    return write_report(out, res)


def write_report(out: str, res: Dict) -> Dict:
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "report.json"), res)
    text = json.dumps({k: res[k] for k in ("finalized", "states", "branch") if k in res}, indent=1, default=str)
    open(os.path.join(out, "report.txt"), "w").write(text + "\n")
    print(text)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["start", "generate"] + list(CONDITIONS) + ["report"])
    ap.add_argument("--identity", default=os.environ.get("CONFIRM_IDENTITY", ""))
    ap.add_argument("--checkpoint", default=os.environ.get("CONFIRM_CHECKPOINT", ""))
    ap.add_argument("--deadline", type=int, default=0)
    ap.add_argument("--finalize", default="")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    if a.stage == "start":
        if not a.identity or not a.checkpoint:
            raise SystemExit("start needs --identity and --checkpoint (the frozen combination)")
        print(json.dumps({k: v for k, v in stage_start(a.identity, a.checkpoint, a.deadline).items() if k != "files"}, indent=1))
    elif a.stage == "generate":
        print(json.dumps(stage_generate(), indent=1))
    elif a.stage == "report":
        try:
            report(finalize=a.finalize)
        except Exception as e:
            os.makedirs(OUT, exist_ok=True)
            open(os.path.join(OUT, "report.txt"), "w").write("REPORT FAILED (invalid, no verdict): %s\n" % e)
            print("REPORT FAILED: %s" % e, file=sys.stderr)
            return 1
    else:
        print(json.dumps(stage_model(a.stage), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
