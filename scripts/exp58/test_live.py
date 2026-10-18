"""exp58 runner with fake objects, no model: every request reaches Reading.decide with the actual picture as image= (never
None, never inside state); a *_mism.jsonl row sends its fixed wrong picture; inside the real Reading.decide a supported
question is decided on the text state + table (image=None after the read) and an unsupported one goes through ordinary
decide WITH the picture; the placeholder endpoint refuses start before anything is written."""
import importlib.util
import json
import os

import pytest
from PIL import Image

HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("exp58_live", os.path.join(HERE, "live.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

TEMP_Q = "Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature?"


def pictures(root):
    (root / "img").mkdir(parents=True)
    Image.new("RGB", (40, 30), (255, 0, 0)).save(root / "img" / "red.png")
    Image.new("RGB", (40, 30), (0, 0, 255)).save(root / "img" / "blue.png")
    return {p: m.sha(str(root / p)) for p in ("img/red.png", "img/blue.png")}


def row(i, image, text):
    return {"id": "s%d" % i, "image_id": "img:%d" % i, "image": image, "state_text": "hint",
            "questions": {"q": {"type": "choice", "instructions": text, "criteria": {"sample A": None, "sample B": None, "neither": None}}},
            "gold": {"q": {"probabilities": {"sample A": 1.0, "sample B": 0.0, "neither": 0.0}}}}


class FakeReading:
    def __init__(self):
        self.calls = []

    def decide(self, state, questions, image=None, diagnostics=None):
        self.calls.append((state, image.getpixel((0, 0)) if image is not None else None))
        return {"answers": {k: {"type": "choice", "probabilities": {"sample A": 0.6, "sample B": 0.3, "neither": 0.1}} for k in questions}}


def test_actual_and_fixed_wrong_pictures_reach_reading_as_image(tmp_path):
    pics = pictures(tmp_path)
    true = [row(0, "img/red.png", TEMP_Q)]
    mism = [row(0, "img/blue.png", TEMP_Q)]                  # *_mism.jsonl: same question, the fixed wrong picture
    fr = FakeReading()
    a = m.run_set(fr, true, str(tmp_path), pics)
    b = m.run_set(fr, mism, str(tmp_path), pics)
    assert fr.calls == [("hint", (255, 0, 0)), ("hint", (0, 0, 255))]
    assert a[0]["routed"] and a[0]["image_used"] == "img/red.png" and b[0]["image_used"] == "img/blue.png" and a[0]["correct"]


def test_real_reading_routes_supported_with_table_and_others_with_picture(tmp_path):
    from devision.reading import Reading
    pics = pictures(tmp_path)
    seen = []

    class Dec:
        def decide(self, state, questions, image=None):
            seen.append((state, sorted(questions), image is not None))
            return {"answers": {k: {"type": "choice", "probabilities": {o: 1 / len(q["criteria"]) for o in q["criteria"]}} for k, q in questions.items()},
                    "usage": {"input_tokens": 1}}
    r = object.__new__(Reading)
    r.decider = Dec()
    reads = []
    r.read = lambda image, templates: (reads.append(image is not None), {"table": [["Sample A", "mass of each particle", "14"]],
                                                                       "reads": [], "seconds": 0.0})[1]
    other = {"type": "choice", "instructions": "Which animal is shown?", "criteria": {"dog": None, "cat": None}}
    s = [row(0, "img/red.png", TEMP_Q)]
    s[0]["questions"]["o"] = other
    s[0]["gold"]["o"] = {"probabilities": {"dog": 1.0, "cat": 0.0}}
    m.run_set(r, s, str(tmp_path), pics)
    assert reads == [True]
    assert seen[0][2] is False and "Values read from the picture" in seen[0][0] and seen[0][1] == ["0:q"]    # table, no picture
    assert seen[1] == ("hint", ["0:o"], True)                                                                  # ordinary, with picture
    seen.clear()
    m.run_set(r, [dict(row(1, "img/blue.png", "Which animal is shown?"), questions={"o": other}, gold={"o": s[0]["gold"]["o"]})], str(tmp_path), pics)
    assert seen == [("hint", ["0:o"], True)] and reads == [True]                                              # nothing routed: no read


def test_placeholder_endpoint_refuses_start(tmp_path):
    assert m.NEW_SHA is None
    with pytest.raises(SystemExit, match="placeholder"):
        m.stage_start(0, out=str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_grouping_is_devision_eval_s(tmp_path):
    from devision.train.evaluate import _requests
    rows = [row(i, "img/red.png" if i < 3 else "img/blue.png", TEMP_Q) for i in range(5)]
    assert m.requests(rows) == _requests(rows, {}, "none", 32) and len(m.requests(rows)) == 2
    json.dumps(m.requests(rows))


# ---------------------------------------------------------------- a fake world: start -> checks -> live -> fidelity (R1-01..03)

class FakeDecider:
    """Deterministic answers from (state, question): the same state gives the same probabilities in any request."""
    class cfg:
        temperature, temperature_by_options = [1.0, 1.0, 1.0], {}

    def decide(self, state, questions, image=None):
        import hashlib
        out = {}
        for k, q in questions.items():
            opts = list(q["criteria"])
            h = int(hashlib.sha256((str(state) + q["instructions"]).encode()).hexdigest(), 16)
            w = [1 + (h >> (8 * i)) % 7 for i in range(len(opts))]
            out[k] = {"type": "choice", "probabilities": {o: x / sum(w) for o, x in zip(opts, w)}}
        return {"answers": out, "usage": {"input_tokens": 1}}


class FakeLiveReading:
    """Reading.decide's contract with fixed decoded values per template."""
    def __init__(self, values):
        self.values, self.decider, self.images = values, FakeDecider(), []

    def decide(self, text, questions, image=None, diagnostics=None):
        from devision.reading import TEMPLATES, fmt, reads_for, serialise
        assert image is not None
        self.images.append(image.getpixel((0, 0)))
        ts = reads_for(questions)
        reads = [{"question": t, "decoded": self.values[t], "valid": True, "value": fmt(self.values[t])} for t in ts]
        table = [[*TEMPLATES[t], r["value"]] for t, r in zip(ts, reads)]
        state = serialise(text, table)
        if diagnostics is not None:
            diagnostics.update(table=table, reads=reads, read_seconds=0.0, state=state)
        return self.decider.decide(state, questions)


@pytest.fixture
def world(tmp_path, monkeypatch):
    from devision.reading import TEMPLATES
    temp = [t for t in TEMPLATES if "sample" in t.lower()]
    data = tmp_path / "data"
    (data / "img").mkdir(parents=True)
    for i, c in enumerate([(200, 0, 0), (0, 200, 0), (0, 0, 200), (90, 90, 90)]):
        Image.new("RGB", (40, 30), c).save(data / "img" / ("%d.png" % i))
    tune = [{"id": "t%d" % i, "image_id": "t%d" % i, "image": "img/%d.png" % i, "state_text": "hint",
             "questions": {"r": {"type": "choice", "instructions": temp[0], "criteria": {"14 u": None, "9 u": None}}},
             "gold": {"r": {"probabilities": {"14 u": 1.0, "9 u": 0.0}}}} for i in range(2)]
    w = lambda p, rows: p.write_text("".join(json.dumps(r) + "\n" for r in rows))  # noqa: E731
    w(tmp_path / "tune.jsonl", tune)
    w(tmp_path / "sqa.jsonl", [row(9, "img/2.png", TEMP_Q)])
    w(tmp_path / "sqa_mism.jsonl", [row(9, "img/3.png", TEMP_Q)])
    monkeypatch.setattr(m.e54, "TUNE", str(tmp_path / "tune.jsonl"))
    monkeypatch.setattr(m, "TRANSFER", {"sqa_temp": str(tmp_path / "sqa.jsonl"), "sqa_temp_mism": str(tmp_path / "sqa_mism.jsonl")})
    monkeypatch.setattr(m, "GUARD", {})
    monkeypatch.setattr(m, "FILES", [])
    monkeypatch.setattr(m, "A_PICTURES", 2)
    monkeypatch.setattr(m, "A_READS", 2 * 4)
    monkeypatch.setattr(m, "A_QUESTIONS", 2)
    values = {t: str(10 + k) for k, t in enumerate(TEMPLATES)}
    # EXP57-like endpoint (dummy weights), frozen, its report on the passing branch
    run57 = tmp_path / "exp57"
    (run57 / "stage2" / "tokenizer").mkdir(parents=True)
    (run57 / "stage2" / "model.safetensors").write_bytes(b"w57")
    (run57 / "stage2" / "config.json").write_text("{}")
    (run57 / "stage2" / "tokenizer" / "t.json").write_text("{}")
    m.fz.freeze(str(run57))
    (run57 / "report.json").write_text(json.dumps({"branch": "fit, G, P, reading retention and exist / science all pass: next"}))
    (run57 / "START.json").write_text("{}")
    model = m.fz.weights(str(run57))
    monkeypatch.setattr(m, "NEW_RUN", str(run57))
    monkeypatch.setattr(m, "E57", str(run57))
    monkeypatch.setattr(m, "NEW_SHA", model)
    monkeypatch.setattr(m.e57, "SCREENS", {})
    # EXP56-like live products: the same fake reading on the old bundle / pictures
    e56d = tmp_path / "exp56"
    e56d.mkdir()
    pics = {r["image"]: m.sha(str(data / r["image"])) for r in tune}
    (e56d / "pictures.json").write_text(json.dumps(pics))
    st56 = {"model": model, "bundle": {"reading.json": "b56"}, "pictures": {"pictures.json": m.sha(str(e56d / "pictures.json"))}}
    (e56d / "START.json").write_text(json.dumps(st56))
    m.live_into(str(e56d), FakeLiveReading(values), st56, pics, str(data))
    monkeypatch.setattr(m, "E56", str(e56d))
    monkeypatch.setattr(m, "E56_SHA", {"START.json": m.sha(str(e56d / "START.json")), "registry.json": m.sha(str(e56d / "registry.json"))})
    monkeypatch.setattr(m.e57, "E56", str(e56d))
    monkeypatch.setattr(m.e57, "E56_SHA", {"inputs_P.jsonl": m.sha(str(e56d / "inputs_P.jsonl"))})
    # EXP57's fixed-table details_P on EXP56's tables (same endpoint)
    rows = {r["id"]: r for r in tune}
    inputs = m.e57.set_inputs("P", rows, {})[1]
    recs = m.e54.decide_condition(FakeDecider(), rows, inputs)
    m.e57.write_set(str(run57), "P", recs, {"set": "P", "start_sha256": m.sha(str(run57 / "START.json")), "inputs_sha256": m.e57.hash_inputs(inputs),
                                           "model": model, "temperature": "none (raw)", "image": None})
    # selector / head files and v0.3 records
    for name in ("sel.pt", "head.pt"):
        (tmp_path / name).write_bytes(name.encode())
    monkeypatch.setattr(m, "SELECTOR", str(tmp_path / "sel.pt"))
    monkeypatch.setattr(m, "SELECTOR_SHA", m.sha(str(tmp_path / "sel.pt")))
    monkeypatch.setattr(m, "HEAD", str(tmp_path / "head.pt"))
    monkeypatch.setattr(m, "HEAD_SHA", m.sha(str(tmp_path / "head.pt")))
    (tmp_path / "v03").mkdir()
    for n in m.TRANSFER:
        (tmp_path / "v03" / (n + ".details.jsonl")).write_text("")
    monkeypatch.setattr(m, "V03", str(tmp_path / "v03"))
    import devision.reading as R
    monkeypatch.setattr(R.Reading, "load", classmethod(lambda cls, *a, **k: FakeLiveReading(values)))
    out = tmp_path / "out"
    m.stage_start(0, out=str(out), data_root=str(data))
    return {"out": str(out), "data": str(data), "values": values, "tmp": tmp_path}


def test_start_load_checks_and_live_fidelity_pass(world):
    out, data = world["out"], world["data"]
    st = json.load(open(os.path.join(out, "START.json")))
    pics = json.load(open(os.path.join(out, "pictures.json")))
    assert set(pics) == {"img/0.png", "img/1.png", "img/2.png", "img/3.png"} and st["pictures"]["count"] == 4   # A + B (true and wrong)
    reading, st2, pics2 = m.load_reading(out, data_root=data)                   # the pre-load checks pass on the real contract
    m.live_into(out, reading, st2, pics2, data)
    res = m.fidelity(out)
    assert res["reproduced"] and res["same_reads"] == 8 and res["max_probability_difference"] == 0
    recs = m.run_set(reading, m.jsonl(m.TRANSFER["sqa_temp_mism"]), data, pics2)
    assert recs[0]["image_used"] == "img/3.png" and reading.images[-1] == (90, 90, 90)          # the fixed wrong picture


@pytest.mark.parametrize("change", ["picture", "manifest", "source"])
def test_changed_inputs_are_refused_before_use(world, change):
    out, data = world["out"], world["data"]
    if change == "picture":
        Image.new("RGB", (40, 30), (1, 2, 3)).save(os.path.join(data, "img", "3.png"))      # a B picture (fixed wrong one)
        match = "pictures differ"
    elif change == "manifest":
        p = os.path.join(out, "pictures.json")
        d = json.load(open(p))
        d["img/3.png"] = d["img/2.png"]
        json.dump(d, open(p, "w"))
        match = "pictures.json differs"
    else:
        open(m.TRANSFER["sqa_temp"], "a").write("\n")
        match = "source files differ"
    with pytest.raises(SystemExit, match=match):
        m.load_reading(out, data_root=data)


def test_b_requests_check_the_bytes(world):
    out, data = world["out"], world["data"]
    reading, st, pics = m.load_reading(out, data_root=data)
    Image.new("RGB", (40, 30), (1, 2, 3)).save(os.path.join(data, "img", "2.png"))          # changed after the load
    with pytest.raises(SystemExit, match="differs from the bound picture"):
        m.run_set(reading, m.jsonl(m.TRANSFER["sqa_temp"]), data, pics)
    with pytest.raises(SystemExit):
        m.run_set(reading, m.jsonl(m.TRANSFER["sqa_temp"]), data, {})                        # outside the manifest


def live(world):
    reading, st, pics = m.load_reading(world["out"], data_root=world["data"])
    m.live_into(world["out"], reading, st, pics, world["data"])
    return st


def edit_jsonl(path, fn):
    lines = [json.loads(x) for x in open(path) if x.strip()]
    lines = fn(lines)
    open(path, "w").write("".join(json.dumps(x) + "\n" for x in lines))


@pytest.mark.parametrize("change", ["read_unregistered", "prediction_unregistered", "duplicate", "missing", "foreign", "table"])
def test_fidelity_refuses_changed_evidence(world, change):
    out = world["out"]
    st = live(world)
    assert m.verify_fidelity(out, st)["reproduced"]
    reads, det, inp = (os.path.join(out, f) for f in ("reads.jsonl", "details_P.jsonl", "inputs_P.jsonl"))

    def rereg(name, f):
        m.e56.merge_register(out, name, {os.path.basename(f): m.sha(f)})
    if change == "read_unregistered":
        edit_jsonl(reads, lambda ls: [dict(ls[0], reads=[dict(ls[0]["reads"][0], decoded="99", value="99")] + ls[0]["reads"][1:])] + ls[1:])
    elif change == "prediction_unregistered":
        edit_jsonl(det, lambda ls: ls[:1] + [dict(ls[1], prediction=next(o for o in ls[1]["options"] if o != ls[1]["prediction"]))] + ls[2:])
    elif change == "duplicate":
        edit_jsonl(det, lambda ls: ls + ls[1:2])
        m.register(out, "decide_P", {"details_P.jsonl": m.sha(det)})
    elif change == "missing":
        edit_jsonl(det, lambda ls: ls[:-1])
        m.register(out, "decide_P", {"details_P.jsonl": m.sha(det)})
    elif change == "foreign":                                   # another run's details, registered here
        edit_jsonl(det, lambda ls: [{"identity": dict(ls[0]["identity"], start_sha256="0" * 64)}] + ls[1:])
        m.register(out, "decide_P", {"details_P.jsonl": m.sha(det)})
    else:                                                       # same read strings, another actual table / state
        edit_jsonl(inp, lambda ls: [dict(ls[0], table=[[*r[:2], "77"] for r in ls[0]["table"]])] + ls[1:])
        rereg("inputs", inp)
    with pytest.raises(SystemExit) as e:
        m.verify_fidelity(out, st)
    print("REFUSED", change, e.value)


def test_stale_fidelity_record_is_not_trusted(world):
    out = world["out"]
    st = live(world)
    m.fidelity(out)
    edit_jsonl(os.path.join(out, "reads.jsonl"), lambda ls: ls[:-1])           # evidence changes after the record
    with pytest.raises(SystemExit):
        m.stage_sets(out, m.TRANSFER, "transfer", data_root=world["data"])
    rep = m.report(out)
    assert rep["fidelity_state"].startswith("missing / invalid") and rep["branch"].startswith("stage A not reproduced")


def test_report_without_start_is_incomplete(tmp_path):
    rep = m.report(str(tmp_path / "none"), finalize=" start=1")
    assert rep["branch"].startswith("incomplete: no START") and (tmp_path / "none" / "report.txt").exists()


STAGE = """import subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)'])
open('runs/exp58/child.pid', 'w').write(str(child.pid))
time.sleep(300)
"""
STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"watch.py"*) shift 2; exec python3 "$@";;
  *"live.py start"*) exit 4;;
  *"live.py guard"*) exec python3 "${STAGE}";;
  *"live.py report --finalize"*) mkdir -p runs/exp58; echo incomplete > runs/exp58/report.txt; exit 0;;
esac
exit 9
"""


def chain(tmp_path, part, limit="60"):
    import shutil
    import stat
    import subprocess
    root = tmp_path / "repo"
    if not (tmp_path / "bin").exists():
        for rel in ("scripts/exp58/run.sh", "scripts/exp53/watch.py", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(os.path.join(HERE, "..", "..", rel), root / rel)
        (tmp_path / "stage.py").write_text(STAGE)
        b = tmp_path / "bin"
        b.mkdir()
        for name, body in {"uv": STUB, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
            (b / name).write_text(body)
            (b / name).chmod((b / name).stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (tmp_path / "bin", os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"),
           "STAGE": str(tmp_path / "stage.py"), "EXP58_LIMIT": limit, "EXP58_RESERVE": "4"}
    return root, subprocess.run(["bash", str(root / "scripts/exp58/run.sh"), part], env=env, capture_output=True, text=True, timeout=120)


def test_failed_start_finalizes_and_part_a_never_starts_twice(tmp_path):
    root, p = chain(tmp_path, "A")
    calls = (tmp_path / "calls").read_text()
    assert p.returncode == 4 and "report --finalize  start=4" in calls and "live.py live" not in calls
    assert (root / "runs/exp58-A.started").exists() and (root / "runs/exp58/report.txt").exists()
    _, p2 = chain(tmp_path, "A")
    assert p2.returncode == 1 and "already started once" in p2.stdout


def test_b2_stopped_before_its_first_set_finalizes_and_never_starts_again(tmp_path):
    import time
    root = tmp_path / "repo"
    (root / "runs" / "exp58").mkdir(parents=True)
    (root / "runs" / "exp58-A.started").write_text("part A\n")
    (root / "runs" / "exp58" / "START.json").write_text("{}")
    t = time.time()
    root, p = chain(tmp_path, "B2")
    assert p.returncode == 124 and time.time() - t < 60, p.stdout + p.stderr
    assert "report --finalize  guard=124" in (tmp_path / "calls").read_text()
    assert not (root / "runs/exp58/details_exist.jsonl").exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int((root / "runs/exp58/child.pid").read_text()), 0)
    assert "continues_start" in (root / "runs/exp58-B2.started").read_text()
    _, p2 = chain(tmp_path, "B2")
    assert p2.returncode == 1 and "already started once" in p2.stdout
