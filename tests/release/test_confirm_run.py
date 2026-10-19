# pyright: reportOptionalIterable=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
"""scripts/confirm/confirm_run.py (critic V04-K-CONFIRM-PREP-02, CONF-RUN-R1-01..04, NOTE-01) without a model and without
the confirmation set. A fake 120-picture set with the real reading templates (temperature: the comparison + four
readings; magnets: the comparison + two distances), an exclusion snapshot, and fake models through the formal evaluation
path (the candidate reports the routes the reading combination takes, v0.3 the ordinary path):
  - the five conditions run once each on the fixed wrong pictures, the report checks every record's content and gives
    the outcome on the 120 ids, readings clustered by picture and the reversed flips;
  - a stage that already ran or started is refused before anything is loaded and its evidence is untouched; a started
    but unregistered stage is incomplete; a missing condition is incomplete;
  - re-registered records with another reading identity, another temperature, a read question answered on the ordinary
    path, a wrong gold, unreversed options, another picture, a non-finite probability or a false "correct" are invalid;
    a wrong answer after a failed read is the model's failure and stays counted;
  - start refuses an unfrozen combination or another v0.3;
  - a generated case in the exclusion snapshot, or a snapshot whose members changed, is refused;
  - run.sh: order, failure, required inputs, a fake process on a newer entry point refused, limit and reserve."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMP = "Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature?"
MAG = "Think about the magnetic force between the magnets in each pair. Which of the following statements is true?"


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Cfg:
    temperature = [1.0, 1.0, 1.0]
    temperature_by_options: dict = {}
    temperature_by_route: dict = {}


class Fake:
    """Right when shown its own picture ("right") or on even pictures only ("half"); `read` reports the route the
    reading combination takes (reading-domain questions are read); `fail_reads` answers the reading questions wrongly."""

    def __init__(self, truth, policy, reading=None, read=False):
        self.truth, self.policy, self.read, self.fail_reads = truth, policy, read, False
        self.reading_status = reading or {"mode": "none"}
        self.cfg = Cfg()

    def decide(self, state=None, questions=None, image=None, diagnostics=None):
        from devision.infer import picture_identity, resolve_request
        from devision.reading import domain_of
        _, pic = resolve_request(state, questions, image)
        ident = picture_identity(pic)
        assert ident is not None
        routes = {k: ("read" if self.read and domain_of(q["instructions"]) else "ordinary") for k, q in questions.items()}
        if diagnostics is not None:
            diagnostics.update(route=routes, picture=ident, states={k: "s" for k, r in routes.items() if r == "read"}, table=[])
        ans = {}
        for k, q in questions.items():
            opts = list(q["criteria"])
            hit = self.truth.get((ident["pixels_sha256"], q["instructions"]))
            failed = self.fail_reads and q["instructions"] not in (TEMP, MAG)
            right = hit is not None and (self.policy == "right" or hit[1] % 2 == 0) and not failed
            pick = hit[0] if right else next(o for o in sorted(opts) if hit is None or o != hit[0])
            probs = {o: (0.8 if o == pick else 0.2 / (len(opts) - 1)) for o in opts}
            ans[k] = {"type": "choice", "choice": pick, "probabilities": probs, "confidence": 0.5}
        return {"model": "fake", "answers": ans, "usage": {"input_tokens": 0, "output_tokens": 0}}


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "build", raising=False)
    m = load("confirm_run_t", "scripts/confirm/confirm_run.py")
    from devision.reading import TEMPLATES
    templ = list(TEMPLATES)
    monkeypatch.chdir(tmp_path)
    os.makedirs("runs")
    for name in ("v03", "cand"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "model.safetensors").write_bytes(name.encode())
        (tmp_path / name / "config.json").write_text(json.dumps({"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}))
    ident = tmp_path / "IDENTITY.json"
    ident.write_text(json.dumps({"reading": {"reading_json_sha256": "frozen-reading"}}))
    monkeypatch.setattr(m, "V03", str(tmp_path / "v03"))
    monkeypatch.setattr(m, "V03_SHA", m.weights(str(tmp_path / "v03")))
    monkeypatch.setattr(m, "ROOT", ROOT)
    monkeypatch.setattr(m.cf, "freeze_problems", lambda *a, **k: [])
    monkeypatch.setattr(m, "excluded_snapshot", lambda: {("temp", "", ((900, 900), (900, 900)))})
    truth = {}
    from devision.infer import picture_identity

    def fake_set(st, used):
        out = m.DATA
        os.makedirs(os.path.join(out, "images"))
        rows = []
        for i in range(120):
            form = "temp" if i < 60 else "force"
            img = Image.new("RGB", (30, 20), (i, 255 - i, (7 * i) % 256))
            img.save(os.path.join(out, "images", "%03d.png" % i))
            texts = [TEMP] + templ[:4] if form == "temp" else [MAG] + templ[4:]
            g = "xyz"[i % 3]                                     # golds differ between pictures, so a wrong picture misleads
            qs = {"q%d" % j: {"type": "choice", "instructions": t, "criteria": {"x": None, "y": None, "z": None}} for j, t in enumerate(texts)}
            gold = {"q%d" % j: {"probabilities": {o: float(o == g) for o in "xyz"}} for j in range(len(texts))}
            rows.append({"id": "v04-confirm:%s%03d" % (form, i), "kind": form, "axis": "a", "image_id": "c%03d" % i,
                         "image": "v04_k_confirm/images/%03d.png" % i, "case": repr((form, "", ((i, i), (i + 1, i)))),
                         "state_text": "", "questions": qs, "gold": gold})
            pid = picture_identity(Image.open(os.path.join(out, "images", "%03d.png" % i)).convert("RGB"))
            assert pid is not None
            for t in texts:
                truth[(pid["pixels_sha256"], t)] = (g, i)
        mism = m.cf.mismatched(rows, 5)
        open(os.path.join(out, "confirm.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in rows))
        open(os.path.join(out, "confirm_mism.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in mism))
        man = {"pictures": 120, "classes": {}, "questions": 480}
        json.dump(man, open(os.path.join(out, "MANIFEST.json"), "w"))
        return man
    monkeypatch.setattr(m, "generate_set", fake_set)
    models = {"candidate": Fake(truth, "right", {"mode": "on", "k_rule": True, "reading_json_sha256": "frozen-reading"}, read=True),
              "v0.3": Fake(truth, "half")}
    monkeypatch.setattr(m, "load_model", lambda who, st, device: models[who])
    return m, str(ident), str(tmp_path / "cand"), models


def run(m, ident, cand, skip=()):
    m.stage_start(ident, cand, 0)
    m.stage_generate()
    for name in m.CONDITIONS:
        if name not in skip:
            m.stage_model(name)
    return m.report()


def test_the_whole_run(ws):
    m, ident, cand, _ = ws
    res = run(m, ident, cand)
    assert all(v == "completed" for v in res["states"].values()), res["states"]
    assert res["primary"]["branch"] == "synthetic confirmation passed"
    assert res["primary"]["candidate_minus_v03"]["value"] == pytest.approx(0.5)
    mism = [json.loads(x) for x in open(os.path.join(m.OUT, "details_cand_mism.jsonl"))][1:]
    rows = {r["id"]: r["image"] for r in map(json.loads, open(os.path.join(m.DATA, "confirm_mism.jsonl")))}
    assert all(x["image_used"] == rows[x["sample_id"]] for x in mism)               # the fixed wrong pictures, not re-drawn
    cand_true = [json.loads(x) for x in open(os.path.join(m.OUT, "details_cand_true.jsonl"))][1:]
    assert len(cand_true) == 480 and {x["route"] for x in cand_true} == {"read"}      # every question here is a reading-domain one
    r = res["readings_by_picture"]["cand_true"]
    assert r["temp"]["pictures"] == 60 and r["force"]["pictures"] == 60 and r["temp"]["value"] == 1.0   # clustered by picture
    assert res["readings_by_picture"]["v03_true"]["temp"]["value"] == pytest.approx(0.5)
    flip = res["reversed_flips"]
    assert flip["temp"]["value"] == 0 and flip["temp"]["pictures"] == 60 and flip["force"]["pictures"] == 60   # same content: no flip
    assert json.load(open(os.path.join(m.OUT, "excluded_cases.json"))) == [repr(("temp", "", ((900, 900), (900, 900))))]


def test_a_stage_runs_once_and_a_broken_run_is_incomplete(ws, monkeypatch):
    m, ident, cand, _ = ws
    m.stage_start(ident, cand, 0)
    m.stage_generate()
    m.stage_model("cand_true")
    f = os.path.join(m.OUT, "details_cand_true.jsonl")
    before = m.sha(f)
    monkeypatch.setattr(m, "load_model", lambda *a, **k: pytest.fail("loaded a model for a stage that already ran"))
    with pytest.raises(SystemExit, match="one run only"):
        m.stage_model("cand_true")
    assert m.sha(f) == before
    open(os.path.join(m.OUT, "stage-cand_mism.started"), "w").write("crashed")        # started, then interrupted
    with pytest.raises(SystemExit, match="already started"):
        m.stage_model("cand_mism")
    res = m.report()
    assert res["states"]["cand_mism"] == "started, not completed" and res["states"]["cand_rev"] == "missing"
    assert res["branch"].startswith("incomplete")
    with pytest.raises(SystemExit, match="one run only"):
        m.stage_generate()


def test_start_refusals(ws, monkeypatch):
    m, ident, cand, _ = ws
    monkeypatch.setattr(m.cf, "freeze_problems", lambda *a, **k: ["not frozen"])
    with pytest.raises(SystemExit, match="not frozen"):
        m.stage_start(ident, cand, 0)
    monkeypatch.setattr(m.cf, "freeze_problems", lambda *a, **k: [])
    monkeypatch.setattr(m, "V03_SHA", {"model.safetensors": "0" * 64, "config.json": "0" * 64})
    with pytest.raises(SystemExit, match="v0.3"):
        m.stage_start(ident, cand, 0)


def tamper(m, name, edit=None, ident_edit=None):
    f = os.path.join(m.OUT, "details_%s.jsonl" % name)
    lines = [json.loads(x) for x in open(f)]
    if ident_edit:
        ident_edit(lines[0]["identity"])
    if edit:
        edit(lines)
    open(f, "w").write("".join(json.dumps(x) + "\n" for x in lines))
    reg = json.load(open(os.path.join(m.OUT, "registry.json")))
    reg[name]["details_%s.jsonl" % name] = m.sha(f)                                     # re-registered: only content checks can catch it
    json.dump(reg, open(os.path.join(m.OUT, "registry.json"), "w"))


CASES = ["reading_identity", "temperature", "read_to_ordinary", "wrong_gold", "unreversed", "other_picture", "non_finite", "false_correct"]


@pytest.mark.parametrize("case", CASES)
def test_re_registered_wrong_content_is_invalid(ws, case):
    m, ident, cand, _ = ws
    run(m, ident, cand)
    name = "cand_rev" if case == "unreversed" else "cand_true"
    if case == "reading_identity":
        tamper(m, name, ident_edit=lambda i: i["reading"].update(reading_json_sha256="another"))
    elif case == "temperature":
        tamper(m, name, ident_edit=lambda i: i["temperature"].update(choice=2.0))
    else:
        def edit(lines):
            x = lines[2]
            if case == "read_to_ordinary":
                x["route"] = "ordinary"
            elif case == "wrong_gold":
                x["gold_answer"] = "z" if x["gold_answer"] != "z" else "y"
            elif case == "unreversed":
                x["options"] = x["options"][::-1]
            elif case == "other_picture":
                x["image_used"] = lines[200]["image_used"]
            elif case == "non_finite":
                x["probabilities"][x["options"][0]] = float("nan")
            else:
                x["correct"] = not x["correct"]
        tamper(m, name, edit)
    res = m.report()
    assert res["states"][name].startswith("invalid") and res["branch"].startswith("invalid"), res["states"][name]


def test_unregistered_change_and_a_missing_condition(ws):
    m, ident, cand, _ = ws
    res = run(m, ident, cand, skip=("v03_mism",))
    assert res["branch"].startswith("incomplete") and res["states"]["v03_mism"] == "missing"
    f = os.path.join(m.OUT, "details_cand_true.jsonl")
    open(f, "a").write("\n")
    assert m.report()["states"]["cand_true"].startswith("invalid")


def test_a_failed_read_is_the_models_failure_and_stays_counted(ws):
    m, ident, cand, models = ws
    models["candidate"].fail_reads = True
    res = run(m, ident, cand)
    assert all(v == "completed" for v in res["states"].values())
    r = res["readings_by_picture"]["cand_true"]["temp"]
    assert r["value"] == 0.0 and r["pictures"] == 60


def test_the_exclusion_snapshot_refuses_a_generated_case(ws, monkeypatch):
    m, ident, cand, _ = ws
    m.stage_start(ident, cand, 0)
    monkeypatch.setattr(m, "excluded_snapshot", lambda: {("temp", "", ((3, 3), (4, 3)))})           # a case the set will hold
    with pytest.raises(SystemExit, match="in the exclusion snapshot"):
        m.stage_generate()


def test_a_snapshot_with_other_members_of_the_same_count_is_refused(ws):
    m, ident, cand, _ = ws
    run(m, ident, cand)
    json.dump([repr(("temp", "", ((901, 901), (901, 901))))], open(os.path.join(m.OUT, "excluded_cases.json"), "w"))
    res = m.report()
    assert res["states"]["generate"].startswith("invalid") and res["branch"].startswith("invalid")


STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"watch.py"*) shift 2; exec python3 "$@";;
  *"confirm_run.py start"*) mkdir -p runs/v04-k-confirm; exit 0;;
  *"confirm_run.py cand_rev"*) [ -n "${FAIL_REV:-}" ] && exit 6; exit 0;;
  *"confirm_run.py report"*) echo "report $*" > runs/v04-k-confirm/report.txt; exit 0;;
  *"confirm_run.py "*) exit 0;;
esac
exit 9
"""


def chain(tmp_path, extra):
    import stat as st_
    root = tmp_path / "repo"
    for rel in ("scripts/confirm/run.sh", "scripts/exp53/watch.py", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    bdir = tmp_path / "bin"
    bdir.mkdir()
    for name, body in {"uv": STUB, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
        (bdir / name).write_text(body)
        (bdir / name).chmod((bdir / name).stat().st_mode | st_.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (bdir, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "CONFIRM_LIMIT": "60",
           "CONFIRM_RESERVE": "4", "CONFIRM_IDENTITY": "x/IDENTITY.json", "CONFIRM_CHECKPOINT": "x/cand", **extra}
    p = subprocess.run(["bash", str(root / "scripts/confirm/run.sh")], env=env, capture_output=True, text=True, timeout=120)
    calls = (tmp_path / "calls").read_text().splitlines() if (tmp_path / "calls").exists() else []
    return p, calls


def test_run_sh(tmp_path):
    p, calls = chain(tmp_path / "a", {})
    assert p.returncode == 0, p.stdout + p.stderr
    order = ["start", "generate", "cand_true", "cand_mism", "cand_rev", "v03_true", "v03_mism", "report"]
    pos = [next(i for i, c in enumerate(calls) if "confirm_run.py %s" % s in c) for s in order]
    assert pos == sorted(pos)
    p, calls = chain(tmp_path / "b", {"FAIL_REV": "1"})
    assert p.returncode == 6 and "report --finalize" in "\n".join(calls) and not any("v03_true" in c for c in calls)
    p, calls = chain(tmp_path / "c", {"CONFIRM_IDENTITY": ""})
    assert p.returncode == 2 and not calls
    s = open(os.path.join(ROOT, "scripts/confirm/run.sh")).read()
    assert "${CONFIRM_LIMIT:-1200}" in s and "${CONFIRM_RESERVE:-120}" in s


def test_run_sh_refuses_a_running_newer_entry_point(tmp_path):
    """NOTE-01: a fake process whose command line names scripts/calib/ (an entry point the pgrep pattern misses)."""
    fake = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "scripts/calib/calib.py", "calibrate"])
    try:
        time.sleep(0.5)
        p, calls = chain(tmp_path / "d", {})
        assert p.returncode == 1 and "other model processes" in p.stdout and "scripts/calib/" in p.stdout and not calls
    finally:
        fake.kill()
        fake.wait()
