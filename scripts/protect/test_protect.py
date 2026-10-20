# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
"""scripts/protect/protect.py without a model (PROTECT-PREP-01 / -R1-01): START binds EXP72-K, the reviewed data and every
exposure file (a changed exposure file is refused); the YAML must be EXP72-K's recipe with mixed_input; the plan's inputs
are 475 text / 923 picture updates; the start's fidelity is checked against EXP72's registered records; with a fake
decider (B / C) and a fake reader (A, two panels) the report walks the pre-written branches -- no cost and both abilities
kept, a protection cost, a lost new ability, a missing set; the run's order, limit and reserve."""
import importlib.util
import json
import os
import shutil

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("protect", os.path.join(HERE, "protect.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
has = all(os.path.exists(os.path.join(ROOT, p)) for p in ("runs/exp72/report.json", "runs/exp73/report.json", "data/protect2/MANIFEST.json",
                                                           "data/protect_p/MANIFEST.json"))
data = pytest.mark.skipif(not has, reason="EXP72 / EXP73 runs or data/protect2 / protect_p not present")
SUBSET = ("count_wide", "vsr")
NEW = {"model.safetensors": "0" * 64, "config.json": "1" * 64}


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)
    sub = {k: m.e58.GUARD[k] for k in SUBSET}
    monkeypatch.setattr(m, "GUARD", sub)
    monkeypatch.setattr(m.e73, "GUARD", sub)


@pytest.fixture
def started(tmp_path):
    import time
    out = str(tmp_path / "protect")
    m.stage_start(int(time.time()) + 7200, out=out)
    return out


@data
def test_start_binds_the_isolation_evidence(started, monkeypatch):
    st = m.check_start(started, code=False)
    assert st["start"] == m.START_SHA and st["data"]["exposure_files"] == 163 and st["candidate_eligible"] is False
    man = json.load(open("data/protect2/MANIFEST.json"))
    victim = sorted(man["exposure_files"])[0]
    real = m.sha
    monkeypatch.setattr(m, "sha", lambda p: "f" * 64 if p == victim else real(p))
    with pytest.raises(SystemExit, match="isolation evidence changed"):
        m.check_start(started, code=False)


@data
def test_yaml_must_be_exp72k_with_mixed_input(tmp_path, monkeypatch):
    import yaml
    m.check_yaml()
    c = yaml.safe_load(open(m.YAML))
    c["stages"][0]["params"]["lr_lora"] = 1e-4
    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump(c))
    monkeypatch.setattr(m, "YAML", str(p))
    with pytest.raises(SystemExit, match="not EXP72-K's recipe"):
        m.check_yaml()


@data
def test_plan_inputs():
    kinds = m.plan_inputs()
    assert len(kinds) == m.UPDATES and kinds.count("text") == 475 and kinds.count("picture") == 923


class FakeDecider:
    """Answers each (state, question) with a fixed table: EXP72-K's registered probabilities where it has them,
    else the gold option at 0.9 (or, policy "wrong", the gold at 0.1)."""

    def __init__(self, table, policy="right"):
        self.table, self.policy = table, policy
        self.cfg = type("C", (), {"temperature": None, "temperature_by_options": None})()

    def decide(self, state, questions, image=None):
        ans = {}
        for k, q in questions.items():
            key = (state, q["instructions"], tuple(q["criteria"]))
            if self.policy == "registered" and key in self.table["registered"]:
                probs = dict(self.table["registered"][key])
            else:
                g = self.table["gold"][key]
                hi = 0.9 if self.policy != "wrong" else 0.1
                probs = {o: (hi if o == g else (1 - hi) / (len(q["criteria"]) - 1)) for o in q["criteria"]}
            ans[k] = {"type": "choice", "probabilities": probs}
        return {"answers": ans}


def table():
    reg, gold = {}, {}
    for name, path in list(m.B_SETS.items()) + [("P", m.P_SET)]:
        rows, inputs = m.rows_inputs(path)
        for r in rows.values():
            for qid, q in r["questions"].items():
                gold[(r["state_text"], q["instructions"], tuple(q["criteria"]))] = max(r["gold"][qid]["probabilities"], key=r["gold"][qid]["probabilities"].get)
        if name in ("slice", "diag", "G", "fit"):
            for rec in m.e57.read_set(m.E72, "k_" + name, rows, m.START_SHA, inputs):
                r = rows[rec["sample_id"]]
                q = r["questions"][rec["qid"]]
                reg[(r["state_text"], q["instructions"], tuple(q["criteria"]))] = rec["probabilities"]
    return {"registered": reg, "gold": gold}


@data
def test_fidelity_against_exp72(started):
    t = table()
    assert m.stage_fid(started, decider=FakeDecider(t, "registered"))["reproduced"]
    with pytest.raises(SystemExit, match="does not reproduce"):
        m.stage_fid(started, decider=FakeDecider(t, "wrong"))


class FakeReading:
    """v0.3's own answers on the panels (policy "v03") or every right count_wide answer turned wrong ("worse")."""

    def __init__(self, policy="v03"):
        st = {"v03": {n: m.sha(os.path.join(m.e58.V03, n + ".details.jsonl")) for n in SUBSET}}
        self.policy, self.v03 = policy, {}
        for n in SUBSET:
            src = {r["id"]: r for r in m.jsonl(m.GUARD[n])}
            for r in m.e58.v03_records(n, st):
                self.v03[(src[r["sample_id"]]["image"], src[r["sample_id"]]["questions"][r["qid"]]["instructions"])] = r
        self.count = {r["image"] for r in m.jsonl(m.GUARD["count_wide"])}
        self.decider = FakeDecider({"registered": {}, "gold": {}})
        self.corrector = object()

    def decide(self, state, questions, image=None, diagnostics=None):
        path = getattr(image, "_path")
        if diagnostics is not None:
            diagnostics.update(reads=[], table=[], state=state)
        ans = {}
        for k, q in questions.items():
            r = self.v03[(path, q["instructions"])]
            probs = dict(r["probabilities"])
            if self.policy == "worse" and path in self.count and r["correct"]:
                g = r["gold_answer"]
                probs = {o: (0.9 if o != g else 0.1) for o in probs}
                s = sum(probs.values())
                probs = {o: v / s for o, v in probs.items()}
            ans[k] = {"type": "noul", "noul": probs["true"]} if q["type"] == "noul" else {"type": "choice", "probabilities": probs}
        return {"answers": ans}


@pytest.fixture
def faked(monkeypatch, started):
    def ask(reading, image_path, text, questions, data_root, pictures, diagnostics=None):
        import io

        from PIL import Image
        raw = open(os.path.join(data_root, image_path), "rb").read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img._path = image_path
        return reading.decide(text, questions, image=img, diagnostics=diagnostics)
    monkeypatch.setattr(m.e58, "ask", ask)
    monkeypatch.setattr(m, "new_weights", lambda: dict(NEW))
    os.makedirs(os.path.join(started, "bundle"))
    open(os.path.join(started, "bundle", "reading.json"), "w").write("{}")
    m.register(started, "bundle", {"bundle/reading.json": m.sha(os.path.join(started, "bundle", "reading.json"))})
    return started


def run_all(out, new_policy="registered", c_policy="right", reading="v03", skip=()):
    t = table()
    m.stage_fid(out, decider=FakeDecider(t, "registered"))
    new = FakeDecider(t, new_policy if new_policy != "registered" else "registered")
    if "decide_b" not in skip:
        m.stage_decide_b(out, deciders={"new": new})
    m.stage_decide_c(out, deciders={"new": FakeDecider(t, c_policy), "start": FakeDecider(t, "right")})
    m.stage_guard(out, reading=FakeReading(reading))
    m.stage_decide_fit(out, deciders={"new": new})
    return m.report(out)


@data
@pytest.mark.parametrize("case,branch", [("pass", "engineering recovery"), ("cost", "recovery failed"),
                                         ("lost", "protection recovered but the new ability is not kept"), ("missing", "incomplete")])
def test_report_branches(faked, case, branch):
    kw = {"pass": {}, "cost": {"reading": "worse"}, "lost": {"c_policy": "wrong"}, "missing": {"skip": ("decide_b",)}}[case]
    res = run_all(faked, **kw)
    assert res["branch"].startswith(branch), res["branch"]
    assert res["candidate_eligible"] is False
    if case == "pass":
        assert res["gates"] == {**res["gates"], "A": True, "B": True, "C": True} and all(not v["cost"] for v in res["A"].values())
    if case == "cost":
        assert res["A"]["count_wide"]["cost"] and res["gates"]["A"] is False
    if case == "lost":
        assert res["gates"]["C"] is False and res["gates"]["B"] is True


STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"watch.py"*) shift 2; exec python3 "$@";;
  *"protect.py start"*) mkdir -p runs/protect; exit 0;;
  *"protect.py guard"*) [ -n "${FAIL_GUARD:-}" ] && exit 5; exit 0;;
  *"protect.py report"*) echo "report $*" > runs/protect/report.txt; exit 0;;
  *"protect.py "*) exit 0;;
esac
exit 9
"""


def chain(tmp_path, extra):
    import stat as st_
    import subprocess
    root = tmp_path / "repo"
    for rel in ("scripts/protect/run.sh", "scripts/exp53/watch.py", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    bdir = tmp_path / "bin"
    bdir.mkdir()
    for name, body in {"uv": STUB, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
        (bdir / name).write_text(body)
        (bdir / name).chmod((bdir / name).stat().st_mode | st_.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (bdir, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "PROTECT_LIMIT": "60",
           "PROTECT_RESERVE": "4", **extra}
    p = subprocess.run(["bash", str(root / "scripts/protect/run.sh")], env=env, capture_output=True, text=True, timeout=120)
    return p, (tmp_path / "calls").read_text().splitlines()


def test_run_order_failed_stage_and_limit(tmp_path):
    p, calls = chain(tmp_path / "a", {})
    assert p.returncode == 0, p.stdout + p.stderr
    order = ["start", "fid", "train", "freeze", "decide_b", "decide_c", "guard", "decide_fit", "report"]
    pos = [next(i for i, c in enumerate(calls) if "protect.py %s" % s in c) for s in order]
    assert pos == sorted(pos)
    p, calls = chain(tmp_path / "b", {"FAIL_GUARD": "1"})
    assert p.returncode == 5 and "report --finalize" in "\n".join(calls) and not any("decide_fit" in c for c in calls)
    s = open(os.path.join(ROOT, "scripts/protect/run.sh")).read()
    assert "${PROTECT_LIMIT:-7200}" in s and "${PROTECT_RESERVE:-180}" in s
