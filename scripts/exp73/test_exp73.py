# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
"""exp73 without a model (a fake reader on two of the panels): START binds EXP71's run and bundle_on (a changed component
is refused); the guard stage answers every question on its own picture through Reading.decide's interface; the report
computes EXP58's cost rule against v0.3 (a v0.3-identical reader: no cost; a reader 10 points worse: cost), and a missing
or tampered panel gives no protection verdict; the run's order and limit."""
import importlib.util
import json
import os
import shutil

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("exp73", os.path.join(HERE, "exp73.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
has = os.path.exists(os.path.join(ROOT, "runs", "exp71", "report.json")) and os.path.exists(os.path.join(ROOT, "runs", "exp58", "pictures.json"))
data = pytest.mark.skipif(not has, reason="EXP71 / EXP58 runs not present")
SUBSET = ("count_wide", "vsr")


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(m, "GUARD", {k: m.e58.GUARD[k] for k in SUBSET})


class FakeReading:
    """Answers with v0.3's own probabilities for the same question (policy "v03"), or always a wrong option on
    count_wide (policy "worse"); reads of the passed picture's size."""

    def __init__(self, policy="v03"):
        self.policy = policy
        st = {"v03": {n: m.sha(os.path.join(m.e58.V03, n + ".details.jsonl")) for n in SUBSET}}
        self.v03 = {}
        for n in SUBSET:
            src = {r["id"]: r for r in m.jsonl(m.GUARD[n])}
            for r in m.e58.v03_records(n, st):
                self.v03[(src[r["sample_id"]]["image"], src[r["sample_id"]]["questions"][r["qid"]]["instructions"], n)] = r
        self.count_imgs = {r["image"] for r in m.jsonl(m.GUARD["count_wide"])}
        self.decider = type("D", (), {"cfg": type("C", (), {"temperature": None, "temperature_by_options": None})()})()
        self.corrector = object()

    def decide(self, state, questions, image=None, diagnostics=None):
        path = getattr(image, "_exp73_path")
        if diagnostics is not None:
            diagnostics.update(reads=[{"size": list(image.size)}], table=[], state=state)
        ans = {}
        for k, q in questions.items():
            r = next((v for (p, ins, n), v in self.v03.items() if p == path and ins == q["instructions"]), None)
            probs = dict(r["probabilities"]) if r else {o: 1 / len(q["criteria"]) for o in q["criteria"]}
            if self.policy == "worse" and path in self.count_imgs and r is not None and r["correct"]:
                g = r["gold_answer"]
                probs = {o: (0.9 if o != g else 0.1 / max(1, len(probs) - 1)) for o in probs}
                s_ = sum(probs.values())
                probs = {o: v / s_ for o, v in probs.items()}
            ans[k] = {"type": "noul", "noul": probs["true"]} if q["type"] == "noul" else {"type": "choice", "probabilities": probs}
        return {"answers": ans}


@pytest.fixture
def tagged_ask(monkeypatch):
    def ask(reading, image_path, text, questions, data_root, pictures, diagnostics=None):
        import hashlib
        import io

        from PIL import Image
        raw = open(os.path.join(data_root, image_path), "rb").read()
        assert hashlib.sha256(raw).hexdigest() == pictures[image_path]
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img._exp73_path = image_path
        return reading.decide(text, questions, image=img, diagnostics=diagnostics)
    monkeypatch.setattr(m.e58, "ask", ask)


@pytest.fixture
def started(tmp_path):
    import time
    out = str(tmp_path / "exp73")
    m.stage_start(int(time.time()) + 2700, out=out)
    return out


@data
def test_start_binds_exp71_on(started, tmp_path, monkeypatch):
    st = m.check_start(started, code=False)
    assert st["exp71"]["bundle_on"]["corrector.pt"] == m.e71.CORR_SHA and st["candidate_eligible"] is False
    copy = tmp_path / "bundle_on"
    shutil.copytree(m.BUNDLE, copy)
    with open(copy / "corrector.pt", "ab") as fh:
        fh.write(b"x")
    monkeypatch.setattr(m, "BUNDLE", str(copy))
    with pytest.raises(SystemExit, match="corrector.pt changed"):
        m.check_start(started, code=False)


@data
@pytest.mark.parametrize("policy,branch", [("v03", "no pre-written protection cost"), ("worse", "protection costs")])
def test_guard_and_cost_rule(started, tagged_ask, policy, branch):
    m.stage_guard(started, reading=FakeReading(policy))
    res = m.report(started)
    assert all(v == "completed" for v in res["states"].values()) and res["branch"].startswith(branch)
    if policy == "v03":
        assert all(g["difference"] == pytest.approx(0) and not g["cost"] for g in res["guard"].values())
    else:
        assert res["costs"] == ["count_wide"] and res["guard"]["count_wide"]["difference"] <= -0.03
    assert res["candidate_eligible"] is False


@data
@pytest.mark.parametrize("case,want", [("missing", "incomplete"), ("other_picture", "invalid")])
def test_missing_or_tampered_panel(started, tagged_ask, case, want):
    m.stage_guard(started, reading=FakeReading())
    f = os.path.join(started, "details_vsr.jsonl")
    if case == "missing":
        os.remove(f)
    else:
        lines = [json.loads(x) for x in open(f)]
        lines[1]["image_used"] = "some/other.png"
        with open(f, "w") as fh:
            fh.write("".join(json.dumps(x) + "\n" for x in lines))
        m.register(started, "decide_vsr", {"details_vsr.jsonl": m.sha(f)})
    res = m.report(started)
    assert res["branch"].startswith(want) and "no protection verdict" in res["branch"]


STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"watch.py"*) shift 2; exec python3 "$@";;
  *"exp73.py start"*) mkdir -p runs/exp73; exit 0;;
  *"exp73.py guard"*) [ -n "${FAIL_GUARD:-}" ] && exit 5; exit 0;;
  *"exp73.py report"*) echo "report $*" > runs/exp73/report.txt; exit 0;;
esac
exit 9
"""


def chain(tmp_path, extra):
    import stat as st_
    import subprocess
    root = tmp_path / "repo"
    for rel in ("scripts/exp73/run.sh", "scripts/exp53/watch.py", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    bdir = tmp_path / "bin"
    bdir.mkdir()
    for name, body in {"uv": STUB, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
        (bdir / name).write_text(body)
        (bdir / name).chmod((bdir / name).stat().st_mode | st_.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (bdir, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "EXP73_LIMIT": "60",
           "EXP73_RESERVE": "4", **extra}
    p = subprocess.run(["bash", str(root / "scripts/exp73/run.sh")], env=env, capture_output=True, text=True, timeout=120)
    return root, p, (tmp_path / "calls").read_text().splitlines()


def test_run_order_failed_stage_and_limit(tmp_path):
    root, p, calls = chain(tmp_path / "a", {})
    assert p.returncode == 0, p.stdout + p.stderr
    first = lambda s: next(i for i, c in enumerate(calls) if s in c)  # noqa: E731
    assert first("exp73.py start") < first("exp73.py guard") < first("exp73.py report")
    root, p, calls = chain(tmp_path / "b", {"FAIL_GUARD": "1"})
    assert p.returncode == 5 and "report --finalize" in "\n".join(calls)
    s = open(os.path.join(ROOT, "scripts/exp73/run.sh")).read()
    assert "${EXP73_LIMIT:-2700}" in s and "${EXP73_RESERVE:-120}" in s
