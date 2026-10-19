# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
"""exp70's runner without a model (a fake decider): START binds EXP68 / X / the data and refuses changes; decide runs the
48 raw fidelity requests first and stops before anything else when they do not reproduce EXP68's registered records;
every set is registered; the report recomputes fidelity and reads the reused raw records through EXP68's checks, gives
incomplete / invalid without a verdict, and otherwise the pre-written branch; the run's order and limit."""
import importlib.util
import json
import os
import shutil

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("exp70_runner", os.path.join(HERE, "exp70.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
has = os.path.exists(os.path.join(ROOT, "runs", "exp68", "details_x_diag68.jsonl")) and os.path.exists(os.path.join(ROOT, "data", "exp70", "MANIFEST.json"))
data = pytest.mark.skipif(not has, reason="EXP68's run / data/exp70 not present")


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


class Cfg:
    temperature, temperature_by_options = [1.0, 1.0, 1.0], {}


class FakeDecider:
    """raw states: EXP68's registered probabilities (perturbed by `shift` on the fidelity requests); K states and the
    training slice: `k_policy` ("right", "raw", "wrong")."""

    def __init__(self, shift=0.0, k_policy="right"):
        self.cfg = Cfg()
        reg = {r["sample_id"]: r["probabilities"] for r in m.raw_registered()}
        rows = m.e68.inputs_of("diag68")[0]
        # keyed by state AND option order: the two orders of an equal combination share one table text
        self.raw = {(rows[i]["state_text"], tuple(next(iter(rows[i]["questions"].values()))["criteria"])): reg[i] for i in reg}
        self.shift, self.k_policy = shift, k_policy

    def decide(self, state, questions, image=None):
        assert image is None
        (qid, q), = questions.items()
        opts = list(q["criteria"])
        base = (state.split("\n" + m.b70.K_HEADER)[0], tuple(opts))
        if base in self.raw:
            p = dict(self.raw[base])
            if m.b70.K_HEADER not in state:
                if self.shift:
                    k = max(p, key=lambda o: p[o])
                    p = {o: (v - self.shift if o == k else v + self.shift / (len(p) - 1)) for o, v in p.items()}
                return {"answers": {qid: {"probabilities": p}}}
            if self.k_policy == "raw":
                return {"answers": {qid: {"probabilities": p}}}
        if m.b70.K_HEADER in state:
            ka, kb = m.b70.k_values(state)
            want = "sample A" if ka > kb else "sample B" if kb > ka else next(o for o in opts if o.startswith("neither"))
        else:
            want = "sample A"
        if self.k_policy == "wrong" and m.b70.K_HEADER in state:
            want = next(o for o in opts if o != want)
        return {"answers": {qid: {"probabilities": {o: (0.9 if o == want else 0.1 / (len(opts) - 1)) for o in opts}}}}


@pytest.fixture
def started(tmp_path):
    import time
    out = str(tmp_path / "exp70")
    m.stage_start(int(time.time()) + 900, out=out)
    return out


@data
def test_start_binds_and_refuses(started, tmp_path, monkeypatch):
    st = m.check_start(started, code=False)
    assert st["requests"] == {"fid_raw": 48, "diag68_K": 480, "train_raw": 120, "train_K": 120} and st["exp68"]["x_end"] == m.X_SHA
    with pytest.raises(SystemExit, match="exists"):
        m.stage_start(0, out=started)
    d = tmp_path / "d70"
    shutil.copytree(m.D70, d)
    with open(d / "train_K.jsonl", "a") as fh:
        fh.write("\n")
    monkeypatch.setattr(m, "D70", str(d))
    with pytest.raises(SystemExit, match="data/exp70"):
        m.check_start(started, code=False)


@data
def test_fidelity_failure_stops_before_any_other_request(started):
    with pytest.raises(SystemExit, match="raw fidelity failed"):
        m.stage_decide(started, device="cpu", decider=FakeDecider(shift=0.01))
    reg = json.load(open(os.path.join(started, "registry.json")))
    assert set(reg) == {"decide_fid_raw", "fidelity"} and not json.load(open(os.path.join(started, "fidelity.json")))["reproduced"]
    res = m.report(started)
    assert res["validity"] == "invalid" and "diag" not in res and res["branch"].startswith("invalid")


@data
@pytest.mark.parametrize("policy,branch", [("right", "the external intermediate quantity helps"), ("raw", "not located"),
                                           ("wrong", "the K block as given disturbs")])
def test_branches(started, policy, branch):
    m.stage_decide(started, device="cpu", decider=FakeDecider(k_policy=policy))
    res = m.report(started)
    assert res["validity"] == "valid" and res["fidelity"]["reproduced"] and res["fidelity"]["max_abs_probability_difference"] == 0
    assert res["branch"].startswith(branch) and res["candidate_eligible"] is False
    if policy == "raw":
        assert res["diag"]["k_minus_raw_conflict"]["value"] == 0 and res["diag"]["k"]["accuracy"] == res["diag"]["raw"]["accuracy"]
    assert "95 %" in res["diag"]["interval"] and set(res["train_slice"]["raw"]) == set(m.CLASSES)


@data
@pytest.mark.parametrize("case,state", [("missing", "incomplete"), ("changed", "invalid")])
def test_missing_or_changed_set(started, case, state):
    m.stage_decide(started, device="cpu", decider=FakeDecider())
    f = os.path.join(started, "details_train_K.jsonl")
    if case == "missing":
        os.remove(f)
    else:
        with open(f, "a") as fh:
            fh.write("\n")
    res = m.report(started)
    assert res["validity"] == state and res["branch"].startswith(state) and "diag" not in res


STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"watch.py"*) shift 2; exec python3 "$@";;
  *"exp70.py start"*) mkdir -p runs/exp70; exit 0;;
  *"exp70.py decide"*) [ -n "${FAIL_DECIDE:-}" ] && exit 5; exit 0;;
  *"exp70.py report"*) echo "report $*" > runs/exp70/report.txt; exit 0;;
esac
exit 9
"""


def chain(tmp_path, extra):
    import stat as st_
    import subprocess
    root = tmp_path / "repo"
    for rel in ("scripts/exp70/run.sh", "scripts/exp53/watch.py", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    bdir = tmp_path / "bin"
    bdir.mkdir()
    for name, body in {"uv": STUB, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
        (bdir / name).write_text(body)
        (bdir / name).chmod((bdir / name).stat().st_mode | st_.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (bdir, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "EXP70_LIMIT": "60",
           "EXP70_RESERVE": "4", **extra}
    p = subprocess.run(["bash", str(root / "scripts/exp70/run.sh")], env=env, capture_output=True, text=True, timeout=120)
    return root, p, (tmp_path / "calls").read_text().splitlines()


def test_run_order_failed_stage_and_limit(tmp_path):
    root, p, calls = chain(tmp_path / "a", {})
    assert p.returncode == 0, p.stdout + p.stderr
    first = lambda s: next(i for i, c in enumerate(calls) if s in c)  # noqa: E731
    assert first("exp70.py start") < first("exp70.py decide") < first("exp70.py report")
    root, p, calls = chain(tmp_path / "b", {"FAIL_DECIDE": "1"})
    assert p.returncode == 5 and "report --finalize" in "\n".join(calls) and "finalize" in (root / "runs/exp70/report.txt").read_text()
    s = open(os.path.join(ROOT, "scripts/exp70/run.sh")).read()
    assert "${EXP70_LIMIT:-900}" in s and "${EXP70_RESERVE:-120}" in s
