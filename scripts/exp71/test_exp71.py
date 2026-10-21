# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
"""exp71 without a model (fake readers): START binds the two bundles, which may differ only by the corrector; stage A's
live reads must reproduce EXP69-RUN-02's registered reads per condition and every state must be its table serialised
(else fidelity fails and B1 does not start); B1 reads each row's own picture (the wrong picture for *_mism) and the
report refuses a record answered on another picture; the report's on - off comparison and branch; the run's order."""
import importlib.util
import json
import os
import shutil

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("exp71", os.path.join(HERE, "exp71.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
has = os.path.exists(os.path.join(ROOT, "runs", "exp69-r2", "report.json")) and os.path.exists(os.path.join(ROOT, "runs", "exp58", "pictures.json"))
data = pytest.mark.skipif(not has, reason="EXP69-RUN-02 / EXP58 runs not present")


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


class FakeReading:
    """Stage A: the reads EXP69-RUN-02 registered for this condition (by the picture actually passed); B1: a read of the
    passed picture's size and a fixed answer policy; the state is the table serialised, as Reading.decide does."""

    def __init__(self, c, policy="gold", break_read=False):
        from devision.reading import TEMPLATES, serialise
        self.c, self.policy, self.break_read = c, policy, break_read
        self.TEMPLATES, self.serialise = TEMPLATES, serialise
        rows = m.a_rows()
        by_image = {}
        for s_, name in m.EXP69_READS[c].items():
            for x in m.jsonl(os.path.join(m.E69, name + ".jsonl"))[1:]:
                by_image[rows[x["id"]]["image"]] = x["reads"]
        self.reads = by_image
        self.gold = {}
        for p in m.TRANSFER.values():
            for r in m.jsonl(p):
                for qid, q in r["questions"].items():
                    self.gold[(r["image"], q["instructions"], tuple(q["criteria"]))] = max(r["gold"][qid]["probabilities"].items(), key=lambda a: a[1])[0]

    def decide(self, state, questions, image=None, diagnostics=None):
        assert image is not None
        reads = None
        for path, rs in self.reads.items():
            if getattr(image, "_exp71_path", None) == path:
                reads = rs
        if reads is None:                                                # B1: a read made from the picture passed
            reads = [{"question": "What is the mass of each particle in sample A?", "box": [0, 0, image.size[0], image.size[1]],
                      "degenerate": False, "decoded": str(image.size[0]), "valid": True, "value": str(image.size[0])}]
        reads = json.loads(json.dumps(reads))
        if self.break_read:
            reads[0]["decoded"], reads[0]["valid"], reads[0]["value"] = "unreadable", False, "unreadable"
        table = [[*self.TEMPLATES[r["question"]], r["value"]] for r in reads]
        st = self.serialise(state if isinstance(state, str) else "", table)
        if diagnostics is not None:
            diagnostics.update(reads=reads, table=table, state=st, read_seconds=0.0)
        ans = {}
        for k, q in questions.items():
            opts = list(q["criteria"])
            g = self.gold.get((getattr(image, "_exp71_path", ""), q["instructions"], tuple(opts)), opts[0])
            pick = g if (self.policy == "gold" or (self.policy == "temp_only" and "temperature" in q["instructions"])) else opts[-1] if opts[-1] != g else opts[0]
            ans[k] = {"type": "choice", "probabilities": {o: (0.9 if o == pick else 0.1 / (len(opts) - 1)) for o in opts}}
        return {"answers": ans}


@pytest.fixture
def tagged_ask(monkeypatch):
    """e58.ask opens the actual picture bytes; tag the opened image with its path so the fake can tell it apart."""
    real = m.e58.ask

    def ask(reading, image_path, text, questions, data_root, pictures, diagnostics=None):
        import hashlib
        import io

        from PIL import Image
        raw = open(os.path.join(data_root, image_path), "rb").read()
        assert hashlib.sha256(raw).hexdigest() == pictures[image_path]
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img._exp71_path = image_path
        return reading.decide(text, questions, image=img, diagnostics=diagnostics)
    monkeypatch.setattr(m.e58, "ask", ask)
    return real


@pytest.fixture
def started(tmp_path):
    import time
    out = str(tmp_path / "exp71")
    m.stage_start(int(time.time()) + 1800, out=out)
    return out


@data
def test_start_binds_two_bundles_differing_only_by_the_corrector(started):
    st = m.check_start(started, code=False)
    assert set(st["bundles"]["on"]) - set(st["bundles"]["off"]) == {"corrector.pt"} and st["bundles"]["on"]["corrector.pt"] == m.CORR_SHA
    assert st["mode"]["name"] == m.e58.EXPLORE and st["mode"]["exp57_gates"]["fit"] is False and st["candidate_eligible"] is False
    with open(os.path.join(started, "bundle_on", "corrector.pt"), "ab") as fh:
        fh.write(b"x")
    with pytest.raises(SystemExit, match="bundle_on"):
        m.check_start(started, code=False)


@data
def test_stage_a_fidelity_and_b1_on_own_pictures(started, tagged_ask):
    m.stage_live(started, readings={c: FakeReading(c) for c in m.CONDS})
    f = m.stage_fidelity(started)
    assert f["reproduced"] and f["off"]["same_reads"] == f["off"]["reads"] == 1872 and f["on"]["consistent_states"] == 624
    m.stage_transfer(started, readings={"off": FakeReading("off", "temp_only"), "on": FakeReading("on", "gold")})
    res = m.report(started)
    assert all(v == "completed" for v in res["states"].values()) and res["fidelity"]["reproduced"]
    assert res["on_minus_off"]["accuracy_change"]["value"] > 0 and res["target"]["on"] is not None
    assert isinstance(res["off_minus_exp58_reader"], dict) and res["candidate_eligible"] is False
    reads = [json.loads(x) for x in open(os.path.join(started, "reads_sqa_temp_mism_on.jsonl"))]
    src = {r["id"]: r["image"] for r in m.jsonl(m.TRANSFER["sqa_temp_mism"])}
    true_src = {r["id"]: r["image"] for r in m.jsonl(m.TRANSFER["sqa_temp"])}
    assert all(x["image"] == src[x["questions"][0][0]] for x in reads)                        # the wrong picture itself was read
    assert set(src) == set(true_src) and all(src[i] != true_src[i] for i in src)                # same questions, other pictures
    assert all(x["image"] != true_src[x["questions"][0][0]] for x in reads)                    # never the true picture's read


@data
def test_unfaithful_read_stops_before_b1(started, tagged_ask):
    m.stage_live(started, readings={"off": FakeReading("off"), "on": FakeReading("on", break_read=True)})
    with pytest.raises(SystemExit, match="stage A not reproduced"):
        m.stage_fidelity(started)
    with pytest.raises(SystemExit, match="stage A not reproduced"):
        m.stage_transfer(started, readings={c: FakeReading(c) for c in m.CONDS})
    assert m.report(started)["branch"].startswith("stage A not reproduced")


@data
def test_answer_on_another_picture_is_refused(started, tagged_ask):
    m.stage_live(started, readings={c: FakeReading(c) for c in m.CONDS})
    m.stage_fidelity(started)
    m.stage_transfer(started, readings={c: FakeReading(c) for c in m.CONDS})
    st = m.check_start(started, code=False)
    f = os.path.join(started, "details_sqa_temp_mism_on.jsonl")
    lines = [json.loads(x) for x in open(f)]
    true_img = {r["id"]: r["image"] for r in m.jsonl(m.TRANSFER["sqa_temp"])}
    lines[1]["image_used"] = true_img.get(lines[1]["sample_id"], "other.png")                    # as if the true picture was used
    with open(f, "w") as fh:
        fh.write("".join(json.dumps(x) + "\n" for x in lines))
    m.register(started, "decide_sqa_temp_mism_on", {"details_sqa_temp_mism_on.jsonl": m.sha(f)})
    with pytest.raises(SystemExit, match="another picture"):
        m.b_records(started, "sqa_temp_mism", "on", st)


def test_branch_texts():
    base = {"fidelity": {"reproduced": True}, "transfer": {"off": {}, "on": {}}}
    d = lambda lo, tlo: {"accuracy_change": {"ci95": [lo, 0.2]}, "temp": {"accuracy_change": {"ci95": [tlo, 0.3]}}}  # noqa: E731
    assert m.branch(dict(base, target={"on": True}, on_minus_off=d(0.01, 0.0))).startswith("with the corrector the v0.3 transfer target")
    assert "overall" in m.branch(dict(base, target={"on": False}, on_minus_off=d(0.01, -0.1)))
    assert "temperature only" in m.branch(dict(base, target={"on": False}, on_minus_off=d(-0.01, 0.02)))
    assert m.branch(dict(base, target={"on": False}, on_minus_off=d(-0.01, -0.02))).startswith("no measurable decision gain")
    assert m.branch({"fidelity": {"reproduced": False}}).startswith("stage A not reproduced")


STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"watch.py"*) shift 2; exec python3 "$@";;
  *"exp71.py start"*) mkdir -p runs/exp71; exit 0;;
  *"exp71.py fidelity"*) [ -n "${FAIL_FID:-}" ] && exit 4; exit 0;;
  *"exp71.py live"*|*"exp71.py transfer"*) exit 0;;
  *"exp71.py report"*) echo "report $*" > runs/exp71/report.txt; exit 0;;
esac
exit 9
"""


def chain(tmp_path, extra):
    import stat as st_
    import subprocess
    root = tmp_path / "repo"
    for rel in ("scripts/exp71/run.sh", "scripts/exp53/watch.py", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    bdir = tmp_path / "bin"
    bdir.mkdir()
    for name, body in {"uv": STUB, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
        (bdir / name).write_text(body)
        (bdir / name).chmod((bdir / name).stat().st_mode | st_.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (bdir, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "EXP71_LIMIT": "60",
           "EXP71_RESERVE": "4", **extra}
    p = subprocess.run(["bash", str(root / "scripts/exp71/run.sh")], env=env, capture_output=True, text=True, timeout=120)
    return root, p, (tmp_path / "calls").read_text().splitlines()


def test_run_order_failed_fidelity_and_limit(tmp_path):
    root, p, calls = chain(tmp_path / "a", {})
    assert p.returncode == 0, p.stdout + p.stderr
    first = lambda s: next(i for i, c in enumerate(calls) if s in c)  # noqa: E731
    assert first("exp71.py start") < first("exp71.py live") < first("exp71.py fidelity") < first("exp71.py transfer") < first("exp71.py report")
    root, p, calls = chain(tmp_path / "b", {"FAIL_FID": "1"})
    joined = "\n".join(calls)
    assert p.returncode == 4 and "exp71.py transfer" not in joined and "report --finalize" in joined
    s = open(os.path.join(ROOT, "scripts/exp71/run.sh")).read()
    assert "${EXP71_LIMIT:-1800}" in s and "${EXP71_RESERVE:-120}" in s
