# pyright: reportOptionalIterable=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
"""scripts/confirm/confirm_run.py (critic V04-K-CONFIRM-PREP-02) without a model and without the confirmation set: a fake
120-picture set (tiny pictures, 480 questions, the fixed wrong pictures from confirm.mismatched) and fake models through
the formal evaluation path: all five conditions run once each with the fixed wrong pictures (never re-drawn), the report
binds and checks every set and gives the pre-written outcome on the 120 ids; a missing condition is incomplete, a changed
or re-pointed record invalid; start refuses an unfrozen combination or another v0.3; a stage cannot run twice; run.sh's
order, required inputs, failure handling, limit and reserve."""
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys

import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Fake:
    """Answers a question right when it is shown its own picture (policy "right"), right on even pictures only ("half")."""

    def __init__(self, truth, policy, reading=None):
        self.truth, self.policy = truth, policy
        self.reading_status = reading or {"mode": "none"}

    def decide(self, state=None, questions=None, image=None, diagnostics=None):
        from devision.infer import picture_identity, resolve_request
        _, pic = resolve_request(state, questions, image)
        ident = picture_identity(pic)
        assert ident is not None
        if diagnostics is not None:
            diagnostics.update(route={k: "ordinary" for k in questions}, picture=ident)
        ans = {}
        for k, q in questions.items():
            opts = list(q["criteria"])
            hit = self.truth.get((ident["pixels_sha256"], q["instructions"], tuple(sorted(opts))))
            right = hit is not None and (self.policy == "right" or hit[1] % 2 == 0)
            pick = hit[0] if right else next(o for o in opts if hit is None or o != hit[0])
            probs = {o: (0.8 if o == pick else 0.2 / (len(opts) - 1)) for o in opts}
            ans[k] = {"type": "choice", "choice": pick, "probabilities": probs, "confidence": 0.5}
        return {"model": "fake", "answers": ans, "usage": {"input_tokens": 0, "output_tokens": 0}}


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "build", raising=False)
    m = load("confirm_run_t", "scripts/confirm/confirm_run.py")
    monkeypatch.chdir(tmp_path)
    os.makedirs("runs")
    v03 = tmp_path / "v03"
    v03.mkdir()
    (v03 / "model.safetensors").write_bytes(b"v03 weights")
    (v03 / "config.json").write_text("{}")
    cand = tmp_path / "cand"
    cand.mkdir()
    (cand / "model.safetensors").write_bytes(b"cand weights")
    (cand / "config.json").write_text("{}")
    ident = tmp_path / "IDENTITY.json"
    ident.write_text('{"frozen": true}')
    monkeypatch.setattr(m, "V03", str(v03))
    monkeypatch.setattr(m, "V03_SHA", m.weights(str(v03)))
    monkeypatch.setattr(m, "ROOT", ROOT)
    monkeypatch.setattr(m.cf, "freeze_problems", lambda *a, **k: [])
    truth = {}
    from devision.infer import picture_identity

    def fake_set(st):
        out = m.DATA
        os.makedirs(os.path.join(out, "images"))
        rows = []
        for i in range(120):
            form = "temp" if i < 60 else "force"
            img = Image.new("RGB", (30, 20), (i, 255 - i, (7 * i) % 256))
            img.save(os.path.join(out, "images", "%03d.png" % i))
            nq = 5 if form == "temp" else 3
            qs = {"q%d" % j: {"type": "choice", "instructions": "question %d of %s" % (j, form), "criteria": {"x": None, "y": None, "z": None}} for j in range(nq)}
            g = "xyz"[i % 3]                                     # golds differ between pictures, so a wrong picture misleads
            gold = {"q%d" % j: {"probabilities": {o: float(o == g) for o in "xyz"}} for j in range(nq)}
            r = {"id": "v04-confirm:%s%03d" % (form, i), "kind": form, "axis": "a", "image_id": "c%03d" % i, "image": "v04_k_confirm/images/%03d.png" % i,
                 "state_text": "", "questions": qs, "gold": gold}
            rows.append(r)
            pid = picture_identity(Image.open(os.path.join(out, "images", "%03d.png" % i)).convert("RGB"))
            assert pid is not None
            for q in qs.values():
                truth[(pid["pixels_sha256"], q["instructions"], tuple(sorted(q["criteria"])))] = (g, i)
        mism = m.cf.mismatched(rows, 5)
        open(os.path.join(out, "confirm.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in rows))
        open(os.path.join(out, "confirm_mism.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in mism))
        man = {"pictures": 120, "classes": {}, "questions": 480}
        json.dump(man, open(os.path.join(out, "MANIFEST.json"), "w"))
        return man
    monkeypatch.setattr(m, "generate_set", fake_set)
    models = {"candidate": Fake(truth, "right", {"mode": "on", "k_rule": True}), "v0.3": Fake(truth, "half")}
    monkeypatch.setattr(m, "load_model", lambda who, st, device: models[who])
    return m, str(ident), str(cand)


def run(m, ident, cand, skip=()):
    m.stage_start(ident, cand, 0)
    m.stage_generate()
    for name in m.CONDITIONS:
        if name not in skip:
            m.stage_model(name)
    return m.report()


def test_the_whole_run_on_a_fake_set(ws):
    m, ident, cand = ws
    res = run(m, ident, cand)
    assert all(v == "completed" for v in res["states"].values()), res["states"]
    assert res["primary"]["branch"] == "synthetic confirmation passed" and res["branch"].startswith("synthetic confirmation passed")
    assert res["primary"]["candidate_minus_v03"]["value"] == pytest.approx(0.5)
    mism = [json.loads(x) for x in open(os.path.join(m.OUT, "details_cand_mism.jsonl"))][1:]
    rows = {r["id"]: r["image"] for r in map(json.loads, open(os.path.join(m.DATA, "confirm_mism.jsonl")))}
    assert all(x["image_used"] == rows[x["sample_id"]] for x in mism)               # the fixed wrong pictures, not re-drawn
    rev = [json.loads(x) for x in open(os.path.join(m.OUT, "details_cand_rev.jsonl"))]
    assert rev[0]["identity"]["control"] == "reversed" and len(rev) - 1 == 480
    assert set(res["report_only"]) == set(m.CONDITIONS)


def test_a_missing_condition_is_incomplete(ws):
    m, ident, cand = ws
    res = run(m, ident, cand, skip=("v03_mism",))
    assert res["branch"].startswith("incomplete") and res["states"]["v03_mism"] == "missing"


@pytest.mark.parametrize("damage", ["changed", "other_picture"])
def test_changed_or_repointed_records_are_invalid(ws, damage):
    m, ident, cand = ws
    run(m, ident, cand)
    f = os.path.join(m.OUT, "details_cand_true.jsonl")
    lines = [json.loads(x) for x in open(f)]
    if damage == "changed":
        lines[3]["correct"] = not lines[3]["correct"]
        open(f, "w").write("".join(json.dumps(x) + "\n" for x in lines))
    else:                                                        # answered on another picture, registration updated
        lines[3]["image_used"] = lines[200]["image_used"]
        open(f, "w").write("".join(json.dumps(x) + "\n" for x in lines))
        reg = json.load(open(os.path.join(m.OUT, "registry.json")))
        reg["cand_true"]["details_cand_true.jsonl"] = m.sha(f)
        json.dump(reg, open(os.path.join(m.OUT, "registry.json"), "w"))
    res = m.report()
    assert res["branch"].startswith("invalid") and res["states"]["cand_true"].startswith("invalid")


def test_start_refusals_and_one_run_only(ws, monkeypatch):
    m, ident, cand = ws
    monkeypatch.setattr(m.cf, "freeze_problems", lambda *a, **k: ["not frozen"])
    with pytest.raises(SystemExit, match="not frozen"):
        m.stage_start(ident, cand, 0)
    monkeypatch.setattr(m.cf, "freeze_problems", lambda *a, **k: [])
    monkeypatch.setattr(m, "V03_SHA", {"model.safetensors": "0" * 64, "config.json": "0" * 64})
    with pytest.raises(SystemExit, match="v0.3"):
        m.stage_start(ident, cand, 0)
    monkeypatch.setattr(m, "V03_SHA", m.weights(m.V03))
    m.stage_start(ident, cand, 0)
    m.stage_generate()
    m.stage_model("cand_true")
    with pytest.raises(SystemExit, match="one run only"):
        m.stage_model("cand_true")


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
