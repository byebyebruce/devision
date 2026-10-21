# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalSubscript=false, reportOptionalMemberAccess=false
"""exp69 without a model: the corrector is c92's network with a picture-token padding mask (unpadded = c92's forward;
padding changes nothing); window tokens / labels / pixel boxes keep their contracts; data/exp69 follows its frozen rule
(the 8 jitter draws recomputed, labels = target within the window, reach, token caps); feature batches and a few
training steps from c92's state; only speed boxes are corrected; START binds and refuses changes; the report
recomputes c92's reproduction and the identical mass / distance reads (missing -> incomplete, different -> invalid) and
the gates; the run's order and limit."""
import importlib.util
import json
import os
import random
import shutil

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("exp69", os.path.join(HERE, "exp69.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
has = os.path.exists(os.path.join(ROOT, "data", "exp69", "MANIFEST.json")) and os.path.exists(os.path.join(ROOT, "runs", "exp64", "rl_tune_new.jsonl"))
data = pytest.mark.skipif(not has, reason="data/exp69 / EXP64's runs not present")


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def test_corrector_is_c92_with_a_padding_mask():
    import torch
    torch.manual_seed(0)
    net = m.sel.Selector(768, 1024).eval()
    x, g = torch.randn(2, 40, 768), torch.rand(2, 40, 4)
    w, wm = torch.randn(2, 7, 1024), torch.ones(2, 7, dtype=torch.bool)
    full = torch.ones(2, 40, dtype=torch.bool)
    with torch.no_grad():
        assert torch.allclose(m.corrector_forward(net, x, g, full, w, wm), net(x, g, w, wm), atol=1e-6)
        xp = torch.cat([x, torch.randn(2, 9, 768) * 50], 1)
        gp = torch.cat([g, torch.rand(2, 9, 4)], 1)
        pm = torch.cat([full, torch.zeros(2, 9, dtype=torch.bool)], 1)
        assert torch.allclose(m.corrector_forward(net, xp, gp, pm, w, wm), net(x, g, w, wm), atol=1e-5)


def test_window_tokens_labels_and_pixels():
    from devision.reading import region
    rng = random.Random(0)
    for _ in range(20):
        w, h = rng.randint(400, 650), rng.randint(300, 420)
        box = [rng.uniform(0, w / 2), rng.uniform(h / 2, h - 40), rng.uniform(100, w / 2), rng.uniform(20, 38)]
        win = m.window_of(box, w, h)
        wt = m.window_tokens(win, w, h)
        r = region([win[0], win[1], win[2] - win[0], win[3] - win[1]], w, h)
        keep = [i for i, v in enumerate(r["view"]) if v > 0]
        assert wt["index"] == [r["index"][i] for i in keep] and wt["geom"] == [r["geom"][i] for i in keep] and len(keep) > 0
        t = [win[0] + 5, win[1] + 3, win[2] - 7, win[3] - 4]
        px = m.to_pixels(m.b69.to_window(t, win), win, w, h)
        assert px == pytest.approx([t[0], t[1], t[2] - t[0], t[3] - t[1]], abs=1e-6)
        tiny = m.to_pixels([0.0, 1.0, 0.0, 0.0], win, w, h)                                   # degenerate -> 1 px, inside
        assert tiny[2] == pytest.approx(1.0) and tiny[3] == pytest.approx(1.0) and tiny[0] >= win[0] and tiny[1] + tiny[3] <= win[3] + 1e-9


@data
def test_data_follows_the_frozen_rule():
    man = json.load(open("data/exp69/MANIFEST.json"))
    tr = [json.loads(x) for x in open("data/exp69/train.jsonl")]
    assert len(tr) == man["samples"] == 7936 and man["dropped_empty_label"] == 0 and man["records"] == 992
    rng = random.Random(m.b69.SEED)
    for s in tr:                                                                       # records by key, 8 x (dy, then dx)
        dy, dx = rng.uniform(-m.b69.DY, m.b69.DY), rng.uniform(-m.b69.DX, m.b69.DX)
        assert (s["dy"], s["dx"]) == (dy, dx)
        assert s["window"] == m.b69.window(s["input"], s["width"], s["height"])
        t, win = s["target"], s["window"]
        assert s["label_px"] == [max(t[0], win[0]), max(t[1], win[1]), min(t[2], win[2]), min(t[3], win[3])]
        assert s["reached"] == m.b69.inside(t, win) and s["template"] in m.SPEED
    assert [s["record"] for s in tr] == sorted(s["record"] for s in tr)
    assert sum(s["reached"] for s in tr) == man["reached"]["all"] == 7902
    assert max(s["local_tokens"] for s in tr) == man["max_tokens_per_window"] == 320 and man["cache"]["float16_bytes"] < 4e8


def fake_features(tmp_path, n_rec=3, per=4):
    out = tmp_path / "f"
    out.mkdir()
    rng = np.random.default_rng(0)
    U = 30
    np.save(out / "union_tokens.npy", rng.standard_normal((n_rec * U, 768)).astype(np.float16))
    pos, geom, soff, srec = [], [], [0], []
    for r in range(n_rec):
        for _ in range(per):
            k = int(rng.integers(5, U))
            pos += sorted(rng.choice(U, k, replace=False).tolist())
            geom += rng.random((k, 4)).tolist()
            soff.append(len(pos))
            srec.append(r)
    np.savez(out / "samples.npz", rec_off=np.arange(n_rec + 1) * U, pos=np.asarray(pos), geom=np.asarray(geom, np.float32), sample_off=np.asarray(soff),
             sample_rec=np.asarray(srec), words=rng.standard_normal((2, 11, 1024)).astype(np.float32), wmask=np.ones((2, 11), bool),
             label=rng.random((n_rec * per, 4)).astype(np.float32) * 0.5 + 0.25, template=rng.integers(0, 2, n_rec * per))
    m.register(str(out), "features", {f: m.sha(str(out / f)) for f in ("union_tokens.npy", "samples.npz")})
    return str(out)


def test_batches_and_training_steps(tmp_path, monkeypatch):
    import torch
    d = m.Samples(fake_features(tmp_path))
    x, g, pm, w, wm, y = d.batch([0, 5, 7])
    z = d.z
    lens = [int(z["sample_off"][i + 1] - z["sample_off"][i]) for i in (0, 5, 7)]
    assert x.shape[1] == max(lens) and pm.sum(1).tolist() == lens and not pm[0, lens[0]:].any()
    a, b_ = int(z["sample_off"][5]), int(z["sample_off"][6])
    assert torch.equal(x[1, :b_ - a], torch.from_numpy(d.flat[int(z["rec_off"][z["sample_rec"][5]]) + z["pos"][a:b_]].astype("float32")))
    torch.manual_seed(1)
    init = m.sel.Selector(768, 1024).state_dict()
    monkeypatch.setattr(m, "BATCH", 4)
    m1, s1 = m.train_corrector(init, d, steps=5)
    m2, s2 = m.train_corrector(init, d, steps=5)
    assert s1 == s2 and all(torch.equal(p, q) for p, q in zip(m1.state_dict().values(), m2.state_dict().values()))
    assert any(not torch.equal(p, init[k]) for k, p in m1.state_dict().items()) and set(m1.state_dict()) == set(init)


def test_template_words_from_a_grad_tracking_embedding():
    """EXP69-RUN-01 failed here: Reading._words runs a trainable-looking embedding outside no_grad. A fake with a real
    nn.Embedding (requires_grad) reproduces that path; template_words must still return plain arrays."""
    import torch
    emb = torch.nn.Embedding(50, 1024)

    class R:
        def _words(self, texts):
            ids = torch.arange(len(texts) * 6).view(len(texts), 6) % 50
            return emb(ids).half(), torch.ones(len(texts), 6, dtype=torch.bool)
    raw, _ = R()._words(list(m.SPEED))
    assert raw.requires_grad or not torch.is_grad_enabled()
    with pytest.raises(RuntimeError, match="requires grad"):
        raw.float().cpu().numpy()                                                    # the original failure
    w, wm = m.template_words(R())
    assert isinstance(w, np.ndarray) and w.shape == (2, 6, 1024) and w.dtype == np.float32 and wm.dtype == bool and wm.all()


def test_only_speed_is_corrected():
    import torch

    class R:
        def _words(self, texts):
            return torch.randn(len(texts), 6, 1024), torch.ones(len(texts), 6, dtype=torch.bool)
    torch.manual_seed(0)
    corr = m.sel.Selector(768, 1024).eval()
    kv = torch.randn(13 * 256, 768)
    reads = [{"question": "What is the mass of each particle in sample A?", "box": [10.0, 300.0, 200.0, 30.0]},
             {"question": m.SPEED[0], "box": [10.0, 340.0, 230.0, 32.0]}]
    out = m.corrected_boxes(R(), corr, kv, 560, 405, reads)
    assert out[reads[0]["question"]] == reads[0]["box"]
    win = m.window_of(reads[1]["box"], 560, 405)
    b = out[m.SPEED[0]]
    assert b != reads[1]["box"] and b[0] >= win[0] - 1e-9 and b[0] + b[2] <= win[2] + 1e-9 and b[1] >= win[1] - 1e-9 and b[1] + b[3] <= win[3] + 1e-9


@pytest.fixture
def started(tmp_path):
    import time
    out = str(tmp_path / "exp69")
    m.stage_start(int(time.time()) + 1800, out=out)
    return out


@data
def test_start_binds_and_refuses(started, tmp_path, monkeypatch):
    m.check_start(started, "data", code=False)
    with pytest.raises(SystemExit, match="exists"):
        m.stage_start(0, out=started)
    d = tmp_path / "d69"
    shutil.copytree(m.D69, d)
    with open(d / "dev_targets.jsonl", "a") as fh:
        fh.write("\n")
    monkeypatch.setattr(m, "D69", str(d))
    with pytest.raises(SystemExit, match="data/exp69"):
        m.check_start(started, "", code=False)


def fake_reads(out, monkeypatch, drop=None, edit=None):
    """Read files without a model: c92 = corr = target = EXP64's registered c92 + head reads (so c92 reproduces and mass /
    distance are identical); written and registered by the runner's own writer; a fake verified corrector."""
    monkeypatch.setattr(m, "corrector_sha", lambda out_: "9" * 64)
    m.write_json(os.path.join(out, "train.json"), {"corrector_sha256": "9" * 64})
    for g, names in m.GROUPS.items():
        name, _ = m.REF[g]
        items = [json.loads(x) for x in open(os.path.join(m.E64, name + ".jsonl"))][1:]
        for n in names:
            if n == drop:
                continue
            xs = json.loads(json.dumps(items))
            if n == edit:
                r = next(r for r in xs[0]["reads"] if r["question"] not in m.SPEED)
                r["decoded"], r["valid"], r["value"] = "unreadable", False, "unreadable"
            m.write_reads(out, n, xs, "9" * 64)


@data
def test_report_valid_and_gates(started, monkeypatch):
    fake_reads(started, monkeypatch)
    res = m.report(started)
    v = res["validity"]
    assert v["state"] == "valid" and all(g["reproduced"] and g["others_differ"] == 0 for g in v["groups"].values())
    assert res["gates"] == {"old_layout": True, "exp61_layout": True, "dev24_speed": False}       # corr == c92: 16 / 24, no gain
    assert res["dev24"]["corr"]["speed"] == 16 and res["branch"].startswith("generated layouts hold")
    assert res["localisation"]["rl_tune/corr"] == res["localisation"]["rl_tune/c92"] and res["candidate_eligible"] is False


@data
@pytest.mark.parametrize("case,state", [("drop", "incomplete"), ("edit", "invalid")])
def test_report_missing_or_changed_other_fields(started, monkeypatch, case, state):
    fake_reads(started, monkeypatch, drop="tune_V_target" if case == "drop" else None, edit="rl_tune_corr" if case == "edit" else None)
    res = m.report(started)
    assert res["validity"]["state"] == state and res["branch"].startswith(state)


def changed_exp49(tmp_path, field):
    """data/exp49/questions.jsonl with one rl_tune speed question's target (a legal other number) or cluster fields
    (family / case) changed, key and template untouched."""
    lines = open(m.e61.QUESTIONS).read().splitlines()
    for i, x in enumerate(lines):
        q = json.loads(x)
        if q["set"] == "rl_tune" and q["field"] == "speed":
            if field == "target":
                q["target"] = str(int(q["target"]) + 10)
            else:
                q["family"], q["case"] = q["family"] + "x", q["case"] + "x"
            lines[i] = json.dumps(q)
            break
    p = tmp_path / ("q49_%s.jsonl" % field)
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def swapped_gold(tmp_path):
    """data/exp28/fields.jsonl with one gold moved to another option whose number is also in the transcription."""
    tr = json.load(open(m.e59.TRANSCRIBED))
    lines = open(m.e59.FIELDS).read().splitlines()
    for i, x in enumerate(lines):
        f = json.loads(x)
        rid = f["id"].split(":", 1)[1].rsplit(":", 2)[0]
        pr = f["gold"]["q"]["probabilities"]
        g = max(pr, key=pr.get)
        other = [o for o in pr if o != g and o in tr.get(rid, "")]
        if other:
            pr[g], pr[other[0]] = pr[other[0]], pr[g]
            lines[i] = json.dumps(f)
            out = tmp_path / "fields.jsonl"
            out.write_text("\n".join(lines) + "\n")
            return str(out)
    raise AssertionError("no swappable gold")


@data
@pytest.mark.parametrize("which", ["target", "cluster", "gold", "transcribed"])
def test_scoring_sources_bound_to_frozen_runs(started, tmp_path, monkeypatch, which):
    """critic EXP69-R2-01: a legal-looking change of a target, a cluster field or a dev24 gold / transcription (keys and
    templates unchanged) is refused at a first start and makes an existing run's report invalid, with no route."""
    fake_reads(started, monkeypatch)
    assert m.bound_sources() == json.load(open(os.path.join(started, "START.json")))["sources"]
    if which in ("target", "cluster"):
        monkeypatch.setattr(m.e61, "QUESTIONS", changed_exp49(tmp_path, which))
    elif which == "gold":
        monkeypatch.setattr(m.e59, "FIELDS", swapped_gold(tmp_path))
        m.e59.gold_numbers(m.e59.items())                                   # passes exp59's own check ...
    else:
        p = tmp_path / "tr.json"
        p.write_text(open(m.e59.TRANSCRIBED).read().replace("1,300", "1,330", 1))
        monkeypatch.setattr(m.e59, "TRANSCRIBED", str(p))
    with pytest.raises(SystemExit, match="scoring source"):
        m.stage_start(0, out=str(tmp_path / "fresh"))                         # ... not a first start
    res = m.report(started)                                                   # ... nor an existing run's report
    assert res["branch"].startswith("invalid") and "gates" not in res and "dev24" not in res


def test_gates_and_branch():
    f = lambda e, x: {"exact": e, **{"exact:" + k: {"value": x} for k in ("mass", "speed", "distance")}}  # noqa: E731
    g = m.gates({"rl_tune": {"corr": f(0.93, 0.9), "c92": f(0.94, 0.92)}, "tune_V": {"corr": f(0.9, 0.85), "c92": f(0.94, 0.92)},
                 "dev24": {"corr": {"speed": 19}, "speed_corr_minus_c92": {"ci95": [0.04, 0.3]}}})
    assert g == {"old_layout": True, "exp61_layout": False, "dev24_speed": True}
    ok = {"old_layout": True, "exp61_layout": True, "dev24_speed": True}
    assert m.branch(ok).startswith("the speed corrector passes") and m.branch(dict(ok, old_layout=False)).startswith("old-layout")
    assert m.branch(dict(ok, dev24_speed=False)).startswith("generated layouts hold") and m.branch(dict(ok, dev24_speed="incomplete")).startswith("incomplete")


STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"watch.py"*) shift 2; exec python3 "$@";;
  *"exp69.py start"*) mkdir -p runs/exp69-r2; exit 0;;
  *"exp69.py train"*) [ -n "${FAIL_TRAIN:-}" ] && exit 6; exit 0;;
  *"exp69.py features"*|*"exp69.py reads"*) exit 0;;
  *"exp69.py report"*) echo "report $*" > runs/exp69-r2/report.txt; exit 0;;
esac
exit 9
"""
OLD = {"runs/exp69/START.json": "old start", "runs/exp69/report.json": "old report", "runs/exp69/union_tokens.npy": "old tokens",
       "runs/exp69.started": "old marker", "runs/exp69.out": "old log", "runs/exp69-status.txt": "old status"}


def chain(tmp_path, extra, pre=None):
    import stat as st_
    import subprocess
    root = tmp_path / "repo"
    for rel in ("scripts/exp69/run.sh", "scripts/exp53/watch.py", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    for rel, body in {**OLD, **(pre or {})}.items():                    # the failed first attempt (and any pre-existing file)
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(body)
    bdir = tmp_path / "bin"
    bdir.mkdir()
    for name, body in {"uv": STUB, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
        (bdir / name).write_text(body)
        (bdir / name).chmod((bdir / name).stat().st_mode | st_.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (bdir, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "EXP69_LIMIT": "60",
           "EXP69_RESERVE": "4", **extra}
    p = subprocess.run(["bash", str(root / "scripts/exp69/run.sh")], env=env, capture_output=True, text=True, timeout=120)
    calls = (tmp_path / "calls").read_text().splitlines() if (tmp_path / "calls").exists() else []
    return root, p, calls


def old_unchanged(root) -> bool:
    return all((root / rel).read_text() == body for rel, body in OLD.items())


def test_retry_chain_writes_only_its_own_paths(tmp_path):
    root, p, calls = chain(tmp_path / "a", {})
    assert p.returncode == 0, p.stdout + p.stderr
    first = lambda s: next(i for i, c in enumerate(calls) if s in c)  # noqa: E731
    assert first("exp69.py start") < first("exp69.py features") < first("exp69.py train") < first("exp69.py reads") < first("exp69.py report")
    assert all("runs/exp69-r2/progress-" in c for c in calls if "watch.py" in c)
    assert (root / "runs/exp69-r2.started").exists() and (root / "runs/exp69-r2-status.txt").exists() and (root / "runs/exp69-r2/report.txt").exists()
    assert old_unchanged(root)
    root, p, calls = chain(tmp_path / "b", {"FAIL_TRAIN": "1"})                     # failure -> finalize, in the new paths
    joined = "\n".join(calls)
    assert p.returncode == 6 and "exp69.py reads" not in joined and "report --finalize" in joined
    assert "finalize" in (root / "runs/exp69-r2/report.txt").read_text() and "train failed" in (root / "runs/exp69-r2-status.txt").read_text()
    assert old_unchanged(root)
    root, p, calls = chain(tmp_path / "c", {}, pre={"runs/exp69-r2.started": "x"})       # the retry also starts only once
    assert p.returncode == 1 and not calls and "already started once" in p.stdout and old_unchanged(root)
    s = open(os.path.join(ROOT, "scripts/exp69/run.sh")).read()
    assert "${EXP69_LIMIT:-1800}" in s and "${EXP69_RESERVE:-120}" in s and "runs/exp69.started" not in s.replace("runs/exp69-r2.started", "")
    assert m.OUT == "runs/exp69-r2"


@data
def test_retry_binds_the_failed_attempt(started, tmp_path, monkeypatch):
    st = json.load(open(os.path.join(started, "START.json")))
    assert st["retry_of"]["files"] == m.RETRY_OF and "ef722f7" in st["retry_of"]["reason"]
    copy = {}
    for f in m.RETRY_OF:
        dst = tmp_path / "old" / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dst)
        copy[str(dst)] = m.RETRY_OF[f]
    monkeypatch.setattr(m, "RETRY_OF", copy)
    m.bound_retry()
    with open(next(iter(copy)), "a") as fh:
        fh.write(" ")
    with pytest.raises(SystemExit, match="failure evidence changed"):
        m.bound_retry()
