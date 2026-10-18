# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalSubscript=false
"""exp66 without a model: the moved-column renderer is the generator at shift 0 (pixels, rows, bands); every variant's
layout is its family's (shared by both views, recomputed from the hash), inside its physical limits, labelled by the
drawn rows; a cut glyph or an ink-removing crop is refused; train_C / tune_C / earlier families are isolated; every
train_C question has its parent; the audit's ink measures on a drawn picture."""
import importlib.util
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b = _load("exp66_build", "build.py")
au = _load("exp66_audit", "audit.py")
rl, b45 = b.rl, b.b45
has = os.path.exists(os.path.join(ROOT, "data", "exp66", "MANIFEST.json"))
data = pytest.mark.skipif(not has, reason="data/exp66 not built")


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def scenes(n=12):
    rng = random.Random(5)
    return [rl.temp_scene(rng) for _ in range(n)]


def test_shift_zero_is_the_generator():
    for s in scenes():
        assert np.array_equal(np.asarray(b.temp_render_at(s, 0)), np.asarray(rl.temp_render(s)))
        assert b.temp_rows_at(s, 0) == b45.temp_rows(s)
        for o in "AB":
            for f in ("mass", "speed"):
                band, text, _ = b45.original_box("temp", s, o, f)
                assert b.temp_band(s, 0, o, f) == pytest.approx(band) and b.temp_texts(s, 0)[o][f] == pytest.approx(text)


def test_shift_moves_the_column_and_its_labels():
    s = scenes(1)[0]
    p = {"top": 0, "shift": 10, "bottom_margin": 6}
    img, boxes, _ = b.render("temp", s, p)
    _, base, _ = b.render("temp", s, dict(p, shift=0))
    left = "A" if s.get("a_left", True) else "B"
    assert boxes[("mass", left)][1][0] == pytest.approx(base[("mass", left)][1][0] - 10)       # the text moved out by 10
    a = np.asarray(img)
    t = [int(round(v)) for v in boxes[("mass", left)][1]]
    assert b.text_mask(a[t[1]:t[3], t[0]:t[2]]).any()                                          # and is drawn there


def test_cut_text_and_ink_crop_are_refused():
    s = scenes(1)[0]
    sm = b.max_shift([s])
    with pytest.raises(SystemExit):
        b.render("temp", s, {"top": 0, "shift": sm + 40, "bottom_margin": 0})
    first, _ = b.ink_rows(rl.temp_render(s))
    with pytest.raises(SystemExit, match="crop"):
        b.render("temp", s, {"top": first + 1, "shift": 0, "bottom_margin": 0})
    img, boxes, _ = b.render("temp", s, {"top": 0, "shift": sm, "bottom_margin": 0})                # the limit itself is fine
    assert all(t[0] >= 0 and t[2] <= img.size[0] for _, t in boxes.values())


@data
def test_variants():
    qs = [json.loads(x) for x in open("data/exp66/questions.jsonl")]
    m = json.load(open("data/exp66/MANIFEST.json"))
    assert m["questions"] == {"train_C": 1200, "tune_C": 1800, "train_C_parent": 1200}
    by = defaultdict(list)
    for q in qs:
        by[(q["set"], q["family"])].append(q)
    sc = b.b61.scenes({f for _, f in by})
    for (st, fam), rows in by.items():
        if st == "train_C_parent":
            continue
        form = sc[(fam, "v0")][0]
        p = b.params(fam, form, [sc[(fam, "v0")][1], sc[(fam, "v1")][1]])
        assert all(q["layout_params"] == p for q in rows)                            # the family's, shared by its views
        assert len({(q["width"], q["height"]) for q in rows}) == 1
        if form == "temp":
            assert -p["shift_in_max"] <= p["shift"] <= p["shift_max"]
        for q in rows:
            t = q["text_box"]
            assert 0 <= t[0] and t[2] <= q["width"] and 0 <= t[1] and t[3] <= q["height"]
            assert q["target"] == b.target_of(form, sc[(fam, q["view"])][1], q["field"], q["object"])
    fams = {st: {f for s, f in by if s == st} for st in ("train_C", "tune_C")}
    used = {json.loads(x)["family"] for x in open("data/exp49/questions.jsonl")}
    used61 = {json.loads(x)["family"] for x in open("data/exp61/questions.jsonl") if json.loads(x)["set"] == "train_V"}
    assert not (fams["train_C"] & fams["tune_C"]) and not (fams["train_C"] & (used | used61))
    cases = {st: {q["case"] for q in qs if q["set"] == st} for st in ("train_C", "tune_C")}
    assert not cases["train_C"] & cases["tune_C"]
    keys = {q["key"] for q in qs if q["set"] == "train_C"}
    pairs = Counter(q["pair"] for q in qs if q["set"] == "train_C_parent")
    assert set(pairs) == keys and set(pairs.values()) == {1}
    assert Counter(sc[(f, "v0")][0] for f in fams["train_C"]) == {"temp": 100, "force": 100}


def test_audit_measures_drawn_ink(tmp_path):
    s = scenes(1)[0]
    img, boxes, _ = b.render("temp", s, {"top": 0, "shift": 0, "bottom_margin": 3})
    p = str(tmp_path / "x.png")
    img.save(p)
    left = "A" if s.get("a_left", True) else "B"
    box, text = boxes[("speed" if not s.get("speed_first") else "mass", left)]
    m = au.measure(p, img.size[0], img.size[1], box, au.xyxy(box))
    assert m["bottom_margin"] == 3 and m["holds"] and abs(m["col_x"] - 0.25) < 0.01
    assert m["side_margin"] == pytest.approx(m["ink"][0]) and 0.3 < m["rel_cy"] < 0.7


# ------------------------------------------------------------------ runner (no model)

def _runner():
    return _load("exp66_runner", "exp66.py")


@data
def test_plan_is_exp62s_generator_with_the_bigger_pool():
    p = _load("exp66_plan_t", "plan.py")
    par = p.jsonl("data/exp62/parents.jsonl")
    rep = sorted(q["key"] for q in p.jsonl("data/exp49/questions.jsonl") if q["set"] in ("fit_O", "fit_L"))
    assert p.plan(sorted(((q["key"], q["pair"]) for q in par), key=lambda x: x[1]), rep) == p.jsonl("data/exp62/plan.jsonl")
    pairs, replay, rec = p.pools("data")
    batches = p.jsonl("data/exp66/plan.jsonl")
    assert batches == p.plan(pairs, replay)
    st = p.plan_stats(batches, pairs, replay, rec)
    assert st["split_pairs"] == 0 and st["pair_exposure_range"] == [20, 20] and st["batch_sizes"] == {"64": 2000}
    assert st["pairs"] == {"exp62": 1200, "exp66": 1200} and st["every_row_seen"]
    assert json.load(open("data/exp66/PLAN.json"))["plan_sha256"] == p.sha("data/exp66/plan.jsonl")


@pytest.fixture
def started(tmp_path):
    import time
    m = _runner()
    out = str(tmp_path / "exp66")
    m.stage_start(int(time.time()) + 600, out=out)
    return m, out


@data
def test_start_binds_and_refuses_changes(started, tmp_path, monkeypatch):
    m, out = started
    m.check_start(out, "data", code=False)
    with pytest.raises(SystemExit, match="exists"):
        m.stage_start(0, out=out)
    plan = tmp_path / "plan.jsonl"
    plan.write_text(open(m.PLAN).read().replace("train_C|", "train_C |", 1))
    monkeypatch.setattr(m, "PLAN", str(plan))
    with pytest.raises(SystemExit, match="plan"):
        m.check_start(out, "", code=False)
    monkeypatch.setattr(m, "M66_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="MANIFEST"):
        m.bound_d66("")


@data
def test_inputs_cover_the_plan_with_same_text_words(started):
    import torch
    m, out = started
    pics = m.train_pictures()
    np.savez(os.path.join(out, "exp66_global.npz"), patches=np.zeros((len(pics), 256, 768), np.float16),
             geom=np.zeros((len(pics), 256, 4), np.float32), pictures=np.array(pics))
    m.register(out, "global", {"exp66_global.npz": m.sha(os.path.join(out, "exp66_global.npz"))})
    inp, recs, at = m.all_inputs(out)
    assert all(k in at for b in m.jsonl(m.PLAN) for k in b)
    new = [i for i, q in enumerate(recs) if q["set"] in ("train_C", "train_C_parent")]
    assert len(new) == 2400
    for i, j in zip(new, m.word_rows(recs, [recs[i]["text"] for i in new])):
        assert recs[j]["text"] == recs[i]["text"] and torch.equal(inp.words[i], inp.words[j]) and torch.equal(inp.mask[i], inp.mask[j])
    assert inp.patches.shape[0] == max(inp.pic) + 1


def fake_reads(m, out, monkeypatch, drop=None, tamper=None, edit_c92=None):
    """A run's read files without a model: c92 / new = EXP64's registered c92 + head reads (so c92 reproduces), tune_C
    from its targets; written and registered by the runner's own writer; a fake verified selector."""
    from devision.reading import fmt
    monkeypatch.setattr(m, "new_selector_sha", lambda out_: "5" * 64)
    m.write_json(os.path.join(out, "train.json"), {"selector_sha256": "5" * 64})
    items = {}
    for g, (name, _) in m.REF.items():
        items[g] = m.jsonl(os.path.join(m.E64, name + ".jsonl"))[1:]
    cby = m.tune_c_by_picture()
    tc = []
    for p, qs in sorted(cby.items()):
        by = {q["text"]: q for q in qs}
        tc.append({"picture": p, "reads": [{"question": t, "box": by[t]["box"], "degenerate": False, "decoded": by[t]["target"], "valid": True,
                                            "value": fmt(by[t]["target"])} for t in m.e61.templates_of(qs)]})
    items["tune_C"] = tc
    for g, names in m.GROUPS.items():
        for n in names:
            if n == drop:
                continue
            xs = json.loads(json.dumps(items[g]))
            if n == edit_c92:
                r = xs[0]["reads"][0]
                r["decoded"], r["valid"], r["value"] = "unreadable", False, "unreadable"
            m.write_reads(out, n, xs, "5" * 64)
    if tamper:
        with open(os.path.join(out, tamper + ".jsonl"), "a") as fh:
            fh.write("\n")


@data
def test_report_valid_run_and_gates(started, monkeypatch):
    m, out = started
    fake_reads(m, out, monkeypatch)
    res = m.report(out)
    assert res["validity"]["state"] == "valid" and all(v["reproduced"] for v in res["validity"]["groups"].values())
    assert res["gates"]["old_layout"] is True and res["gates"]["exp61_layout"] is True                 # new == c92
    assert res["dev24"]["new"]["speed"] == 16 and res["gates"]["dev24_speed"] is False                 # EXP64's 16 / 24
    assert res["branch"].startswith("generated layouts hold") and res["candidate_eligible"] is False
    assert res["tune_C"]["new"]["exact"] == 1.0 and res["tune_C_boxes_hold"]["new"]["speed"] == 1.0


@data
@pytest.mark.parametrize("case,state", [("drop", "incomplete"), ("tamper", "invalid"), ("c92", "invalid")])
def test_report_missing_tampered_or_unfaithful(started, monkeypatch, case, state):
    m, out = started
    fake_reads(m, out, monkeypatch, drop="tune_C_label" if case == "drop" else None, tamper="tune_V_new" if case == "tamper" else None,
               edit_c92="rl_tune_c92" if case == "c92" else None)
    res = m.report(out)
    assert res["validity"]["state"] == state and res["branch"].startswith(state) and "no verdict" in res["branch"]


def test_gates_and_branch():
    m = _runner()
    f = lambda e, x: {"exact": e, **{"exact:" + k: {"value": x} for k in ("mass", "speed", "distance")}}  # noqa: E731
    assert m.no_worse(f(0.92, 0.9), f(0.94, 0.92)) is True and m.no_worse(f(0.90, 0.9), f(0.94, 0.92)) is False
    assert m.no_worse(f(0.95, None), f(0.94, 0.92)) == "incomplete"
    ok = {"old_layout": True, "exp61_layout": True, "dev24_speed": True, "dev24_distance": True}
    assert m.branch(ok).startswith("the selector recipe passes")
    assert m.branch(dict(ok, old_layout=False)).startswith("old-layout") and m.branch(dict(ok, exp61_layout=False)).startswith("EXP61-layout")
    assert m.branch(dict(ok, dev24_distance=False, dev24_speed=False)).startswith("the real-picture distance")
    assert m.branch(dict(ok, dev24_speed=False)).startswith("generated layouts hold")
    assert m.branch(dict(ok, dev24_speed="incomplete")).startswith("incomplete")
    g = m.gates({"rl_tune": {"new": f(0.95, 0.9), "c92": f(0.94, 0.92)}, "tune_V": {}, "dev24": {"new": {"speed": 18, "distance": 19}}})
    assert g == {"old_layout": True, "exp61_layout": "incomplete", "dev24_speed": True, "dev24_distance": False}


STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"watch.py"*) shift 2; exec python3 "$@";;
  *"exp66.py start"*) mkdir -p runs/exp66; exit 0;;
  *"exp66.py global"*) exit 0;;
  *"exp66.py train"*) [ -n "${FAIL_TRAIN:-}" ] && exit 6; exit 0;;
  *"exp66.py reads"*) exit 0;;
  *"exp66.py report"*) echo report > runs/exp66/report.txt; exit 0;;
esac
exit 9
"""


def chain(tmp_path, extra):
    import shutil
    import stat as st_
    import subprocess
    root = tmp_path / "repo"
    for rel in ("scripts/exp66/run.sh", "scripts/exp53/watch.py", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    bdir = tmp_path / "bin"
    bdir.mkdir()
    for name, body in {"uv": STUB, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
        (bdir / name).write_text(body)
        (bdir / name).chmod((bdir / name).stat().st_mode | st_.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (bdir, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "EXP66_LIMIT": "60",
           "EXP66_RESERVE": "4", **extra}
    p = subprocess.run(["bash", str(root / "scripts/exp66/run.sh")], env=env, capture_output=True, text=True, timeout=120)
    return root, p, (tmp_path / "calls").read_text().splitlines()


def test_run_order_failed_stage_and_limit(tmp_path):
    root, p, calls = chain(tmp_path / "a", {})
    assert p.returncode == 0, p.stdout + p.stderr
    first = lambda s: next(i for i, c in enumerate(calls) if s in c)  # noqa: E731
    assert first("exp66.py start") < first("exp66.py global") < first("exp66.py train") < first("exp66.py reads") < first("exp66.py report")
    assert (root / "runs/exp66.started").exists()
    root, p, calls = chain(tmp_path / "b", {"FAIL_TRAIN": "1"})
    joined = "\n".join(calls)
    assert p.returncode == 6 and "exp66.py reads" not in joined and "report --finalize" in joined
    s = open(os.path.join(ROOT, "scripts/exp66/run.sh")).read()
    assert "${EXP66_LIMIT:-2400}" in s and "${EXP66_RESERVE:-120}" in s
