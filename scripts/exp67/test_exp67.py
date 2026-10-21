# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalSubscript=false, reportOptionalMemberAccess=false
"""exp67 without a model: same-source conditions are region()'s own output; mixed conditions keep index / view / rel / mask
and change only geom, by the formula, unclipped and finite; the head's tensor build equals EXP64's read path; START
binds the boxes; the report recomputes AA / OO's reproduction of EXP64 (missing -> incomplete, wrong -> invalid); the
effects and the pre-written explanation; the run's order and bounded finalize."""
import importlib.util
import json
import math
import os
import random

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("exp67", os.path.join(HERE, "exp67.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
has = os.path.exists(os.path.join(ROOT, "runs", "exp64", "dev24_auto_new.jsonl"))
data = pytest.mark.skipif(not has, reason="EXP64's runs not present")


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def boxes_pair(rng, w, h):
    def one():
        x, y = rng.uniform(0, w * 0.6), rng.uniform(h * 0.5, h * 0.9)
        return [x, y, rng.uniform(20, w - x), rng.uniform(8, h - y)]
    return {"A": one(), "O": one()}


def test_geometry_contract():
    from devision.reading import region
    rng = random.Random(0)
    for _ in range(20):
        w, h = rng.randint(400, 700), rng.randint(120, 420)
        b = boxes_pair(rng, w, h)
        regs = {s: region(b[s], w, h) for s in "AO"}
        for s in "AO":
            assert m.geom_against(regs[s], b[s], w, h) == regs[s]["geom"]                      # bit-identical on its own box
            assert m.condition(regs, b, s + s, w, h)["geom"] == regs[s]["geom"]
        for c in ("AO", "OA"):
            s, g = c
            x = m.condition(regs, b, c, w, h)
            assert x["index"] == regs[s]["index"] and x["view"] == regs[s]["view"] and x["rel"] == [list(r) for r in regs[s]["rel"]]
            bx, by, bw, bh = b[g]
            sx, sy, sw, sh = b[s]
            for gs, gg in zip(regs[s]["geom"], x["geom"]):                                    # same absolute patch, other box
                cx, cy = gs[0] * sw + sx, gs[1] * sh + sy
                assert gg[0] == pytest.approx((cx - bx) / bw) and gg[1] == pytest.approx((cy - by) / bh)
                assert gg[2] == pytest.approx(gs[2] * sw / bw) and gg[3] == pytest.approx(gs[3] * sh / bh)
            sm = m.summary(x)
            assert sm["finite"] and sm["tokens"] == len(regs[s]["index"])
    far = {"A": [0, 0, 50, 20], "O": [400, 300, 50, 20]}
    regs = {s: region(far[s], 500, 330) for s in "AO"}
    sm = m.summary(m.condition(regs, far, "AO", 500, 330))
    assert sm["centres_outside_box"] == sm["tokens"] and sm["geom_min"][0] < -1                  # kept, counted, not clipped


def test_head_tensor_build_is_exp64s_read_path():
    import torch

    from devision.reading import CharHead, region
    torch.manual_seed(0)
    head = CharHead(768).eval()
    w, h = 560, 405
    kv = torch.randn(13 * 256, 768)
    rng = random.Random(1)
    ts = ["What is the mass of each particle in sample A?", "What is the average particle speed in sample A?"]
    bx = {t: boxes_pair(rng, w, h)["A"] for t in ts}
    ref = m.e64.read_heads(None, kv, w, h, ts, bx, {"h": head})["h"]
    conds = [m.condition({"A": region(bx[t], w, h)}, {"A": bx[t]}, "AA", w, h) for t in ts]
    got = m.reads_of(m.head_logits(head, kv, conds), ts, [bx[t] for t in ts])
    assert [(r["decoded"], r["valid"], r["value"]) for r in got] == [(r["decoded"], r["valid"], r["value"]) for r in ref]
    assert torch.equal(m.head_logits(head, kv, conds).argmax(-1), m.head_logits(head, kv, conds).argmax(-1))


def test_effects_and_explanation():
    pics = ["p%02d" % i for i in range(12)]
    t = "What is the average particle speed in sample A?"
    mk = lambda v: {p: {t: v(i)} for i, p in enumerate(pics)}  # noqa: E731
    hs = {"AA": mk(lambda i: False), "AO": mk(lambda i: True), "OA": mk(lambda i: False), "OO": mk(lambda i: True)}
    e = m.effects(hs, "speed")
    assert e["geom_AO_minus_AA"]["value"] == 1.0 and e["S_OA_minus_AA"]["value"] == 0.0
    assert m.explain(e).startswith("continuous geometry")
    hs = {"AA": mk(lambda i: False), "AO": mk(lambda i: False), "OA": mk(lambda i: True), "OO": mk(lambda i: True)}
    assert m.explain(m.effects(hs, "speed")).startswith("S package")
    hs = {"AA": mk(lambda i: False), "AO": mk(lambda i: False), "OA": mk(lambda i: False), "OO": mk(lambda i: True)}
    assert m.explain(m.effects(hs, "speed")).startswith("positive interaction")
    hs = {"AA": mk(lambda i: False), "AO": mk(lambda i: i in (0, 1)), "OA": mk(lambda i: i in (2, 3)), "OO": mk(lambda i: True)}
    e = m.effects(hs, "speed")                                          # single factors positive in point, not stable
    assert e["geom_AO_minus_AA"]["value"] > 0 and e["geom_AO_minus_AA"]["ci95"][0] <= 0 and e["S_OA_minus_AA"]["ci95"][0] <= 0
    assert e["interaction"]["ci95"][0] > 0 and e["OO_minus_AA"]["ci95"][0] > 0
    x = m.explain(e)
    assert x.startswith("positive interaction") and "uncertain" in x and "only" not in x
    hs = {c: mk(lambda i: i % 2 == 0) for c in m.CONDS}
    assert m.explain(m.effects(hs, "speed")).startswith("not located")
    tr = m.transitions(mk(lambda i: i < 3), mk(lambda i: i < 6), "speed")
    assert tr == {"right_right": 3, "right_wrong": 0, "wrong_right": 3, "wrong_wrong": 6}


@pytest.fixture
def started(tmp_path):
    import time
    out = str(tmp_path / "exp67")
    m.stage_start(int(time.time()) + 600, out=out)
    return out


@data
def test_start_binds_boxes(started, tmp_path, monkeypatch):
    m.check_start(started, "data", code=False)
    with pytest.raises(SystemExit, match="exists"):
        m.stage_start(0, out=started)
    ob = json.load(open(m.e60.BOXES))
    k = next(x for x in ob if not x.startswith("_"))
    t = next(iter(ob[k]["boxes"]))
    ob[k]["boxes"][t][1] += 1.0
    p = tmp_path / "boxes.json"
    p.write_text(json.dumps(ob))
    monkeypatch.setattr(m.e60, "BOXES", str(p))
    with pytest.raises(SystemExit, match="EXP60 differ from START"):            # EXP60's identity hashes the box file
        m.check_start(started, "", code=False)
    with pytest.raises(SystemExit, match="oracle read used another box"):        # and the boxes must be EXP64's oracle ones
        m.boxes()


def logits_for(decoded, valid):
    """One-hot logits whose argmax + strict decode give exactly (decoded, valid)."""
    from devision.reading import CHARS, CLASSES, EOS, SLOTS
    ids = [CHARS.index(ch) for ch in decoded]
    if not (len(ids) == SLOTS and not valid):                       # a read without EOS fills every slot
        ids = ids + [EOS] * (SLOTS - len(ids))
    return [[5.0 if k == i else 0.0 for k in range(CLASSES)] for i in ids]


def fake_conditions(out, drop=None, edit=None, status="completed", broken=None):
    """Condition files without a model: AA / AO = EXP64's c92 reads, OA / OO = its oracle reads (so AA / OO reproduce),
    with input summaries and zero logits; written and registered by the runner's own writer."""
    src = {"AA": m.reference("AA"), "OO": m.reference("OO")}
    src["AO"], src["OA"] = src["AA"], src["OO"]
    res = {}
    for c in m.CONDS:
        items = []
        for rid in sorted(src[c]):
            rs = []
            for q, r in src[c][rid].items():
                rs.append({"question": q, "box": r["box"], "decoded": r["decoded"], "valid": r["valid"], "value": r["value"], "geom_box": r["box"],
                           "inputs": {"tokens": 1, "finite": True, "centres_outside_box": 0, "geom_min": [0, 0, 0, 0], "geom_max": [1, 1, 1, 1]},
                           "logits": logits_for(r["decoded"], r["valid"])})
            items.append({"id": rid, "reads": rs})
        if c == edit:
            r = items[0]["reads"][0]
            r["decoded"], r["valid"], r["value"] = "unreadable", False, "unreadable"
            r["logits"] = logits_for("", False)
        if broken and c == broken[0]:
            r = items[0]["reads"][0]
            if broken[1] == "nan":
                r["logits"][0][0] = float("nan")
            elif broken[1] == "inf":
                r["logits"][1][2] = float("inf")
            elif broken[1] == "shape":
                r["logits"] = r["logits"][:-1]
            else:                                                       # a legal read that is not its logits' decoding
                r["decoded"], r["valid"], r["value"] = "7", True, "7"
                r["logits"] = logits_for("8", True)
        res[c] = items
    m.write_conditions(out, res, m.sha(os.path.join(out, "START.json")), status)
    if drop:
        os.remove(os.path.join(out, "cond_%s.jsonl" % drop))


@data
def test_report_valid_and_effects(started):
    fake_conditions(started)
    res = m.report(started)
    assert res["validity"] == "valid" and res["states"]["repro_AA"] == "reproduced" and res["states"]["repro_OO"] == "reproduced"
    assert res["correct"]["AA"]["speed"] == 16 and res["correct"]["OO"]["speed"] == 24 and res["correct"]["OO"]["distance"] == 10
    assert res["correct"]["AO"] == res["correct"]["AA"] and res["effects"]["speed"]["geom_AO_minus_AA"]["value"] == 0
    assert res["explanation"]["speed"].startswith("S package") and res["candidate_eligible"] is False
    assert all(math.isfinite(v) for c in m.CONDS for v in res["gold_logprob"][c].values())


@data
@pytest.mark.parametrize("cond,kind", [("AO", "nan"), ("OA", "inf"), ("AO", "shape"), ("OA", "decode")])
def test_report_refuses_bad_logits(started, cond, kind):
    fake_conditions(started, broken=(cond, kind))
    res = m.report(started)
    assert res["states"]["repro_AA"] == "reproduced" and res["states"]["repro_OO"] == "reproduced"      # AA / OO fine ...
    assert res["validity"] == "invalid" and res["states"][cond].startswith("invalid") and "effects" not in res   # ... still no verdict


def swapped_gold(tmp_path):
    """data/exp28/fields.jsonl with one gold moved to another option whose number is also in the transcription."""
    tr = json.load(open(m.e59.TRANSCRIBED))
    lines = open(m.e59.FIELDS).read().splitlines()
    for i, x in enumerate(lines):
        f = json.loads(x)
        rid = f["id"].split(":", 1)[1].rsplit(":", 2)[0]
        p = f["gold"]["q"]["probabilities"]
        g = max(p, key=p.get)
        other = [o for o in p if o != g and o in tr.get(rid, "")]
        if other:
            p[g], p[other[0]] = p[other[0]], p[g]
            lines[i] = json.dumps(f)
            out = tmp_path / "fields.jsonl"
            out.write_text("\n".join(lines) + "\n")
            return str(out)
    raise AssertionError("no swappable gold")


@data
@pytest.mark.parametrize("which", ["fields", "transcribed"])
def test_gold_sources_bound(started, tmp_path, monkeypatch, which):
    fake_conditions(started)
    assert m.bound_gold() == json.load(open(os.path.join(started, "START.json")))["gold"]
    if which == "fields":
        path = swapped_gold(tmp_path)
        monkeypatch.setattr(m.e59, "FIELDS", path)
        m.e59.gold_numbers(m.e59.items())                               # the swap passes exp59's own check ...
    else:
        p = tmp_path / "tr.json"
        p.write_text(open(m.e59.TRANSCRIBED).read().replace("1,300", "1,330", 1))
        monkeypatch.setattr(m.e59, "TRANSCRIBED", str(p))
    with pytest.raises(SystemExit, match="gold source"):
        m.bound_gold()
    res = m.report(started)                                             # ... but not the run's identity
    assert res["branch"].startswith("invalid") and "correct" not in res


@data
@pytest.mark.parametrize("case,state", [("drop", "incomplete"), ("edit", "invalid"), ("stopped", "invalid")])
def test_report_missing_wrong_or_stopped(started, case, state):
    fake_conditions(started, drop="AO" if case == "drop" else None, edit="AA" if case == "edit" else None,
                    status="stopped: x" if case == "stopped" else "completed")
    res = m.report(started)
    assert res["validity"] == state and res["branch"].startswith(state) and "no verdict" in res["branch"]


STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"watch.py"*) shift 2; exec python3 "$@";;
  *"exp67.py start"*) mkdir -p runs/exp67; exit 0;;
  *"exp67.py run"*) [ -n "${FAIL_RUN:-}" ] && exit 5; exit 0;;
  *"exp67.py report"*) echo report > runs/exp67/report.txt; exit 0;;
esac
exit 9
"""


def chain(tmp_path, extra):
    import shutil
    import stat as st_
    import subprocess
    root = tmp_path / "repo"
    for rel in ("scripts/exp67/run.sh", "scripts/exp53/watch.py", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    bdir = tmp_path / "bin"
    bdir.mkdir()
    for name, body in {"uv": STUB, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
        (bdir / name).write_text(body)
        (bdir / name).chmod((bdir / name).stat().st_mode | st_.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (bdir, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "EXP67_LIMIT": "60",
           "EXP67_RESERVE": "4", **extra}
    p = subprocess.run(["bash", str(root / "scripts/exp67/run.sh")], env=env, capture_output=True, text=True, timeout=120)
    return root, p, (tmp_path / "calls").read_text().splitlines()


def test_run_order_failed_stage_and_limit(tmp_path):
    root, p, calls = chain(tmp_path / "a", {})
    assert p.returncode == 0, p.stdout + p.stderr
    first = lambda s: next(i for i, c in enumerate(calls) if s in c)  # noqa: E731
    assert first("exp67.py start") < first("exp67.py run") < first("exp67.py report")
    root, p, calls = chain(tmp_path / "b", {"FAIL_RUN": "1"})
    assert p.returncode == 5 and "report --finalize" in "\n".join(calls)
    s = open(os.path.join(ROOT, "scripts/exp67/run.sh")).read()
    assert "${EXP67_LIMIT:-600}" in s and "${EXP67_RESERVE:-120}" in s
