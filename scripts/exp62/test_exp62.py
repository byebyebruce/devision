# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false
"""exp62 without a model: every parent is the generator's own picture whose rows [t, t + h) are its EXP61 crop and whose
field box, moved by -t and clipped, is the crop's box; families / cases isolated from the dev slices; the batch plan
rebuilds to the committed file with 24 whole pairs + 16 replay per update and the fixed exposure counts; training follows
the plan from the given selector state; changed parents / plan / EXP61 products after START are refused."""
import importlib.util
import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("exp62", os.path.join(HERE, "exp62.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
bspec = importlib.util.spec_from_file_location("exp62_build", os.path.join(HERE, "build.py"))
b = importlib.util.module_from_spec(bspec)
bspec.loader.exec_module(b)
has_data = os.path.exists(os.path.join(ROOT, m.P_MANIFEST))
data = pytest.mark.skipif(not has_data, reason="data/exp62 not built")


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


@data
def test_parents_map_to_their_crops():
    par = m.jsonl(m.PARENTS)
    crops = {q["key"]: q for q in m.jsonl(m.e61.V_QUESTIONS)}
    assert len(par) == 1200 and len({q["picture"] for q in par}) == 400
    for q in par[::37]:
        c = crops[q["pair"]]
        assert {k: q[k] for k in ("field", "object", "target", "family", "view", "text")} == {k: c[k] for k in ("field", "object", "target", "family", "view", "text")}
        p = np.asarray(Image.open(os.path.join("data", q["picture"])).convert("RGB"))
        v = np.asarray(Image.open(os.path.join("data", c["picture"])).convert("RGB"))
        t = q["crop_top"]
        assert np.array_equal(p[t:t + v.shape[0]], v)                                  # only blank rows differ
        x, y, w, h = q["box"]
        moved = [x, max(0.0, y - t), x + w, min(float(c["height"]), y + h - t)]
        cx = [c["box"][0], c["box"][1], c["box"][0] + c["box"][2], c["box"][1] + c["box"][3]]
        assert moved == pytest.approx(cx)
        tb = q["text_box"]
        assert x - 1 <= tb[0] and y - 1 <= tb[1] and x + w + 1 >= tb[2] and y + h + 1 >= tb[3]
        if q["field"] == "distance":
            assert w / q["width"] > 0.3                                                    # the wide label band


@data
def test_isolation_and_records():
    man = json.load(open(m.P_MANIFEST))
    assert man["isolation"] == {"families_in_tune_V": 0, "cases_in_tune_V": 0, "families_in_exp49": 0}
    assert man["records"] == {"replay": 1152, "train_V": 1200, "train_P": 1200, "total": 3552}


@data
def test_plan_rebuilds_and_keeps_pairs_whole():
    par = m.jsonl(m.PARENTS)
    replay = [q for q in m.jsonl(m.e61.QUESTIONS) if q["set"] in ("fit_O", "fit_L")]
    plan = b.plan(sorted(q["pair"] for q in par), sorted(q["key"] for q in replay))
    assert plan == m.jsonl(m.PLAN)
    st = b.plan_stats(plan, par, replay)
    assert (st["updates"], st["exposures"], st["parent"], st["crop"], st["replay"], st["split_pairs"]) == (2000, 128000, 48000, 48000, 32000, 0)
    assert st["parent_exposure_range"] == [40, 40] and st["every_row_seen"]
    for bt in plan[:50]:
        assert len(bt) == 64 and all(bt[2 * i] == bt[2 * i + 1].replace("train_V|", "train_P|", 1) for i in range(24))
        assert all(k.split("|")[0] in ("fit_O", "fit_L") for k in bt[48:])


def test_training_follows_the_plan_from_the_given_selector():
    torch.manual_seed(1)
    inp = m.sel.Inputs(torch.randn(4, 256, 768).half(), torch.rand(4, 256, 4), [i % 4 for i in range(8)],
                       torch.randn(8, 5, 1024).half(), torch.ones(8, 5, dtype=torch.bool))
    init = m.sel.Selector().state_dict()
    targets = torch.rand(8, 4) * 0.5 + 0.25
    same, st0 = m.train_plan(init, inp, targets, [])
    assert all(torch.equal(same.state_dict()[k], init[k]) for k in init) and st0["exposures"] == 0
    a, st = m.train_plan(init, inp, targets, [[0, 1, 2], [3, 4], [5, 6, 7]])
    b2, _ = m.train_plan(init, inp, targets, [[0, 1, 2], [3, 4], [5, 6, 7]])
    assert st["steps"] == 3 and st["exposures"] == 8
    assert any(not torch.equal(a.state_dict()[k], init[k]) for k in init)
    assert all(torch.equal(a.state_dict()[k], b2.state_dict()[k]) for k in init) and set(a.state_dict()) == set(init)


@pytest.fixture
def started(tmp_path, monkeypatch):
    if not has_data or not os.path.exists(os.path.join(ROOT, m.E61, "START.json")):
        pytest.skip("data / runs not here")
    pars, plan, man = (tmp_path / n for n in ("parents.jsonl", "plan.jsonl", "MANIFEST.json"))
    pars.write_text(open(m.PARENTS).read())
    plan.write_text(open(m.PLAN).read())
    man.write_text(open(m.P_MANIFEST).read())
    monkeypatch.setattr(m, "PARENTS", str(pars))
    monkeypatch.setattr(m, "PLAN", str(plan))
    monkeypatch.setattr(m, "P_MANIFEST", str(man))
    out = str(tmp_path / "out")
    m.stage_start(0, out=out)
    return out, pars, plan, man


@pytest.mark.parametrize("what", ["parent_box", "plan"])
def test_changed_parents_or_plan_after_start_are_refused(started, what):
    out, pars, plan, man = started
    m.check_start(out, data_root="")
    f = pars if what == "parent_box" else plan
    lines = f.read_text().splitlines()
    r = json.loads(lines[0])
    if what == "parent_box":
        r["box"][1] += 3
    else:
        r = list(reversed(r))
    lines[0] = json.dumps(r)
    f.write_text("\n".join(lines) + "\n")
    mm = json.loads(man.read_text())
    mm["parents_sha256" if what == "parent_box" else "plan_sha256"] = m.sha(str(f))   # the current MANIFEST updated too
    man.write_text(json.dumps(mm))
    with pytest.raises(SystemExit, match="differ from START"):
        m.check_start(out, data_root="")
    assert m.report(out)["branch"].startswith("invalid")


def test_changed_exp61_run_is_refused(started, monkeypatch):
    out = started[0]
    monkeypatch.setattr(m, "E61_SHA", dict(m.E61_SHA, **{"report.json": "0" * 64}))
    with pytest.raises(SystemExit, match="EXP61's"):
        m.check_start(out, data_root="")
