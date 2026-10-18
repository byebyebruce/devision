# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false
"""exp68 data without a model: magnets byte for byte; 6 x 100 new temperature combinations, two views each, every table
number = its reading gold and the comparison gold = m * v^2; the 100 conflict pairs differ in one number and flip the
answer; nothing overlaps earlier data, EXP65's diag, the transcribed pictures or the new diag; the new diag's format;
the real batch plan (9,600 / 1,200, no unit split); text-only rows never open their image path."""
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("exp68_build_t", os.path.join(HERE, "build.py"))
assert spec is not None and spec.loader is not None
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
has = os.path.exists(os.path.join(ROOT, "data", "exp68", "MANIFEST.json"))
data = pytest.mark.skipif(not has, reason="data/exp68 not built")


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def rows():
    return [json.loads(x) for x in open("data/exp68/train.jsonl")]


@data
def test_magnets_unchanged_and_counts():
    old = [x for x in open("data/facts/train.jsonl").read().splitlines() if json.loads(x)["kind"] == "force"]
    new = open("data/exp68/train.jsonl").read().splitlines()
    assert new[:len(old)] == old and len(old) == 1200
    rs = rows()
    temp = [r for r in rs if r["kind"] == "temp"]
    assert len(temp) == 1200 and sum(len(r["questions"]) for r in rs) == 9600
    assert all(r["image"] == b.IMAGE for r in temp) and all(len(r["questions"]) == 5 for r in temp)


@data
def test_temperature_rows_follow_the_rule():
    temp = [r for r in rows() if r["kind"] == "temp"]
    fams = defaultdict(list)
    for r in temp:
        b.check_row(r)
        fams[r["family"]].append(r)
    assert len(fams) == 600 and Counter(rs[0]["axis"] for rs in fams.values()) == {c: 100 for c in b.CLASSES}
    keys = set()
    for rs in fams.values():
        v0, v1 = sorted(rs, key=lambda r: r["id"])
        a, c = b.b65.values_of(v0)
        assert b.b65.values_of(v1) == (c, a)                                          # A / B swapped
        keys.add(b.key(a, c))
        assert v0["questions"]["q0"]["instructions"] == b.rl.TEMP_Q
    assert len(keys) == 600


@data
def test_conflict_pairs_flip_on_one_number():
    temp = {r["family"]: b.b65.values_of(r) for r in rows() if r["kind"] == "temp" and r["id"].endswith(":v0")}
    m = json.load(open("data/exp68/MANIFEST.json"))
    assert m["pairs"]["n"] == 100 and m["pairs"]["shared_numbers"] == [[3, 100]]
    fam_cls = {r["family"]: r["axis"] for r in rows() if r["kind"] == "temp"}
    mw = sorted(f for f, c in fam_cls.items() if c == "opposite_mass_wins")
    sw = sorted(f for f, c in fam_cls.items() if c == "opposite_speed_wins")
    for i, (x, y) in enumerate(zip(mw, sw)):                                        # pair i = mass-wins i, speed-wins i
        a = sorted(temp[x])
        c = sorted(temp[y])
        same = [p == q for x_, y_ in zip(a, c) for p, q in zip(x_, y_)]
        other = [p == q for x_, y_ in zip(a, c[::-1]) for p, q in zip(x_, y_)]
        assert 3 in (sum(same), sum(other))                                          # exactly one of the four numbers changed
        assert b.class_of(*temp[x]) == "opposite_mass_wins" and b.class_of(*temp[y]) == "opposite_speed_wins"


@data
def test_isolation_and_diag():
    ex = b.excluded("data")
    banned = set().union(*ex.values())
    train = {b.key(*b.b65.values_of(r)) for r in rows() if r["kind"] == "temp"}
    d = [json.loads(x) for x in open("data/exp68/diag.jsonl")]
    dk = {b.key(*b.b65.state_values(r["state_text"])) for r in d}
    assert not train & banned and not dk & banned and not train & dk
    assert len(d) == 480 and Counter(r["axis"] for r in d) == {c: 80 for c in b.CLASSES}
    for r in d:
        a, c = b.b65.state_values(r["state_text"])
        g = next(o for o, p in r["gold"]["q0"]["probabilities"].items() if p == 1.0)
        w = b.b65.winner(a, c)
        assert (g.startswith("neither") if w == "neither" else g == w) and b.class_of(a, c) == r["axis"]
    fam = defaultdict(list)
    for r in d:
        fam[r["family"]].append(b.b65.state_values(r["state_text"]))
    assert all(x == (y[1], y[0]) for x, y in fam.values())


@data
def test_real_batch_plan():
    p = json.load(open("data/exp68/PLAN.json"))
    assert p["items"] == 9600 and p["updates"] == 1200 and p["split_units"] == 0 and p["every_item_once"]
    assert p["unit_sizes"] == {"2": 4800} and p["train_sha256"] == b.sha("data/exp68/train.jsonl")
    assert all(not v["state_dropped"] and not v["options_not_distinct"] for v in p["truncation"].values())


@data
def test_text_only_never_opens_the_image(tmp_path, monkeypatch):
    sys.path[:0] = [os.path.join(ROOT, "tests"), os.path.join(ROOT, "tests", "train")]
    from test_distill import CONFIG  # pyright: ignore[reportMissingImports]
    from test_text_only import exp50_like  # pyright: ignore[reportMissingImports]

    from devision.train import rlcd
    from devision.train.rlcd import TrainConfig, train
    monkeypatch.setattr(rlcd._Images, "__call__", lambda *a, **k: (_ for _ in ()).throw(AssertionError("an image was opened")))
    ss = [r for r in rows() if r["kind"] == "temp"][:4]
    cfg = {**CONFIG, "epochs": 1, "objective": "ce", "micro_batch": 4, "save_every": 0, "text_only": True}
    rep = train(exp50_like(seed=1), ss, data_root=tmp_path / "nowhere", out_dir=tmp_path / "o", val_samples=ss[:2], config=TrainConfig(**cfg))
    assert rep["losses"]
