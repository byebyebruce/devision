"""scripts/release/route_temps.py (critic V04-ROUTE-TEMP-PREP-01): the routed temperatures are written only from a
registered branch-1 calibration -- its registry hashes, CALIB rows / candidate config, records recomputed equal to the
written report -- into exactly the candidate it was registered for, before binding, changing nothing else; another
candidate, a bound candidate, an edited or re-registered report, a non-branch-1 result and a second write are refused."""
import copy
import importlib.util
import json
import os
import shutil

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A registered calibration run around the tiny registered calibration of test_k_source, its analysis made branch 1
    (the statistics themselves are route_calib's, tested there)."""
    ks = load("ks_rt", "tests/release/test_k_source.py")
    rc, pins, cand, recs, rows, ident, paths = ks.tiny_calibration(tmp_path, monkeypatch)
    m = load("route_temps_t", "scripts/release/route_temps.py")
    real_rc = load("route_calib_rt", "scripts/release/route_calib.py")
    from types import SimpleNamespace
    fake = SimpleNamespace(completeness=real_rc.completeness, analyse=None)
    branch1 = {"branch": "1 separate routed temperature: ...", "pooled": {}, "domains": {},
               "routed_temperature_all_data": {"temperature": 3.238142943837961, "magnets": 4.664590270988126}}
    fake.analyse = lambda recs_, cfg: copy.deepcopy(branch1)
    monkeypatch.setattr(m, "_load", lambda name, rel: fake)
    calib = tmp_path / "calibrun"
    (calib).mkdir()
    shutil.copytree(tmp_path / "calib", calib / "calib")
    json.dump(branch1, open(calib / "route_calibration.json", "w"))
    reg = {"collect": {"calib/CALIB.json": m.sha(str(calib / "calib" / "CALIB.json")), "calib/records.jsonl": m.sha(str(calib / "calib" / "records.jsonl"))},
           "analyse": {"route_calibration.json": m.sha(str(calib / "route_calibration.json"))}}
    json.dump(reg, open(calib / "registry.json", "w"))
    return m, str(calib), str(cand), branch1, fake


def test_writes_the_registered_temperatures_and_nothing_else(run):
    m, calib, cand, branch1, _ = run
    before = json.load(open(os.path.join(cand, "config.json")))
    out = m.write(calib, cand)
    after = json.load(open(os.path.join(cand, "config.json")))
    assert after["temperature_by_route"] == branch1["routed_temperature_all_data"] == out["temperature_by_route"]
    assert {k: v for k, v in after.items() if k != "temperature_by_route"} == before
    with pytest.raises(SystemExit, match="already has routed temperatures"):
        m.write(calib, cand)


def test_refusals(run, tmp_path, monkeypatch):
    m, calib, cand, branch1, fake = run
    other = tmp_path / "other_cand"
    shutil.copytree(cand, other)
    with pytest.raises(SystemExit, match="registered for"):
        m.write(calib, str(other))
    os.makedirs(os.path.join(cand, "reading"))
    with pytest.raises(SystemExit, match="already has reading"):
        m.write(calib, cand)
    os.rmdir(os.path.join(cand, "reading"))
    rp = os.path.join(calib, "route_calibration.json")
    edited = dict(branch1, routed_temperature_all_data={"temperature": 2.0, "magnets": 2.0})
    json.dump(edited, open(rp, "w"))
    with pytest.raises(SystemExit, match="not the registered calibration evidence"):
        m.write(calib, cand)
    reg = json.load(open(os.path.join(calib, "registry.json")))
    reg["analyse"]["route_calibration.json"] = m.sha(rp)                 # re-registered: the recomputation still differs
    json.dump(reg, open(os.path.join(calib, "registry.json"), "w"))
    with pytest.raises(SystemExit, match="differs from a recomputation"):
        m.write(calib, cand)
    two = dict(branch1, branch="2 keep the current temperatures")
    json.dump(two, open(rp, "w"))
    reg["analyse"]["route_calibration.json"] = m.sha(rp)
    json.dump(reg, open(os.path.join(calib, "registry.json"), "w"))
    monkeypatch.setattr(fake, "analyse", lambda recs_, cfg: copy.deepcopy(two))
    with pytest.raises(SystemExit, match="not branch 1"):
        m.write(calib, cand)
    assert "temperature_by_route" not in json.load(open(os.path.join(cand, "config.json")))
