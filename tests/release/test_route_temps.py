"""scripts/release/route_temps.py (critic V04-ROUTE-TEMP-PREP-01, ROUTE-TEMP-R1-01): the routed temperatures are read,
as written and without any fitting, only from the accepted calibration run -- its pinned START / registry / final
report, the registry's route_calibration / CALIB / records and rl_calib questions -- into exactly the candidate config
it was registered with, before binding, changing nothing else. Another self-consistent branch-1 calibration, a replaced
report or registry, a non-branch-1 result, another / changed / bound candidate and a second write are refused; with
route_calib (analyse, fit_t) made unloadable the read and the write still work."""
import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMPS = {"temperature": 3.238142943837961, "magnets": 4.664590270988126}


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def calib_run(m, d, cand, temps=TEMPS, branch="1 separate routed temperature: ...", tag="a"):
    """A registered calibration run's files (the evidence layout of runs/v04-k-calib), and its pins."""
    from devision.model import ModelConfig
    os.makedirs(d / "calib")
    if not cand.exists():
        cand.mkdir()
        (cand / "config.json").write_text(json.dumps(ModelConfig().to_dict(), indent=2))
    rows = d / "questions.jsonl"
    rows.write_text('{"id": "%s"}\n' % tag)
    (d / "START.json").write_text(json.dumps({"run": tag}))
    (d / "calib" / "records.jsonl").write_text('{"sample_id": "%s"}\n' % tag)
    (d / "calib" / "CALIB.json").write_text(json.dumps({"paths": {"candidate": str(cand), "rows": str(rows)}, "rows": m.sha(str(rows)),
                                                        "candidate": {"config.json": m.sha(str(cand / "config.json"))}}))
    (d / "route_calibration.json").write_text(json.dumps({"branch": branch, "routed_temperature_all_data": temps}))
    (d / "report.json").write_text(json.dumps({"branch": branch, "run": tag}))
    reg = {"start": {"START.json": m.sha(str(d / "START.json"))}, "calibrate": {"config.json": m.sha(str(cand / "config.json"))},
           "rl_calib": {"questions.jsonl": m.sha(str(rows))},
           "collect": {"calib/CALIB.json": m.sha(str(d / "calib" / "CALIB.json")), "calib/records.jsonl": m.sha(str(d / "calib" / "records.jsonl"))},
           "analyse": {"route_calibration.json": m.sha(str(d / "route_calibration.json"))}}
    (d / "registry.json").write_text(json.dumps(reg))
    return {f: m.sha(str(d / f)) for f in ("START.json", "registry.json", "report.json")}


@pytest.fixture
def run(tmp_path, monkeypatch):
    m = load("route_temps_t", "scripts/release/route_temps.py")
    real = importlib.util.spec_from_file_location

    def no_route_calib(name, path, *a, **k):                    # analyse / fit_t: one call and it fails
        if str(path).endswith("route_calib.py"):
            raise AssertionError("route_calib loaded: a fit would run")
        return real(name, path, *a, **k)
    monkeypatch.setattr(importlib.util, "spec_from_file_location", no_route_calib)
    cand = tmp_path / "cand"
    pins = calib_run(m, tmp_path / "calibrun", cand)
    return m, tmp_path, str(tmp_path / "calibrun"), str(cand), pins


def test_writes_the_accepted_temperatures_exactly_and_nothing_else(run):
    m, _, calib, cand, pins = run
    assert m.source_temperatures(calib, pins) == TEMPS
    before = json.load(open(os.path.join(cand, "config.json")))
    out = m.write(calib, cand, pins)
    after = json.load(open(os.path.join(cand, "config.json")))
    assert after["temperature_by_route"] == TEMPS == out["temperature_by_route"]
    assert after["temperature_by_route"]["temperature"] == 3.238142943837961 and after["temperature_by_route"]["magnets"] == 4.664590270988126
    assert {k: v for k, v in after.items() if k != "temperature_by_route"} == before
    with pytest.raises(SystemExit, match="already has routed temperatures"):
        m.write(calib, cand, pins)


def test_the_real_pins_are_the_accepted_run():
    m = load("route_temps_p", "scripts/release/route_temps.py")
    assert m.ACCEPTED == {"START.json": "2c02a535cb580dab0f1771b13147600e821326709f7152ac9ec05794b9c1a877",
                          "registry.json": "5b35da89bb245cc680b118522eb66f3b3346858fd81515c232331005a7c001f6",
                          "report.json": "1b54ff353ffe59b91fe503d4cbd4683e33965645655a025e4f33c854911b778b"}


def test_another_self_consistent_calibration_is_refused(run):
    m, tmp_path, _, cand, pins = run
    other = tmp_path / "other"
    own = calib_run(m, other, tmp_path / "cand", temps={"temperature": 2.0, "magnets": 2.5}, tag="b")
    assert m.source_temperatures(str(other), own) == {"temperature": 2.0, "magnets": 2.5}      # consistent in itself ...
    with pytest.raises(SystemExit, match="not the accepted calibration run"):
        m.write(str(other), cand, pins)                                                         # ... but not the accepted run
    with pytest.raises(SystemExit, match="not the accepted calibration run"):
        m.write(str(other), cand)                                                               # nor the real pins


@pytest.mark.parametrize("damage", ["report", "registry", "route_calibration", "records", "rows"])
def test_replaced_evidence_is_refused(run, damage):
    m, _, calib, cand, pins = run
    if damage == "report":
        open(os.path.join(calib, "report.json"), "a").write(" ")
    elif damage == "registry":
        reg = json.load(open(os.path.join(calib, "registry.json")))
        open(os.path.join(calib, "route_calibration.json"), "w").write(json.dumps({"branch": "1 ...", "routed_temperature_all_data": {"temperature": 2.0, "magnets": 2.0}}))
        reg["analyse"]["route_calibration.json"] = m.sha(os.path.join(calib, "route_calibration.json"))
        json.dump(reg, open(os.path.join(calib, "registry.json"), "w"))                        # re-registered: the registry is no longer pinned
    elif damage == "route_calibration":
        open(os.path.join(calib, "route_calibration.json"), "a").write(" ")
    elif damage == "records":
        open(os.path.join(calib, "calib", "records.jsonl"), "a").write("\n")
    else:
        rows = json.load(open(os.path.join(calib, "calib", "CALIB.json")))["paths"]["rows"]
        open(rows, "a").write("\n")
    with pytest.raises(SystemExit, match="not the accepted|not the registered|rows are not"):
        m.write(calib, cand, pins)
    assert "temperature_by_route" not in json.load(open(os.path.join(cand, "config.json")))


def test_not_branch_1_is_refused(tmp_path, monkeypatch):
    m = load("route_temps_b", "scripts/release/route_temps.py")
    pins = calib_run(m, tmp_path / "c2", tmp_path / "cand", branch="2 keep the current temperatures")
    with pytest.raises(SystemExit, match="not branch 1"):
        m.write(str(tmp_path / "c2"), str(tmp_path / "cand"), pins)


def test_candidate_refusals(run):
    import shutil
    m, tmp_path, calib, cand, pins = run
    other = tmp_path / "other_cand"
    shutil.copytree(cand, other)
    with pytest.raises(SystemExit, match="registered for"):
        m.write(calib, str(other), pins)
    os.makedirs(os.path.join(cand, "reading"))
    with pytest.raises(SystemExit, match="already has reading"):
        m.write(calib, cand, pins)
    os.rmdir(os.path.join(cand, "reading"))
    c = json.load(open(os.path.join(cand, "config.json")))
    c["temperature"] = [1.9, 1.0, 1.1]
    json.dump(c, open(os.path.join(cand, "config.json"), "w"))
    with pytest.raises(SystemExit, match="not the one the calibration was registered with"):
        m.write(calib, cand, pins)
    assert "temperature_by_route" not in json.load(open(os.path.join(cand, "config.json")))
