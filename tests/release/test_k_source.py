"""The LIVE-K release source (critic V04-K-PREP-01; scripts/release/rebind.py kind "livek"): a LIVE-K-like run -- pinned
START / registry / report, every state completed, gates live_P / transfer / protection true -- with its K bundle, PROTECT-
like end and EXP71-like components goes source -> clone -> calibrate -> bind -> freeze, and the candidate keeps the K rule
and the source's rules hash with components and weights unchanged; a wrong end, a failed gate or incomplete state, a
changed registered file, a source bundle without the K rule, an edited K rule, K dropped after binding and a no-K source
in the release flow are refused; the no-K EXP71 flow is untouched (tests/release/test_reading_lifecycle.py)."""
import importlib.util
import json
import os

import pytest
import torch
import yaml
from conftest import tiny_decider

import devision
from devision import reading as R

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def script(name):
    spec = importlib.util.spec_from_file_location("k_" + name, os.path.join(ROOT, "scripts", "release", name + ".py"))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rb, ce = script("rebind"), script("check_eval")


def local_view(seed):
    d = tiny_decider(seed=seed)
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    d.model.enable_local_views(12)
    d.cfg.local_views, d.cfg.local_overlap, d.cfg.local_source = 12, 0.25, "T"
    return d


def livek_run(tmp, k_rule=True, gate=True, states=True, edit_rule=False):
    """A LIVE-K-like run in tmp/livek over the end tmp/end and a K bundle tmp/bundle_k; returns its pins."""
    if not (tmp / "end").exists():
        local_view(3).save(tmp / "end")
        local_view(11).save(tmp / "other_end")
        torch.manual_seed(7)
        for n, m in (("sel.pt", R.Selector(32, 64)), ("head.pt", R.CharHead(32)), ("corr.pt", R.Selector(32, 64))):
            torch.save(m.state_dict(), tmp / n)
    import shutil
    for d in ("bundle_k", "livek"):
        if (tmp / d).exists():
            shutil.rmtree(tmp / d)
    ident = R.write_bundle(str(tmp / "bundle_k"), str(tmp / "end"), str(tmp / "sel.pt"), str(tmp / "head.pt"), {},
                           dims={"selector": [32, 64], "head": 32}, corrector_pt=str(tmp / "corr.pt"), k_rule=k_rule)
    if edit_rule:                                            # an edited K rule (START records the edited file)
        p = tmp / "bundle_k" / "reading.json"
        j = json.loads(p.read_text())
        j["k_rule"]["formula"] = "mass * speed"
        p.write_text(json.dumps(j))
    run = tmp / "livek"
    run.mkdir()
    (run / "details_x.jsonl").write_text("{}\n")
    st = {"k_bundle": {"reading.json": R.sha(str(tmp / "bundle_k" / "reading.json")), "rules_sha256": ident["rules_sha256"]},
          "exp71": {"bundle_on": {c: ident["sha256"][c] for c in ("selector.pt", "head.pt", "corrector.pt")}},
          "protect": {"end": {k: ident["sha256"][k] for k in ("model.safetensors", "config.json")}}}
    (run / "START.json").write_text(json.dumps(st))
    (run / "registry.json").write_text(json.dumps({"decide_x": {"details_x.jsonl": R.sha(str(run / "details_x.jsonl"))}}))
    (run / "report.json").write_text(json.dumps({"states": {"s%d" % i: "completed" if states or i else "missing" for i in range(13)},
                                                 "gates": {"live_P": True, "transfer": gate, "protection": True}}))
    (tmp / "round").mkdir(exist_ok=True)
    (tmp / "round" / "run.json").write_text("{}")
    return {k: R.sha(str(run / k)) for k in ("START.json", "registry.json", "report.json")}


def source(tmp, pins, decision="end"):
    return rb.source(str(tmp / "livek"), str(tmp / decision), str(tmp / "bundle_k"), str(tmp / "round"), pins=pins, kind="livek")


def candidate(tmp, pins):
    cand = tmp / "round" / "cand"
    rb.clone(str(tmp / "round"), str(cand), pins)
    c = json.loads((cand / "config.json").read_text())
    c.update(temperature=[1.3, 1.0, 1.1], temperature_by_options={"choice:2": 1.2}, model_name="devision-v0.4")
    (cand / "config.json").write_text(json.dumps(c))
    return cand


def test_the_k_rule_survives_the_lifecycle(tmp_path):
    pins = livek_run(tmp_path)
    rec = source(tmp_path, pins)
    assert rec["kind"] == "livek"
    cand = candidate(tmp_path, pins)
    lin = rb.bind(str(cand), pins)
    src = json.loads((tmp_path / "bundle_k" / "reading.json").read_text())
    new = json.loads((cand / "reading" / "reading.json").read_text())
    assert new["k_rule"] == src["k_rule"] and new["rules_sha256"] == src["rules_sha256"] == lin["rules_sha256"] and lin["k_rule"] is True
    assert {c: new["sha256"][c] for c in ("selector.pt", "head.pt", "corrector.pt")} == {c: src["sha256"][c] for c in ("selector.pt", "head.pt", "corrector.pt")}
    assert new["sha256"]["model.safetensors"] == src["sha256"]["model.safetensors"] and set(lin["config_changes"]) <= rb.ALLOWED
    rd = devision.load(str(cand), device="cpu")
    assert isinstance(rd, R.Reading) and rd.k_rule and rd.reading_status["k_rule"] is True
    assert rb.lineage_problems(str(cand), pins) == []
    conf = tmp_path / "full.yaml"
    (tmp_path / "set.jsonl").write_text("")
    conf.write_text(yaml.safe_dump({"name": "full", "stages": [], "evaluate": {"checkpoint": str(cand), "sets": {"a": {"path": str(tmp_path / "set.jsonl"), "role": "heldout"}}}}))
    setattr(ce, "SOURCE_PINS", pins)
    assert ce.main(["freeze", str(conf)]) == 0                                  # freeze accepts the K candidate


@pytest.mark.parametrize("case,why", [("wrong_end", "are not the files"), ("gate", "gates"), ("state", "not every state completed"),
                                      ("registered", "registered details_x.jsonl changed"), ("no_k", "no K rule"), ("edited_k", "rules differ")])
def test_wrong_sources_are_refused(tmp_path, case, why):
    pins = livek_run(tmp_path, k_rule=case != "no_k", gate=case != "gate", states=case != "state", edit_rule=case == "edited_k")
    if case == "registered":
        (tmp_path / "livek" / "details_x.jsonl").write_text("{} changed\n")
    with pytest.raises(SystemExit, match=why):
        source(tmp_path, pins, decision="other_end" if case == "wrong_end" else "end")


def test_k_dropped_after_binding_is_refused(tmp_path):
    pins = livek_run(tmp_path)
    source(tmp_path, pins)
    cand = candidate(tmp_path, pins)
    rb.bind(str(cand), pins)
    rj = cand / "reading" / "reading.json"
    j = json.loads(rj.read_text())
    j.pop("k_rule")
    rj.write_text(json.dumps(j))
    lp = rb.lineage_dir(str(cand)) + "/lineage.json"
    lin = json.loads(open(lp).read())
    lin["reading_json_sha256"] = R.sha(str(rj))                                # even with the lineage re-pointed
    open(lp, "w").write(json.dumps(lin))
    assert any("K rule" in x for x in rb.lineage_problems(str(cand), pins))


def test_the_release_flow_requires_the_livek_source(tmp_path):
    pins = livek_run(tmp_path)
    rec = source(tmp_path, pins)
    assert any("not the required source" in x for x in rb.check_source(rec))      # default: the real LIVE-K pins, not these
    old = dict(rec, kind="exp71")
    assert any("requires 'livek'" in x for x in rb.check_source(old))
    assert rb.RELEASE_KIND == "livek" and rb.SOURCES["livek"] == rb.PINS_LIVEK and rb.PINS_LIVEK["START.json"].startswith("f9a10d6a")


def test_temporary_calibration_bundle_is_outside_and_bound_to_the_current_config(tmp_path):
    pins = livek_run(tmp_path)
    source(tmp_path, pins)
    cand = candidate(tmp_path, pins)
    tb = rb.temp_bundle(str(cand), str(tmp_path / "calib_bundle"), pins)
    assert tb["k_rule"] and tb["config.json"] == R.sha(str(cand / "config.json")) and not (cand / "reading").exists()
    rd = R.Reading.load(str(cand), str(tmp_path / "calib_bundle"), device="cpu", full_rules=True)
    assert rd.k_rule
    with pytest.raises(SystemExit, match="outside the candidate"):
        rb.temp_bundle(str(cand), str(cand / "tmp"), pins)
    c = json.loads((cand / "config.json").read_text())
    c["model_name"] = "devision-v0.4-final"                                    # the final config: the temporary bundle no longer binds it
    (cand / "config.json").write_text(json.dumps(c))
    with pytest.raises(ValueError, match="does not bind"):
        R.Reading.load(str(cand), str(tmp_path / "calib_bundle"), device="cpu")


def tiny_calibration(tmp_path, monkeypatch):
    """A tiny registered calibration: LIVE-K-like source -> candidate -> temporary K bundle -> a 6 + 6 family rl_calib drawn
    by the real builder (short options / hint for the tiny model) -> collect through the verified entry -> register."""
    import sys
    rc = script("route_calib")
    rl = script("rl_calib")
    monkeypatch.setattr(rl, "used_cases", lambda root: set())
    monkeypatch.delitem(sys.modules, "build", raising=False)
    pins = livek_run(tmp_path)
    source(tmp_path, pins)
    cand = candidate(tmp_path, pins)
    rb.temp_bundle(str(cand), str(tmp_path / "calib_bundle"), pins)
    rl.build(str(tmp_path), want={"temp": 6, "force": 6})
    rows = [json.loads(x) for x in open(tmp_path / "rl_calib" / "questions.jsonl")]
    short = ["red", "blue", "green"]
    for r in rows:
        opts = list(r["questions"]["q0"]["criteria"])
        r["state_text"] = ""
        r["questions"]["q0"]["criteria"] = {short[i]: None for i in range(len(opts))}
        r["gold"]["q0"]["probabilities"] = {short[i]: r["gold"]["q0"]["probabilities"][o] for i, o in enumerate(opts)}
    rows_path = tmp_path / "rl_calib" / "rows_short.jsonl"
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ident = rc.identity(str(cand), str(tmp_path / "calib_bundle"), str(rows_path), str(tmp_path), pins, ROOT)
    rd = R.Reading.load(str(cand), str(tmp_path / "calib_bundle"), device="cpu", full_rules=True)
    recs = rc.collect(rd, rows, str(tmp_path))
    paths = {"candidate": str(cand), "bundle": str(tmp_path / "calib_bundle"), "rows": str(rows_path), "data_root": str(tmp_path)}
    from devision.train.evaluate import runtime_code
    rc.register(str(tmp_path / "calib"), ident, recs, paths, runtime_code(), ROOT)
    return rc, pins, cand, recs, rows, ident, paths


def test_registered_calibration_analyses_only_complete_bound_records(tmp_path, monkeypatch):
    rc, pins, cand, recs, rows, ident, paths = tiny_calibration(tmp_path, monkeypatch)
    res = rc.analyse_registered(str(tmp_path / "calib"), pins, ROOT)
    assert res["branch"][0] in "1234" and set(res["domains"]) == {"temperature", "magnets"} and res["pooled"]["families"] == 12
    assert ident["temporary_bundle"] and len(ident["pictures"]) == 24 and "scripts/release/route_calib.py" in ident["code"]


@pytest.mark.parametrize("damage,why", [("config", "identity refused|does not verify"), ("records", "does not verify"), ("rows", "does not verify"),
                                        ("domain", "not both"), ("view", "not v0 \\+ v1"), ("duplicate", "duplicate"), ("gold", "differ from the source"),
                                        ("missing", "missing")])
def test_registered_calibration_refusals(tmp_path, monkeypatch, damage, why):
    rc, pins, cand, recs, rows, ident, paths = tiny_calibration(tmp_path, monkeypatch)
    out = str(tmp_path / "calib")
    if damage == "config":                                       # an allowed key, but not the registered config any more
        c = json.loads((cand / "config.json").read_text())
        c["temperature"] = [1.9, 1.0, 1.1]
        (cand / "config.json").write_text(json.dumps(c))
    elif damage == "records":
        with open(os.path.join(out, "records.jsonl"), "a") as fh:
            fh.write("\n")
    elif damage == "rows":
        with open(paths["rows"], "a") as fh:
            fh.write("\n")
    else:                                                        # registered again, so only completeness can catch it
        rws, rcs = list(rows), list(recs)
        if damage == "domain":
            keep = {r["id"] for r in rws if r["kind"] == "temp"}
            rws = [r for r in rws if r["id"] in keep]
            rcs = [x for x in rcs if x["sample_id"] in keep]
        elif damage == "view":
            drop = rws[0]["id"]
            rws = [r for r in rws if r["id"] != drop]
            rcs = [x for x in rcs if x["sample_id"] != drop]
        elif damage == "duplicate":
            rcs = rcs + [rcs[0]]
        elif damage == "gold":
            rcs = [dict(rcs[0], gold_answer=next(o for o in rcs[0]["probabilities"] if o != rcs[0]["gold_answer"]))] + rcs[1:]
        else:
            rcs = rcs[1:]
        rows_path = tmp_path / "rows_edited.jsonl"
        rows_path.write_text("".join(json.dumps(r) + "\n" for r in rws))
        p2 = dict(paths, rows=str(rows_path))
        ident2 = rc.identity(paths["candidate"], paths["bundle"], str(rows_path), paths["data_root"], pins, ROOT)
        out = str(tmp_path / "calib2")
        from devision.train.evaluate import runtime_code
        rc.register(out, ident2, rcs, p2, runtime_code(), ROOT)
    with pytest.raises(SystemExit, match=why):
        rc.analyse_registered(out, pins, ROOT)


def test_temporary_bundle_refuses_a_changed_candidate(tmp_path):
    pins = livek_run(tmp_path)
    source(tmp_path, pins)
    cand = candidate(tmp_path, pins)
    c = json.loads((cand / "config.json").read_text())
    c["max_len"] = 64                                            # not a whitelisted key
    (cand / "config.json").write_text(json.dumps(c))
    with pytest.raises(SystemExit, match="not allowed"):
        rb.temp_bundle(str(cand), str(tmp_path / "calib_bundle"), pins)


def test_a_self_consistent_foreign_component_is_refused_before_loading(tmp_path, monkeypatch):
    """R1-03 A: another head with the temporary bundle's own hashes updated (fully self-consistent, same candidate)."""
    rc, pins, cand, recs, rows, ident, paths = tiny_calibration(tmp_path, monkeypatch)
    import shutil
    foreign = tmp_path / "foreign_bundle"
    shutil.copytree(paths["bundle"], foreign)
    torch.manual_seed(99)
    torch.save(R.CharHead(32).state_dict(), foreign / "head.pt")
    j = json.loads((foreign / "reading.json").read_text())
    j["sha256"]["head.pt"] = R.sha(str(foreign / "head.pt"))
    (foreign / "reading.json").write_text(json.dumps(j))
    R.Reading.load(str(cand), str(foreign), device="cpu", full_rules=True)            # the bundle itself loads: it is self-consistent
    with pytest.raises(SystemExit, match="head.pt is not the approved source's"):
        rc.identity(str(cand), str(foreign), paths["rows"], paths["data_root"], pins, ROOT)


def test_a_component_changed_after_registration_is_refused(tmp_path, monkeypatch):
    rc, pins, cand, recs, rows, ident, paths = tiny_calibration(tmp_path, monkeypatch)
    assert set(ident["temporary_components"]) == {"selector.pt", "head.pt", "corrector.pt"}
    with open(os.path.join(paths["bundle"], "selector.pt"), "ab") as fh:
        fh.write(b"x")
    with pytest.raises(SystemExit, match="identity refused|does not verify"):
        rc.analyse_registered(str(tmp_path / "calib"), pins, ROOT)


@pytest.mark.parametrize("case", ["other_content", "missing_module", "same_content_other_path", "changed_after"])
def test_runtime_code_must_be_this_implementation(tmp_path, monkeypatch, case):
    """R1-03 B: the modules collect ran, by name and content (paths only traced)."""
    import copy
    rc, pins, cand, recs, rows, ident, paths = tiny_calibration(tmp_path, monkeypatch)
    from devision.train.evaluate import runtime_code
    rt = copy.deepcopy(runtime_code())
    if case == "changed_after":                                  # the registered run, then this implementation changes
        fake = tmp_path / "reading_edited.py"
        fake.write_text(open(os.path.join(ROOT, "src", "devision", "reading.py")).read() + "\n# edited\n")
        real = rc.module_source
        monkeypatch.setattr(rc, "module_source", lambda n, root: str(fake) if n == "devision.reading" else real(n, root))
        with pytest.raises(SystemExit, match="runtime code"):
            rc.analyse_registered(str(tmp_path / "calib"), pins, ROOT)
        return
    if case == "other_content":
        rt["modules"]["devision.reading"]["sha256"] = "0" * 64
    elif case == "missing_module":
        rt["modules"].pop("devision.image")
    else:
        copy_path = tmp_path / "elsewhere" / "reading.py"
        copy_path.parent.mkdir()
        copy_path.write_text(open(rt["modules"]["devision.reading"]["path"]).read())
        rt["modules"]["devision.reading"]["path"] = str(copy_path)
    out = str(tmp_path / "calib_rt")
    if case == "same_content_other_path":
        rc.register(out, ident, recs, paths, rt, ROOT)
        assert rc.analyse_registered(out, pins, ROOT)["branch"][0] in "1234"
    else:
        with pytest.raises(SystemExit, match="did not run this implementation"):
            rc.register(out, ident, recs, paths, rt, ROOT)


def test_fixed_sources_cannot_be_silently_missing(tmp_path, monkeypatch):
    rc, pins, cand, recs, rows, ident, paths = tiny_calibration(tmp_path, monkeypatch)
    monkeypatch.setattr(rc, "CODE", rc.CODE + ("scripts/release/not_there.py",))
    with pytest.raises(SystemExit, match="fixed source scripts/release/not_there.py missing"):
        rc.identity(paths["candidate"], paths["bundle"], paths["rows"], paths["data_root"], pins, ROOT)
