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
