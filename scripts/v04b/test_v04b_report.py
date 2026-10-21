"""Candidate B screen and endpoint identity without a model (critic V04-B-R2-01 / 02): the start is allowed only when
exp39's report, the actual start file and the given sha agree; screening files are tied to the frozen final weights
and config, and a change of the weights, of the config (temperatures) or of a recorded file at the same path is
refused; missing training / required sets / the whole run directory give a written "incomplete"."""
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(__file__)


def load(name):
    spec = importlib.util.spec_from_file_location("v04b_" + name, os.path.join(HERE, name + ".py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r, fz = load("report"), load("freeze")


def setup(tmp_path, statuses, required_ok=True):
    run = tmp_path / "v04b-fit"
    (run / "stage2").mkdir(parents=True)
    (run / "stage2" / "model.safetensors").write_bytes(b"w")
    (run / "stage2" / "config.json").write_text(json.dumps({"temperature": [1.5, 1.0, 1.2]}))
    (run / "eval").mkdir()
    st = tmp_path / "status.txt"
    st.write_text("".join("%s %s\n" % x for x in statuses))
    return str(run), str(st)


def all_done():
    return [("train", "completed")] + [("eval:" + n, "completed") for n in r.REQUIRED]


def write_evals(run, checkpoint):
    for n in r.REQUIRED:
        json.dump({"run": {"checkpoint": checkpoint, "temperature": "fitted"}}, open(os.path.join(run, "eval", n + ".json"), "w"))
        open(os.path.join(run, "eval", n + ".details.jsonl"), "w").write("{}\n")


def test_start_needs_report_file_and_given_sha_to_agree(tmp_path):
    start = tmp_path / "h.safetensors"
    start.write_bytes(b"h-weights")
    h = fz.sha(str(start))
    rep = tmp_path / "exp39-report.json"
    rep.write_text(json.dumps({"invest_in_512": True, "identity": {"H": {"model.safetensors": h}}}))
    assert fz.start_allowed(str(rep), str(start), h) == ""
    rep.write_text(json.dumps({"invest_in_512": True, "identity": {"H": {"model.safetensors": "other"}}}))
    assert fz.start_allowed(str(rep), str(start), h).startswith("start weights disagree")      # another H's report
    rep.write_text(json.dumps({"invest_in_512": False, "identity": {"H": {"model.safetensors": h}}}))
    assert fz.start_allowed(str(rep), str(start), h).startswith("exp39 did not meet")


def test_frozen_identity_refuses_changes_at_the_same_path(tmp_path):
    run, st = setup(tmp_path, all_done())
    write_evals(run, os.path.join(run, "stage2"))
    fz.freeze(run)
    for n in r.REQUIRED:
        fz.record(run, n)
    assert fz.check(run, list(r.REQUIRED))["weights"]["config.json"]
    open(os.path.join(run, "eval", "dev16.details.jsonl"), "w").write("{\"changed\": 1}\n")     # a recorded file
    with pytest.raises(SystemExit):
        fz.check(run, list(r.REQUIRED))
    run2, _ = setup(tmp_path / "b", all_done())
    write_evals(run2, os.path.join(run2, "stage2"))
    fz.freeze(run2)
    open(os.path.join(run2, "stage2", "config.json"), "w").write(json.dumps({"temperature": [9.0, 1.0, 1.2]}))
    with pytest.raises(SystemExit):                                                           # temperatures changed
        fz.record(run2, "sqa_temp")
    run3, _ = setup(tmp_path / "c", all_done())
    fz.freeze(run3)
    open(os.path.join(run3, "stage2", "model.safetensors"), "wb").write(b"other weights")
    with pytest.raises(SystemExit):
        fz.check(run3)


def test_report_incomplete_cases_and_identity(tmp_path):
    run, st = setup(tmp_path, [("train", "failed")])
    out = r.build(run, "abc", st)
    assert out["outcome"].startswith("incomplete: train") and out["identity"]["model.safetensors"] is None
    run, st = setup(tmp_path / "b", [s for s in all_done() if s[0] != "eval:syn_erased"])
    assert r.build(run, "abc", st)["outcome"] == "incomplete: syn_erased"
    run, st = setup(tmp_path / "c", all_done())                       # everything completed but nothing frozen
    assert r.build(run, "abc", st)["outcome"].startswith("incomplete")
    run, st = setup(tmp_path / "d", all_done())
    write_evals(run, "runs/other/stage2")                             # frozen and recorded, but another checkpoint
    fz.freeze(run)
    for n in r.REQUIRED:
        fz.record(run, n)
    with pytest.raises(SystemExit):
        r.build(run, "abc", st)


def test_no_run_directory_still_writes_incomplete(tmp_path):
    (tmp_path / "status.txt").write_text("train missing\n")
    run = str(tmp_path / "v04b-fit")                                   # never created
    out = r.build(run, "abc", str(tmp_path / "status.txt"))
    assert out["outcome"].startswith("incomplete") and out["identity"]["model.safetensors"] is None
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        os.makedirs("runs")
        os.rename("status.txt", os.path.join("runs", "v04b-status.txt"))
        assert r.main(["runs/v04b-fit", "abc"]) == 0
        written = json.load(open(os.path.join("runs", "v04b-screen.json")))
    finally:
        os.chdir(cwd)
    assert written["outcome"].startswith("incomplete") and not os.path.exists(os.path.join(tmp_path, "runs", "v04b-fit"))


def test_delivery_reads_the_screen_identity(tmp_path, monkeypatch):
    """release/delivery.report_weights on the written screen: a checked complete screen gives its frozen weights, so
    it labels that candidate only; an incomplete screen gives no identity."""
    spec = importlib.util.spec_from_file_location("delivery", os.path.join(HERE, "..", "release", "delivery.py"))
    assert spec is not None and spec.loader is not None
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    run, st = setup(tmp_path, all_done())
    write_evals(run, os.path.join(run, "stage2"))
    fz.freeze(run)
    for n in r.REQUIRED:
        fz.record(run, n)
    real_check = fz.check
    stub = {"transfer": {"accuracy_change": {"value": 0.0, "ci95": [0, 0]}, "gain_change": {"value": 0, "ci95": [0, 0]}}}
    monkeypatch.setattr(r, "transfer", lambda a, b: stub["transfer"])                  # statistics are not the point here
    monkeypatch.setattr(r, "recs", lambda d_, n: [])
    monkeypatch.setattr(r, "paired_metric", lambda *a, **k: {"difference": 0.0, "ci95": [-0.01, 0.01]})
    monkeypatch.setattr(r, "syn_index", lambda rs: {})
    monkeypatch.setattr(r, "syn_stat", lambda *a, **k: {"value": 0.9, "ci95": [0.85, 0.95]})
    monkeypatch.setattr(r, "binding", lambda rs, p: {})
    monkeypatch.setattr(r.r32, "by_family", lambda rs: rs)
    out = r.build(run, "abc", st)
    assert fz.check is real_check and out["model_sha256"] == out["identity"]["model.safetensors"]
    screen = tmp_path / "screen.json"
    screen.write_text(json.dumps(out, default=str))
    w, src = d.report_weights(str(screen), None)
    assert w == out["identity"]["model.safetensors"] and src == "the report itself"
    own = {"files": {"model.safetensors": w}}
    other = {"files": {"model.safetensors": "another candidate"}}
    assert d.screening(own, "v04b", out, w, src)["status"].startswith("bound")
    assert d.screening(other, "v04b", out, w, src)["status"].startswith("unverifiable")     # cannot borrow the label
    run2, st2 = setup(tmp_path / "b", [("train", "failed")])
    inc = r.build(run2, "abc", st2)
    screen.write_text(json.dumps(inc, default=str))
    assert "model_sha256" not in inc and d.report_weights(str(screen), None)[0] is None
