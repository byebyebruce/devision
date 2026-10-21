"""The reading combination's identity hand-over on tiny files (critic V04-EVIDENCE-R3-02, V04-ENTRY-R1-01..03;
docs/experiments/2026-10-19-v04-reader-evidence-prep03.md 5.2): source (the pinned run's START bundles.on) -> clone ->
calibrate / name in the candidate -> bind (scripts/release/rebind.py) -> freeze -> evaluate -> seal / check
(scripts/release/check_eval.py) -> build (scripts/release/hf.py). The bound candidate loads as "on"; another
self-consistent source, a bundle left bound to the old config, a changed component, a disallowed config change or a
changed weight is refused; an incomplete bundle is refused by load / freeze / build (only an explicit "off" loads);
an "off" output, other code, or a route / domain hidden in the details is not evidence; deleted or changed lineage and
a rebound checkpoint are refused after the freeze; build copies exactly the frozen files and refuses to rename."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import pytest
import torch
import yaml
from conftest import tiny_decider
from PIL import Image, ImageDraw

import devision
from devision import reading as R

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", "release", name + ".py"))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rb, ce, hf = script("rebind"), script("check_eval"), script("hf")
TEMP = "Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature?"


def local_view(seed):
    d = tiny_decider(seed=seed)
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    d.model.enable_local_views(12)
    d.cfg.local_views, d.cfg.local_overlap, d.cfg.local_source = 12, 0.25, "T"
    return d


@pytest.fixture
def src(tmp_path, monkeypatch):
    """A decision checkpoint and a bundle bound to it (as EXP57 + bundle_on), and a run whose START records them as
    bundles.on (as EXP71); the release scripts are pinned to that run."""
    local_view(3).save(tmp_path / "exp57")
    torch.manual_seed(7)
    for n, m in (("sel.pt", R.Selector(32, 64)), ("head.pt", R.CharHead(32)), ("corr.pt", R.Selector(32, 64)), ("sel2.pt", R.Selector(32, 64))):
        torch.save(m.state_dict(), tmp_path / n)
    R.write_bundle(str(tmp_path / "bundle_on"), str(tmp_path / "exp57"), str(tmp_path / "sel.pt"), str(tmp_path / "head.pt"), {},
                   dims={"selector": [32, 64], "head": 32}, corrector_pt=str(tmp_path / "corr.pt"))
    run = tmp_path / "exp71"
    run.mkdir()
    (run / "START.json").write_text(json.dumps({"bundles": {"on": rb.files_of(str(tmp_path / "exp57"), str(tmp_path / "bundle_on"))}}))
    (run / "registry.json").write_text("{}")
    (run / "report.json").write_text("{}")
    pins = {k: R.sha(str(run / k)) for k in ("START.json", "registry.json", "report.json")}
    monkeypatch.setattr(ce, "SOURCE_PINS", pins)
    monkeypatch.setattr(hf, "SOURCE_PINS", pins)
    (tmp_path / "round").mkdir()
    (tmp_path / "round" / "run.json").write_text("{}")     # a round directory: results go to round/eval
    return tmp_path, pins


def registered(s):
    tmp, pins = s
    return rb.source(str(tmp / "exp71"), str(tmp / "exp57"), str(tmp / "bundle_on"), str(tmp / "round"), pins=pins)


def calibrated(s, name="devision-v9.9", **extra):
    """source + clone + what calibration and naming change in the candidate's config."""
    tmp, pins = s
    registered(s)
    cand = tmp / "round" / "cand"
    rb.clone(str(tmp / "round"), str(cand), pins)
    c = json.loads((cand / "config.json").read_text())
    c.update(temperature=[1.3, 1.0, 1.1], temperature_by_options={"choice:2": 1.2}, model_name=name, **extra)
    (cand / "config.json").write_text(json.dumps(c))
    return cand


def test_calibrated_copy_binds_and_loads_on(src):
    tmp, pins = src
    cand = calibrated(src)
    lineage = rb.bind(str(cand), src[1])
    assert set(lineage["config_changes"]) == {"temperature", "temperature_by_options", "model_name"}
    assert lineage["source_json_sha256"] == R.sha(str(tmp / "round" / "reading_lineage" / "source.json"))
    rd = devision.load(str(cand), device="cpu")
    assert isinstance(rd, R.Reading) and rd.corrector is not None and rd.cfg.temperature == [1.3, 1.0, 1.1]
    assert rd.reading_status["reading_json_sha256"] == lineage["reading_json_sha256"]
    assert rb.lineage_problems(str(cand), pins) == [] and not (tmp / "exp57" / "reading").exists()


def test_another_self_consistent_source_is_refused(src):
    tmp, pins = src
    R.write_bundle(str(tmp / "bundle_other"), str(tmp / "exp57"), str(tmp / "sel2.pt"), str(tmp / "head.pt"), {},
                   dims={"selector": [32, 64], "head": 32}, corrector_pt=str(tmp / "corr.pt"))   # binds EXP57 correctly
    with pytest.raises(SystemExit, match="are not the files"):
        rb.source(str(tmp / "exp71"), str(tmp / "exp57"), str(tmp / "bundle_other"), str(tmp / "round"), pins=pins)
    with pytest.raises(SystemExit, match="not the pinned source"):
        rb.source(str(tmp / "exp71"), str(tmp / "exp57"), str(tmp / "bundle_on"), str(tmp / "round"), pins=dict(pins, **{"START.json": "0" * 64}))


@pytest.mark.parametrize("damage,why", [("old_config", "does not bind"), ("component", "records as bundles.on"),
                                        ("config_key", "not allowed"), ("weight", "differs from the source")])
def test_bind_refusals(src, damage, why):
    tmp, _ = src
    if damage == "old_config":                     # calibrated again after binding: the bundle binds the old config
        cand = calibrated(src)
        rb.bind(str(cand), src[1])
        c = json.loads((cand / "config.json").read_text())
        c["temperature"] = [2.0, 1.0, 2.0]
        (cand / "config.json").write_text(json.dumps(c))
        with pytest.raises(ValueError, match=why):
            devision.load(str(cand), device="cpu")
        return
    cand = calibrated(src, max_len=64) if damage == "config_key" else calibrated(src)
    if damage == "component":
        with open(tmp / "bundle_on" / "head.pt", "ab") as fh:
            fh.write(b"x")
    if damage == "weight":
        with open(cand / "model.safetensors", "ab") as fh:
            fh.write(b"x")
    with pytest.raises(SystemExit, match=why):
        rb.bind(str(cand), src[1])
    assert not (cand / "reading").exists()


def samples(root):
    (root / "img").mkdir(exist_ok=True)
    rows = []
    for i in range(2):
        img = Image.new("RGB", (560, 405), ("white", "lightyellow")[i])
        ImageDraw.Draw(img).text((20, 380), "Mass %d u" % (10 + i), fill="black")
        img.save(root / "img" / ("%d.png" % i))
        rows.append({"id": "s%d" % i, "image": "img/%d.png" % i, "image_id": "p%d" % i,
                     "questions": {"t": {"type": "choice", "instructions": TEMP, "criteria": {"red": None, "blue": None}},
                                   "o": {"type": "choice", "instructions": "is there a dog ?", "criteria": {"red": None, "blue": None}}},
                     "gold": {"t": {"probabilities": {"red": 1.0, "blue": 0.0}}, "o": {"probabilities": {"red": 0.0, "blue": 1.0}}}})
    (root / "set.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(root / "set.jsonl")


def plan_for(tmp, cand, data):
    conf = tmp / "full.yaml"
    conf.write_text(yaml.safe_dump({"name": "full", "stages": [], "evaluate": {"checkpoint": str(cand), "sets": {"a": {"path": data, "role": "heldout"}}}}))
    return str(conf)


@pytest.fixture
def frozen(src):
    tmp, _ = src
    cand = calibrated(src)
    rb.bind(str(cand), src[1])
    data = samples(tmp)
    conf = plan_for(tmp, cand, data)
    assert ce.main(["freeze", conf]) == 0
    ident = json.loads((tmp / "round" / "eval" / "IDENTITY.json").read_text())
    assert "reading/reading.json" in ident["files"] and set(ident["reading"]["lineage"]) == {"source.json", "lineage.json"}
    assert {"devision.reading", "devision.train.evaluate"} <= set(ident["code"]["modules"])
    return tmp, cand, data, conf


def evaluate(cand, data, root, out, *extra):
    """devision-eval in its own process, as the pipeline runs it (the code identity is that process's)."""
    args = ["--checkpoint", str(cand), "--data", data, "--data-root", str(root), "--out", out, "--role", "heldout", "--device", "cpu", *extra]
    subprocess.run([sys.executable, "-c", "import sys; from devision.train.cli import eval_main; eval_main(sys.argv[1:])", *args],
                   check=True, capture_output=True)


def content_of(conf, out, control="none"):
    p = ce.Plan(conf)
    ident, _ = ce.identity_problems(p)
    data = p.outputs[0].data
    return ce.content(ce.Output(out, data, control, "heldout"), p.checkpoint, ident["temperature"], {}, ident)


def rewrite(a, recs=None, summary=None):
    if recs is not None:
        with open(ce.details_of(a), "w") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in recs))
    if summary is not None:
        with open(a, "w") as fh:
            json.dump(summary, fh)


def test_on_output_is_sealed_off_and_other_code_are_not(frozen):
    tmp, cand, data, conf = frozen
    a = str(tmp / "round" / "eval" / "a.json")
    evaluate(cand, data, tmp, a)
    s = json.loads(open(a).read())
    assert s["run"]["reading"]["mode"] == "on" and s["run"]["reading"]["read_questions"] == 2
    assert ce.main(["seal", conf]) == 0 and ce.main(["check", conf]) == 0
    s["run"]["code_identity"]["modules"]["devision.reading"]["sha256"] = "0" * 64     # as if other code had run
    rewrite(a, summary=s)
    assert any("devision.reading: content differs" in x for x in content_of(conf, a))
    os.remove(a)
    os.remove(ce.details_of(a))
    evaluate(cand, data, tmp, a, "--reading", "off")
    assert ce.main(["check", conf]) == 1                                              # off: not evidence of "on"
    assert any("not an output of the frozen reading combination" in w for w in content_of(conf, a))


@pytest.mark.parametrize("hide", ["route", "domain", "all_zero"])
def test_hidden_unread_questions_are_refused(frozen, hide):
    """V04-ENTRY-R1-02: whether a question had to be read comes from the data file's own text, not the record."""
    tmp, cand, data, conf = frozen
    a = str(tmp / "round" / "eval" / "a.json")
    evaluate(cand, data, tmp, a)
    assert content_of(conf, a) == []
    recs = [json.loads(x) for x in open(ce.details_of(a))]
    s = json.loads(open(a).read())
    for r in recs:
        if r["qid"] == "t":
            r["route"] = "ordinary"
            if hide in ("domain", "all_zero"):
                r["reading_domain"] = None
    if hide == "all_zero":
        s["run"]["reading"].update(routed_questions=0, read_questions=0)
    rewrite(a, recs, s)
    assert any("routed questions (from the data file) not read" in w for w in content_of(conf, a))
    assert ce.main(["seal", conf]) == 1                                               # never sealed as evidence


def test_reversed_and_mismatched_on_outputs_pass(frozen):
    tmp, cand, data, conf = frozen
    for control in ("reversed", "mismatched"):
        out = str(tmp / "round" / "eval" / ("a.%s.json" % control))
        evaluate(cand, data, tmp, out, "--control", control)
        assert content_of(conf, out, control) == [], control


@pytest.mark.parametrize("damage", ["delete_source", "edit_lineage", "rebind"])
def test_lineage_changes_after_the_freeze_are_refused(frozen, damage):
    tmp, cand, data, conf = frozen
    meta = tmp / "round" / "reading_lineage"
    if damage == "delete_source":
        os.remove(meta / "source.json")
    elif damage == "edit_lineage":
        lin = json.loads((meta / "lineage.json").read_text())
        lin["note"] = "edited"
        (meta / "lineage.json").write_text(json.dumps(lin))
    else:                                       # recalibrated after the freeze and bound again, self-consistently
        shutil.rmtree(cand / "reading")
        c = json.loads((cand / "config.json").read_text())
        c["temperature"] = [1.7, 1.0, 1.7]
        (cand / "config.json").write_text(json.dumps(c))
        rb.bind(str(cand), ce.SOURCE_PINS)
    assert ce.main(["preflight", conf]) == 1 and ce.main(["check", conf]) == 1
    _, bad = ce.identity_problems(ce.Plan(conf))
    assert bad
    with pytest.raises(SystemExit):
        build(tmp, "devision-v9.9")


def test_incomplete_bundle_is_refused_everywhere(src):
    tmp, _ = src
    cand = calibrated(src)
    rb.bind(str(cand), src[1])
    os.remove(cand / "reading" / "reading.json")                 # components kept, manifest gone
    assert R.bundle_state(str(cand)) == "incomplete"
    for mode in ("auto", "on"):
        with pytest.raises(ValueError, match="incomplete reading/ bundle"):
            devision.load(str(cand), device="cpu", reading=mode)
    off = devision.load(str(cand), device="cpu", reading="off")
    assert off.reading_status["mode"] == "off" and off.reading_status["bundle"] == "incomplete"
    conf = plan_for(tmp, cand, samples(tmp))
    assert ce.main(["freeze", conf]) == 1 and not (tmp / "round" / "eval" / "IDENTITY.json").exists()
    with pytest.raises(SystemExit, match="incomplete reading/ bundle"):
        build(tmp, "devision-v9.9")


def build(tmp, name, out="build"):
    rel = tmp / "release" / "v9.9"
    rel.mkdir(parents=True, exist_ok=True)
    (rel / "release.yaml").write_text(yaml.safe_dump({
        "version": "v9.9", "git_tag": "model-v9.9", "model_name": name, "round": str(tmp / "round"), "stage": "cand",
        "hub_repo": "someone/devision-test", "license": "apache-2.0", "base_model": ["convaiinnovations/laya"], "datasets": ["x"],
        "smoke_questions": {"dog": {"type": "noul", "instructions": "Is there a dog in the image?"}}}))
    (rel / "notes.md").write_text("## training\nTrained.\n\n## limitations\n- Tiny.\n")
    return hf.build(str(rel), str(tmp / out))


def test_build_only_after_freeze_and_exactly_the_frozen_files(src):
    tmp, _ = src
    cand = calibrated(src)
    rb.bind(str(cand), src[1])
    with pytest.raises(SystemExit, match="no frozen identity"):
        build(tmp, "devision-v9.9")
    conf = plan_for(tmp, cand, samples(tmp))
    assert ce.main(["freeze", conf]) == 0
    out = build(tmp, "devision-v9.9")
    ident = json.loads((tmp / "round" / "eval" / "IDENTITY.json").read_text())
    for f in ("config.json", "model.safetensors", "reading/reading.json", "reading/corrector.pt"):
        assert R.sha(os.path.join(out, f)) == ident["files"][f], f
    assert isinstance(devision.load(out, device="cpu"), R.Reading)
    with pytest.raises(SystemExit, match="set it before binding"):
        build(tmp, "devision-other", "build2")
    with open(cand / "reading" / "head.pt", "ab") as fh:            # changed after the freeze
        fh.write(b"x")
    with pytest.raises(SystemExit, match="differs from its frozen identity"):
        build(tmp, "devision-v9.9", "build3")


@pytest.mark.parametrize("swap", ["other_run", "other_bundle", "pinned_file_changed"])
def test_source_replaced_or_changed_before_the_freeze_is_refused(src, swap):
    """V04-ENTRY-R2: source.json is checked against the pinned run every time it is used -- a record replaced after
    `source` by another self-consistent one, or a pinned run file changed, stops clone, bind and freeze."""
    tmp, pins = src
    registered(src)
    path = tmp / "round" / "reading_lineage" / "source.json"
    R.write_bundle(str(tmp / "bundle_other"), str(tmp / "exp57"), str(tmp / "sel2.pt"), str(tmp / "head.pt"), {},
                   dims={"selector": [32, 64], "head": 32}, corrector_pt=str(tmp / "corr.pt"))
    other = rb.files_of(str(tmp / "exp57"), str(tmp / "bundle_other"))
    rec = json.loads(path.read_text())
    if swap == "other_run":                  # a run of its own whose START names the other bundle; record self-consistent
        run2 = tmp / "exp71b"
        run2.mkdir()
        (run2 / "START.json").write_text(json.dumps({"bundles": {"on": other}}))
        for k in ("registry.json", "report.json"):
            (run2 / k).write_text("{}")
        rec.update(run=str(run2), bundle=str(tmp / "bundle_other"), bundle_on=other,
                   pins={k: R.sha(str(run2 / k)) for k in ("START.json", "registry.json", "report.json")})
    elif swap == "other_bundle":             # the pinned run kept, the record pointed at another bundle
        rec.update(bundle=str(tmp / "bundle_other"), bundle_on=other)
    path.write_text(json.dumps(rec))
    if swap == "pinned_file_changed":
        with open(tmp / "exp71" / "registry.json", "a") as fh:
            fh.write(" ")
    cand = tmp / "round" / "cand"
    with pytest.raises(SystemExit, match="refusing to clone"):
        rb.clone(str(tmp / "round"), str(cand), pins)
    shutil.copytree(tmp / "exp57", cand, ignore=shutil.ignore_patterns("reading"))    # a candidate made by hand
    with pytest.raises(SystemExit, match="refusing to bind"):
        rb.bind(str(cand), pins)
    assert rb.lineage_problems(str(cand), pins)
    if swap == "pinned_file_changed":        # bound before the change: freeze still refuses
        with open(tmp / "exp71" / "registry.json", "w") as fh:
            fh.write("{}")
        rb.bind(str(cand), pins)
        with open(tmp / "exp71" / "registry.json", "a") as fh:
            fh.write(" ")
        assert ce.main(["freeze", plan_for(tmp, cand, samples(tmp))]) == 1
