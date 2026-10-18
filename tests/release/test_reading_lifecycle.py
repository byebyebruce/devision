"""The reading combination's identity hand-over on tiny files (critic V04-EVIDENCE-R3-02; docs/experiments/2026-10-19-
v04-reader-evidence-prep03.md 5.2): clone -> calibrate / name in the candidate -> bind (scripts/release/rebind.py) ->
freeze -> evaluate -> seal / check (scripts/release/check_eval.py) -> build (scripts/release/hf.py). The bound candidate
loads as "on" with its components; lineage records exactly the allowed config changes; a bundle left bound to the old
config, a changed component, a disallowed config change or a changed weight is refused; an "off" output or one from
other code is not evidence of the combination; build copies the evaluated inference files unchanged and refuses to
rename a bound checkpoint."""
import importlib.util
import json
import os

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


@pytest.fixture
def source(tmp_path):
    """A source decision checkpoint (local views 12 / 0.25 / T) and a bundle bound to it, as EXP57 + bundle_on."""
    d = tiny_decider(seed=3)
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    d.model.enable_local_views(12)
    d.cfg.local_views, d.cfg.local_overlap, d.cfg.local_source = 12, 0.25, "T"
    d.save(tmp_path / "exp57")
    torch.manual_seed(7)
    for n, m in (("sel.pt", R.Selector(32, 64)), ("head.pt", R.CharHead(32)), ("corr.pt", R.Selector(32, 64))):
        torch.save(m.state_dict(), tmp_path / n)
    R.write_bundle(str(tmp_path / "bundle_on"), str(tmp_path / "exp57"), str(tmp_path / "sel.pt"), str(tmp_path / "head.pt"), {},
                   dims={"selector": [32, 64], "head": 32}, corrector_pt=str(tmp_path / "corr.pt"))
    return tmp_path


def calibrated(src, name="devision-v9.9", **extra):
    """clone + what calibration and naming change in the candidate's config."""
    (src / "round").mkdir(exist_ok=True)
    (src / "round" / "run.json").write_text("{}")         # a round directory: results go to round/eval, not into the checkpoint
    cand = src / "round" / "cand"
    rb.clone(str(src / "exp57"), str(cand))
    c = json.loads((cand / "config.json").read_text())
    c.update(temperature=[1.3, 1.0, 1.1], temperature_by_options={"choice:2": 1.2}, model_name=name, **extra)
    (cand / "config.json").write_text(json.dumps(c))
    return cand


def bind(src, cand):
    return rb.bind(str(src / "exp57"), str(src / "bundle_on"), str(cand), str(src / "meta"))


def test_calibrated_copy_binds_and_loads_on(source):
    cand = calibrated(source)
    lineage = bind(source, cand)
    assert set(lineage["config_changes"]) == {"temperature", "temperature_by_options", "model_name"}
    assert lineage["components"] == {k: json.loads((source / "bundle_on" / "reading.json").read_text())["sha256"][k]
                                     for k in ("selector.pt", "head.pt", "corrector.pt")}
    rd = devision.load(str(cand), device="cpu")
    assert isinstance(rd, R.Reading) and rd.corrector is not None and rd.cfg.temperature == [1.3, 1.0, 1.1]
    assert rd.reading_status["reading_json_sha256"] == lineage["reading_json_sha256"]
    assert not (source / "exp57" / "reading").exists()                     # the source stays as it was


@pytest.mark.parametrize("damage,why", [("old_config", "does not bind"), ("component", "changed since it was bound"),
                                        ("config_key", "not allowed"), ("weight", "differs from the source")])
def test_refusals(source, damage, why):
    if damage == "old_config":                     # calibrated again after binding: the bundle binds the old config
        cand = calibrated(source)
        bind(source, cand)
        c = json.loads((cand / "config.json").read_text())
        c["temperature"] = [2.0, 1.0, 2.0]
        (cand / "config.json").write_text(json.dumps(c))
        with pytest.raises(ValueError, match=why):
            devision.load(str(cand), device="cpu")
        return
    cand = calibrated(source, max_len=64) if damage == "config_key" else calibrated(source)
    if damage == "component":
        with open(source / "bundle_on" / "head.pt", "ab") as fh:
            fh.write(b"x")
    if damage == "weight":
        with open(cand / "model.safetensors", "ab") as fh:
            fh.write(b"x")
    with pytest.raises(SystemExit, match=why):
        bind(source, cand)
    assert not (cand / "reading").exists()


def samples(root):
    (root / "img").mkdir()
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


@pytest.fixture
def frozen(source):
    cand = calibrated(source)
    bind(source, cand)
    data = samples(source)
    conf = source / "full.yaml"
    conf.write_text(yaml.safe_dump({"name": "full", "stages": [], "evaluate": {"checkpoint": str(cand),
                                                                               "sets": {"a": {"path": data, "role": "heldout"}}}}))
    assert ce.main(["freeze", str(conf)]) == 0
    ident = json.loads((source / "round" / "eval" / "IDENTITY.json").read_text())
    assert "reading/reading.json" in ident["files"] and ident["reading"]["reading_json_sha256"] == R.sha(str(cand / "reading" / "reading.json"))
    assert {"devision.reading", "devision.train.evaluate"} <= set(ident["code"]["modules"])
    return source, cand, data, conf


def evaluate(cand, data, root, out, *extra):
    """devision-eval in its own process, as the pipeline runs it (the code identity is that process's)."""
    import subprocess
    import sys
    args = ["--checkpoint", str(cand), "--data", data, "--data-root", str(root), "--out", out, "--role", "heldout", "--device", "cpu", *extra]
    subprocess.run([sys.executable, "-c", "import sys; from devision.train.cli import eval_main; eval_main(sys.argv[1:])", *args],
                   check=True, capture_output=True)


def test_on_output_is_sealed_off_and_other_code_are_not(frozen):
    src, cand, data, conf = frozen
    a = str(src / "round" / "eval" / "a.json")
    evaluate(cand, data, src, a)
    s = json.loads(open(a).read())
    assert s["run"]["reading"]["mode"] == "on" and s["run"]["reading"]["read_questions"] == 2
    assert ce.main(["seal", str(conf)]) == 0 and ce.main(["check", str(conf)]) == 0
    s["run"]["code_identity"]["modules"]["devision.reading"]["sha256"] = "0" * 64     # as if other code had run
    rows = ce.Plan(str(conf))
    why = ce.reading_problems(a, s["run"], [json.loads(x) for x in open(ce.details_of(a))], json.loads((src / "round" / "eval" / "IDENTITY.json").read_text()))
    assert any("devision.reading: content differs" in x for x in why) and rows.outputs
    os.remove(a)
    os.remove(ce.details_of(a))
    evaluate(cand, data, src, a, "--reading", "off")
    assert ce.main(["check", str(conf)]) == 1                                         # off: not evidence of "on"
    st = ce.status(ce.Plan(str(conf)), *ce.identity_problems(ce.Plan(str(conf))), {})
    assert st[0][1] == "invalid" and any("not an output of the frozen reading combination" in w for w in st[0][2])


def test_build_copies_the_evaluated_files_and_never_renames(source, tmp_path):
    round_dir = tmp_path / "runs" / "cand-round"
    round_dir.mkdir(parents=True)
    (round_dir / "run.json").write_text("{}")
    cand = calibrated(source, name="devision-v9.9")
    bind(source, cand)
    os.rename(cand, round_dir / "stage2")
    rel = tmp_path / "release" / "v9.9"
    rel.mkdir(parents=True)
    spec = {"version": "v9.9", "git_tag": "model-v9.9", "model_name": "devision-v9.9", "round": str(round_dir), "stage": "stage2",
            "hub_repo": "someone/devision-test", "license": "apache-2.0", "base_model": ["convaiinnovations/laya"], "datasets": ["x"],
            "smoke_questions": {"dog": {"type": "noul", "instructions": "Is there a dog in the image?"}}}
    (rel / "release.yaml").write_text(yaml.safe_dump(spec))
    (rel / "notes.md").write_text("## training\nTrained.\n\n## limitations\n- Tiny.\n")
    out = tmp_path / "build"
    hf.build(str(rel), str(out))
    for f in ("config.json", "model.safetensors", "reading/reading.json", "reading/corrector.pt"):
        assert R.sha(str(out / f)) == R.sha(str(round_dir / "stage2" / f)), f
    assert isinstance(devision.load(str(out), device="cpu"), R.Reading)
    spec["model_name"] = "devision-other"
    (rel / "release.yaml").write_text(yaml.safe_dump(spec))
    with pytest.raises(SystemExit, match="set it before binding"):
        hf.build(str(rel), str(tmp_path / "build2"))
