"""Clean-install check (scripts/release/install_check.py; V04-ENTRY-PENDING-01, V04-INSTALL-R1-01..03) on a tiny bound
checkpoint. The wheel built from this tree, installed with uv.lock's versions into a new environment and run in
isolated mode, agrees with the source environment on the fixed smoke (ordinary / temperature / magnets / mismatched,
through load, off and a plain Decider); a changed module, an import from the source tree, a different table or read,
an off that is not the plain Decider, an invalid or re-chosen answer, a wrong shared route is reported; a smoke that
misses a role, repeats a name, leaves out a question or shows the same picture as its mismatch is refused before
anything is installed; the release check needs a complete frozen identity of this very candidate (or a content-equal
copy) and keeps a record; a code-only identity, another checkpoint or a file changed after the freeze is refused."""
import copy
import importlib.util
import json
import math
import os
import shutil

import pytest
import torch
import yaml
from conftest import tiny_decider
from PIL import Image, ImageDraw

from devision import reading as R

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def script(name):
    spec = importlib.util.spec_from_file_location("t_" + name, os.path.join(ROOT, "scripts", "release", name + ".py"))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ic, rb, ce = script("install_check"), script("rebind"), script("check_eval")
TEMP = "Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature?"
MAG = "Think about the magnetic force between the magnets in each pair. Which of the following statements is true?"


def q(text, opts=("red", "blue")):
    return {"type": "choice", "instructions": text, "criteria": {o: None for o in opts}}


def frozen_candidate(tmp):
    """EXP57-like checkpoint + bundle, an EXP71-like run naming them, then source -> clone -> bind -> freeze."""
    d = tiny_decider(seed=3)
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    d.model.enable_local_views(12)
    d.cfg.local_views, d.cfg.local_overlap, d.cfg.local_source = 12, 0.25, "T"
    d.save(tmp / "exp57")
    torch.manual_seed(7)
    for n, m in (("sel.pt", R.Selector(32, 64)), ("head.pt", R.CharHead(32)), ("corr.pt", R.Selector(32, 64)), ("sel2.pt", R.Selector(32, 64))):
        torch.save(m.state_dict(), tmp / n)
    R.write_bundle(str(tmp / "bundle_on"), str(tmp / "exp57"), str(tmp / "sel.pt"), str(tmp / "head.pt"), {},
                   dims={"selector": [32, 64], "head": 32}, corrector_pt=str(tmp / "corr.pt"))
    run = tmp / "exp71"
    run.mkdir()
    (run / "START.json").write_text(json.dumps({"bundles": {"on": rb.files_of(str(tmp / "exp57"), str(tmp / "bundle_on"))}}))
    (run / "registry.json").write_text("{}")
    (run / "report.json").write_text("{}")
    pins = {k: R.sha(str(run / k)) for k in ("START.json", "registry.json", "report.json")}
    (tmp / "round").mkdir()
    (tmp / "round" / "run.json").write_text("{}")
    rb.source(str(run), str(tmp / "exp57"), str(tmp / "bundle_on"), str(tmp / "round"), pins=pins)
    cand = tmp / "round" / "cand"
    rb.clone(str(tmp / "round"), str(cand), pins)
    rb.bind(str(cand), pins)
    conf = tmp / "full.yaml"
    (tmp / "set.jsonl").write_text("")
    conf.write_text(yaml.safe_dump({"name": "full", "stages": [], "evaluate": {"checkpoint": str(cand), "sets": {"a": {"path": str(tmp / "set.jsonl"), "role": "heldout"}}}}))
    setattr(ce, "SOURCE_PINS", pins)
    assert ce.main(["freeze", str(conf)]) == 0
    return cand, pins, tmp / "round" / "eval" / "IDENTITY.json"


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("install")
    cand, pins, ident = frozen_candidate(tmp)
    pics = {}
    for name, size, text in (("temp", (560, 405), "Mass 14 u  speed 1,360 m/s"), ("mag", (650, 140), "2 cm   3 cm"),
                             ("other", (560, 405), "Mass 9 u  speed 700 m/s")):
        img = Image.new("RGB", size, "white")
        ImageDraw.Draw(img).text((20, size[1] - 25), text, fill="black")
        img.save(tmp / ("%s.png" % name))
        pics[name] = str(tmp / ("%s.png" % name))
    cases = [{"name": "ordinary", "role": "ordinary", "picture": pics["temp"], "state": "",
              "questions": {"a": q("is there a dog ?"), "b": {"type": "noul", "instructions": "Is it red?"}}},
             {"name": "temperature", "role": "temperature", "picture": pics["temp"], "state": "Two samples.", "questions": {"t": q(TEMP)}},
             {"name": "magnets", "role": "magnets", "picture": pics["mag"], "state": "", "questions": {"m": q(MAG)}},
             {"name": "mismatched", "role": "mismatched", "of": "temperature", "picture": pics["other"], "state": "Two samples.",
              "questions": {"t": q(TEMP)}}]
    work = tmp / "work"
    work.mkdir()
    py = ic.install(str(work))
    got = ic.probe(py, str(cand), cases, str(work), isolated=True)
    ref = ic.probe(ic.sys.executable, str(cand), cases, ROOT, isolated=False)
    return {"tmp": tmp, "ck": str(cand), "pins": pins, "identity": str(ident), "cases": cases, "venv": str(work / "venv"), "got": got, "ref": ref}


def check(s, got=None, reference=None, cases=None):
    return ic.compare(s["ref"], got or s["got"], cases or s["cases"], reference or s["ref"]["code"], s["venv"], bound=True)


def test_the_clean_install_agrees(setup):
    got = setup["got"]
    assert check(setup) == [] and ic.case_problems(setup["cases"]) == []
    assert got["file"].startswith(setup["venv"]) and got["status"]["mode"] == "on"
    assert got["on"]["temperature"]["diagnostics"]["route"] == {"t": "read"}
    assert got["on"]["mismatched"]["diagnostics"]["picture"] != got["on"]["temperature"]["diagnostics"]["picture"]


def damaged(setup, damage):
    got = copy.deepcopy(setup["got"])
    ans = got["on"]["temperature"]["answers"]["t"]
    p = ans["probabilities"]
    lo, hi = sorted(p, key=lambda o: p[o])
    if damage == "nan":
        p[lo] = float("nan")
    elif damage == "inf":
        p[lo] = float("inf")
    elif damage == "out_of_range":
        p[lo], p[hi] = -0.1, 1.1
    elif damage == "unnormalised":
        p[lo] = p[lo] + 0.2
    elif damage == "wrong_choice":
        ans["choice"] = lo
    elif damage == "noul_nan":
        got["on"]["ordinary"]["answers"]["b"]["noul"] = float("nan")
    elif damage == "module":
        pass
    elif damage == "source_tree":
        got["file"] = os.path.join(ROOT, "src", "devision", "__init__.py")
    elif damage == "table":
        got["on"]["temperature"]["diagnostics"]["table"][0][2] = "999"
    elif damage == "reads":
        got["on"]["temperature"]["diagnostics"]["reads"][0]["decoded"] = "998"
    elif damage == "off":
        got["off"]["ordinary"]["answers"]["b"]["noul"] += 0.01
    elif damage == "probability":
        p[lo], p[hi] = p[lo] + 0.01, p[hi] - 0.01
    elif damage == "plain_mode":
        got["status"] = {"mode": "none"}
    return got


@pytest.mark.parametrize("damage,why", [("nan", "probabilities"), ("inf", "probabilities"), ("out_of_range", "probabilities"),
                                        ("unnormalised", "sum to"), ("wrong_choice", "not the arg-max"), ("noul_nan", "noul"),
                                        ("module", "content differs"), ("source_tree", "not the new environment"),
                                        ("table", "table differs"), ("reads", "reads differ"), ("off", "off is not the plain Decider"),
                                        ("probability", "probabilities differ"), ("plain_mode", "not the reading combination")])
def test_differences_and_invalid_answers_are_reported(setup, damage, why):
    reference = copy.deepcopy(setup["ref"]["code"])
    if damage == "module":
        reference["modules"]["devision.reading"]["sha256"] = "0" * 64
    assert any(why in x for x in check(setup, damaged(setup, damage), reference)), check(setup, damaged(setup, damage), reference)


def test_a_choice_change_with_equal_probabilities_is_reported():
    a = {"t": {"type": "choice", "choice": "red", "probabilities": {"red": 0.6, "blue": 0.4}}}
    b = {"t": {"type": "choice", "choice": "blue", "probabilities": {"red": 0.6, "blue": 0.4}}}
    assert ic.differ(a, b) and ic.differ(a, {"t": {"type": "choice", "choice": "red", "probabilities": {"red": math.nan, "blue": 0.4}}})
    assert ic.differ(a, {"t": {"type": "choice", "choice": "red", "probabilities": {"red": 0.60005, "blue": 0.39995}}}) == []


def test_a_wrong_route_on_both_sides_is_reported(setup):
    ref, got = copy.deepcopy(setup["ref"]), copy.deepcopy(setup["got"])
    for r in (ref, got):
        r["on"]["magnets"]["diagnostics"]["route"] = {"m": "ordinary"}
    assert any("route" in x for x in ic.compare(ref, got, setup["cases"], ref["code"], setup["venv"], bound=True))


@pytest.mark.parametrize("change,why", [("empty", "no ordinary case"), ("no_temperature", "no temperature case"),
                                        ("no_magnets", "no magnets case"), ("no_mismatched", "no mismatched case"),
                                        ("duplicate", "unique"), ("same_picture", "the same picture"),
                                        ("other_questions", "not the same state and questions"), ("one_ordinary", "too few questions"),
                                        ("wrong_domain", "not of the magnets domain")])
def test_the_smoke_must_be_the_fixed_one(setup, change, why):
    cases = copy.deepcopy(setup["cases"])
    by = {c["name"]: c for c in cases}
    if change == "empty":
        cases = []
    elif change.startswith("no_"):
        cases = [c for c in cases if c["role"] != change[3:]]
    elif change == "duplicate":
        cases.append(dict(by["magnets"]))
    elif change == "same_picture":
        by["mismatched"]["picture"] = by["temperature"]["picture"]
    elif change == "other_questions":
        by["mismatched"]["state"] = "Other text."
    elif change == "one_ordinary":
        by["ordinary"]["questions"].pop("b")
    else:
        by["magnets"]["questions"]["x"] = q("is there a dog ?")
    assert any(why in x for x in ic.case_problems(cases))
    if change in ("empty", "no_magnets"):        # and the comparison does not accept a smaller result either
        assert any("cases" in x for x in ic.compare(setup["ref"], setup["got"], setup["cases"][:1] if change == "empty" else
                                                     [c for c in setup["cases"] if c["role"] != "magnets"], setup["ref"]["code"],
                                                     setup["venv"], bound=True))


def test_release_check_on_the_frozen_candidate_and_its_copy_keeps_a_record(setup, tmp_path):
    record = tmp_path / "record.json"
    assert ic.install_check(setup["ck"], setup["cases"], setup["identity"], record=str(record), pins=setup["pins"]) == []
    rec = json.loads(record.read_text())
    ident = json.loads(open(setup["identity"]).read())
    assert rec["kind"] == "release" and rec["verdict"] == "agrees" and rec["identity"]["sha256"] == R.sha(setup["identity"])
    assert rec["inference_files"]["reading/reading.json"] == ident["files"]["reading/reading.json"] and rec["wheel"]["sha256"]
    assert [c["picture_sha256"] for c in rec["cases"]] == [R.sha(c["picture"]) for c in setup["cases"]]
    copy_dir = tmp_path / "release_copy"
    shutil.copytree(setup["ck"], copy_dir)
    (copy_dir / "README.md").write_text("card")                     # a release copy carries other files too
    assert ic.install_check(str(copy_dir), setup["cases"], setup["identity"], record=str(tmp_path / "r2.json"), pins=setup["pins"]) == []


def test_development_self_check_is_marked(setup, tmp_path):
    record = tmp_path / "dev.json"
    assert ic.install_check(setup["ck"], setup["cases"], record=str(record)) == []
    assert json.loads(record.read_text())["kind"].startswith("development self-check")


@pytest.mark.parametrize("bad,why", [("code_only", "not a complete frozen identity"), ("other_reading", "differ from IDENTITY.json"),
                                     ("changed_after_freeze", "differs from IDENTITY.json")])
def test_release_check_refuses_before_installing(setup, tmp_path, monkeypatch, bad, why):
    monkeypatch.setattr(ic, "install", lambda *a: pytest.fail("installed despite a wrong identity"))
    ident, ck = setup["identity"], setup["ck"]
    if bad == "code_only":
        p = tmp_path / "IDENTITY.json"
        p.write_text(json.dumps({"code": setup["ref"]["code"]}))
        ident = str(p)
    elif bad == "other_reading":                      # same code, another bundle bound to the same decision files
        ck = str(tmp_path / "other")
        shutil.copytree(setup["ck"], ck)
        shutil.rmtree(os.path.join(ck, "reading"))
        R.write_bundle(os.path.join(ck, "reading"), ck, str(setup["tmp"] / "sel2.pt"), str(setup["tmp"] / "head.pt"), {},
                       dims={"selector": [32, 64], "head": 32}, corrector_pt=str(setup["tmp"] / "corr.pt"))
    else:                                             # the frozen candidate itself, changed after the freeze
        head = os.path.join(ck, "reading", "head.pt")
        saved = open(head, "rb").read()
        with open(head, "ab") as fh:
            fh.write(b"x")
    try:
        problems = ic.install_check(ck, setup["cases"], ident, record=str(tmp_path / "r.json"), pins=setup["pins"])
    finally:
        if bad == "changed_after_freeze":
            with open(head, "wb") as fh:
                fh.write(saved)
    assert any(why in x for x in problems), problems
    assert json.loads((tmp_path / "r.json").read_text())["verdict"] == "refused"
