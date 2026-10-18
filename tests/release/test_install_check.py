"""Clean-install check (scripts/release/install_check.py; V04-ENTRY-PENDING-01) on a tiny bound checkpoint: the wheel
built from this tree, installed with uv.lock's versions into a new environment and run in isolated mode, gives the same
routes, tables, states, pictures and probabilities as the source environment for ordinary, temperature, magnet and
mismatched-picture requests; its code is the reference's; "off" is the plain Decider. A changed module, an import from
the source tree, a different table or an "off" that is not the plain Decider is reported."""
import copy
import importlib.util
import json
import os

import pytest
import torch
from conftest import tiny_decider
from PIL import Image, ImageDraw

from devision import reading as R

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("install_check", os.path.join(ROOT, "scripts", "release", "install_check.py"))
assert spec and spec.loader
ic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ic)

TEMP = "Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature?"
MAG = "Think about the magnetic force between the magnets in each pair. Which of the following statements is true?"


def q(text, opts=("red", "blue")):
    return {"type": "choice", "instructions": text, "criteria": {o: None for o in opts}}


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("install")
    d = tiny_decider(seed=3)
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    d.model.enable_local_views(12)
    d.cfg.local_views, d.cfg.local_overlap, d.cfg.local_source = 12, 0.25, "T"
    d.save(tmp / "ck")
    torch.manual_seed(7)
    torch.save(R.Selector(32, 64).state_dict(), tmp / "sel.pt")
    torch.save(R.CharHead(32).state_dict(), tmp / "head.pt")
    torch.save(R.Selector(32, 64).state_dict(), tmp / "corr.pt")
    R.write_bundle(str(tmp / "ck" / "reading"), str(tmp / "ck"), str(tmp / "sel.pt"), str(tmp / "head.pt"), {},
                   dims={"selector": [32, 64], "head": 32}, corrector_pt=str(tmp / "corr.pt"))
    pics = {}
    for name, size, text in (("temp", (560, 405), "Mass 14 u  speed 1,360 m/s"), ("mag", (650, 140), "2 cm   3 cm"),
                             ("other", (560, 405), "Mass 9 u  speed 700 m/s")):
        img = Image.new("RGB", size, "white")
        ImageDraw.Draw(img).text((20, size[1] - 25), text, fill="black")
        img.save(tmp / ("%s.png" % name))
        pics[name] = str(tmp / ("%s.png" % name))
    cases = [{"name": "ordinary", "picture": pics["temp"], "state": "", "questions": {"a": q("is there a dog ?"),
                                                                                      "b": {"type": "noul", "instructions": "Is it red?"}}},
             {"name": "temperature", "picture": pics["temp"], "state": "Two samples.", "questions": {"t": q(TEMP)}},
             {"name": "magnets", "picture": pics["mag"], "state": "", "questions": {"m": q(MAG)}},
             {"name": "mismatched", "picture": pics["other"], "state": "Two samples.", "questions": {"t": q(TEMP)}}]
    work = tmp / "work"
    work.mkdir()
    py = ic.install(str(work))
    got = ic.probe(py, str(tmp / "ck"), cases, str(work), isolated=True)
    ref = ic.probe(ic.sys.executable, str(tmp / "ck"), cases, ROOT, isolated=False)
    return {"ck": str(tmp / "ck"), "cases": cases, "venv": str(work / "venv"), "got": got, "ref": ref}


def test_the_clean_install_agrees(installed):
    got, ref = installed["got"], installed["ref"]
    assert ic.compare(ref, got, ref["code"], installed["venv"], bound=True) == []
    assert got["file"].startswith(installed["venv"]) and got["status"]["mode"] == "on"
    assert got["on"]["temperature"]["diagnostics"]["route"] == {"t": "read"}
    assert got["on"]["ordinary"]["diagnostics"]["route"] == {"a": "ordinary", "b": "ordinary"}
    assert got["on"]["mismatched"]["diagnostics"]["picture"] != got["on"]["temperature"]["diagnostics"]["picture"]
    assert {"devision.reading", "devision.train.evaluate", "devision.train.cli"} <= set(got["code"]["modules"])


@pytest.mark.parametrize("damage,why", [("module", "content differs"), ("source_tree", "not the new environment"),
                                        ("table", "table differs"), ("off", "off is not the plain Decider"),
                                        ("probability", "probabilities differ"), ("plain_mode", "not the reading combination")])
def test_differences_are_reported(installed, damage, why):
    got, ref = copy.deepcopy(installed["got"]), installed["ref"]
    reference = copy.deepcopy(ref["code"])
    if damage == "module":
        reference["modules"]["devision.reading"]["sha256"] = "0" * 64       # the frozen code was another reading.py
    elif damage == "source_tree":
        got["file"] = os.path.join(ROOT, "src", "devision", "__init__.py")
    elif damage == "table":
        got["on"]["temperature"]["diagnostics"]["table"][0][2] = "999"
    elif damage == "off":
        got["off"]["ordinary"]["answers"]["b"]["noul"] += 0.1
    elif damage == "probability":
        p = got["on"]["temperature"]["answers"]["t"]["probabilities"]
        p["red"], p["blue"] = p["red"] + 0.01, p["blue"] - 0.01
    else:
        got["status"] = {"mode": "none"}
    assert any(why in x for x in ic.compare(ref, got, reference, installed["venv"], bound=True))


def test_the_whole_command_with_a_frozen_identity(installed, tmp_path):
    ident = tmp_path / "IDENTITY.json"
    ident.write_text(json.dumps({"code": installed["ref"]["code"]}))
    assert ic.install_check(installed["ck"], installed["cases"], str(ident)) == []
