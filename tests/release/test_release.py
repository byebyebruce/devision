"""Release tooling (scripts/release/hf.py) on a tiny checkpoint: the built folder has the Hub layout and a filled
card, verify accepts it, and verify catches a changed file, a missing file and changed answers. No network."""
import importlib.util
import json
import os

import pytest
import yaml
from conftest import tiny_decider

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("release_hf", os.path.join(HERE, "..", "..", "scripts", "release", "hf.py"))
assert spec and spec.loader
hf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf)


@pytest.fixture()
def release(tmp_path):
    round_dir = tmp_path / "runs" / "v0-tiny"
    tiny_decider().save(str(round_dir / "stage2"))
    (round_dir / "run.json").write_text(json.dumps({"git_commit": "abc", "started": "now", "stages": []}))
    rel = tmp_path / "release" / "v9.9"
    rel.mkdir(parents=True)
    (rel / "release.yaml").write_text(yaml.safe_dump({
        "version": "v9.9", "git_tag": "model-v9.9", "model_name": "devision-v9.9", "round": str(round_dir),
        "stage": "stage2", "hub_repo": "someone/devision-test", "license": "apache-2.0",
        "base_model": ["convaiinnovations/laya"], "datasets": ["lmms-lab/GQA"],
        "smoke_questions": {"person": {"type": "noul", "instructions": "Is there a dog in the image?"},
                            "color": {"type": "choice", "instructions": "What color is it?",
                                      "criteria": {"red": None, "green": None, "blue": None}}}}))
    (rel / "notes.md").write_text("## training\nTrained.\n\n## limitations\n- Tiny.\n")
    out = tmp_path / "build"
    hf.build(str(rel), str(out))
    return rel, out


def test_build_lays_out_the_hub_repository_and_verify_accepts_it(release):
    rel, out = release
    for f in hf.REQUIRED:
        assert (out / f).exists(), f
    text = (out / "README.md").read_text()
    meta = yaml.safe_load(text.split("---")[1])
    assert meta["license"] == "apache-2.0" and meta["library_name"] == "devision" and "{{" not in text
    assert json.loads((out / "devision_config.json").read_text())["model_name"] == "devision-v9.9"
    assert (rel / "MANIFEST.json").exists() and (rel / "smoke.json").exists() and (rel / "results.json").exists()
    assert hf.verify(str(rel), str(out), latency=False) == []


def test_verify_catches_a_changed_or_missing_file(release):
    rel, out = release
    with open(out / "README.md", "a") as f:
        f.write("edited on the Hub\n")
    (out / "LICENSE").unlink()
    problems = hf.verify(str(rel), str(out), latency=False)
    assert any("README.md" in p and "sha256" in p for p in problems)
    assert any("LICENSE" in p for p in problems)


def test_verify_catches_different_answers(release):
    rel, out = release
    smoke = json.loads((rel / "smoke.json").read_text())
    pic = sorted(smoke["answers"])[0]
    smoke["answers"][pic]["person"]["noul"] += 0.01
    (rel / "smoke.json").write_text(json.dumps(smoke))
    assert any(p.startswith("smoke:") for p in hf.verify(str(rel), str(out), latency=False))
