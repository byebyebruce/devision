"""Release tooling (scripts/release/hf.py) on a tiny checkpoint: the built folder has the Hub layout and a filled
card, verify accepts it, and verify catches a changed file, a missing file and changed answers. No network."""
import importlib.util
import hashlib
import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

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
    assert json.loads((out / "config.json").read_text())["model_name"] == "devision-v9.9"
    assert not (out / "devision_config.json").exists()
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


def test_verify_accepts_only_the_hub_added_logo_lfs_rule(release):
    rel, out = release
    # v0.3's source manifest preceded HF's automatic logo rule addition.
    attrs = (out / ".gitattributes").read_bytes()
    source = attrs.removesuffix(hf.LOGO_LFS.encode())
    manifest = json.loads((rel / "MANIFEST.json").read_text())
    manifest[".gitattributes"] = {"size": len(source), "sha256": hashlib.sha256(source).hexdigest()}
    (rel / "MANIFEST.json").write_text(json.dumps(manifest))
    assert hf.verify(str(rel), str(out), latency=False) == []
    (out / ".gitattributes").write_bytes(b"*.json -filter\n" + attrs)
    assert any(".gitattributes" in p for p in hf.verify(str(rel), str(out), latency=False))


def test_build_refuses_to_replace_a_folder_holding_the_source_checkpoint(release):
    rel, out = release
    cfg = yaml.safe_load((rel / "release.yaml").read_text())
    with pytest.raises(SystemExit):
        hf.build(str(rel), cfg["round"])
    assert os.path.exists(os.path.join(cfg["round"], "stage2", "model.safetensors"))


@pytest.mark.parametrize("exists,private", [(True, False), (True, True), (False, True)])
def test_publish_preserves_existing_visibility_and_creates_missing_repo_private(tmp_path, monkeypatch, exists, private):
    import huggingface_hub
    api = Mock()
    api.repo_exists.return_value = exists
    api.model_info.return_value = SimpleNamespace(private=private)
    api.list_repo_refs.return_value = SimpleNamespace(tags=[])
    api.upload_folder.return_value = SimpleNamespace(oid="uploaded-commit")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    monkeypatch.setattr(hf, "verify", lambda *a, **k: [])
    monkeypatch.setattr(hf, "release_config", lambda _: {
        "hub_repo": "someone/model", "version": "v9.9", "git_tag": "model-v9.9"})
    hf.publish("unused", True, str(tmp_path))
    if exists:
        api.create_repo.assert_not_called()
    else:
        api.create_repo.assert_called_once_with("someone/model", private=True, repo_type="model")
    api.upload_folder.assert_called_once()
    api.create_tag.assert_called_once_with("someone/model", tag="v9.9", revision="uploaded-commit",
                                          tag_message="github model-v9.9")


def test_publish_refuses_to_overwrite_an_existing_tag(tmp_path, monkeypatch):
    import huggingface_hub
    api = Mock()
    api.repo_exists.return_value = True
    api.model_info.return_value = SimpleNamespace(private=False)
    api.list_repo_refs.return_value = SimpleNamespace(tags=[SimpleNamespace(name="v9.9")])
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    monkeypatch.setattr(hf, "verify", lambda *a, **k: [])
    monkeypatch.setattr(hf, "release_config", lambda _: {
        "hub_repo": "someone/model", "version": "v9.9", "git_tag": "model-v9.9"})
    with pytest.raises(SystemExit, match="already exists"):
        hf.publish("unused", True, str(tmp_path))
    api.upload_folder.assert_not_called()
