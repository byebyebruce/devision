"""The cards' install lines (critic V04-REL-B-PREP-01): plain `devision` for releases that any package can load,
`"devision>=X"` when release.yaml sets min_package_version (a 512-input model needs the interpolation support); the
ModelScope card keeps modelscope-hub on that same line. Old and new model configs load with their own input size
and interpolation setting."""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "release"))


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", "release", name + ".py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_install_lines():
    hf = load("hf")
    assert hf.package_spec({}) == "devision"
    assert hf.package_spec({"min_package_version": "0.3.0"}) == '"devision>=0.3.0"'
    sys.modules["hf"] = hf
    ms = load("ms")
    cfg = {"modelscope_repo": "o/devision", "hub_repo": "u/devision", "min_package_version": "0.3.0"}
    text = "pip install %s\n" % hf.package_spec(cfg)
    assert 'pip install "devision>=0.3.0" modelscope-hub\n' in ms.model_card(text, cfg)
    cfg.pop("min_package_version")
    assert "pip install devision modelscope-hub\n" in ms.model_card("pip install devision\n", cfg)
    template = open(os.path.join(ROOT, "release", "card_template.md")).read()
    assert "pip install {{package}}\n" in template


def test_old_and_new_model_configs():
    from devision.model import ModelConfig
    old = ModelConfig.from_dict({"model_name": "devision-v0.3", "image_size": 256, "visual_shuffle": 2})
    new = ModelConfig.from_dict({"model_name": "devision-v0.4", "image_size": 512, "visual_shuffle": 2,
                                 "interpolate_pos_encoding": True})
    assert (old.image_size, old.interpolate_pos_encoding) == (256, False)
    assert (new.image_size, new.interpolate_pos_encoding) == (512, True)
