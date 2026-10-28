"""Native-resolution tiles (spec "实验性、可选：原生分辨率切块"), through decide: off by default (saved config and
answers as before); a model without tiles that gains them answers small pictures exactly as before and reads large
ones with 4 x 64 more visual tokens; a batch mixing both answers as each request alone; a tiles model trains, saves,
loads and answers the same."""
import json

import pytest
from conftest import image_b64, tiny_decider
from PIL import Image

from devision.infer import Decider
from devision.model import ModelConfig
from devision.train.rlcd import TrainConfig, train

QUESTIONS = {"a": {"type": "noul", "instructions": "Is there a dog in the image?"},
             "b": {"type": "choice", "instructions": "What color is it?",
                   "criteria": {"red": None, "green": None, "blue": None}}}
SMALL, LARGE = image_b64((300, 200)), image_b64((1024, 800), (30, 200, 60))


def ask(d, img):
    return d.decide(state=[{"type": "image", "base64": img}], questions=QUESTIONS)


def gain_tiles(d, n=2, min_side=512):
    d.model.enable_tiles(n)
    d.cfg.tiles, d.cfg.tile_min_side = n, min_side


def test_without_tiles_the_saved_config_and_answers_are_unchanged(tmp_path):
    d = tiny_decider(seed=5)
    before = {"small": ask(d, SMALL), "large": ask(d, LARGE)}
    d.save(tmp_path / "c")
    saved = json.loads((tmp_path / "c" / "config.json").read_text())
    assert "tiles" not in saved and "tile_min_side" not in saved
    loaded = Decider.load(tmp_path / "c", device="cpu")
    assert {"small": ask(loaded, SMALL), "large": ask(loaded, LARGE)} == before


def test_an_old_checkpoint_given_tiles_answers_small_pictures_exactly_as_before(tmp_path):
    tiny_decider(seed=6).save(tmp_path / "old")
    old = Decider.load(tmp_path / "old", device="cpu")
    small, large = ask(old, SMALL), ask(old, LARGE)
    gain_tiles(old)

    assert ask(old, SMALL) == small                       # below tile_min_side: the 64-token path, bit for bit
    tiled = ask(old, LARGE)
    assert tiled["usage"]["input_tokens"] == large["usage"]["input_tokens"] + 2 * 4 * 64   # two questions, 4 tiles each
    assert tiled["answers"] != large["answers"]           # the tiles are read


def test_a_batch_of_tiled_and_untiled_pictures_answers_as_each_alone(tmp_path):
    tiny_decider(seed=7).save(tmp_path / "old")
    d = Decider.load(tmp_path / "old", device="cpu")
    gain_tiles(d)
    alone = [ask(d, img) for img in (SMALL, LARGE, SMALL)]
    batch = d.decide_batch([{"state": [{"type": "image", "base64": img}], "questions": QUESTIONS}
                            for img in (SMALL, LARGE, SMALL)])
    for a, b in zip(alone, batch):
        assert a["usage"] == b["usage"]
        assert a["answers"]["a"]["noul"] == pytest.approx(b["answers"]["a"]["noul"], abs=1e-5)
        for o in QUESTIONS["b"]["criteria"]:
            assert a["answers"]["b"]["probabilities"][o] == pytest.approx(b["answers"]["b"]["probabilities"][o], abs=1e-5)


def tiled_samples(root, n):
    """Red / blue pictures, every third one large enough to be tiled."""
    (root / "images").mkdir(parents=True)
    out = []
    for i in range(n):
        color = "red" if i % 2 else "blue"
        size = (600, 520) if i % 3 == 0 else (64, 48)
        Image.new("RGB", size, {"red": (220, 20, 20), "blue": (20, 20, 220)}[color]).save(root / "images" / ("%d.png" % i))
        is_red = float(color == "red")
        out.append({"id": "t:%d" % i, "image_id": "t:%d" % i, "image": "images/%d.png" % i, "source": "synthetic",
                    "questions": {"q": {"type": "noul", "instructions": "Is the image red?"}},
                    "gold": {"q": {"probabilities": {"false": 1 - is_red, "true": is_red}}}})
    return out


def test_a_tiles_model_trains_saves_loads_and_answers_the_same(tmp_path):
    root = tmp_path / "data"
    samples = tiled_samples(root, 12)
    d = tiny_decider(seed=8)
    train(d, samples[:8], data_root=root, out_dir=tmp_path / "ckpt", val_samples=samples[8:],
          config=TrainConfig(epochs=2, micro_batch=4, warmup=1, eval_every=0, save_every=0, device="cpu",
                             tiles=2, tile_min_side=512))
    saved = json.loads((tmp_path / "ckpt" / "config.json").read_text())
    assert (saved["tiles"], saved["tile_min_side"]) == (2, 512)

    loaded = Decider.load(tmp_path / "ckpt", device="cpu")
    big = image_b64((600, 520), (220, 20, 20))
    for img in (SMALL, big):
        assert ask(loaded, img) == ask(d, img)
    assert ask(loaded, big)["usage"]["input_tokens"] == ask(loaded, SMALL)["usage"]["input_tokens"] + 2 * 4 * 64


def test_a_checkpoint_with_tiles_refuses_training_with_other_tile_settings(tmp_path):
    root = tmp_path / "data"
    samples = tiled_samples(root, 4)
    d = tiny_decider(seed=9)
    gain_tiles(d)
    with pytest.raises(ValueError, match="tiles"):
        train(d, samples, data_root=root, out_dir=tmp_path / "x",
              config=TrainConfig(epochs=1, device="cpu", tiles=2, tile_min_side=300))


def test_one_tile_is_refused():
    with pytest.raises(ValueError, match="tiles"):
        ModelConfig(tiles=1)
