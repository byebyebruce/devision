"""The counterfactual generator: answers follow the drawn scene, every kind is asked, held-out colour x shape
combinations stay out of train / dev and appear in every test family, the three salience strata are balanced,
variants share their family's group, and boxes cover the asked kind."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import synth  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    for split, n in (("train", 30), ("dev", 8), ("test", 8)):
        synth.build_split(str(root), split, n)
    read = lambda name: [json.loads(l) for l in open(root / "r13" / "synth" / name)]
    return read("train.jsonl"), read("dev_bind_cf.jsonl"), read("test_bind_cf.jsonl")


def answer(gold):
    return max(gold["probabilities"], key=gold["probabilities"].get)


def test_answers_follow_the_scene_and_every_kind_is_asked(built):
    for r in sum(built, []):
        for i, k in enumerate(r["scene"]):
            sh = k["shape"]
            assert answer(r["gold"]["colour_%s" % sh]) == k["colour"]
            assert answer(r["gold"]["count_%s" % sh]) == synth.NUMBERS[k["count"]]
            assert answer(r["gold"]["exists_%s" % sh]) == "true"
            assert answer(r["gold"]["trap_%s" % sh]) == "false"
            assert len(r["boxes"]["colour_%s" % sh]) == k["count"]
            assert r["meta"]["colour_%s" % sh]["kind"] == i


def test_held_out_combinations(built):
    train, dev, test = built
    for r in train + dev:
        assert not {(k["colour"], k["shape"]) for k in r["scene"]} & synth.HELD_OUT
    for group in {r["group"] for r in test}:
        base = next(r for r in test if r["group"] == group and r["variant"] == "base")
        assert {(k["colour"], k["shape"]) for k in base["scene"]} & synth.HELD_OUT


def test_strata_are_balanced_and_families_are_grouped(built):
    train = built[0]
    strata = [m["stratum"] for r in train for m in r["meta"].values()]
    for s in ("target_salient", "target_minor", "close"):
        assert 0.2 < strata.count(s) / len(strata) < 0.45
    groups = {}
    for r in train:
        groups.setdefault(r["group"], set()).add(r["variant"])
    assert all(v == {"base", "recolour", "recolour2", "count"} for v in groups.values())
    assert not {r["group"] for r in built[0]} & {r["group"] for r in built[1]}


def test_samples_are_valid_training_input(built):
    from devision.train.samples import sample_errors
    assert all(sample_errors(r) == [] for r in sum(built, []))
