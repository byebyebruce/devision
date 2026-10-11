"""The counterfactual generator (v2): one layout per family and minimal edits (critic R13-I01), one fixed
question set per family with gold that flips on target edits and holds on distractor edits, grounding boxes
only for referents in the picture (R13-I03), held-out combinations, balanced strata, valid samples."""
import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import synth  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    for split, n in (("train", 40), ("dev", 8), ("test", 8)):
        synth.build_split(str(root), split, n)
    read = lambda name: [json.loads(l) for l in open(root / "r13" / "synth-v2" / name)]
    return root, read("train.jsonl"), read("dev_bind_cf.jsonl"), read("test_bind_cf.jsonl")


def answer(gold):
    return max(gold["probabilities"], key=gold["probabilities"].get)


def families(rows):
    out = {}
    for r in rows:
        out.setdefault(r["group"], {})[r["variant"]] = r
    return out


def test_answers_follow_the_scene_and_every_kind_is_asked(built):
    for r in sum(built[1:], []):
        for i, k in enumerate(r["scene"]):
            sh = k["shape"]
            assert answer(r["gold"]["colour_%s" % sh]) == k["colour"]
            assert answer(r["gold"]["count_%s" % sh]) == synth.NUMBERS[k["count"]]
            assert sum(1 for x in r["instances"] if x["kind"] == i) == k["count"]


def test_variants_keep_the_layout_and_change_only_the_edited_instances(built):
    root = built[0]
    for fam in families(built[1]).values():
        base = fam["base"]
        for v in ("recolour", "recolour2"):
            assert fam[v]["instances"] == base["instances"]
            edited = 0 if v == "recolour" else 1
            a = np.asarray(Image.open(root / base["image"]).convert("RGB")).astype(int)
            b = np.asarray(Image.open(root / fam[v]["image"]).convert("RGB")).astype(int)
            outside = np.ones(a.shape[:2], bool)
            for x in base["instances"]:
                if x["kind"] == edited:
                    x0, y0, x1, y1 = x["box"]
                    outside[y0:y1 + 1, x0:x1 + 1] = False
            assert (a[outside] == b[outside]).all()
        kept = [x for x in base["instances"] if x in fam["count"]["instances"]]
        assert len(kept) >= len(base["instances"]) - 1   # one removed, or all kept and one added


def test_one_question_set_per_family_flips_on_target_edits_and_holds_on_distractor_edits(built):
    for fam in families(built[1]).values():
        qs = fam["base"]["questions"]
        assert all(r["questions"] == qs for r in fam.values())
        k0, k1 = (k["shape"] for k in fam["base"]["scene"])
        g = lambda v, q: answer(fam[v]["gold"][q])
        assert g("base", "own_" + k0) == "true" and g("recolour", "own_" + k0) == "false"
        assert g("base", "new_" + k0) == "false" and g("recolour", "new_" + k0) == "true"
        assert g("recolour2", "own_" + k0) == g("base", "own_" + k0)          # editing kind 1 leaves kind 0
        assert g("recolour2", "colour_" + k0) == g("base", "colour_" + k0)
        assert g("recolour", "colour_" + k0) != g("base", "colour_" + k0)
        assert g("base", "trap_" + k0) == "false" and fam["base"]["meta"]["trap_" + k0]["task"] == "exists_trap"


def test_grounding_boxes_only_for_referents_in_the_picture(built):
    for r in sum(built[1:], []):
        for qid, m in r["meta"].items():
            if m["task"].startswith("exists"):
                assert (qid in r["boxes"]) == (m["task"] == "exists_true")
            else:
                assert r["boxes"][qid]
            assert m["referent_present"] == (qid in r["boxes"])


def test_held_out_combinations_and_strata(built):
    _, train, dev, test = built
    for r in train + dev:
        assert not {(k["colour"], k["shape"]) for k in r["scene"]} & synth.HELD_OUT
    for fam in families(test).values():
        assert {(k["colour"], k["shape"]) for k in fam["base"]["scene"]} & synth.HELD_OUT
    strata = [m["stratum"] for r in train for m in r["meta"].values() if m["task"] in ("colour", "count")]
    for s in ("target_salient", "target_minor", "close"):
        assert 0.2 < strata.count(s) / len(strata) < 0.45
    assert not {r["group"] for r in train} & {r["group"] for r in dev}


def test_samples_are_valid_training_input(built):
    from devision.train.samples import sample_errors
    assert all(sample_errors(r) == [] for r in sum(built[1:], []))
