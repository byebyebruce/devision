"""Round 13 GQA binding rules and the identity list, on hand-made scene graphs (no network, no data files)."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gqa_bind as G  # noqa: E402
import identity as I  # noqa: E402


def obj(name, attrs=(), x=0, y=0, w=100, h=100):
    return {"name": name, "attributes": list(attrs), "x": x, "y": y, "w": w, "h": h, "relations": []}


def graph(**objects):
    return {"width": 400, "height": 300, "objects": objects}


CATS = {"color": {"red", "green", "blue", "white", "black", "silver", "gray"}, "material": {"metal", "glass", "wood"},
        "length": {"long", "short"}}


def verify(qid, oid, name, cat, value, answer, detailed="verifyAttrC", filters=()):
    prog = [{"operation": "select", "argument": "%s (%s)" % (name, oid), "dependencies": []}]
    for f_cat, f_val in filters:
        prog.append({"operation": "filter " + f_cat, "argument": f_val, "dependencies": [len(prog) - 1]})
    prog.append({"operation": "verify " + cat, "argument": value, "dependencies": [len(prog) - 1]})
    return {"id": qid, "imageId": "1", "question": "Is the %s %s?" % (name, value), "answer": answer,
            "types": {"detailed": detailed}, "semantic": prog}


def choose(qid, oid, name, cat, a, b, answer):
    return {"id": qid, "imageId": "1", "question": "Is the %s %s or %s?" % (name, a, b), "answer": answer,
            "types": {"detailed": "chooseAttr"},
            "semantic": [{"operation": "select", "argument": "%s (%s)" % (name, oid), "dependencies": []},
                         {"operation": "choose " + cat, "argument": "%s|%s" % (a, b), "dependencies": [0]}]}


def test_negative_from_missing_annotation_is_rejected():
    g = graph(**{"1": obj("car", []), "2": obj("tree", ["green"], x=200)})
    v, reason, _ = G.judge(verify("q", "1", "car", "color", "red", "no"), g, CATS)
    assert v is None and reason == "no_from_missing"


def test_negative_with_a_different_annotated_value_is_kept():
    g = graph(**{"1": obj("car", ["blue"]), "2": obj("tree", ["red"], x=200)})
    v, reason, info = G.judge(verify("q", "1", "car", "color", "red", "no"), g, CATS)
    assert reason == "kept" and v == {"type": "noul", "truth": False}
    assert info["competitors"] == ["2"]          # the asked colour is on another object: the binding case


def test_negative_rules_for_colour_families_regions_and_materials():
    g = graph(**{"1": obj("car", ["silver"])})
    assert G.judge(verify("q", "1", "car", "color", "white", "no"), g, CATS)[1] == "no_colour_same_family"
    g = graph(**{"1": obj("field", ["green"], w=400, h=300)})
    assert G.judge(verify("q", "1", "field", "color", "red", "no"), g, CATS)[1] == "colour_of_region"
    g = graph(**{"1": obj("mirror", ["metal"])})
    assert G.judge(verify("q", "1", "mirror", "material", "glass", "no"), g, CATS)[1] == "no_material"


def test_yes_needs_the_annotation_and_no_contradiction():
    g = graph(**{"1": obj("car", ["blue"])})
    assert G.judge(verify("q", "1", "car", "color", "blue", "yes", "verifyAttr"), g, CATS)[0] == {"type": "noul", "truth": True}
    assert G.judge(verify("q", "1", "car", "color", "red", "yes", "verifyAttr"), g, CATS)[1] == "yes_not_annotated"
    assert G.judge(verify("q", "1", "car", "color", "blue", "no"), g, CATS)[1] == "no_but_annotated"


def test_non_unique_target_is_rejected():
    g = graph(**{"1": obj("car", ["blue"]), "2": obj("cars", ["red"], x=200)})     # plural counts as the same name
    assert G.judge(verify("q", "1", "car", "color", "red", "no"), g, CATS)[1] == "name_not_unique"
    g = graph(**{"1": obj("car", ["blue"])})
    assert G.judge(verify("q", "1", "car", "color", "red", "no"), g, CATS, {"car": 2})[1] == "coco_not_one"
    assert G.judge(verify("q", "1", "car", "color", "red", "no"), g, CATS, {"car": 1})[1] == "kept"
    # a shirt in a picture of several people is one labelled shirt among unlabelled ones
    g = graph(**{"1": obj("shirt", ["blue"]), "2": obj("man", x=200), "3": obj("woman", x=300)})
    assert G.judge(verify("q", "1", "shirt", "color", "red", "no"), g, CATS)[1] == "worn_by_one_of_several_people"


def test_small_objects_and_unannotated_filters_are_rejected():
    g = graph(**{"1": obj("cup", ["blue"], w=10, h=10)})
    assert G.judge(verify("q", "1", "cup", "color", "red", "no"), g, CATS)[1] == "too_small"
    g = graph(**{"1": obj("cup", ["blue"])})
    rec = verify("q", "1", "cup", "material", "glass", "yes", "materialVerify", filters=[("color", "red")])
    assert G.judge(rec, g, CATS)[1] == "filter_not_annotated"


def test_choose_needs_both_options_backed():
    g = graph(**{"1": obj("shorts", ["black"]), "2": obj("bench", ["green"], x=200)})
    v, reason, _ = G.judge(choose("q", "1", "shorts", "color", "black", "green", "black"), g, CATS)
    assert reason == "kept" and v["answer"] == "black"
    g = graph(**{"1": obj("shorts", ["black"])})                                    # green backed by nothing
    assert G.judge(choose("q", "1", "shorts", "color", "black", "green", "black"), g, CATS)[1] == "other_unbacked"
    g = graph(**{"1": obj("hair", ["long"])})                                       # antonym backs the other option
    assert G.judge(choose("q", "1", "hair", "length", "long", "short", "long"), g, CATS)[1] == "kept"
    g = graph(**{"1": obj("train", ["long"])})                                      # length only for hair
    assert G.judge(choose("q", "1", "train", "length", "long", "short", "long"), g, CATS)[1] == "length_not_hair"
    g = graph(**{"1": obj("shorts", ["black", "green"])})
    assert G.judge(choose("q", "1", "shorts", "color", "black", "green", "black"), g, CATS)[1] == "both_annotated"


def test_same_needs_one_value_each_and_agreement():
    rec = {"id": "q", "imageId": "1", "question": "Are the hat and the bucket the same color?", "answer": "no",
           "types": {"detailed": "twoSameC"},
           "semantic": [{"operation": "select", "argument": "hat (1)", "dependencies": []},
                        {"operation": "select", "argument": "bucket (2)", "dependencies": []},
                        {"operation": "same color", "argument": "", "dependencies": [0, 1]}]}
    g = graph(**{"1": obj("hat", ["red"]), "2": obj("bucket", ["blue"], x=200)})
    assert G.judge(rec, g, CATS)[0] == {"type": "noul", "truth": False}
    g = graph(**{"1": obj("hat", ["red"]), "2": obj("bucket", ["red", "blue"], x=200)})
    assert G.judge(rec, g, CATS)[1] == "same_value_not_single"
    g = graph(**{"1": obj("hat", ["red"]), "2": obj("bucket", ["red"], x=200)})
    assert G.judge(rec, g, CATS)[1] == "answer_disagrees"


def test_spatial_answer_follows_the_box():
    rec = {"id": "q", "imageId": "1", "question": "Is the cup on the left or right?", "answer": "left",
           "types": {"detailed": "positionChoose"},
           "semantic": [{"operation": "select", "argument": "cup (1)", "dependencies": []},
                        {"operation": "choose hposition", "argument": "left|right", "dependencies": [0]}]}
    assert G.judge(rec, graph(**{"1": obj("cup", x=10)}), CATS)[0]["answer"] == "left"
    assert G.judge(rec, graph(**{"1": obj("cup", x=150)}), CATS)[1] == "box_not_clearly_on_a_side"
    assert G.judge(rec, graph(**{"1": obj("cup", x=290)}), CATS)[1] == "box_disagrees"


def test_sample_has_group_boxes_and_shuffled_options():
    g = graph(**{"1": obj("shorts", ["black"], x=40, y=30), "2": obj("bench", ["green"], x=200)})
    rec = choose("0123", "1", "shorts", "color", "black", "green", "black")
    v, _, info = G.judge(rec, g, CATS)
    s = G.to_sample("0123", rec, v, info, "vg:1", "gqa/images/1.jpg", "vg:1",
                    G.scaled_boxes(g, info["targets"], (800, 600)), (800, 600))
    assert s["group"] == "vg:1" and s["boxes"] == {"q": [[80.0, 60.0, 200.0, 200.0]]} and s["image_size"] == [800, 600]
    assert set(s["questions"]["q"]["criteria"]) == {"black", "green"}
    assert s["gold"]["q"]["probabilities"] == {"black": 1.0, "green": 0.0}
    assert G.scaled_boxes(g, info["targets"], (800, 300)) is None      # aspect ratio differs: no boxes


def test_splits_are_disjoint_and_dev_test_unseen():
    pic_split = {"coco:1": "train", "coco:2": "train", "vg:3": "train", "vg:4": "val", "vg:5": "val",
                 "coco:6": "both", "vg:7": "train"}
    seen = {"coco:1", "vg:5"}.__contains__
    pools = G.candidate_pools(pic_split, seen, random.Random(0))
    assert set(pools["dev"]) == {"coco:2", "vg:3", "vg:7"} and set(pools["test"]) == {"vg:4"}
    dev, test = set(pools["dev"][:2]), set(pools["test"])
    train = G.train_pictures(pic_split, dev, test)
    assert not any(G.intersections({"dev": dev, "test": test, "train": train}).values())
    assert "coco:6" in train and "coco:6" not in dev | test          # in both GQA splits: train only
    assert not any(seen(p) for p in dev | test)


def test_identity_canonical_and_history_files():
    v2c = {"vg:10": "coco:42"}
    assert I.canonical("vg:10", v2c) == "coco:42"
    assert I.canonical("coco:42:flip", v2c) == "coco:42"
    assert I.canonical("vg:11", v2c) == "vg:11"
    assert I.canonical("objects365:5", v2c) == "objects365:5"
    ident = I.Identity({"coco:42": {"sources": ["a.jsonl"], "seen_in_training": True}}, v2c, [0b1111])
    assert ident.seen("vg:10") and not ident.seen("vg:11")
    assert ident.near_seen(0b1100) and not ident.near_seen(0b11110000)
    config = {"stages": [{"name": "align", "kind": "align", "data": "data/a.jsonl", "val": "data/av.jsonl"},
                         {"name": "s2", "kind": "train", "init": "align", "data": "data/t.jsonl",
                          "val": "data/v.jsonl", "eval": {"pope": "data/p.jsonl"}}]}
    assert [f["path"] for f in I.stage_files(config)] == ["data/a.jsonl", "data/av.jsonl", "data/t.jsonl",
                                                          "data/v.jsonl", "data/p.jsonl"]
    configs = {"r2": {"stages": [{"init": "runs/r1/stage2"}]}, "r1": {"stages": [{"init": "align"}]}}
    assert I.init_chain(configs, "r2") == ["r2", "r1"]
