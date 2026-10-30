"""Caption hard negatives: each rule changes the meaning in one controlled way, symmetric links are not swapped,
object replacements never name an object annotated in the image, and the per-image cap / kind cap hold."""
from v05_board_caption import build_questions, negatives, replace_obj, swap_att, swap_obj, tokens


def test_swap_obj_needs_an_asymmetric_relation():
    assert " ".join(swap_obj(tokens("A man riding a horse on the beach.")) or []) == "a horse riding a man on the beach"
    assert swap_obj(tokens("A man and a horse on the beach.")) is None          # symmetric
    assert swap_obj(tokens("A dog next to a cat.")) is None
    assert swap_obj(tokens("A man holding two dogs.")) is None                   # number would not agree


def test_swap_att_swaps_same_group_attributes_once():
    assert " ".join(swap_att(tokens("A red car near a white truck")) or []) == "a white car near a red truck"
    assert swap_att(tokens("A young woman holding a large knife")) is None        # age vs size
    assert swap_att(tokens("A white toilet and a white and brown sink")) is None  # attribute used twice


def test_negatives_differ_and_fix_articles():
    negs = negatives("An old man sits above a red bus.", {"person", "bus"}, "k")
    assert negs and all(n.casefold() != "an old man sits above a red bus" for n in negs.values())
    assert negs["replace_rel"] == "an old man sits below a red bus"
    assert negs["replace_att"].startswith("a young man")                       # an -> a
    assert "replace_rel" not in negatives("Two people walking down the beach", set(), "k")      # up / down not used


def test_replace_obj_never_names_an_annotated_object():
    for k in range(30):
        out = replace_obj(tokens("A dog on a bench"), {"dog", "bench", "cat", "horse"}, "k%d" % k)
        assert out and out[1] not in ("cat", "horse", "dog")


def test_build_questions_caps_per_image_and_per_kind():
    caps = {i: ["A man riding a horse above a red car", "A red car near a white truck", "A man sitting under a tree"]
            for i in range(50)}
    qs, stats = build_questions(caps, {i: {"person", "horse", "car", "truck"} for i in range(50)}, cap_kind=20,
                                per_image=2)
    per_image = {}
    assert len({(iid, true) for iid, _, true, _ in qs}) == len(qs)       # a caption is gold once per image
    for iid, kind, true, neg in qs:
        per_image[iid] = per_image.get(iid, 0) + 1
        assert true != neg
    assert max(per_image.values()) <= 2
    kinds = {}
    for _, kind, _, _ in qs:
        kinds[kind] = kinds.get(kind, 0) + 1
    assert max(kinds.values()) <= 20 and len(kinds) >= 3
