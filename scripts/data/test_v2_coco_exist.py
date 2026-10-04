"""v2 COCO existence questions: balanced per category, no wrong "no" answers we can catch."""
import random
from collections import Counter

from v2_coco_exist import build

NAMES = {1: "person", 2: "dog", 3: "chair"}


def image(i):
    return {"id": i, "width": 100, "height": 100}


def box(cat, size=30, crowd=0):
    return {"category_id": cat, "area": size * size, "iscrowd": crowd}


def answers(rows):
    return Counter((r["category"], r["gold"]["q"]["probabilities"]["true"] == 1.0) for r in rows)


def world(n=40):
    """Images 0..n-1: even ones hold a person and a dog, odd ones a person and a chair."""
    images = [image(i) for i in range(n)]
    anns = {i: [box(1), box(2 if i % 2 == 0 else 3)] for i in range(n)}
    return images, anns


def test_every_category_is_asked_as_often_yes_as_no():
    images, anns = world()
    rows = build(images, anns, NAMES, {}, "train2014", per_category=5, held_out=set(), rng=random.Random(0))
    count = answers(rows)
    for cat in ("dog", "chair"):
        assert count[(cat, True)] == count[(cat, False)] > 0


def test_a_category_the_captions_mention_is_never_a_no():
    images, anns = world()
    captions = {i: ["a man sitting on a chair next to a dog"] for i in range(40)}
    rows = build(images, anns, NAMES, captions, "train2014", per_category=5, held_out=set(), rng=random.Random(0))
    assert not [r for r in rows if r["gold"]["q"]["probabilities"]["false"] == 1.0
                and r["category"] in ("dog", "chair")]


def test_a_crowd_box_counts_as_present():
    images, anns = world()
    for i in range(1, 40, 2):
        anns[i].append(box(2, crowd=1))  # odd images: a crowd of dogs, no single dog box
    rows = build(images, anns, NAMES, {}, "train2014", per_category=5, held_out=set(), rng=random.Random(0))
    dog_no = [r for r in rows if r["category"] == "dog" and r["gold"]["q"]["probabilities"]["false"] == 1.0]
    assert not [r for r in dog_no if int(r["image_id"].split(":")[1]) % 2 == 1]


def test_too_small_objects_are_not_asked_as_yes():
    images, anns = world()
    for i in range(0, 40, 2):
        anns[i] = [box(1), box(2, size=5)]  # the dog covers 0.25% of the image
    rows = build(images, anns, NAMES, {}, "train2014", per_category=5, held_out=set(), rng=random.Random(0))
    assert ("dog", True) not in answers(rows)


def test_evaluation_images_are_skipped():
    images, anns = world()
    held = {"coco:%d" % i for i in range(0, 40, 4)}
    rows = build(images, anns, NAMES, {}, "train2014", per_category=50, held_out=held, rng=random.Random(0))
    assert rows and not {r["image_id"] for r in rows} & held


def test_an_image_gets_at_most_two_questions():
    images, anns = world()
    rows = build(images, anns, NAMES, {}, "train2014", per_category=50, held_out=set(), rng=random.Random(0))
    assert max(Counter(r["image_id"] for r in rows).values()) <= 2
