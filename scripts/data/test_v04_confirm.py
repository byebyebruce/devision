"""v0.4 confirmation set: the mismatched control moves every picture inside its subset, the near-duplicate check
catches a resized + mirrored copy, and the converted gold is always one of the options."""
import numpy as np
from PIL import Image, ImageDraw

from v04_confirm import convert, derangement, is_near, mask_hashes, mismatched, nearest, perceptual


def picture(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (240, 180), "white")
    d = ImageDraw.Draw(im)
    for _ in range(12):
        x, y = int(rng.integers(0, 200)), int(rng.integers(0, 140))
        d.rectangle([x, y, x + int(rng.integers(10, 60)), y + int(rng.integers(10, 40))],
                    fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    return im


def test_derangement_moves_every_picture_and_two_swap():
    for n in range(2, 12):
        ids = ["img:%02d" % i for i in range(n)]
        m = derangement(ids, 0)
        assert sorted(m) == sorted(m.values()) == ids
        assert all(k != v for k, v in m.items())
        assert m == derangement(ids, 0)
    assert derangement(["a", "b"]) == {"a": "b", "b": "a"}


def test_mismatched_stays_inside_the_subset():
    rows = [{"id": "q%d" % i, "kind": kind, "image_id": "%s-%d" % (kind, i // 2), "image": "%s-%d.png" % (kind, i // 2)}
            for kind in ("a", "b") for i in range(8)]
    for orig, m in zip(rows, mismatched(rows)):
        assert m["mismatched_from"] == orig["image_id"] != m["image_id"]
        assert m["image_id"].startswith(orig["kind"]) and m["image"] == m["image_id"] + ".png"
        assert m["id"] == orig["id"]


def test_near_duplicate_catches_a_resized_mirrored_copy_but_not_another_picture():
    a, b = picture(1), picture(2)
    copy = a.resize((173, 130), Image.Resampling.BILINEAR).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    refs = [perceptual(copy), perceptual(b)]
    ref_d = np.array([h[0] for h in refs], dtype=np.uint64)
    ref_p = np.array([h[2] for h in refs], dtype=np.uint64)
    near = nearest(perceptual(a), ref_d, ref_p)
    assert is_near(near) and near["dhash"][1] == 0
    alone = nearest(perceptual(a), ref_d[1:], ref_p[1:])
    assert not is_near(alone)


def test_gold_is_always_one_of_the_options():
    row = {"question": "Which is harder?", "choices": ["rock", "pillow", "cotton"], "answer": 0,
           "hint": "Select the better answer.", "skill": "Compare properties of objects"}
    for i in range(50):
        s = convert(i, row, "img:x", "v04_confirm/images/x.png", "object-property")
        assert s is not None
        probs = s["gold"]["q"]["probabilities"]
        assert set(probs) == set(s["questions"]["q"]["criteria"]) == {"rock", "pillow", "cotton"}
        assert probs["rock"] == 1.0 and sum(probs.values()) == 1.0
        assert s["id"] == "v04-confirm:test-%d" % i and s["kind"] == "object-property"
        assert s["state_text"] == "Select the better answer."


def chart(heights, colours) -> Image.Image:
    im = Image.new("RGB", (300, 200), "white")
    d = ImageDraw.Draw(im)
    for k, (h, c) in enumerate(zip(heights, colours)):
        d.rectangle([20 + 22 * k, 190 - h, 36 + 22 * k, 190], fill=c, outline="gray")
    return im


def test_recoloured_chart_is_caught_by_the_colour_agnostic_hash():
    heights = [120, 110, 80, 40, 30, 30, 30, 30, 30, 30, 45, 85]
    pal = [(170, 170, 250), (180, 240, 150), (120, 220, 200), (250, 170, 120), (240, 160, 240), (250, 130, 130)]
    a = chart(heights, [pal[k % 6] for k in range(12)])
    b = chart(heights, [pal[(k * 5 + 3) % 6] for k in range(12)])
    other = chart(heights[::-1][3:] + heights[:3], [pal[k % 6] for k in range(12)])
    ma, mb, mo = mask_hashes(a), mask_hashes(b), mask_hashes(other)
    assert ma is not None and mb is not None and mo is not None
    refs_d = np.array([mb[0], mo[0]], dtype=np.uint64)
    refs_p = np.array([mb[2], mo[2]], dtype=np.uint64)
    near = nearest(ma, refs_d, refs_p)
    assert is_near(near) and near["both"][1] == 0
    assert mask_hashes(Image.new("RGB", (100, 100), (128, 128, 128))) is None   # no saturated pixels: no mask hash
