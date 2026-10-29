"""Board exclusion and shared sample rules: a board image is caught as exact bytes, as re-encoded pixels and as a
resized copy, an unrelated image is not; gold is unique; dev / train never share an image; baselines are image-blind."""
import io
import json

from v05_board_common import (BoardIndex, choice_sample, gold_text, image_hashes, place_gold, text_only, write_split)


def picture(seed: int, size=(320, 240), fmt="PNG", quality=90) -> bytes:
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (6, 8, 3), dtype=np.uint8)   # blocky picture: survives resizing like a photo
    im = Image.fromarray(base).resize(size, Image.Resampling.BILINEAR)
    b = io.BytesIO()
    im.save(b, fmt, **({"quality": quality} if fmt == "JPEG" else {}))
    return b.getvalue()


def index_of(*images: bytes, ids=()):
    return BoardIndex({"images": [{"path": "full/full_x/img/%d.png" % i, **image_hashes(b)}
                                  for i, b in enumerate(images)], "source_ids": list(ids)})


def test_board_image_is_caught_exactly_reencoded_and_resized():
    board = picture(1)
    idx = index_of(board, picture(2), ids=["coco:42"])
    assert (idx.hit(board) or ("",))[0] == "sha256"
    from PIL import Image
    im = Image.open(io.BytesIO(board))
    b = io.BytesIO()
    im.save(b, "BMP")
    assert (idx.hit(b.getvalue()) or ("",))[0] == "pixels"
    b = io.BytesIO()
    im.resize((640, 480)).save(b, "JPEG", quality=80)
    assert (idx.hit(b.getvalue()) or ("",))[0] == "near"
    assert idx.hit(picture(3)) is None
    assert idx.hit_id("coco:42") and not idx.hit_id("coco:43")


def test_choice_sample_has_one_gold_and_rejects_duplicate_options():
    s = choice_sample("x", "src", "img:1", "a.jpg", "Q?", place_gold("yes it is", ["no", "maybe"], 2), "yes it is")
    assert list(s["questions"]["q"]["criteria"]) == ["A", "B", "C"]
    assert gold_text(s) == "yes it is" and sum(s["gold"]["q"]["probabilities"].values()) == 1.0
    try:
        choice_sample("x", "src", "img:1", "a.jpg", "Q?", ["Same", "same"], "Same")
    except AssertionError:
        pass
    else:
        raise AssertionError("duplicate options accepted")


def test_dev_and_train_never_share_an_image(tmp_path):
    rows = [choice_sample("q%d:%d" % (i, k), "s", "img:%d" % i, "a.jpg", "Q?", ["a", "b"], "a")
            for i in range(300) for k in range(3)]
    counts = write_split(str(tmp_path), rows, dev_frac=0.2)
    split = {sp: [json.loads(l) for l in open(tmp_path / (sp + ".jsonl"))] for sp in ("train", "dev")}
    assert counts["dev"] > 0 and counts["train"] > counts["dev"]
    assert not {r["image_id"] for r in split["train"]} & {r["image_id"] for r in split["dev"]}


def test_text_only_prior_does_not_memorise_unique_options():
    rows = [choice_sample("q%d" % i, "s", "i%d" % i, "a.jpg", "Which caption?", place_gold("true %d" % i,
                          ["false %d" % i], i % 2), "true %d" % i) for i in range(100)]
    t = text_only(rows)
    assert t["question_prior"] == t["first"] == 0.5 and t["gold_position"] == {"A": 50, "B": 50}
    yes = [choice_sample("y%d" % i, "s", "i%d" % i, "a.jpg", 'Is "f%d" filled?' % i, ["Yes", "No"],
                         "Yes" if i < 70 else "No") for i in range(100)]
    assert text_only(yes)["question_prior"] == 0.7   # a real answer prior is still measured
