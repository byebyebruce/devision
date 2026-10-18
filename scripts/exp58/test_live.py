"""exp58 runner with fake objects, no model: every request reaches Reading.decide with the actual picture as image= (never
None, never inside state); a *_mism.jsonl row sends its fixed wrong picture; inside the real Reading.decide a supported
question is decided on the text state + table (image=None after the read) and an unsupported one goes through ordinary
decide WITH the picture; the placeholder endpoint refuses start before anything is written."""
import importlib.util
import json
import os

import pytest
from PIL import Image

HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("exp58_live", os.path.join(HERE, "live.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

TEMP_Q = "Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature?"


def pictures(root):
    (root / "img").mkdir(parents=True)
    Image.new("RGB", (40, 30), (255, 0, 0)).save(root / "img" / "red.png")
    Image.new("RGB", (40, 30), (0, 0, 255)).save(root / "img" / "blue.png")


def row(i, image, text):
    return {"id": "s%d" % i, "image_id": "img:%d" % i, "image": image, "state_text": "hint",
            "questions": {"q": {"type": "choice", "instructions": text, "criteria": {"sample A": None, "sample B": None, "neither": None}}},
            "gold": {"q": {"probabilities": {"sample A": 1.0, "sample B": 0.0, "neither": 0.0}}}}


class FakeReading:
    def __init__(self):
        self.calls = []

    def decide(self, state, questions, image=None, diagnostics=None):
        self.calls.append((state, image.getpixel((0, 0)) if image is not None else None))
        return {"answers": {k: {"type": "choice", "probabilities": {"sample A": 0.6, "sample B": 0.3, "neither": 0.1}} for k in questions}}


def test_actual_and_fixed_wrong_pictures_reach_reading_as_image(tmp_path):
    pictures(tmp_path)
    true = [row(0, "img/red.png", TEMP_Q)]
    mism = [row(0, "img/blue.png", TEMP_Q)]                  # *_mism.jsonl: same question, the fixed wrong picture
    fr = FakeReading()
    a = m.run_set(fr, true, str(tmp_path))
    b = m.run_set(fr, mism, str(tmp_path))
    assert fr.calls == [("hint", (255, 0, 0)), ("hint", (0, 0, 255))]
    assert a[0]["routed"] and a[0]["image_used"] == "img/red.png" and b[0]["image_used"] == "img/blue.png" and a[0]["correct"]


def test_real_reading_routes_supported_with_table_and_others_with_picture(tmp_path):
    from devision.reading import Reading
    pictures(tmp_path)
    seen = []

    class Dec:
        def decide(self, state, questions, image=None):
            seen.append((state, sorted(questions), image is not None))
            return {"answers": {k: {"type": "choice", "probabilities": {o: 1 / len(q["criteria"]) for o in q["criteria"]}} for k, q in questions.items()},
                    "usage": {"input_tokens": 1}}
    r = object.__new__(Reading)
    r.decider = Dec()
    reads = []
    r.read = lambda image, templates: (reads.append(image is not None), {"table": [["Sample A", "mass of each particle", "14"]],
                                                                       "reads": [], "seconds": 0.0})[1]
    other = {"type": "choice", "instructions": "Which animal is shown?", "criteria": {"dog": None, "cat": None}}
    s = [row(0, "img/red.png", TEMP_Q)]
    s[0]["questions"]["o"] = other
    s[0]["gold"]["o"] = {"probabilities": {"dog": 1.0, "cat": 0.0}}
    m.run_set(r, s, str(tmp_path))
    assert reads == [True]
    assert seen[0][2] is False and "Values read from the picture" in seen[0][0] and seen[0][1] == ["0:q"]    # table, no picture
    assert seen[1] == ("hint", ["0:o"], True)                                                                  # ordinary, with picture
    seen.clear()
    m.run_set(r, [dict(row(1, "img/blue.png", "Which animal is shown?"), questions={"o": other}, gold={"o": s[0]["gold"]["o"]})], str(tmp_path))
    assert seen == [("hint", ["0:o"], True)] and reads == [True]                                              # nothing routed: no read


def test_placeholder_endpoint_refuses_start(tmp_path):
    assert m.NEW_SHA is None
    with pytest.raises(SystemExit, match="placeholder"):
        m.stage_start(0, out=str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_grouping_is_devision_eval_s(tmp_path):
    from devision.train.evaluate import _requests
    rows = [row(i, "img/red.png" if i < 3 else "img/blue.png", TEMP_Q) for i in range(5)]
    assert m.requests(rows) == _requests(rows, {}, "none", 32) and len(m.requests(rows)) == 2
    json.dumps(m.requests(rows))
