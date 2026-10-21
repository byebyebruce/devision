import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_ext import choice_row, cv_choice, naturalbench_choice, strip_yesno, yes_no  # noqa: E402


def test_repeated_option_texts_drop_the_question():
    assert choice_row("x", "s", "k", "i.png", "Q?", ["a", "a"], 0) is None
    r = choice_row("x", "s", "k", "i.png", "Q?", ["a", "b"], 1, state_text="hint")
    assert r["gold"]["q"]["probabilities"] == {"a": 0.0, "b": 1.0} and r["state_text"] == "hint"


def test_yes_no_tail_is_removed_and_answers_are_parsed():
    assert strip_yesno("Is there a dog? Please answer yes or no.") == "Is there a dog?"
    assert strip_yesno("Is there a dog? Answer with yes or no") == "Is there a dog?"
    assert yes_no("Yes") is True and yes_no("no.") is False and yes_no("maybe") is None


def test_cvbench_letter_answer_points_into_the_choice_list():
    opts, gold = cv_choice("['3', '2', '1', '0']", "(C)")
    assert opts == ["3", "2", "1", "0"] and gold == 2


def test_naturalbench_options_are_read_from_the_question():
    r = naturalbench_choice("n", "i.png", "What colour is the car?\nOption: A:red; B:blue;", "B")
    assert r is not None
    assert list(r["questions"]["q"]["criteria"]) == ["red", "blue"]
    assert r["questions"]["q"]["instructions"] == "What colour is the car?"
    assert r["answer_key"] == "blue"
    assert naturalbench_choice("n", "i.png", "No options here?", "A") is None


def test_blind_gate_drops_choices_the_wrong_picture_still_answers():
    from blind import blind_ids
    d = lambda sid, p, kind="v7w", opts=("a", "b", "c", "d"): {  # noqa: E731
        "sample_id": sid, "source": "v7w", "kind": kind, "type": "choice", "options": list(opts),
        "gold": {o: float(o == "a") for o in opts}, "probabilities": {o: (p if o == "a" else (1 - p) / 3) for o in opts}}
    ids, seen = blind_ids([d("x", 0.95), d("y", 0.5), d("z", 0.99, kind="count"), d("w", 0.99, opts=("a", "b"))], 0.9)
    assert ids == ["x"] and seen["v7w"] == 2


def test_n2_own_dev_splits_are_not_held_out(tmp_path):
    import held
    (tmp_path / "n2").mkdir()
    for name in ("dev_v7w", "dev_sqa", "test_count_coco"):
        (tmp_path / "n2" / (name + ".jsonl")).write_text("")
    names = {os.path.basename(f) for f in held.eval_files(str(tmp_path))}
    assert names == {"dev_sqa.jsonl", "test_count_coco.jsonl"}


def test_drawn_pictures_are_held_out_only_by_identical_pixels(tmp_path):
    from PIL import Image
    import held
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    im = Image.new("RGB", (64, 64), "white")
    im.save(a)
    im.putpixel((5, 5), (250, 250, 250))   # almost the same picture: same dHash, different pixels
    im.save(b)
    assert held.dhash_file(str(a)) == held.dhash_file(str(b))
    assert held.pixel_hash_file(str(a)) != held.pixel_hash_file(str(b))
