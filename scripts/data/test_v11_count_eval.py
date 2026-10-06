"""Round 11 counting sets: one question per picture, dev and test split by stratum, mirrored copies are their
original picture, every project file outside raw downloads is scanned."""
import random

import pytest
from v11_count_eval import base_picture, bucket, one_per_picture, project_files, split


def row(i, answer, n_options, pic=None):
    return {"id": "q%d" % i, "image_id": "coco:%d" % (i if pic is None else pic), "answer_key": answer,
            "questions": {"q": {"criteria": {str(k): None for k in range(n_options)}}}}


def test_buckets_and_mirrored_pictures():
    assert [bucket(a) for a in ("0", "1", "2", "3", "4", "5", "10")] == ["0", "1-2", "1-2", "3-4", "3-4", "5-10", "5-10"]
    assert base_picture("coco:12:flip") == "coco:12" and base_picture("vg:3") == "vg:3"


def test_one_question_per_picture():
    rows = [row(i, "2", 3, pic=i // 3) for i in range(30)]
    got = one_per_picture(rows, random.Random(0))
    assert len(got) == 10 and len({r["image_id"] for r in got}) == 10


def test_split_sizes_strata_and_seed():
    rng = random.Random(1)
    rows = [row(i, str(rng.randint(0, 10)), rng.randint(2, 5)) for i in range(2000)]
    dev, test = split(rows, 300, 600, random.Random(5))
    assert len(dev) == 300 and len(test) == 600 and not {r["id"] for r in dev} & {r["id"] for r in test}
    share = lambda rs, b: sum(bucket(r["answer_key"]) == b for r in rs) / len(rs)
    for b in ("0", "1-2", "3-4", "5-10"):
        assert abs(share(dev, b) - share(test, b)) < 0.02
    assert split(rows, 300, 600, random.Random(5)) == (dev, test)
    with pytest.raises(SystemExit):
        split(rows[:800], 300, 600, random.Random(5))


def test_project_files_skip_raw_downloads_and_round_11(tmp_path):
    for d in ("v2", "raw/vqav2", "coco", "v11", "lv_bench"):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "x.jsonl").write_text("")
    (tmp_path / "align_full_train.jsonl").write_text("")
    got = {p[len(str(tmp_path)) + 1:] for p in project_files(str(tmp_path))}
    assert got == {"v2/x.jsonl", "lv_bench/x.jsonl", "align_full_train.jsonl"}


def test_held_files_include_round_11s_sets(tmp_path):
    from v5_build import held_files
    (tmp_path / "v11").mkdir()
    for name in ("dev_count_transfer", "test_count_fresh", "dev_count_existing"):
        (tmp_path / "v11" / (name + ".jsonl")).write_text("")
    files = {p.replace(str(tmp_path), "") for p in held_files(str(tmp_path))}
    assert files == {"/v11/dev_count_transfer.jsonl", "/v11/test_count_fresh.jsonl", "/v11/dev_count_existing.jsonl"}
