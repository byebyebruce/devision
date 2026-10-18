"""exp43 without a model: the spatial questions change only the object's name into its drawn side (options, order,
gold, picture, state_text kept; sides from the generator, never A / 1 assumed left); the transcripts give both objects'
values equal to the golds, one per picture, the same object order in both views, changing with the value twins; the
report refuses a wrong file or a transcript answered on a picture, says "incomplete" for missing sets and reads the
pre-written screens."""
import importlib.util
import json
import os
import re
from collections import defaultdict

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b = load("exp43_build", os.path.join(HERE, "build.py"))
rep = load("exp43_report", os.path.join(HERE, "report.py"))
t41 = load("t41", os.path.join(HERE, "..", "exp41", "test_exp41.py"))


def rows(name):
    return [json.loads(x) for x in open(os.path.join(DATA, "exp43", name + ".jsonl"))]


def test_spatial_changes_only_the_object_name():
    sources = {"fit_spatial_O": "exp42/fit_O", "fit_spatial_L": "exp42/fit_L", "tune_spatial_O": "real_layout/questions_tune",
               "tune_spatial_L": "exp40/rl_tune_L"}
    sides = 0
    for n, p in sources.items():
        src = {r["id"]: r for r in (json.loads(x) for x in open(os.path.join(DATA, p + ".jsonl")))}
        for r in rows(n):
            o = src[r["orig_id"]]
            assert (r["image"], r["image_id"], r["state_text"]) == (o["image"], o["image_id"], o["state_text"])
            for q, v in r["questions"].items():
                ov = o["questions"][q]
                assert {k: x for k, x in v.items() if k != "instructions"} == {k: x for k, x in ov.items() if k != "instructions"}
                assert list(v["criteria"]) == list(ov["criteria"]) and r["gold"][q] == o["gold"][q]
                t, obj = b.split_question(ov["instructions"])
                assert v["instructions"] == b.TEMPLATES[t] % r["sides"][q] and "sample " not in v["instructions"].replace("the sample", "")
                assert not re.search(r"Pair [12]|sample [AB]", v["instructions"])
                sides += r["sides"][q] == "right" and obj in ("A", "1")
    assert sides > 0                                   # A / Pair 1 drawn on the right does occur


def test_side_follows_the_layout_flag():
    assert b.side("temp", {"a_left": True}, "A") == "left" and b.side("temp", {"a_left": False}, "A") == "right"
    assert b.side("temp", {"a_left": False}, "B") == "left" and b.side("force", {"p1_left": False}, "1") == "right"


def test_transcripts_carry_both_values_one_per_picture():
    gold = lambda g: next(o for o, p in g["probabilities"].items() if p == 1.0)  # noqa: E731
    for n in ("fit_transcript", "tune_transcript"):
        per_picture, order = defaultdict(set), {}
        for r in rows(n):
            text = r["state_text"].split("\n", 1)[1]
            per_picture[(r["family"], r["view"])].add(text)
            assert r["image"] == b.BLANK and "answer" not in text.lower()
            objs = re.findall(r"(Sample [AB]|Pair [12]):", text)
            assert len(objs) == 2 and len(set(objs)) == 2                      # both objects, never only the asked one
            order.setdefault(r["family"], set()).add(tuple(objs))
            for q, v in r["questions"].items():
                name = "Sample " + r["objects"][q][-1] if r["kind"] == "temp" else r["objects"][q]
                parts = dict(re.findall(r"(Sample [AB]|Pair [12]): (.*?)(?= (?:Sample [AB]|Pair [12]):|$)", text))
                g = gold(r["gold"][q])
                want = ("mass of each particle %s" % g if "mass" in v["instructions"] else "average particle speed %s" % g
                        if "speed" in v["instructions"] else "the magnets are %s apart" % g)
                assert want in parts[name]                                       # the asked object's own value
        assert all(len(v) == 1 for v in per_picture.values())               # one transcript per picture
        assert all(len(v) == 1 for v in order.values())                     # same object order in both views
    firsts = {o for r in rows("tune_transcript") for o in re.findall(r"^(Sample [AB]|Pair [12])", r["state_text"].split("\n", 1)[1])}
    assert firsts == {"Sample A", "Sample B", "Pair 1", "Pair 2"}            # not always A / Pair 1 first


def fake(tmp_path, monkeypatch, right=lambda n, r, q: True, image_used=None, missing=()):
    """A frozen fake exp42-C with name records, and exp43 records for the new sets."""
    fz = rep.r41.frozen
    c = tmp_path / "exp42-c"
    (c / "stage2").mkdir(parents=True)
    (c / "eval").mkdir()
    (c / "stage2" / "model.safetensors").write_bytes(b"c")
    (c / "stage2" / "config.json").write_text("{}")
    out = tmp_path / "exp43"
    (out / "eval").mkdir(parents=True)

    def write(run, n, path, blank):
        data = [json.loads(x) for x in open(os.path.join(ROOT, path))]
        recs = t41.fake_details(data, lambda r, q: not right(n, r, q))
        for x in recs:
            x["image_used"] = image_used or (b.BLANK if blank else "x.png")
        open(run / "eval" / (n + ".details.jsonl"), "w").writelines(json.dumps(x) + "\n" for x in recs)
        json.dump({"run": {"checkpoint": str(c / "stage2"), "temperature": "none (raw)", "control": "none",
                           "data_sha256": rep.r41.sha(os.path.join(ROOT, path))}}, open(run / "eval" / (n + ".json"), "w"))
    fz.freeze(str(c))
    for n, p in rep.NAME.items():
        write(c, n, p, False)
        fz.record(str(c), n)
    status = []
    for n, p in rep.NEW.items():
        if n not in missing:
            write(out, n, p, "transcript" in n)
            status.append("eval:%s completed" % n)
    (tmp_path / "exp43-status.txt").write_text("\n".join(status) + "\n")
    w = fz.weights(str(c))
    monkeypatch.setattr(rep, "C_RUN", str(c))
    monkeypatch.setattr(rep, "CKPT", str(c / "stage2"))
    monkeypatch.setattr(rep, "C_SHA", w)
    os.chdir(ROOT)
    return str(out)


def test_report_screens_and_pattern(tmp_path, monkeypatch):
    spatial_only = lambda n, r, q: "spatial" in n or "transcript" in n  # noqa: E731
    out = fake(tmp_path, monkeypatch, right=spatial_only)
    res = rep.build(out)
    assert res["missing"] == [] and res["screens"] == {"text_usable": True, "spatial_rescue_O": True, "spatial_rescue_L": True}
    assert res["pattern"] == 1
    t = res["transitions"]["tune_spatial_O-rl_tune"]
    assert t["wrong_to_right"] == t["questions"] == 828 and t["right_to_wrong"] == 0
    assert res["readout"]["rl_tune"]["both:speed"]["of"] == 191


def test_report_refuses_pictures_for_transcripts_and_reports_missing(tmp_path, monkeypatch):
    out = fake(tmp_path / "a", monkeypatch, image_used="real_layout/images/rl00003_v0.png")
    with pytest.raises(SystemExit):
        rep.build(out)
    out = fake(tmp_path / "b", monkeypatch, missing=("tune_transcript",))
    res = rep.build(out)
    assert res["missing"] == ["tune_transcript"] and res["screens"]["text_usable"] == "incomplete" and res["pattern"] == "incomplete"
