"""bind_report refuses incomplete or mismatched predictions (critic R13-I05) and its gate is attainable by a
model that is right and not by a name-blind policy (R13-I04)."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "data", "r13"))
import bind_report as br  # noqa: E402
import synth  # noqa: E402


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("d")
    synth.build_split(str(root), "dev", 12)
    path = root / "r13" / "synth-v2" / "dev_bind_cf.jsonl"
    return str(path), br.rows(str(path))


def details(samples, pick):
    out = []
    for s in samples:
        for qid in s["questions"]:
            gold = s["gold"][qid]["probabilities"]
            out.append({"sample_id": s["id"], "qid": qid, "prediction": pick(s, qid), "gold": gold, "correct": None})
    return out


def write(tmp_path, name, rows_, sha=None):
    p = tmp_path / (name + ".details.jsonl")
    p.write_text("".join(json.dumps(r) + "\n" for r in rows_))
    if sha:
        (tmp_path / (name + ".json")).write_text(json.dumps({"run": {"data_sha256": sha}}))
    return str(p)


right = lambda s, q: br.answer(s["gold"][q])


def test_the_gate_passes_a_right_model_and_fails_both_name_blind_policies(data, tmp_path):
    path, samples = data
    r = br.report(path, [write(tmp_path, "right", details(samples, right))])
    assert r["models"][0]["gate"]["passes"] and r["models"][0]["summary"]["binding"]["accuracy"] == 1.0
    for policy in ("salient", "minor"):
        assert not r["policy_" + policy]["gate"]["passes"]
        assert r["policy_" + policy]["summary"]["binding"]["accuracy"] < 0.8


def test_missing_duplicate_extra_or_other_version_predictions_are_refused(data, tmp_path):
    path, samples = data
    full = details(samples, right)
    bad = {"missing": full[1:], "duplicate": full + full[:1],
           "extra": full + [dict(full[0], sample_id="nope")],
           "not_an_option": [dict(full[0], prediction="maybe")] + full[1:]}
    flipped = dict(full[0])
    flipped["gold"] = {k: 1.0 - v for k, v in flipped["gold"].items()}
    bad["other_gold"] = [flipped] + full[1:]
    for name, rows_ in bad.items():
        with pytest.raises(br.ReportError):
            br.report(path, [write(tmp_path, name, rows_)])
    with pytest.raises(br.ReportError, match="ran on data"):
        br.report(path, [write(tmp_path, "sha", full, sha="0" * 64)])
    # the critic's example: deleting one wrong answer must not raise binding -- it is refused instead
    wrong_one = [dict(r, prediction=[o for o in samples[0]["questions"][r["qid"]]["criteria"] if o != br.answer({"probabilities": r["gold"]})][0])
                 if r["sample_id"] == samples[0]["id"] and r["qid"].startswith("colour_") else r for r in full[:9]] + full[9:]
    br.report(path, [write(tmp_path, "one_wrong", wrong_one)])
    with pytest.raises(br.ReportError):
        br.report(path, [write(tmp_path, "one_deleted", [r for r in wrong_one if r is not wrong_one[0] or not r["qid"].startswith("colour_")])])


def test_paired_difference_matches_a_hand_count():
    fa = {"f1": {"binding": [1, 2]}, "f2": {"binding": [0, 2]}}
    fb = {"f1": {"binding": [2, 2]}, "f2": {"binding": [1, 2]}}
    fa = {g: dict(v) for g, v in fa.items()}
    r = br.paired(fa, fb, "binding")
    assert r["difference"] == 0.5 and r["units"] == 4      # (3 - 1) / 4
    with pytest.raises(br.ReportError):
        br.paired(fa, {"f1": fb["f1"]}, "binding")
    with pytest.raises(br.ReportError):
        br.paired(fa, {"f1": {"binding": [2, 3]}, "f2": fb["f2"]}, "binding")


def test_two_complete_models_with_different_error_counts_are_compared(data, tmp_path):
    """Critic R13-I06: B answers one colour question with the salient kind's colour; both files are complete, so
    the full report runs, pairs the fixed-unit metrics, and keeps wrong_is_salient per model only."""
    path, samples = data
    s0 = samples[0]
    q = next(q for q, m in s0["meta"].items() if m["task"] == "colour" and m["salient"] != br.answer(s0["gold"][q]))
    wrong = lambda s, qid: s["meta"][qid]["salient"] if (s["id"], qid) == (s0["id"], q) else right(s, qid)
    a = write(tmp_path, "a", details(samples, right))
    b = write(tmp_path, "b", details(samples, wrong))
    r = br.report(path, [a, b])
    vs = r["models"][1]["vs_baseline"]
    assert "wrong_is_salient" not in vs
    assert vs["binding"]["difference"] < 0 and vs["task:colour"]["difference"] < 0
    assert r["models"][0]["summary"].get("wrong_is_salient") is None
    assert r["models"][1]["summary"]["wrong_is_salient"] == {"accuracy": 1.0, "n": 1}
    with pytest.raises(br.ReportError):                       # the coverage guard still holds
        br.report(path, [a, write(tmp_path, "b_short", details(samples, wrong)[1:])])
