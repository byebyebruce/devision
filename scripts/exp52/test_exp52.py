"""exp52 scripts without the real model: the number sidecar covers every training question once with a strict target;
the readout runs the ordinary forward on a tiny reader checkpoint over real pictures and writes EXP49-format details
whose donor is the twin's own prediction, plus with / without content logits."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tests"))


def test_sidecar_covers_training_once():
    side = [json.loads(x) for x in open(os.path.join(ROOT, "data", "exp52", "numbers.jsonl"))]
    train = [json.loads(x) for x in open(os.path.join(ROOT, "data", "exp42", "train.jsonl"))]
    assert len(side) == sum(len(r["questions"]) for r in train) == len({(s["id"], s["qid"]) for s in side}) == 4608
    gold = {(r["id"], q): next(o for o, p in r["gold"][q]["probabilities"].items() if p == 1.0) for r in train for q in r["questions"]}
    for s in side:
        assert gold[(s["id"], s["qid"])].split(" ")[0].replace(",", "") == s["target"]


def test_readout_smoke(tmp_path):
    import torch
    from conftest import tiny_decider  # type: ignore
    d = tiny_decider(seed=4)
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    d.model.enable_local_views(12)
    d.cfg.local_views, d.cfg.local_overlap, d.cfg.local_source = 12, 0.25, "T"
    torch.manual_seed(5)
    d.model.add_number_reader()
    d.cfg.number_reader = True
    run = tmp_path / "arm"
    d.save(run / "stage2")
    p = subprocess.run([sys.executable, os.path.join(HERE, "readout.py"), "--run", str(run), "--device", "cpu", "--families", "2"],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    lines = [json.loads(x) for x in open(run / "eval" / "readout_details.jsonl")]
    rows = lines[1:]
    assert lines[0]["identity"]["path"].startswith("rlcd._forward") and rows
    by = {r["key"]: r for r in rows}
    qs = {q["key"]: q for q in map(json.loads, open(os.path.join(ROOT, "data", "exp49", "questions.jsonl"))) if q["key"] in by}
    for k, r in by.items():
        assert len(r["pred"]) == 5
        if qs[k]["twin_changed"] and qs[k]["twin"] in by:
            assert r["donor"] == by[qs[k]["twin"]]["pred"]                       # the same question on the twin picture
    use = [json.loads(x) for x in open(run / "eval" / "readout_use.jsonl")][1:]
    assert len(use) == len(rows) and all(len(u["on"]) == len(u["off"]) >= 2 for u in use)
    assert any(u["on"] != u["off"] for u in use)                                 # the content is read by the scoring
